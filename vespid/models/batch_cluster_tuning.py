from argparse import ArgumentParser
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import optuna
import umap
from tqdm import tqdm

from vespid import setup_logger, get_secure_key
from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler
)
from vespid.models.clustering import HdbscanEstimator
from vespid.models.mlflow_tools import setup_mlflow
from vespid.data.aws import test_s3_access
from vespid.models.optuna import Hyperparameter, Criterion, Objectives
from vespid.models.static_communities import hydrate_best_optuna_solution

logger = setup_logger(module_name=__name__)


def get_embeddings(
    year,
    db_ip=None,
    db_username='neo4j',
    db_password_secret=None,
    db_password=None,
    connection_handler=None,
    drivers=None,
    to_list=True,
    batch_size=None
):
    '''
    Gets embeddings of papers from a given year from Neo4j.

    Parameters
    ----------
    year : int
        The year of papers from which to grab embeddings
    db_ip : str, optional
        The IP address of the database, by default None
    db_username : str, optional
        The username to use for connecting to Neo4j, by default 'neo4j'
    db_password_secret : str, optional
        The name of the database password secret stored in AWS Secrets Manager, 
        by default None
    db_password : str, optional
        If `db_password_secret` is not provided, this must be, by default None
    connection_handler : Neo4jConnectionHandler object, optional
        If desired, a pre-made connection to the target Neo4j database can be
        provided to avoid constructing one here, by default None
    drivers : str or iterable of str, optional
        Driver(s) to use when querying Neo4j. If None, will try each of the 
        possible drivers in vespid.data.neo4j_tools.Neo4jConnectionHandler in 
        a pre-defined order until one succeeds or they are exhausted, 
        by default None
    to_list : bool, optional
        Whether to return numpy array (True) or a pandas dataframe (False), 
        by default True
    batch_size : int, optional
        For particularly large datasets, a batch size (measured in number of 
        records/embeddings) may be set to ensure the Neo4j server doesn't 
        queue too much up in memory before sending it to the client 
        application. If None, no batching will be done, by default None

    Returns
    -------
    numpay array or pandas DataFrame
        Returns the embeddings in a format dictated by the `to_list` kwarg
    '''
    if connection_handler is None:
        if db_password_secret is not None:
            db_password = get_secure_key(
                db_password_secret,
                aws_secret=True,
                bypass_safety_check=True
            )[db_username]
        graph = Neo4jConnectionHandler(
            db_ip,
            db_username=db_username,
            db_password=db_password
        )
    else:
        graph = connection_handler

    # The publicationDate and semanticScholarID non-null *look* like overkill, 
    # but they're not. They leverage indexes we've generated for both that 
    # will let us do this query quicker (can't have an embedding if you don't 
    # have a semanticScholarID and can't have a publication year if you don't 
    # have a publication date!)
    
    # NOTE: may be expensive, but sorting is needed so we can use rehydrated models consistently
    query = f"""
    MATCH (n:Publication)
    WHERE n.publicationDate IS NOT NULL
    AND n.semanticScholarID IS NOT NULL
    AND n.publicationDate.year = {year}
    AND n.embedding IS NOT NULL
    RETURN n.embedding AS embedding
    ORDER BY n.id ASC
    """
    if to_list:
        return np.array(
            graph.cypher_query_to_dataframe(query, drivers=drivers)['embedding'].tolist()
        )
    else:
        return graph.cypher_query_to_dataframe(query, drivers=drivers)


def objective(
    trial,
    features,
    criteria,
    mlflow_experiment,
    mlflow_client,
    experiments_per_trial=None,
    umap_n_neighbors=None,
    umap_n_components=None,
    umap_min_dist=None,
    umap_metric=None,
    hdbscan_min_cluster_size=None,
    hdbscan_min_samples=None,
    hdbscan_cluster_selection_method=None
):
    '''
    Wraps the entire model building process into a single function
    that can be called every time a new hyperparameter permutation
    is attempted.
    '''
    if experiments_per_trial.constant < 1:
        return ValueError(f"Got {experiments_per_trial} for "
                            "`experiments_per_trial`. "
                            "Must be an integer >= 1.")
        
    # Suggest values for the hyperparameters using a trial object.
    # Options for suggest_* are:
        # suggest_float(name, low, high, step=1, log=False): uniform sampling of floats in range [low, high) that can be weighted towards lower values via log=True
        # suggest_int(name, low, high, step=1, log=False): same as preceding, but for integers
        # suggest_uniform(name, low, high): good for uniform sampling of floats as it is treated as fully continuous (not contrained to ``step`` args as in preceding two)
        # suggest_loguniform(name, low, high): same as preceding but effectively selects values towards the low end of the distro more often
        # suggest_discrete_uniform(): similar to suggest_int, but arguably less explicit about what it does
        # suggest_categorical(): as indicated, picks among a list of categorical options uniformly
       
    #FIXME: stop pushing experiment tags to runs once experiment-level tags are available in MLFlow UI 
    run = mlflow_client.create_run(
        mlflow_experiment.experiment_id,
        tags=mlflow_experiment.tags
    )
    run_id = run.info.run_id
    
    experiments_per_trial = experiments_per_trial.constant
    
    umap_n_neighbors = trial.suggest_int(
        umap_n_neighbors.name,
        umap_n_neighbors.min,
        umap_n_neighbors.max,
        log=False
    )
    umap_n_components = trial.suggest_int(
        umap_n_components.name,
        umap_n_components.min,
        umap_n_components.max,
        log=False
    )
    umap_metric = trial.suggest_categorical(
        umap_metric.name,
        umap_metric.categories
    )
    umap_min_dist = trial.suggest_uniform(
        umap_min_dist.name,
        umap_min_dist.min,
        umap_min_dist.max
    )

    hdbscan_min_cluster_size = trial.suggest_int(
        hdbscan_min_cluster_size.name,
        hdbscan_min_cluster_size.min,
        hdbscan_min_cluster_size.max,
        log=False
    )  # Keeping lower minimizes DBCV cross-fold stddev
    
    hdbscan_min_samples = trial.suggest_int(
        hdbscan_min_samples.name,
        hdbscan_min_samples.min,
        hdbscan_min_samples.max,
        log=False
    )  # Keeping higher minimizes DBCV cross-fold stddev
    
    hdbscan_cluster_selection_method = trial.suggest_categorical(
        hdbscan_cluster_selection_method.name,
        # EOM = better use of persistence and variable density clusters
        # leaf = more likely to get finely resolved, 
        #   more homogeneous clusters
        hdbscan_cluster_selection_method.categories
    )
    
    # Make sure we log all of these selected hyperparameters to mlflow
    params = {
        'experiments_per_trial': experiments_per_trial,
        'umap_n_neighbors': umap_n_neighbors,
        'umap_n_components': umap_n_components,
        'umap_metric': umap_metric,
        'umap_min_dist': umap_min_dist,
        'hdbscan_min_cluster_size': hdbscan_min_cluster_size,
        'hdbscan_min_samples': hdbscan_min_samples,
        'hdbscan_cluster_selection_method': hdbscan_cluster_selection_method
    }
    for name, value in params.items():
        mlflow_client.log_param(run_id, name, value)

    # Keep track of metrics across repeat runs
    dbcvs = []
    num_clusters_list = []
    mean_cluster_persistences = []
    std_cluster_persistences = []

    for i in range(experiments_per_trial):
        # Setup the models using the selected hyperparameters
        dim_reducer = umap.UMAP(
            n_jobs=1,
            n_neighbors=umap_n_neighbors,
            n_components=umap_n_components,
            metric=umap_metric,
            min_dist=umap_min_dist
        )

        clusterer = HdbscanEstimator(
            gen_min_span_tree=True,
            prediction_data=True,
            algorithm='boruvka_kdtree',
            core_dist_n_jobs=1,
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            cluster_selection_method=hdbscan_cluster_selection_method
        )

        pipe = Pipeline(
            steps=[('umap', dim_reducer), ('hdbscan', clusterer)])
        pipe.fit(features, hdbscan__soft_cluster=True)

        #TODO: remove this extra debug code
        reducer = pipe.named_steps['umap']
        cluster_model = pipe.named_steps['hdbscan']

        # Score the results
        dbcvs.append(cluster_model.relative_validity_)
        logger.debug(f"dbcv = {cluster_model.relative_validity_}")

        num_clusters_list.append(np.unique(cluster_model.labels_).shape[0])
        logger.debug(
            f"num_clusters = {np.unique(cluster_model.labels_).shape[0]}")

        mean_cluster_persistences.append(
            cluster_model.cluster_persistence_.mean())  # Maximize
        logger.debug(
            f"mean_cluster_persistence = {cluster_model.cluster_persistence_.mean()}")

        std_cluster_persistences.append(
            cluster_model.cluster_persistence_.std())  # Minimize
        logger.debug(
            f"std_cluster_persistence = {cluster_model.cluster_persistence_.std()}\n")

    # Use metrics across multiple identical runs to come up 
    # with an "average run"
    dbcvs = pd.Series(dbcvs).dropna().values
    num_clusters_list = pd.Series(num_clusters_list).dropna().values
    mean_cluster_persistences = pd.Series(
        mean_cluster_persistences).dropna().values
    std_cluster_persistences = pd.Series(
        std_cluster_persistences).dropna().values

    if experiments_per_trial > 1:
        dbcv_mean = dbcvs.mean()  # Maximize
        dbcv_std = dbcvs.std()  # Minimize
        num_clusters_mean = num_clusters_list.mean()  # Maximize
        num_clusters_std = num_clusters_list.std()  # Minimize
        mean_cluster_persistences_mean = mean_cluster_persistences.mean()  # Maximize
        # Minimize, may be redundant with DBCV tho
        #mean_cluster_persistences_std = mean_cluster_persistences.std()
        #std_cluster_persistences_mean = std_cluster_persistences.mean()  # Minimize

    else:
        dbcv_mean = dbcvs[0]  # Maximize
        dbcv_std = 0  # Minimize
        num_clusters_mean = num_clusters_list[0]  # Maximize
        num_clusters_std = 0  # Minimize
        # Maximize
        mean_cluster_persistences_mean = mean_cluster_persistences[0]
        # mean_cluster_persistences_std = 0 # Minimize, may be redundant with DBCV tho
        # std_cluster_persistences_mean = std_cluster_persistences[0] # Minimize
        
    output = tuple([
        dbcv_mean,
        dbcv_std,
        num_clusters_mean,
        num_clusters_std,            
        mean_cluster_persistences_mean,
        # mean_cluster_persistences_std,
        # std_cluster_persistences_mean
    ])
    
    #TODO: consider logging metrics in a nested fashion at the individual iteration level for full data in addition to the means, etc.
    # Track results of trial to mlflow
    for name, value in zip(criteria, output):
        mlflow_client.log_metric(run_id, name, value)
    
    # End the run
    mlflow_client.set_terminated(run_id)

    return output

#TODO: convert to optuna's database-based parallelization approach (or something with ray maybe?)
def bayesian_tune(
    experiment_parameters,
    n_trials,
    garbage_collect=True,
    n_jobs=1,
    study_name=None
):
    '''
    Takes ranges of values for different hyperparameters of our
    modeling pipeline and tries each one out in an intelligent
    optimization-driven fashion to focus in on the hyperparameter
    permutations that do the best job meeting our objectives.


    Parameters
    ----------
    experiment_parameters: vespid.models.optuna.Objectives object. Defines 
        the hyperparamter types and ranges to be tried, how to fit and score
        the model(s), and the optimization criteria.
            
    n_trials: Indicates how many different hyperparameter
        permutations to test.

    garbage_collect: bool. If True, indicates that you want
        optuna to garbage collect after every trial is run
        (more time-intensive but less likely to run out of 
        memory).

    n_jobs: Indicates how many parallel tuning trials
        can be run.
        
    mlflow_experiment: mlflow.entities.experiment.Experiment object.
        If provided, indicates the experiment under which the optimization
        trials should be logged. Will supersede `study_name` if that is also
        set.
        
    study_name: str. If provided, indicates to optuna the name of the study
        it should use. Only used if `mlflow_experiment` is None.


    Returns
    -------
    optuna Study object representing the completed tuning trials.
    '''
    
    if experiment_parameters.experiment is not None and study_name is not None:
        raise UserWarning("Both `mlflow_experiment` and `study_name` were "
                         "set; defaulting to using `mlflow_experiment`")
    
    # Create a study object and optimize the objective function.
    if experiment_parameters.experiment is not None:
        study = optuna.create_study(
            study_name=experiment_parameters.experiment.name,
            directions=experiment_parameters.get_criteria_directions()
        )
    elif study_name is not None:
        study = optuna.create_study(
            study_name=study_name,
            directions=experiment_parameters.get_criteria_directions()
        )
    else:
        raise ValueError("Both `mlflow_experiment` and `study_name` "
                         "cannot be set to None")

    study.optimize(
        experiment_parameters, 
        n_trials=n_trials, 
        n_jobs=n_jobs,
        gc_after_trial=garbage_collect,
        show_progress_bar=False
    )

    return study

if __name__ == '__main__':
    parser = ArgumentParser(description='Bayesian tuning of cluster solutions')
    parser.add_argument('experiment_name', 
                        type=str,
                        help='Name of a new or existing MLFlow experiment '
                        'to log results to. Also serves as optuna study name.'
                        ' Note that it will automatically have " - <year>" '
                        'added to it for every year of tuning done'
                        )
    parser.add_argument('project', type=str,
                        help='Name of project to use in MLFlow tags')
    parser.add_argument('dataset', type=str,
                        help='Name of dataset to use in MLFlow tags')
    
    parser.add_argument('--n_trials', type=int, default=100,
                        help='number of parameter combinations to test')
    parser.add_argument('--n_jobs', type=int, default=25,
                        help='# parallel jobs in search')
    parser.add_argument('--cv', type=int, default=5,
                        help='# of times to fit a cluster solution to ensure '
                        'robustness')
    
    parser.add_argument('--n_hdb_jobs', type=int, default=1,
                        help='# of parallel jobs used by hdbscan')
    parser.add_argument('--n_umap_jobs', type=int, default=1,
                        help='# of parallel jobs used by umap')
    
    parser.add_argument('--min_mcs', type=int, default=10,
                        help='min for HDBSCAN min cluster size search')
    parser.add_argument('--max_mcs', type=int, default=500,
                        help='max for HDBSCAN min cluster size search')
    parser.add_argument('--min_ms', type=int, default=5,
                        help='min for HDBSCAN min samples search')
    parser.add_argument('--max_ms', type=int, default=100,
                        help='max for HDBSCAN min samples search')
    parser.add_argument('--min_nc', type=int, default=5,
                        help='min for UMAP components search')
    parser.add_argument('--max_nc', type=int, default=25,
                        help='max for UMAP components search') # possibly reduce further
    parser.add_argument('--min_nn', type=int, default=5,
                        help='min for UMAP n_neighbors search')
    parser.add_argument('--max_nn', type=int, default=50,
                        help='max for UMAP n_neighbors search')
    parser.add_argument('--min_min_dist', type=float, default=0.01,
                        help='min for UMAP min_dist search')
    parser.add_argument('--max_min_dist', type=float, default=0.09,
                        help='max for UMAP min_dist search')
    parser.add_argument('--cluster_method', type=str, 
                        default=None, action='append',
                        help='Cluster selection methods to use. Allowed '
                        'values are ["eom", "leaf"]. Should be '
                        'entered as `--cluster_method=eom --cluster_method=leaf` or just '
                        'one of those if desired. If left at default of None, '
                        'will use both eom and leaf')
    
    
    parser.add_argument('--start_year', type=int, default=2016,
                        help='starting year for study. Can be greater than end '
                        'year.')
    parser.add_argument('--end_year', type=int, default=2018,
                        help='ending year you want in study')
    
    parser.add_argument('--db_ip', type=str,
                        default=None,
                        help='IP address or domain in which the Neo4j graph '
                        'containing embeddings is located')
    parser.add_argument('--db_password_secret', type=str,
                        default=None,
                        help='Name of AWS Secrets Manager secret to use for '
                        'grabbing the Neo4j database password. If None, will '
                        'look for a value provided via --db_password')
    parser.add_argument('--db_password', type=str,
                        default=None,
                        help='Plaintext database password. Not recommended '
                        'for security reasons, ideally you should save the '
                        'password via a tool like AWS Secrets Manager and '
                        'then access using the --db_password_secret arg')
    parser.add_argument('--mlflow_tracking_uri', type=str,
                        default=None,
                        help='IP address/domain + port of the MLFlow tracking'
                        ' server for logging optuna studies and resultant '
                        'models')
    
    args = parser.parse_args()
    
    if args.cluster_method is None:
        cluster_methods = ['eom', 'leaf']
    else:
        cluster_methods = args.cluster_method
    
    # Check that we can save to mlflow
    test_s3_access('mlflow-qs2')

    if args.start_year > args.end_year:
        step_size = -1
    else:
        step_size = 1

    for year in tqdm(range(
        args.start_year, args.end_year + step_size, step_size
    )):

        logger.info(
            f"creating embeddings array from dataframe for year {year}...")
        embeddings = get_embeddings(
            db_ip=args.db_ip,
            year=year,
            db_password_secret=args.db_password_secret
        )
        
        # Setup the mlflow experiment
        full_experiment_name = args.experiment_name + f' - {year}'        
        experiment, client = setup_mlflow(
            full_experiment_name,
            tags={
                'project': args.project,
                'dataset': args.dataset
            },
            tracking_server=args.mlflow_tracking_uri,
            return_client=True
        )
        
        logger.info("Creating Objectives object defining experimental "
                    "parameters...")        
        experiment_parameters = Objectives(
            [
                Hyperparameter(
                    'experiments_per_trial', 
                    constant=args.cv
                    ),
                Hyperparameter(
                    'umap_n_neighbors', 
                    min=args.min_nn, 
                    max=args.max_nn
                    ),
                Hyperparameter(
                    'umap_n_components',
                    min=args.min_nc, 
                    max=args.max_nc
                    ), 
                Hyperparameter(
                    'umap_min_dist', 
                    min=args.min_min_dist, 
                    max=args.max_min_dist
                    ),
                Hyperparameter(
                    'umap_metric', 
                    categories=['euclidean', 'cosine']
                    ),
                Hyperparameter(
                    'hdbscan_min_cluster_size', 
                    min=args.min_mcs, 
                    max=args.max_mcs
                    ),
                Hyperparameter(
                    'hdbscan_min_samples', 
                    min=args.min_ms, 
                    max=args.max_ms
                ),
                Hyperparameter(
                    'hdbscan_cluster_selection_method',
                    categories=cluster_methods
                )
            ],
            [
                Criterion(
                    'DBCV Mean', 
                    'maximize'
                    ),
                Criterion(
                    'DBCV StDev', 
                    'minimize'
                    ),
                Criterion(
                    'NumClusters Mean', 
                    'maximize',
                    range=(0, np.inf)
                    ),
                Criterion(
                    'NumClusters StDev', 
                    'minimize',
                    range=(0, np.inf)
                    ),
                Criterion(
                    'Mean of Cluster Persistence Means', 
                    'maximize'
                )
            ],
            objective,
            embeddings,
            mlflow=(experiment, client)
        )      

        logger.info(f"Starting tuning for year {year}...")
        study = bayesian_tune(
            experiment_parameters,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            garbage_collect=True
        )
        
        #TODO: manual garbage collection here?

        # Generate the final, close-to-average model we found and log it
        logger.info(f"Building best model for year {year} and logging it...")
        best_model = hydrate_best_optuna_solution(
            experiment_parameters,
            hydration_tolerance=0.05,
            log_to_mlflow=True
        )
