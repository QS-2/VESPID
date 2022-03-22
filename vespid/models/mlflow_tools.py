import mlflow
from mlflow.tracking import MlflowClient
from sys import version_info
import cloudpickle
from smart_open import open
import importlib
from importlib.metadata import PackageNotFoundError
from packaging import version
import yaml
import pandas as pd

from vespid import setup_logger

logger = setup_logger(module_name=__name__)

MODEL_ARTIFACT_PATH = 'model'
MODEL_STAGES = ["Staging", "Production", "Archived"]

def _check_tag_overlap(old_tags, new_tags, return_only_new=True):
    '''
    Checks on overlap of keys between two dicts and reports on it.
    If instructed, will return only the new key-value pairs after comparison.

    Parameters
    ----------
    old_tags : dict
        The current tags
    new_tags : dict
        The new tags being considered (may have overlap with old tags)
    return_only_new : bool
        Indicates if only new tags should be returned

    Returns
    -------
    dict
        If there is no overlap in old and new tags, returns all new tags.
        If `return_only_new=False`, also returns all new tags, after flagging
        how much overlap there is. If `return_only_new=True`, only returns
        the new tags that have non-overlapping keys with the old tags.
    '''
    original_keys = set(old_tags.keys())
    new_keys = set(new_tags.keys())
    tag_overlap = len(original_keys.intersection(new_keys))
    
    if tag_overlap > 0 and not return_only_new:
        logger.info(f"{tag_overlap} overlapping tag keys detected, "
                    "overwriting them as instructed...")
            
    elif tag_overlap > 0 and return_only_new:
        logger.warning(f"{tag_overlap} overlapping tag keys detected, "
                    "skipping those tags as instructed...")

        output = {}
        for k in new_keys.difference(original_keys):
            output[k] = new_tags[k]
        return output
    
    return new_tags


def setup_mlflow(
    experiment_name,
    tags=None,
    set_only_new_tags=True,
    tracking_server=None,
    end_active_run=True,
    return_client=False  
):
    '''
    Sets up mlflow tracking server and, if it didn't previously exist,
    a new experiment. If it did, sets that as the active experiment.

    Parameters
    ----------
    experiment_name : str
        Name of your mlflow experiment that will contain all of your model 
        runs.
    tags : dict
        Of form {'key': value}. Experiment-level (and thus run-level) tags.
    set_only_new_tags : bool, optional
        Indicates if overlapping experiment tags should be updated with the 
        values provided by `tags`, by default True
    tracking_server : str, optional
        URI for the tracking server/location being used, 
        by default None
    end_active_run : bool, optional
        Indicates if an active model run should be terminated if one is 
        identified, to allow a new run to spin up, by default True
    return_client : bool, optional
        If True, function returns a 2-tuple of 
        (Experiment, mlflow.tracking.MlflowClient). Useful if you want to have
        a client you can feed to other functions that will be using it to 
        manually create runs.

    Returns
    -------
    mlflow Experiment object or 2-tuple of form (Experiment, MlflowClient)
        The experiment that has been set or, if `return_client=True`, 
        the experiment as well as the client used to set it.
    '''        
    mlflow.set_tracking_uri(tracking_server)
    client = MlflowClient()
    
    # Bare minimum tag key checks
    required_tag_keys = ['project', 'dataset']
    
    # Determine if experiment already exists, making a new one if it doesn't
    if mlflow.get_experiment_by_name(experiment_name) is None:
        for k in required_tag_keys:
            if k not in tags.keys() or tags is None:
                raise ValueError("`tags` must at least have keys "
                                f"{required_tag_keys}")
        
        logger.info(f"Creating new experiment '{experiment_name}'...")
        experiment_id = mlflow.create_experiment(
            experiment_name, 
            tags=tags
        )
        
    else:
        logger.info(f"Experiment '{experiment_name}' already exists.")
        experiment = client.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        existing_tags = experiment.tags
        if tags is None: 
            tags = {}
        
        if experiment.lifecycle_stage == 'deleted':
            raise RuntimeError(f"Experiment '{experiment_name}' is in a "
                               "deleted state")
        for k in required_tag_keys:
            if k not in list(tags.keys()) + list(existing_tags.keys()):
                raise ValueError("Experiment does not have one of "
                                f"{required_tag_keys} tags and none were given")
            
                
        tags_to_write = _check_tag_overlap(
            experiment.tags,
            tags,
            return_only_new=set_only_new_tags
        )
        for k,v in tags_to_write.items():
            client.set_experiment_tag(experiment_id, k, v)
    
    experiment = mlflow.set_experiment(experiment_id=experiment_id)
    
    # End model training run, if it wasn't ended automatically, as we are starting a new one
    if mlflow.active_run() is not None and end_active_run:
        logger.debug("Active model run detected, ending it as instructed...")
        mlflow.end_run()
    
    if return_client:
        return experiment, client
    else:
        return experiment

def register_model(
    run_id, 
    registered_model_name,
    description,
    tags=None,
    tracking_server=None
):
   
    client =  MlflowClient(tracking_uri=tracking_server)
    results = mlflow.register_model(
        f'runs:/{run_id}/{MODEL_ARTIFACT_PATH}',
        registered_model_name
    )
    model = client.get_registered_model(registered_model_name)
        
    if tags is not None:
        logger.info("Only using new tags for registered model...")
        tags_to_use = _check_tag_overlap(model.tags, tags)
        for k,v in tags_to_use.items():
            client.set_registered_model_tag(model.name, k, v)
        
    client.update_registered_model(
        model.name, 
        description=description
    )
    
def get_current_python_version():
    
    version_str = \
        f"{version_info.major}.{version_info.minor}.{version_info.micro}"
        
    return version.parse(version_str)
    
def generate_conda_env(model_packages=None):
    '''
    Generates a dictionary representation of the bare-minimum conda environment
    needed to run a given model such that mlflow knows how to rehydrate it.

    Parameters
    ----------
    model_packages : list of str, optional
        Names of packages you need in the same format as they are returned 
        when running `conda env export` (e.g. 'umap-learn' not 'umap'), 
        by default None

    Returns
    -------
    dict
        Representation of the conda environment needed. At a bare minimum, 
        ensures that the packages needed for mlflow and pip installation
        (and the right python version) are included.
    '''
    python_version = get_current_python_version()
        
    if model_packages is not None:
        model_packages = set(model_packages) 
    else: 
        model_packages = set()
        
    model_packages.add('cloudpickle')
    
    package_versions = ['mlflow']
    
    # We pretty much always need pandas and numpy...
    # and numba is always a problem child...
    for p in ['numpy', 'pandas', 'numba']:
        if p not in model_packages:
            logger.debug(f"{p} not found in `model_packages`, adding it...")
            model_packages.add(p)
        
    
    for m in model_packages:
        version = importlib.metadata.version(m)
        package_versions.append(f"{m}=={version}")
    
        
    conda_env = {
        'channels': ['defaults', 'conda-forge'],
        'dependencies': [
        f'python={python_version}',
        'pip',
        {
            'pip': package_versions,
        },
        ],
        'name': 'mlflow-env'
    }
    
    return conda_env

def _check_model_dependencies(run_id):
    
    base_model_uri = mlflow.get_run(run_id).info.artifact_uri
    model_object_uri = base_model_uri + '/model/model.pkl'
    conda_file_uri = base_model_uri + '/model/conda.yaml'

    with open(conda_file_uri, 'rb') as f:
        conda = yaml.full_load(f)

    packages = []
    for p in conda['dependencies']:
        if isinstance(p, dict):
            if 'pip' not in p.keys() or len(p.keys()) > 1:
                raise RuntimeError(f"Found more sub-dependencies in model conda file than just 'pip': {p.keys()}")
            else:
                packages += p['pip']
        else:
            packages.append(p)
    
    # Separate version numbers from package names
    packages = pd.DataFrame(pd.Series(packages).str.replace("==", '=')\
                            .str.split('=', expand=True))\
    .rename(columns={0: 'name', 1: 'version'})
    
    # Check each package name and version number against current environment
    # Thanks for the semver comparison code!:
    # https://stackoverflow.com/a/11887885/8630238

    for _, name, required_version in packages.itertuples():
        # Make something we can do a comparison on
        if required_version is not None:
            required_version = version.parse(required_version)
        
        # Package existence checking
        if name == 'python':
            current_version = get_current_python_version()
        else:
            try:
                current_version = version.parse(
                    importlib.metadata.version(name)
                )
            except PackageNotFoundError as e:
                raise PackageNotFoundError(f"Couldn't find package '{name}' "
                                           "installed, please install to use "
                                           "this model")
        
        # Version checking
        if required_version is None:
            logger.debug(f"Package {name} is installed and no version was "
                        "specified for the model")
        elif required_version < current_version:
            logger.warning(f"Your version of the package {name} "
                           f"({current_version}) is newer than the required "
                           f"version {required_version}")
        elif required_version > current_version:
            raise RuntimeError(f"Your version of the package {name} "
                               f"({current_version}) is older than the "
                               f"required version {required_version}. "
                            "Likely this will cause conflicts, please update "
                            "your package.")
        else:
            logger.debug(f"Package {name}=={current_version} is a match!")
            
    return True

def _build_model_uri(
    client, 
    run_id=None, 
    model_name=None, 
    model_stage=None, 
    model_version=None
):
    
    if run_id is not None:
        if model_name is not None or model_version is not None:
            raise ValueError("`run_id` must be the only parameter "
                                "specified if it is not None")
        model_uri = client.get_run(run_id).info.artifact_uri + '/model'
    elif model_name is not None:
        if run_id is not None:
            raise ValueError("`run_id` and `model_name` cannot both be "
                                "specified")
        elif model_stage is None and model_version is None:
            raise ValueError("One of `model_stage` or `model_version` must "
                             "be provided when `model_name` is used.")
        models = client.get_registered_model(model_name).latest_versions
        stage_to_version_mapping = {
            model.current_stage: model.version for model in models
        }
        
        if model_stage is not None and model_version is None:
            if model_stage not in MODEL_STAGES:
                raise ValueError(f"`model_stage` value of '{model_stage}' is "
                                 f"not one of allowed types: {MODEL_STAGES}")
            
            try:
                model_version = stage_to_version_mapping[model_stage]
            except KeyError as e:
                logger.error(f"Stage '{model_stage}' not found")
                raise e
            
        model_uri = client.get_model_version_download_uri(
            model_name,
            str(model_version)
        )
    else:
        raise ValueError("If not using `run_id`, both `model_name` and "
                         "`model_version` must be provided")
        
    return model_uri
    

def load_model(
    run_id=None, 
    model_name=None,
    model_version=None,
    model_stage=None
):
    '''
    Loads a model with all of its original bits from the artifact store

    Parameters
    ----------
    run_id : str, optional
        Run to load up. If None, `registered_model_name` must not be None, 
        by default None
    model_name : str, optional
        Registered model to load into memory. Should be in the form 
        'name/version' or 'name/stage'. If None, `run_id` must not be None, 
        by default None
    model_version : str or int, optional
        Registered model version to use. Should not be None if `model_stage` 
        is None.
    model_stage : str, optional
        Name of Stage that registered model is in. 
    

    Returns
    -------
    Trained python model object
        Can be any kind of model object, serialized by cloudpickle. Operating
        environment it is loaded into must be the same as the one in which it 
        was originally logged. If this is difficult, try setting up this model
        as an inference server instead.
    '''
    client = MlflowClient()
        
    model_uri = _build_model_uri(
        client,
        run_id=run_id,
        model_name=model_name,
        model_version=model_version,
        model_stage=model_stage
    )
    
    if run_id is None:
        run_id =  (model_uri).split("/")[-3]
    
    # Check model's conda requirements against current environment 
    # and throw warnings accordingly
    packages_ready = _check_model_dependencies(run_id)
    
    with open(model_uri + '/model.pkl', 'rb') as f:
        loaded_model = cloudpickle.load(f)
        
    return loaded_model
