from debugpy import log_to
import pandas as pd
import numpy as np
import mlflow
from mlflow.models.signature import infer_signature
from tqdm import tqdm
import umap
from sklearn.metrics.pairwise import cosine_similarity
from dateutil.relativedelta import relativedelta

from vespid.models.clustering import build_cluster_pipeline
from vespid.models.mlflow_tools import generate_conda_env, _build_model_uri
from vespid import setup_logger
from vespid.data.neo4j_tools import Nodes, Relationships
from vespid.features.interdisciplinarity import calculate_interdisciplinarity_score
from vespid.features.keyphrases import extract_cluster_keyphrases

logger = setup_logger(module_name=__name__)
        
def hydrate_best_optuna_solution(
    experiment_parameters,
    hydration_tolerance=0.05,
    log_to_mlflow=True
):
    '''
    Uses information provided about successful optuna trials in MLFlow,
    as provided by an instantiated Objectives object, to find the
    best hyperparameter combination tested and create a model most closely
    approximating the average model seen in that optimal optuna trial.
    

    Parameters
    ----------
    experiment_parameters : vespid.models.optuna.Objectives object
        Information about the tuning trials used to generate the MLFlow-based
        results. Note that, if you don't have the original Objectives object,
        you should only need to make sure you have properly-named criteria
        and hyperparameter objects instantiated, but their values (for tuning)
        are not relevant as the tuning has already occurred.
    
    hydration_tolerance : float, optional
        Indicates percentage tolerance of deviation from tuned DBCV, 
        number of clusters, and mean persistence per cluster the 
        hydrated solution is allowed, by default 0.05

    Returns
    -------
    Trained optimal model
    '''
    best_hyperparameters = experiment_parameters.find_best_multiobj_solution()
    
    # Pull out full info about best trial so we can use its metrics
    best_trial = experiment_parameters.find_best_multiobj_solution(
        return_type='full_best_trial'
    )
    
    # Fit until you've reached a suitably similar solution to what we got 
    # in tuning (or better!)
    # Assume criteria are [DBCV mean, DBCV stdev, NumClusters mean, 
    # NumClusters stdev, Mean of cluster persistence means]    
    #TODO: avoid assuming criteria names somehow?
    metric_names = experiment_parameters.get_criteria_names()
    dbcv = best_trial[metric_names[0]]
    num_clusters = best_trial[metric_names[2]]
    cluster_persistences_mean = best_trial[metric_names[-1]]
    
    i=0
    while True:
        logger.info(f"Fitting pipeline, i = {i}")
        pipe = build_cluster_pipeline(
            **best_hyperparameters
        ).fit(experiment_parameters._features, hdbscan__soft_cluster=True)
        
        clusterer = pipe.named_steps['hdbscan']
        
        # Is this solution good enough?
        close_enough_dbcv = \
            clusterer.relative_validity_ >= dbcv * (1 - hydration_tolerance)
        close_enough_num_clusters = \
            np.unique(clusterer.labels_).shape[0] >= num_clusters * (1 - hydration_tolerance) \
            and np.unique(clusterer.labels_).shape[0] <= num_clusters * (1 + hydration_tolerance)
        close_enough_mean_persistence = \
            clusterer.cluster_persistence_.mean() >= cluster_persistences_mean * (1 - hydration_tolerance)
        
        if close_enough_dbcv \
        and close_enough_num_clusters \
        and close_enough_mean_persistence:
            logger.info("Found a great pipeline solution!")
            break        
        i += 1
    
    # Log the result
    if log_to_mlflow:
        client = experiment_parameters.client
        experiment = experiment_parameters.experiment
        logged_model_tags = experiment.tags
        logged_model_tags['best_model'] = 'true'
        
        with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        tags=logged_model_tags
        ) as run:
            run_id = run.info.run_id
            
            # Make sure we log all of these selected hyperparameters to mlflow
            for name, value in best_hyperparameters.items():
                mlflow.log_param(name, value)
                
            metrics = [
                clusterer.relative_validity_, # DBCV "mean"
                0, # DBCV stdev
                np.unique(clusterer.labels_).shape[0], # num clusters "mean"
                0, # num clusters stdev
                clusterer.cluster_persistence_.mean() # Cluster persistence "mean of means"
            ]
            
            for name, value in zip(metric_names, metrics):
                mlflow.log_metric(name, value)
                
            # Log the model
            # Build out metadata we need to save the model
            signature = infer_signature(
                experiment_parameters._features, 
                pipe.predict(experiment_parameters._features)
            )
            conda_env = generate_conda_env([
                'scikit-learn',
                'hdbscan',
                'umap-learn'
            ])
            
            #TODO: determine if it's better to generate our own conda env or use the default parser
            mlflow.sklearn.log_model(
                pipe, 
                'model', 
                conda_env=None,
                signature=signature
            )
        
    return pipe

def compare_clusters_to_communities(
    year, 
    graph, 
    similarity_threshold=0.1, 
    network_community_attribute='louvainCommunityID',
    include_noise = False
):
    '''
    Compares membership (via Jaccard similarity) of language-based clusters
    to those identified via Louvain modularity of the citation network.
    
    Note that this function assumes GDS graph projections named as 
    "citations_<year>" exist and are appropriate for doing network
    community detection on.

    Parameters
    ----------
    year : int
        Year to analyze
    graph : Neo4jConnectionHandler object
        Connection to the online graph to use for datasets
    similarity_threshold : float in the range (0.0, 1.0), optional
        Dictates how similar a cluster and network community need
        to be in order to be considered linked, by default 0.1
    network_community_attribute: str, optional
        Network-derived node attribute name for identifying community
        membership.
    include_noise: bool. Indicates if papers labeled as noise by language-based
        clustering should be included in the comparison of language clusters
        to network communities via the `HAS_SIMILAR_LANGUAGE_AS` edge type.

    Returns
    -------
    pandas DataDFrame of shape (n_clusters, n_network_communities)
        Language-cluster to network-community similarity values. 
        
        Note that the labels used for the index and columns correspond to the 
        year-specific language-cluster/network-community labels and should not
        be considered unique across time.
    '''
    
    assert isinstance(include_noise, bool)
    #TODO: re-tool queries to make Neo4j COLLECT the results as lists itself
    if not include_noise:
        query = f"""
        MATCH (p:Publication)-[r:IS_CLUSTER_MEMBER_OF]->(q:LanguageCluster)
        WHERE p.publicationDate.year = {year}
        AND p.{network_community_attribute} IS NOT NULL
        WITH p.id AS paperID, 
            q.id AS clusterID, 
            p.{network_community_attribute} AS networkCommunityLabel

        MATCH (n:NetworkCommunity)
        WHERE n.year = {year}
        AND n.label = networkCommunityLabel
        RETURN paperID, clusterID, networkCommunityLabel, n.id AS networkCommunityID
        ORDER BY clusterID ASC, networkCommunityID ASC
        """
        output = graph.cypher_query_to_dataframe(query)
        # Ignore noise points, as they're not part of a "cluster"
        cluster_members = output[output['clusterID'] > -1].groupby(
            'clusterID', 
            sort=True
            )['paperID'].agg(list)
        
    else:
        query = f"""
        MATCH (p:Publication)-[r]->(q:LanguageCluster)
        WHERE p.publicationDate.year = {year}
        AND (r:IS_CLUSTER_MEMBER_OF OR r:HAS_SIMILAR_LANGUAGE_AS)
        AND p.{network_community_attribute} IS NOT NULL
        WITH p.id AS paperID, 
            q.id AS clusterID, 
            p.{network_community_attribute} AS networkCommunityLabel

        MATCH (n:NetworkCommunity)
        WHERE n.year = {year}
        AND n.label = networkCommunityLabel
        RETURN paperID, clusterID, networkCommunityLabel, n.id AS networkCommunityID
        ORDER BY clusterID ASC, networkCommunityID ASC
        """
        output = graph.cypher_query_to_dataframe(query)
        cluster_members = output.groupby(
            'clusterID',
            sort=True
            )['paperID'].agg(list)
    
    network_community_members = output.groupby(
        'networkCommunityID', 
        sort=True
        )['paperID'].agg(list)
    
    similarity_matrix = np.full(
        (cluster_members.shape[0], network_community_members.shape[0]), 
        np.nan)
    
    for i, cluster_list in tqdm(enumerate(cluster_members), 
                                desc='Parsing each language-based cluster',
                                total=similarity_matrix.shape[0]):
        for j, network_comm_list in enumerate(network_community_members):
            similarity_matrix[i][j] = jaccard_coefficient(cluster_list, network_comm_list)
    
    # the values for index and column come from the relevant "label" 
    # property for network- vs. language-based groups, resp.
    df_similarity_matrix = pd.DataFrame(
        similarity_matrix,
        columns=[f"NetworkCommunity_{label}" for label in network_community_members.index],
        index=[f"Knowledge_{label}" for label in cluster_members.index]
    )
    
    num_clusters_with_one_or_more_matches = \
        ((df_similarity_matrix >= similarity_threshold).sum(axis=1) == 1).sum()
    pct_clusters_with_one_or_more_matches = \
        round(num_clusters_with_one_or_more_matches / similarity_matrix.shape[0] * 100, 2)

    logger.info(f"Found {num_clusters_with_one_or_more_matches} "
                f"({pct_clusters_with_one_or_more_matches}%) "
                "clusters with 1+ matches to a network-derived community")
    
    return df_similarity_matrix

def compare_clusters_within_time_window(data, year):
    '''
    For a single time window (often a year), build a
    similarity matrix describing how similar each cluster
    in that window is to all the other clusters.
    
    Note that the time window should correspond to 
    the same window used for the static cluster
    solution to begin with. Also note that this function
    currently assumes you will be comparing clusters
    via embedding vectors.

    Parameters
    ----------
    data : pandas DataFrame with Year and ClusterEmbedding columns
        The data about the clusters (one per row) to use for comparison.
    year : int
        Year describing the time window of interest.

    Returns
    -------
    numpy array of shape (n_clusters, n_clusters)
        The cosine similarity of each cluster's embedding
        to every other cluster in the same year (including
        itself, making the diagonal always 1.0).
    '''
    return cosine_similarity(
        np.array(data.loc[data['Year'] == year, 
                          'ClusterEmbedding'].tolist())
    )
    
def get_cluster_data(year, graph):
    '''
    For a given static time window (e.g. a year),
    generate metadata for all clusters.

    Parameters
    ----------
    year : int
        Year of the time window of interest.
    graph : Neo4jConnectionHandler object
        Neo4j graph to use for getting the cluster
        member data.

    Returns
    -------
    pandas DataFrame with one row per cluster
        Metadata and labels for each cluster in the 
        specified time window.
    '''
    query = f"""
    MATCH (p:Publication)
    WHERE p.clusterID IS NOT NULL
    AND p.publicationDate.year = {year}
    AND toInteger(p.clusterID) > -1
    RETURN toInteger(p.clusterID) AS ClusterLabel, 
    {year} AS Year,
    p.embedding AS embedding,
    p.clusterKeyphrases AS ClusterKeyphrases
    ORDER BY ClusterLabel ASC
    """
    df = graph.cypher_query_to_dataframe(query, 
    verbose=False)
    
    def lists_to_aggregate_embeddings(cluster_data, metric='mean', weights=None):
        '''
        Take a pandas Series of lists and take
        the element-wise aggregation (e.g. take the mean) to 
        aggregate the lists into a single array.
        '''
        cluster_embeddings = np.array(cluster_data['embedding'].tolist())
        if metric == 'mean':
            return np.average(cluster_embeddings, axis=0, weights=weights)
        
        elif metric == 'std':
            if weights is not None:
                logger.warning("Note that ``weights`` is ignored for metric='std'")
            return np.std(cluster_embeddings, axis=0)
            
    embeddings = df.groupby('ClusterLabel').apply(lists_to_aggregate_embeddings, metric='mean')\
        .reset_index(drop=False)\
            .rename(columns={0: 'ClusterEmbedding'})
            
    embeddings['ClusterEmbeddingStdDev'] = \
        df.groupby('ClusterLabel').apply(lists_to_aggregate_embeddings, metric='std')

    embeddings['Year'] = year
    #TODO: change below to *.count() to skip NULLs?
    embeddings['ClusterSize'] = df.groupby('ClusterLabel').size()
    
    # Grab the keyphrases from a single member paper of each cluster
    embeddings['ClusterKeyphrases'] = df.groupby('ClusterLabel')['ClusterKeyphrases'].first()
    
    # For easy analysis elsewhere
    embeddings['C_t_label'] = 'cluster' + embeddings['ClusterLabel'].astype(str)\
        + '_' + embeddings['Year'].astype(str)
    
    return embeddings


def cosine_similarities(
        embeddings_t1,
        embeddings_t2
):
    '''
    Calculates the pairwise cosine similarity of embedding
    vectors at time t1 to those at time t2.

    Parameters
    ----------
    embeddings_t1 : sequence of numpy arrays of length n_clusters_t1
        Cluster embeddings at time t
    embeddings_t2 : sequence of numpy arrays of length n_clusters_t2
        Cluster embeddings at time t+1

    Returns
    -------
    numpy array of shape (n_clusters_t1, n_clusters_t2)
        Cosine similarities of the form:
            [
                [t1_0 to t2_0, t1_0 to t2_1, ..., t1_0 to t2_N],
                ...,
                [t1_N to t2_0, t1_N t t2_1, ... t1_N to t2_N]
            ]
    '''
    array_t1 = np.array(embeddings_t1.tolist())
    array_t2 = np.array(embeddings_t2.tolist())

    return cosine_similarity(array_t1, array_t2)


def jaccard_coefficient(
        array1,
        array2
):
    '''
    Calculates the intersection over union of cluster membership.
    Flexible enough to work on arrays of mismatched lengths.


    Parameters
    ----------
    array1: numpy array or list.

    array2: numpy array or list to compare to.


    Returns
    -------
    Float value between 0.0 and 1.0, with 1.0 indicating complete
    overlap (identical vectors, except potentially in sort order).
    '''
    if not isinstance(array1, np.ndarray) and not isinstance(array1, list):
        raise ValueError(f"``array1`` must be a numpy array or list, got type {type(array1)} instead")

    if not isinstance(array2, np.ndarray) and not isinstance(array1, list):
        raise ValueError(f"``array2`` must be a numpy array or list, got type {type(array1)} instead")

    intersection = np.intersect1d(array1, array2)
    union = np.union1d(array1, array2)

    return len(intersection) / len(union)

def make_language_clusters(
    year,
    clusterer,
    graph,
    model_id=None,
    batch_size=None
):
    
    logger.info("Pulling down needed Publication data...")
    query = f"""
    MATCH (n:Publication)
    WHERE n.publicationDate IS NOT NULL
    AND n.semanticScholarID IS NOT NULL
    AND n.publicationDate.year = {year}
    AND n.embedding IS NOT NULL
    RETURN n.id AS id, n.title as title, n.abstract as abstract, 
    n.embedding AS embedding
    ORDER BY id ASC
    """
    df_papers = graph.cypher_query_to_dataframe(query)
    df_papers['clusterLabel'] = clusterer.labels_
    num_clusters = df_papers['clusterLabel'].nunique() - 1 # -1 for noise

    # Add soft cluster vectors property to each paper
    df_papers['clusterMembershipProbabilities'] = pd.Series(
        list(clusterer.soft_cluster_probabilities.values.tolist())
    )

    logger.info("Calculating interdisciplinarity scores for each paper...")
    df_papers['interdisciplinarityScore'] = calculate_interdisciplinarity_score(
        clusterer.soft_cluster_probabilities.values
    )
    
    # Setup necessary paper node data
    properties = pd.DataFrame([
        ['clusterMembershipProbabilities', 'clusterMembershipProbabilities', 'float[]'],
        ['interdisciplinarityScore', 'interdisciplinarityScore', 'float'],
        ['clusterLabel', 'clusterLabel', 'int']
    ], columns=['old', 'new', 'type'])
    
    # Make sure we don't end up with weird variable names 
    # when saving straight to graph
    properties['type'] = np.nan

    # Setup paper Nodes object for making edges to clusters
    paper_nodes = Nodes(
        parent_label='Publication', 
        data=df_papers, 
        id_column='id', 
        reference='paper',
        properties=properties
    )
    
    logger.info(f"Finding descriptive keyphrases for all {num_clusters} clusters...")
    # Get keyphrases, aggregate embeddings at the cluster level
    df_clusters = extract_cluster_keyphrases(
        df_papers, 
        cluster_label='clusterLabel'
    ).rename(columns={'clusterLabel': 'label'})
    
    logger.info("Making sure cluster node IDs are unique over time...")
    # Query Neo4j for cluster nodes, if any, and pull down 
    # the max ID so we can increment off of that for our IDs
    query = """
    MATCH (n:LanguageCluster)
    RETURN MAX(n.id)
    """
    max_node_id = graph.cypher_query_to_dataframe(query, verbose=False).iloc[0,0]

    # If no cluster nodes in graph
    if max_node_id is not None and not np.isnan(max_node_id):
        starting_node_id = max_node_id + 1
    else:
        starting_node_id = 0
        
    df_clusters['id'] = range(starting_node_id, len(df_clusters) + starting_node_id)
    df_clusters['year'] = year
    df_clusters['start_date'] = pd.to_datetime(
        df_clusters['year'], 
        format="%Y"
    )
    df_clusters['end_date'] = pd.to_datetime(
        df_clusters['start_date'].dt.date + relativedelta(years=1, days=-1)
    )
    
    # Build the cluster nodes
    properties = pd.DataFrame([
        ['label', 'label', 'int'], # this will be specific to its year
        ['start_date', 'startDate', 'datetime'],
        ['end_date', 'endDate', 'datetime'],
        ['year', 'year', 'int'],
        ['top_keyphrases', 'keyphrases', 'string[]'],
        ['embedding', 'embedding', 'float[]']
    ], columns=['old', 'new', 'type'])
    
    # Make sure we don't end up with weird variable names 
    # when saving straight to graph
    properties['type'] = np.nan

    cluster_nodes = Nodes(
        parent_label='LanguageCluster',
        data=df_clusters,
        id_column='id',
        reference='cluster',
        additional_labels=None,
        properties=properties
    )

    # Make sure we grab this now so we can have it as an edge property
    df_papers['membershipProbability'] = clusterer.probabilities_
    if model_id is not None:
        df_papers['modelID'] = model_id

    papers_to_clusters = df_papers[[
        'id', 
        'clusterLabel', 
        'membershipProbability', 
        'modelID'
        ]].merge(
        df_clusters[['label', 'id']].rename(columns={'id': 'cluster_id'}), 
        how='inner', 
        left_on='clusterLabel',
        right_on='label'
    )

    properties = pd.DataFrame([
        ['membershipProbability', 'membershipProbability', 'float'],
        ['modelID', 'modelID', 'string']
    ], columns=['old', 'new', 'type'])
    
    # Make sure we don't end up with weird variable names 
    # when saving straight to graph
    properties['type'] = np.nan

    paper_to_cluster_edges = Relationships(
        type='IS_CLUSTER_MEMBER_OF',
        data=papers_to_clusters,
        start_node=paper_nodes,
        id_column_start='id',
        end_node=cluster_nodes,
        id_column_end='cluster_id',
        properties=properties
    )
    
    # Ensure we don't accidentally try to create duplicate cluster nodes
    query = """
    CREATE CONSTRAINT clusters 
    IF NOT EXISTS 
    ON (n:LanguageCluster) 
    ASSERT n.id IS UNIQUE
    """
    _ = graph.cypher_query_to_dataframe(query, verbose=False)
    
    # Write to the graph!
    paper_nodes.export_to_neo4j(graph, batch_size=1_000)
    cluster_nodes.export_to_neo4j(graph, batch_size=1_000)
    paper_to_cluster_edges.export_to_neo4j(graph, batch_size=1_000)


def match_noise_nodes_to_cluster(
    year, 
    graph, 
    model_id=None,
    node_label='Publication', 
    cluster_label='LanguageCluster',
    add_edges_to_graph=False
):
    '''
    Given a set of nodes labeled as noise by a clustering model
    (which is assumed to have given them a clusterID property of '-1'),
    and a set of cluster nodes (e.g. LanguageCluster nodes), and given that
    both node types have a vector embedding property that can be compared
    via cosine similarity, generate a mapping of regular nodes to cluster
    nodes, including the strength of that mapping, in order to weakly connect
    noise nodes to existing clusters.

    Parameters
    ----------
    year : int
        Year to analyze
    graph : Neo4jConnectionHandler object
        Connection to the graph containing the nodes and clusters to study
    model_id : str, optional
        Identifier of the clustering model used to generate the language-based
        clusters that are now being compared against. If None, the property 
        isn't written to the edges, by default None
    node_label : str, optional
        Label associated with the noisy nodes, by default 'Publication'
    cluster_label : str, optional
        Label associated with the cluster nodes, by default 'LanguageCluster'
    add_edges_to_graph : bool, optional
        Indicates if resultant data should be added straight into the
        provided graph. If True, note that the noise nodes and cluster nodes
        are assumed to use the `'id'` property as their true unique IDs.

    Returns
    -------
    pandas DataFrame
        A DataFrame describing which cluster node each noise node maps to
        (based on maximum cosine similarity score) and the score for that 
        mapping.
    '''
    logger.debug("Running node query...")
    node_query = f"""
    MATCH (n:{node_label})
    WHERE n.clusterLabel = '-1'
    AND n.publicationDate.year = {year}
    RETURN n.id AS nodeID, n.embedding AS embedding
    """    
    df_nodes = graph.cypher_query_to_dataframe(node_query, verbose=False)
    
    logger.debug("Running cluster query...")
    cluster_query = f"""
    MATCH (n:{cluster_label})
    WHERE n.startDate.year = {year} AND n.endDate.year = {year}
    RETURN n.id AS clusterID, n.embedding AS embedding
    """
    df_clusters = graph.cypher_query_to_dataframe(cluster_query, verbose=False)
    
    df_similarities = pd.DataFrame(
        cosine_similarities(
            df_nodes['embedding'], 
            df_clusters['embedding']
            ),
        index=df_nodes['nodeID'].values,
        columns=df_clusters['clusterID'].values
    )
    
    node_to_cluster_mapping = pd.DataFrame(
        df_similarities.idxmax(axis=1), 
        columns=['clusterID']
    )
    node_to_cluster_mapping['similarity'] = df_similarities.max(axis=1)
    node_to_cluster_mapping = node_to_cluster_mapping.reset_index(drop=False)\
        .rename(columns={'index': 'nodeID'})
        
    if add_edges_to_graph:
        noise_nodes = Nodes(
            parent_label='Publication',
            data=node_to_cluster_mapping.rename(columns={'nodeID': 'id'}),
            id_column='id',
            reference='noise_node',
            additional_labels=None,
            properties=None
        )

        cluster_nodes = Nodes(
            parent_label='LanguageCluster',
            data=pd.DataFrame(
                node_to_cluster_mapping['clusterID'].unique(), 
                columns=['id']
                ),
            id_column='id',
            reference='cluster',
            additional_labels=None,
            properties=None
        )
        
        properties = pd.DataFrame({
                'old': ['similarity'], 
                'new': ['similarity'], 
                'type': ['float']
                })
        
        if model_id is not None:
            node_to_cluster_mapping['modelID'] = model_id
            
            properties = pd.concat([
                properties,
                pd.DataFrame({
                    'old': ['modelID'],
                    'new': ['modelID'],
                    'type': ['string']
                })
            ], ignore_index=True)
        
        noise_edges = Relationships(
            type='HAS_SIMILAR_LANGUAGE_AS',
            data=node_to_cluster_mapping,
            start_node=noise_nodes,
            end_node=cluster_nodes,
            id_column_start='nodeID',
            id_column_end='clusterID',
            allow_unknown_nodes=False,
            properties=properties,
        )
        
        noise_edges.export_to_neo4j(graph, 1_000)
    
    return node_to_cluster_mapping
