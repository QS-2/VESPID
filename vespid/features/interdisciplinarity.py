import logging
from vespid import setup_logger
logger = setup_logger(__name__)

import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_interdisciplinarity_score(
    membership_vectors
):
    '''
    Given a set of entities and
    one vector for each representing the (ordered) strength
    of membership of that entity across a set of clusters,
    calculate the level of interdisciplinarity for each entity.

    NOTE: length of membership_vectors should be the same for
    all entities for an accurate calculation.


    Parameters
    ----------
    membership_vectors: numpy array of shape (n_samples, n_clusters) 
        that indicates how strongly each sample/entity belongs to
        a cluster (e.g. membership_vectors[0] = [0.1, 0.2, 0.3, 0.4] 
        would indicate the strongest association for sample 0 with
        cluster 3 and the weakest with cluster 0).


    Returns
    -------
    numpy array of float scores of shape (n_samples,) in the range
    [0.0, 1.0].
    '''
    num_clusters = membership_vectors.shape[1]

    # (N / N-1) * (1 - max(P)) * (1 - stdev(P))
    id_scores = (num_clusters / (num_clusters - 1)) * \
        (1 - membership_vectors.max(axis=1)) * \
            (1 - membership_vectors.std(axis=1))

    # In some instances, the score can go higher than 1.0
    # Make sure that doesn't happen but we alert on it
    over_max = id_scores[id_scores > 1.0].sum()
    if over_max > 0:
        logger.warn(f"Found {over_max} instances in which score is above 1.0. "
        "Forcing these to be 1.0...")
        id_scores[id_scores > 1.0] = 1.0

    return id_scores


def interdisciplinarity_from_citation_clusters(
    graph, 
    year,
    cluster_attribute='clusterID'
    ):
    '''
    Uses Cypher query with Neo4j instance (enriched with paper cluster labels
    e.g. from HDBSCAN clustering) to determine how interdisciplinary
    papers' references and citations are. Uses a similar scoring
    logic as what is used in vespid.models.clustering with
    HDBSCAN soft clustering probabilities.


    Parameters
    ----------
    graph: Neo4jConnectionHandler object. Used for querying the
        graph for citation information.
        
    year: int. Indicates the maximum year of publication of
        interest.

    cluster_attribute: str. Indicates the node attribute to use
        for determining the cluster membership of the node
        (e.g. 'cluster_id_2019').


    Returns
    -------
    pandas DataFrame with columns ['paperID', 'id_score'] of 
    length n_nodes, with id_score being interdisciplinarity 
    scores of shape (n_nodes,)
    '''

    def fill_out_vector(cluster_identifiers, cluster_values, num_total_clusters):
        '''
        Takes a partial membership vector and fills out the missing 
        elements with zeros, placing the nonzero elements properly.


        Parameters
        ----------
        cluster_identifiers: numpy array of ints. Indicates which clusters
            map to the values given in ``cluster_values`` (and thus must
            be the same length as ``cluster_values``) for the node
            in question.

        cluster_values: numpy array of float. Indicates the strength
            of membership the entity has to each cluster for the node
            in question.

        num_total_clusters: int. Indicates how many clusters there
            are in the total solution. Must be greater than or 
            equal to the values provided in ``cluster_identifiers``.


        Returns
        -------
        numpy array of shape (num_total_clusters,) representing
        the cluster membership strengths/probabilities of the
        node.
        '''
        
        if len(cluster_identifiers) != len(cluster_values):
            raise ValueError("cluster_identifiers and cluster_values "
            f"must be of the same length, but got {len(cluster_identifiers)} "
            f"and {len(cluster_values)}, resp.")

        if num_total_clusters < np.max(cluster_identifiers):
            raise ValueError(f"num_total_clusters ({num_total_clusters}) "
            "must not be less than the maximum "
            f"cluster_identifiers value ({np.max(cluster_identifiers)})")

        if len(cluster_identifiers) > len(np.unique(cluster_identifiers)):
            raise ValueError("cluster_identifiers contains duplicate values")
        
        # Build out an all-zeros vector of the proper length
        cluster_vector = np.zeros(num_total_clusters)
        
        # Fill in the right zeros to reflect cluster membership values
        cluster_vector[cluster_identifiers] = cluster_values
        
        return cluster_vector
    
    # Query in the same fashion as what is used to generate BW centrality scores
    # Effectively insures that all papers are either published in `year` or 
    # are referenced by ones published in `year`
    # also ignores publications that lack a cluster ID or are noise (clusterID = -1)
    query = f"""
    MATCH (p:Publication)<-[c:CITED_BY]-(m:Publication) 
    WHERE c.publicationDate.year = {year} 
    AND m.publicationDate.year <= {year}
    AND p.{cluster_attribute} IS NOT NULL 
    AND toInteger(p.{cluster_attribute}) > -1
    AND m.{cluster_attribute} IS NOT NULL
    AND toInteger(m.{cluster_attribute}) > -1
    WITH DISTINCT p AS p, COUNT(c) AS NumTotalCitations
    
    MATCH (p)<-[c:CITED_BY]-(m:Publication)
    WHERE c.publicationDate.year = {year} 
    AND m.publicationDate.year <= {year}
    AND m.{cluster_attribute} IS NOT NULL
    AND toInteger(m.{cluster_attribute}) > -1
    WITH p, 
    NumTotalCitations, 
    toInteger(m.{cluster_attribute}) AS CitationClusterLabel, 
    COUNT(m) AS NumCitationsInCluster
    
    RETURN p.id AS paperID, 
    p.publicationDate.year AS Year, 
    toInteger(p.{cluster_attribute}) AS PrimaryClusterLabel, 
    CitationClusterLabel, 
    toFloat(NumCitationsInCluster) / NumTotalCitations AS FractionalMembership
    """
    
    df = graph.cypher_query_to_dataframe(query, verbose=False)
    logger.debug(f"Years covered by network-ID-scoring query are {df['Year'].min()} to {df['Year'].max()}")
    
    # Which papers didn't have a membership value for the cluster they're assigned to?
    # AKA which ones failed to have any citations/references from within their own cluster?
    df['PrimaryLabelMatchesCitation'] = df['PrimaryClusterLabel'] == df['CitationClusterLabel']
    num_zero_primary_membership = \
        df['paperID'].nunique() - df.loc[df['PrimaryLabelMatchesCitation'], 'paperID'].nunique()
    fraction_zero_primary_membership = round(num_zero_primary_membership / df['paperID'].nunique() * 100, 2)
    if num_zero_primary_membership > 0:
        logger.warn(f"No citations from host cluster found for "
                    f"{num_zero_primary_membership} ({fraction_zero_primary_membership}%) papers! "
                    "This suggests that the clustering solution may not be very good or "
                    "that the citation network was undersampled")
    
    query = f"""
    MATCH (p:Publication)
    WHERE p.{cluster_attribute} IS NOT NULL
    AND p.publicationDate.year = {year}
    RETURN MAX(toInteger(p.{cluster_attribute}))
    """
    # cluster labels are zero-indexed, so need +1
    num_clusters = graph.cypher_query_to_dataframe(query, verbose=False).iloc[0,0] + 1

    tqdm.pandas(desc="Building full cluster membership vectors from citation-based membership per paper")
    # Group membership into list for each paper
    cluster_vectors = df.groupby('paperID', sort=False).agg(list).progress_apply(
        lambda row: fill_out_vector(
            row['CitationClusterLabel'],
            row['FractionalMembership'],
            num_clusters
        ), 
        axis=1
    )

    id_scores = calculate_interdisciplinarity_score(
        np.array(cluster_vectors.tolist())
    )

    output = pd.DataFrame({
        'paperID': df['paperID'].unique(),
        'scoreInterDNetwork': id_scores
    })

    #TODO: maybe additional weighting from dendrogram distance/cluster exemplar-exemplar distance?

    return output