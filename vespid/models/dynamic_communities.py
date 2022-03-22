import pandas as pd
import numpy as np
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import RegressorChain
from sklearn.metrics import fbeta_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from vespid.data.neo4j_tools import Nodes, Relationships
from vespid.models.static_communities import get_cluster_data
from vespid.models.static_communities import jaccard_coefficient
from vespid.models.static_communities import cosine_similarities
from vespid import setup_logger
logger = setup_logger(__name__)

class DynamicCommunities():
    '''
    Class designed to track, over an entire dynamic graph,
    the birth, death, merging, splitting, or simple
    continuation of dynamic communities.
    '''
    DEATH = 'death'
    BIRTH = 'birth'
    SPLIT = 'split'
    MERGE = 'merge'
    CONTINUATION = 'continuation'
    CONTRACTION = 'contraction'
    EXPANSION = 'expansion'
    

    def __init__(
        self, 
        graph, 
        start_year, 
        end_year, 
        window_size=3, 
        similarity_threshold=0.95, 
        similarity_scoring='embeddings',
        size_change_threshold=0.1, 
        expire_cycles=3
    ):
        '''
        Parameters
        ----------
        graph: Neo4jConnectionHandler object. The graph of interest.

        start_year: int. Indicates the beginning year from which data will
            be pulled for community building.

        end_year: int. Same as start year, but defines the end of the period
            of interest. Inclusive.

        window_size: int. Number of years to include in a single analytical frame.
            Note that this is currently not being used.

        similarity_threshold: float in range [0.01, 0.99].
            Dictates the minimum similarity score required
            between C_t and C_(t+1) to indicate connected
            clusters. Note that recommended value for membership-type
            scoring is 0.1. For embeddings-type scoring,
            recommended value is 0.95.

        similarity_scoring: str. Can be one of ['embeddings', 'membership'].
            Indicates what you want to compare in order to detect cluster
            evolution events. 

            'embeddings': Use BERT/SPECTER/GraphSAGE/whatever vector
                embeddings of the nodes to assess their similarity scores.
                The actual similarity mechanism to be used is cosine similarity.
                This is most directly useful when you need the embeddings to
                make the cluster comparisons over time stateful, e.g.
                if your static clustering approach is based on a graph
                at a fixed time window `t` such that nodes
                that existed before that window began aren't included
                in the solution.

            'membership': Use actual membership (e.g. unique node IDs) vectors
                of each cluster to assess changes. Uses Jaccard similarity as
                the metric. This really only works when the static clustering
                solutions come from a cumulative graph in which the graph at
                time `t` is the result of all graph information prior to `t`.

        size_change_threshold: float in range [0.01, 0.99].
            Dictates the minimum change in size of a cluster
            from t to t+1 to indicate that it has expanded
            or contracted.

        expire_cycles: int. Number of timesteps a cluster
            should be missing from the timeline before declaring
            it dead.
        '''
        self.graph = graph
        self.start_year = start_year
        self.end_year = end_year
        self.window_size = window_size
        if similarity_scoring not in ['embeddings', 'membership']:
            raise ValueError(f"``embeddings`` received an invalid value of '{self.embeddings}'")
        self.similarity_threshold = similarity_threshold
        self.similarity_scoring = similarity_scoring
        self.size_change_threshold = size_change_threshold
        self.expire_cycles = expire_cycles

        self._c_t1_column = 'C_t'
        self._c_t2_column = 'C_(t+1)'
        self._event_column = 'event_type'

    def __repr__(self):
        output = {k:v for k,v in self.__dict__.items() if k not in ['graph', 'jaccard_scores'] and k[0] != '_'}

        return str(output)

    def clusters_over_time(self):
        '''
        Gets the cluster labels associated with each year of the
        graph and maps them to the list of papers in that cluster
        for that year, generating useful cluster-level metadata
        along the way.
        
        
        Parameters
        ----------
        None.
        
        
        Returns
        -------
        pandas DataFrame describing each cluster found in each time window 
        (e.g. cluster0_2016).
        '''
        dfs = []
        #TODO: change queries below if we stop putting the year in the cluster ID attribute name
        for year in tqdm(range(self.start_year, self.end_year + 1), 
                         desc='Pulling down year-by-year data from Neo4j'):
            if self.similarity_scoring == 'membership':
                query = f"""
                MATCH (p:Publication)
                WHERE p.clusterID IS NOT NULL
                AND p.publicationDate.year = {year}
                RETURN toInteger(p.clusterID) AS ClusterLabel, 
                {year} AS Year, 
                COUNT(p) AS ClusterSize, 
                COLLECT(ID(p)) AS Papers
                ORDER BY ClusterLabel ASC
                """

                dfs.append(self.graph.cypher_query_to_dataframe(query, 
                verbose=False))

            elif self.similarity_scoring == 'embeddings':
                dfs.append(get_cluster_data(year, self.graph))

            else:
                raise ValueError(f"``embeddings`` received an invalid value of '{self.embeddings}'")

        output = pd.concat(dfs, ignore_index=True)
        
        return output


    def track_cluster_similarity(self, clusters_over_time=None):
        '''
        Computes the Jaccard coefficient for consecutive year
        pairs (e.g. 2017-2018) pairwise between each year's clusters
        (e.g. cluster0_2017 compared to cluster1_2018) to determine
        how similar each cluster in year t is those in year t+1.
        
        
        Parameters
        ----------
        clusters_over_time: pandas DataFrame that is equivalent
            to the output of DynamicCommunities.clusters_over_time().
            If not None, this will be used as the pre-computed result
            of running that method. If None, clusters_over_time() will
            be run.
            
            Useful for saving time if you already have the pre-computed
            result in memory.
        
        
        Returns
        -------
        pandas DataFrame that is the result of self.clusters_over_time(). 
        
        Also, a dict of the form {integer_t+1_year: pandas DataFrame}, wherein
        the DataFrame has rows representing each cluster in the year t 
        and columns representing each cluster in t+1, with the values
        reflective the Jaccard coefficient of similarity between each
        cluster pair is written to self.similarity_scores. Note that keys 
        are the t+1 year value, so output[2018] covers the 2017-2018 
        comparison.
        '''
        if clusters_over_time is None:
            df_clusters_over_time = self.clusters_over_time()
        else:
            df_clusters_over_time = clusters_over_time
        
        results = {}
        #TODO: make this robust to start year > end year
        #TODO: combine with to_dataframe functionality so we can loop only once
        for year in tqdm(range(self.start_year, self.end_year), 
        desc='Calculating cluster similarity scores for each t/t+1 pair'):
            # Setup DataFrames for t and t+1 that have all cluster labels in them
            df_t1 = df_clusters_over_time[
                df_clusters_over_time['Year'] == year
            ].set_index('ClusterLabel')

            df_t2 = df_clusters_over_time[
                df_clusters_over_time['Year'] == year + 1
            ].set_index('ClusterLabel')

            if self.similarity_scoring == 'membership':
                # This will produce np.nan for any cluster label not present in a given year
                df_years = pd.DataFrame({
                    year: df_t1['Papers'],
                    year + 1: df_t2['Papers']
                })

                # shape is (max_cluster_num, max_cluster_num)
                # Form of [[cluster0_year0 to cluster0_year1], [cluster1_year0 to cluster0_year1], [cluster2_year0 to cluster0_year1],
                #         [cluster0_year0 to cluster1_year1], [cluster1_year0 to cluster1_year1], [cluster2_year0 to cluster1_year1],
                #         [cluster0_year0 to cluster2_year1], [], []]
                scores = np.full((df_years.shape[0], df_years.shape[0]), np.nan)

                #TODO: make this more efficient by avoiding loops!
                # Go through each C_t vs. C_(t+1) pair and calculate Jaccard coefficient
                for i, papers_past in enumerate(df_years[year]):
                    # Check for nulls
                    if isinstance(papers_past, list):
                        for j, papers_current in enumerate(df_years[year + 1]):
                            if isinstance(papers_current, list):
                                scores[i][j] = jaccard_coefficient(papers_past, papers_current)
                                
                results[year + 1] = pd.DataFrame(
                    scores, 
                    index=[f"cluster{i}_{year}" for i in range(scores.shape[1])],
                    columns=[f"cluster{i}_{year + 1}" for i in range(scores.shape[0])]
                ).dropna(how='all', axis=0).dropna(how='all', axis=1) # Drop past, then future, clusters that don't exist (all null)
                                
            elif self.similarity_scoring == 'embeddings':
                #TODO: consider plugging this straight into scoring function if memory is tight
                t1 = df_clusters_over_time.loc[
                    df_clusters_over_time['Year'] == year, 
                    'ClusterEmbedding'
                    ]
                
                t2 = df_clusters_over_time.loc[
                    df_clusters_over_time['Year'] == year + 1, 
                    'ClusterEmbedding'
                    ]
                
                scores = cosine_similarities(t1, t2)
                
                results[year + 1] = pd.DataFrame(
                    scores, 
                    index=[f"cluster{i}_{year}" for i in range(scores.shape[0])],
                    columns=[f"cluster{i}_{year + 1}" for i in range(scores.shape[1])]
                )
            
        self.similarity_scores = results

        return df_clusters_over_time
    
    def _format_events(self, events):
        '''
        Reformats events DataFrame to be consistent output

        Parameters
        ----------
        events : pandas DataFrame
            Original output of any given flagging method

        Returns
        -------
        pandas DataFrame
            Formatted output with consistent column order, etc.
        '''
        return events[[
            self._c_t1_column, 
            self._c_t2_column, 
            self._event_column
            ]]

    def flag_merge_events(self, year):
        '''
        Given a set of C_t (cluster at time t) to C_(t+1)
        similarity scores and a threshold to dictate what
        clusters are similar enough to be connected to one
        another through time, return the ones that appear
        to be the result of a merge event.


        Parameters
        ----------
        year: int. Year `t` that should be compared to
            year t+1.
            
            
        Returns
        -------
        pandas DataFrame tracking the type of event, the focal cluster
        (e.g. the result of a merge or the cause of a split) and the
        parents/children of the focal cluster (for merging and splitting, 
        resp.), if any.
        '''
        above_threshold_scores = (self.similarity_scores[year] >= self.similarity_threshold)
        num_matching_past_clusters = (above_threshold_scores).sum(axis=0)
        resulting_merged_clusters = num_matching_past_clusters[num_matching_past_clusters >= 2].index.tolist()
        
        if np.any(num_matching_past_clusters > 1):
            # For each column 
            merge_results = above_threshold_scores[resulting_merged_clusters]\
                .apply(lambda column: [column[column].index.tolist()])\
                    .iloc[0].to_dict()
            
            output = pd.DataFrame([merge_results])\
                .transpose().reset_index(drop=False).rename(columns={
                    'index': self._c_t2_column,
                    0: self._c_t1_column
                })

            output[self._event_column] = self.MERGE
            return self._format_events(output)

        else:
            return pd.DataFrame()


    def flag_split_events(self, year):
        '''
        Given a set of C_t (cluster at time t) to C_(t+1)
        similarity scores and a threshold to dictate what
        clusters are similar enough to be connected to one
        another through time, return the ones that appear
        to be the result of a split event.
        
        
        Parameters
        ----------
        year: int. Year `t` that should be compared to
            year t+1.
            
            
        Returns
        -------
        pandas DataFrame tracking the type of event, the focal cluster
        (e.g. the result of a merge or the cause of a split) and the
        parents/children of the focal cluster (for merging and splitting, 
        resp.), if any.
        '''
        
        above_threshold_scores = (self.similarity_scores[year] >= self.similarity_threshold)
        num_matching_current_clusters = (above_threshold_scores).sum(axis=1)
        resulting_split_clusters = num_matching_current_clusters[num_matching_current_clusters >= 2].index.tolist()
        
        if np.any(num_matching_current_clusters > 1):
            # For each row AKA C_t cluster that qualified as being above threshold in 2+ cases,
            # pull out the column names for the C_(t+1) clusters that are its children
            merge_results = above_threshold_scores.loc[resulting_split_clusters]\
                .apply(lambda row: row[row].index.tolist(), 
                axis=1).to_dict()
            
            output = pd.DataFrame([merge_results])\
                .transpose().reset_index(drop=False).rename(columns={
                    'index': self._c_t1_column,
                    0: self._c_t2_column
                })

            output[self._event_column] = self.SPLIT
            return self._format_events(output)

        else:
            return pd.DataFrame()

    def flag_birth_events(self, year):
        '''
        Given a set of C_t (cluster at time t) to C_(t+1)
        similarity scores and a threshold to dictate what
        clusters are similar enough to be connected to one
        another through time, return the ones that appear
        to have been created for the first time in t+1.
        
        
        Parameters
        ----------
        year: int. Year `t` that should be compared to
            year t+1.
            
            
        Returns
        -------
        pandas DataFrame tracking the type of event, the focal cluster
        (e.g. the result of a merge or the cause of a split) and the
        parents/children of the focal cluster (for merging and splitting, 
        resp.), if any.
        '''
        # The question: do any t+1 clusters have no t cluster they are similar to?
        # Put in terms of the jaccard_scores DataFrame structure: any column for C_(t+1) 
        # that is all < similarity_threshold?
        above_threshold_scores = (self.similarity_scores[year] >= self.similarity_threshold)
        num_matching_current_clusters = (above_threshold_scores).sum(axis=0)
        resulting_birth_clusters = num_matching_current_clusters[num_matching_current_clusters < 1].index.tolist()
        
        if np.any(num_matching_current_clusters < 1):
            output = pd.DataFrame({
                self._c_t1_column: np.nan,
                self._c_t2_column: resulting_birth_clusters,
                self._event_column: self.BIRTH
            })
            return self._format_events(output)

        else:
            return pd.DataFrame()

    def flag_death_events(self, year):
        '''
        Given a set of C_t (cluster at time t) to C_(t+1)
        similarity scores and a threshold to dictate what
        clusters are similar enough to be connected to one
        another through time, return the ones that appear
        to have not continued in any form into t+1.
        
        
        Parameters
        ----------
        year: int. Year `t` that should be compared to
            year t+1.
            
            
        Returns
        -------
        pandas DataFrame tracking the type of event, the focal cluster
        (e.g. the result of a merge or the cause of a split) and the
        parents/children of the focal cluster (for merging and splitting, 
        resp.), if any.
        '''
        # The question: do any t+1 clusters have no t cluster they are similar to?
        # Put in terms of the jaccard_scores DataFrame structure: any column for C_(t+1) 
        # that is all < similarity_threshold?
        above_threshold_scores = (self.similarity_scores[year] >= self.similarity_threshold)
        num_matching_current_clusters = (above_threshold_scores).sum(axis=1)
        resulting_dead_clusters = num_matching_current_clusters[num_matching_current_clusters < 1].index.tolist()
        
        if np.any(num_matching_current_clusters < 1):
            output = pd.DataFrame({
                self._c_t1_column: resulting_dead_clusters,
                self._c_t2_column: np.nan,
                self._event_column: self.DEATH
            })
            return self._format_events(output)

        else:
            return pd.DataFrame()

    def flag_continuity_events(self, year, cluster_metadata, other_events):
        '''
        Given a set of C_t (cluster at time t) to C_(t+1)
        similarity scores and a threshold to dictate what
        clusters are similar enough to be connected to one
        another through time, return the ones that appear
        to have continued on as a single cluster into t+1, 
        but that have increased above the relative change
        threshold.
        
        
        Parameters
        ----------
        year: int. Year `t` that should be compared to
            year t+1.

        cluster_metadata: pandas DataFrame with columns
            ['ClusterLabel', 'Year', 'ClusterSize'].

        other_events: pandas DataFrame of split/merge/etc.
            events that can be used to determine what clusters
            are left and thus likely continuity events.
            
            
        Returns
        -------
        pandas DataFrame tracking the type of event, the focal cluster
        (e.g. the result of a merge or the cause of a split) and the
        parents/children of the focal cluster (for merging and splitting, 
        resp.), if any.
        '''

        above_threshold_scores = (self.similarity_scores[year] >= self.similarity_threshold)

        # Find clusters that qualify as very similar to one another
        # Need to check that there's only one-to-one mapping from t to t+1
        num_matching_t1_clusters = (above_threshold_scores).sum(axis=1)
        num_matching_t2_clusters = (above_threshold_scores).sum(axis=0)

        if np.any(num_matching_t1_clusters == 1) \
        and np.any(num_matching_t2_clusters == 1) \
        and not other_events.empty:
            # There were other flagged events, so we need to skip them
            # Expand cluster columns so we have 1:1 C_t to C_(t+1) mappings
            events_expanded = other_events\
                .explode(self._c_t1_column)\
                    .explode(self._c_t2_column)
                    
            # Drop any C_t that are part of another event already
            num_matching_t1_clusters.drop(
                labels=events_expanded[self._c_t1_column], 
                errors='ignore',
                inplace=True
                )
            
            # No more events to investigate?
            if num_matching_t1_clusters.empty:
                return pd.DataFrame()
            
            # Identify clusters at time `t` that only match one cluster in time `t+1`
            continued_clusters = num_matching_t1_clusters[num_matching_t1_clusters == 1]\
                .index.tolist()

            # Make a dict mapping {C_t: C_(t+1)}
            continuity_mapping = above_threshold_scores.loc[continued_clusters]\
                .apply(lambda row: row[row].index.tolist()[0], 
                    axis=1).to_dict()

            # Put it all into an events-record format
            events = pd.DataFrame([continuity_mapping])\
                .transpose().reset_index(drop=False).rename(columns={
                    'index': self._c_t1_column,
                    0: self._c_t2_column
                })
                
            # Make sure everything gets flagged as continuing 
            # if it made it this far, only change flag if needed
            events[self._event_column] = self.CONTINUATION

            # Get an events-records-friendly cluster label
            cluster_metadata['C_t_label'] = 'cluster' + cluster_metadata['ClusterLabel'].astype(str) \
                + "_" + cluster_metadata['Year'].astype(str)

            # Match cluster sizes to the cluster labels
            cluster_label_columns = [self._c_t1_column, self._c_t2_column]
            for column in cluster_label_columns:
                events = events.merge(
                    cluster_metadata[['C_t_label', 'ClusterSize']],
                    how='left',
                    left_on=column,
                    right_on='C_t_label',
                    sort=False
                ).rename(columns={'ClusterSize': column + '_size'})\
                    .drop(columns=['C_t_label'])
                
            # bool Series indicating if expansion has occurred at or above our threshold
            expanded = events[f"{cluster_label_columns[0]}_size"] * (1 + self.size_change_threshold) \
                <= events[f"{cluster_label_columns[1]}_size"]

            events.loc[expanded, self._event_column] = self.EXPANSION

            contracted = events[f"{cluster_label_columns[0]}_size"] * (1 - self.size_change_threshold) \
                >= events[f"{cluster_label_columns[1]}_size"]

            events.loc[contracted, self._event_column] = self.CONTRACTION

            return self._format_events(events)

        else:
            return pd.DataFrame()
    

    def to_dataframe(self, clusters_over_time=None):
        '''
        Produces a tabular record of the various dynamic community events. 
        Does so for all t/t+1 pairs of time windows.


        Parameters
        ----------
        clusters_over_time: pandas DataFrame that is equivalent
            to the output of DynamicCommunities.clusters_over_time().
            If not None, this will be used as the pre-computed result
            of running that method. If None, clusters_over_time() will
            be run.
            
            Useful for saving time if you already have the pre-computed
            result in memory.


        Returns
        -------
        pandas DataFrame with relevant columns for each event.
        '''
        # Get the data
        if clusters_over_time is None:
            df_clusters_over_time = self.track_cluster_similarity()
        else:
            df_clusters_over_time = self.track_cluster_similarity(clusters_over_time)
        all_events = []
        step_size = -1 if self.start_year > self.end_year else 1
        for year in tqdm(range(self.start_year + 1, self.end_year + 1, step_size),
                         desc="Identifying events over each consecutive pair of years"):
            merges = self.flag_merge_events(year)
            splits = self.flag_split_events(year)
            births = self.flag_birth_events(year)
            deaths = self.flag_death_events(year)
            
            events = pd.concat([
                births, 
                splits, 
                merges, 
                deaths
            ], 
            ignore_index=True)
            
            # Continuity events are the only ones left after these, but 
            # need knowledge of other flagged events to work properly
            continuity = self.flag_continuity_events(year, df_clusters_over_time, events)
            events = events.append(continuity, ignore_index=True)

            if events.empty:
                raise RuntimeError("No community events detected...")
            
            elif self._missing_cluster_events(year, events):
                raise RuntimeError("Some clusters were not accounted for")
            
            all_events.append(events)
        
        return pd.concat(all_events, ignore_index=True)
    
    def _missing_cluster_events(self, year, events):
        '''
        Detects if any clusters have not been accounted for
        in the events logging.

        Parameters
        ----------
        year : int
            Year of interest.
        events : pandas DataFrame
            Results of flagging methods combined together

        Returns
        -------
        bool
            True if there were any missing clusters detected,
            False otherwise.
        '''
        events_expanded = events.explode('C_t').explode('C_(t+1)')
        C_t1 = self.similarity_scores[year].index
        C_t2 = self.similarity_scores[year].columns

        missing_C_t1 = C_t1[~C_t1.isin(events_expanded['C_t'])]
        missing_C_t2 = C_t2[~C_t2.isin(events_expanded['C_(t+1)'])]
        
        return not missing_C_t1.empty or not missing_C_t2.empty
    
    def export_to_neo4j(self, clusters_over_time=None):
        '''
        Takes Knowledge nodes generated and pushes them to a target graph,
        along with edges between the member nodes and Knowledge nodes. 

        Parameters
        ----------
        clusters_over_time : pandas DataFrame, optional
            This is equivalent to the output of 
            DynamicCommunities.clusters_over_time(). 
            If not None, this will be used as the pre-computed result
            of running that method. If None, clusters_over_time() will
            be run.
            
            Useful for saving time if you already have the pre-computed
            result in memory, by default None
        '''
        if clusters_over_time is None:
            clusters_over_time = self.clusters_over_time()
        else:
            clusters_over_time = clusters_over_time.copy()
            
        # Set start and end dates to be 1/1/YEAR and 
        # 12/31/YEAR, resp., to allow for future 
        # non-year-long time windows to be used
        clusters_over_time['start_date'] = pd.to_datetime(clusters_over_time['Year'], format="%Y")
        clusters_over_time['end_date'] = pd.to_datetime(clusters_over_time['start_date'].dt.date + relativedelta(years=1, days=-1))
        
        # Query Neo4j for Knowledge nodes, if any, and pull down 
        # the max ID so we can increment off of that for our IDs
        query = """
        MATCH (n:LanguageCluster)
        RETURN MAX(n.id)
        """
        max_node_id = self.graph.cypher_query_to_dataframe(query, verbose=False).iloc[0,0]

        # If no Knowledge nodes in graph
        if max_node_id is not None:
            starting_node_id = max_node_id + 1
        else:
            starting_node_id = 0
        clusters_over_time['id'] = range(starting_node_id, len(clusters_over_time))
        
        # Get the events records
        events_all_years = self.to_dataframe(clusters_over_time=clusters_over_time)
        
        # Birth or death: make sure proper node gets this label!
        # Drop nulls to make it so we retain index while still getting only relevant rows
        birth_events = events_all_years[events_all_years['event_type'] == 'birth']
        birth_index = clusters_over_time.merge(
            birth_events[self._c_t2_column],
            how='left', 
            left_on='C_t_label', 
            right_on=self._c_t2_column
            ).dropna(subset=[self._c_t2_column]).index
        clusters_over_time['born'] = False
        clusters_over_time.loc[birth_index, 'born'] = True

        # Drop nulls to make it so we retain index while still getting only relevant rows
        death_events = events_all_years[events_all_years['event_type'] == 'death']
        death_index = clusters_over_time.merge(
            death_events[self._c_t1_column], 
            how='left', 
            left_on='C_t_label', 
            right_on=self._c_t1_column
            ).dropna(subset=[self._c_t1_column]).index
        clusters_over_time['died'] = False
        clusters_over_time.loc[death_index, 'died'] = True
        
        #TODO: figure out how to re-do this such that we are setting the birth/death properties only, since LanguageCluster nodes should already exist at this point in analysis
        properties = pd.DataFrame([
            ['ClusterLabel', 'label', np.nan], # this will be specific to its year
            ['start_date', 'startDate', 'datetime'],
            ['end_date', 'endDate', 'datetime'],
            ['ClusterKeyphrases', 'keyphrases', 'string[]'],
            ['ClusterEmbedding', 'embedding', 'float[]'],
            ['born', 'born', 'boolean'],
            ['died', 'died', 'boolean']
        ], columns=['old', 'new', 'type'])
        
        # 'type' is in case we do CSV saving, but not necessary right now
        properties['type'] = np.nan

        knowledge_nodes = Nodes(
            parent_label='LanguageCluster',
            data=clusters_over_time,
            id_column='id',
            reference='knowledge',
            properties=properties,
            additional_labels=None
        )
        
        # Get the edges

        # Drop birth and death, no edges there
        events = events_all_years[~events_all_years['event_type'].isin(['birth', 'death'])].copy()

        # Map event types to planned Neo4j relationship types
        # Also set all expansion and contraction events to just continuation
        events['event_type'] = events['event_type'].replace({
            'continuation': 'CONTINUES_AS',
            'expansion': 'CONTINUES_AS',
            'contraction': 'CONTINUES_AS',
            'split': 'SPLITS_INTO',
            'merge': 'MERGES_INTO'
        }).values

        events = events.explode('C_t').explode('C_(t+1)')
        
        # Map Knowledge IDs to events
        # First t1
        events = events.merge(
            clusters_over_time[['id', 'C_t_label']], 
            left_on=self._c_t1_column,
            right_on='C_t_label', 
            how='left'
        ).drop(columns=['C_t_label']).rename(columns={'id': 'id_t1'})

        # Now t2
        events = events.merge(
            clusters_over_time[['id', 'C_t_label']], 
            left_on=self._c_t2_column,
            right_on='C_t_label', 
            how='left'
        ).drop(columns=['C_t_label']).rename(columns={'id': 'id_t2'})
        
        events['similarity_threshold'] = self.similarity_threshold
        events['similarity_scoring'] = self.similarity_scoring
        
        all_edges = []

        properties = pd.DataFrame([
            ['similarity_threshold', 'similarityThreshold', 'float'],
            ['similarity_scoring', 'similarityMethod', np.nan]
        ], columns=['old', 'new', 'type'])

        for type in events['event_type'].unique():
            all_edges.append(Relationships(
                type=type,
                data=events[events['event_type'] == type],
                start_node=knowledge_nodes,
                id_column_start='id_t1',
                end_node=knowledge_nodes,
                id_column_end='id_t2', # Need this so they don't both try to use 'id'
                properties=properties
            ))
                                     
        # Connect papers to their Knowledge node
        logger.debug("Creating Knowledge node constraint if it doesn't exist...")
        query = "CREATE CONSTRAINT clusters IF NOT EXISTS ON (n:LanguageCluster) ASSERT n.id IS UNIQUE"
        _ = self.graph.cypher_query_to_dataframe(query, verbose=False)
        
        #TODO: consider removing all of this, as this should be done upstream
        query = """
        MATCH (p:Publication)
        WHERE NOT (p)-[:IS_CLUSTER_MEMBER_OF]-(:LanguageCluster)
        AND p.clusterID IS NOT NULL
        AND toInteger(p.clusterID) > -1
        RETURN p.id AS id_paper, 'cluster' + p.clusterID + '_' + p.publicationDate.year AS C_t_label
        """

        papers_to_knowledge = self.graph.cypher_query_to_dataframe(query, verbose=False)
        # Merge C_t_label on to nodes we have to get IDs
        papers_to_knowledge = papers_to_knowledge.merge(
            clusters_over_time[['C_t_label', 'id']], 
            how='left', 
            on='C_t_label'
            ).rename(columns={'id': 'id_cluster'})
        
        papers = Nodes(
            parent_label='Publication',
            data=papers_to_knowledge,
            id_column='id_paper',
            reference='paper'
        )
        
        all_edges.append(Relationships(
            type='IS_CLUSTER_MEMBER_OF',
            data=papers_to_knowledge,
            start_node=papers,
            id_column_start='id_paper',
            end_node=knowledge_nodes,
            id_column_end='id_cluster',
            properties=None #TODO: add info about clustering pipeline model version used
        ))
        
        # Export Knowledge nodes
        knowledge_nodes.export_to_neo4j(self.graph, batch_size=1_000)
        
        # Export all edges
        for edges in all_edges:
            edges.export_to_neo4j(self.graph, batch_size=1_000)
        
        return clusters_over_time, knowledge_nodes, all_edges

def process_data_for_event_modeling(
    graph, 
    features_query, 
    targets_query=None,
    time_variable='Year',
    sort_key_t1='KnowledgeID_t1',
    sort_key_t2='KnowledgeID_t2'
):
    '''
    Given existing Knowledge nodes in a Neo4j graph, pull down the dynamic
    events data via the edges between Knowledge nodes and vectorize the
    event counts for predictive modeling of event evolutions. Also merges
    the vectorized event information (AKA the modeling targets) with a custom
    feature query result to generate the full dataset and returns it all
    sorted by time (ascending) for easy time-aware train/test splitting.

    Parameters
    ----------
    graph : Neo4jConnectionHandler
        The graph providing the feature and target data
    features_query : str, optional
        Cypher query to generate the input features for modeling
    targets_query : str, optional
        Cypher query to generate the prediction targets
    time_variable : str, optional
        Variable name used in `features_query` to describe the time window
        used for generating the cluster events, by default 'Year'

    Returns
    -------
    pandas DataFrame
        DataFrame with identfier columns and feature + target columns.
        Target columns are called the same as the edge types between 
        `Knowledge` nodes (e.g. 'SPLITS_INTO').
    '''
    
    if targets_query is None:
        targets_query = """
        MATCH (n:LanguageCluster)
        OPTIONAL MATCH (n)-[e]->(n2:LanguageCluster)
        
        RETURN 
        DISTINCT 'cluster' + toString(n.label) + '_' + toString(n.startDate.year) AS cluster_t1,
        n.id AS KnowledgeID_t1,
        n.startDate.year AS Year1,
        n.born AS born_t1, n.died AS died_t1,
        type(e) AS event_type, 
        'cluster' + toString(n2.label) + '_' + toString(n2.startDate.year) AS cluster_t2,
        n2.id AS KnowledgeID_t2,
        n2.startDate.year AS Year2,
        n2.born AS born_t2, n2.died AS died_t2
        ORDER BY cluster_t1 ASC
        """
    
    df_events = graph.cypher_query_to_dataframe(targets_query)\
        .replace({None: np.nan})
        
    # Transform so we have a new row for each death event
    # Valid events include death at time t2 OR birth + death at time t1, 
    # but death at time t1 alone is over-counting

    # As we don't default to assuming a cluster is born in the first year of 
    # analysis, 
    # need to find min year of clusters so we can use that as a parameter for 
    # tracking deaths in that year
    death_criteria = (
        (df_events['born_t1']) & (df_events['died_t1'])
    ) | ( #TODO: is this over-counting? We only want t1 stuff don't we?
        df_events['died_t2']
    ) | (
        (df_events['died_t1']) & (df_events['Year1'] == df_events['Year1'].min())
    )
    death_events = df_events[death_criteria].copy()
    death_events['event_type'] = 'DIES'

    # For clusters born in t1 and dead in t2, keep their clusterN_year labels 
    # as they are
    # BUT for those that have an event associated with them, 
    # need to generate new event records
    # with DIES type and cluster_t2 in cluster_t1 position
    death_events.loc[
        death_events['died_t2'] == True, 
        sort_key_t1
        ] = death_events.loc[
            death_events['died_t2'] == True, 
            sort_key_t2
        ]
    death_events[sort_key_t2] = np.nan

    # Make sure we drop duplicates t1 clusters, since split/merge events 
    # can generate a bunch of extra death events for a single t2 cluster
    death_events.drop_duplicates(subset=[sort_key_t1], inplace=True)
    
    # Merge death events with normal events
    df_events = df_events.append(death_events, ignore_index=True)\
        .drop(columns=['born_t1', 'died_t1', 'born_t2', 'died_t2'])

    # Get rid of event_types that are null, 
    # as these are likely birth + death at t1 holdovers
    df_events.dropna(subset=['event_type'], inplace=True)
    
    ordered_columns = [
        'SPLITS_INTO',
        'MERGES_INTO',
        'CONTINUES_AS',
        'DIES'
    ]
    
    # Vectorize the records so we have one row per cluster at t1
    df_events = pd.get_dummies(
        df_events[[sort_key_t1, 'event_type']], 
        columns=['event_type'], 
        prefix='', 
        prefix_sep=''
        ).groupby(sort_key_t1).sum()[ordered_columns]
    df_events.columns.name = 'events_at_t2'
    
    # Generate the features
    features = graph.cypher_query_to_dataframe(features_query)
    
    # Combine features and labels to make sure we align the data row-wise properly
    # Should preserve sort order
    data = features.merge(
        df_events, 
        how='right', 
        left_on='KnowledgeID',
        right_on=sort_key_t1
        ).reset_index(drop=True)
    
    return data

def temporal_train_test_split(
    data, 
    feature_columns, 
    target_columns=None, 
    train_fraction=0.6, 
    time_variable='Year'
):
    '''
    Given a dataset with features and targets and a timestamp-like column,
    split the observations as closely to the desired training fraction as 
    possible without breaking up any time window to do it. 
    
    Note: the observations are *not* shuffled.

    Parameters
    ----------
    data : pandas DataFrame
        The features and targets to be split
    feature_columns : list of str
        The columns to be considered features
    target_columns : list of str, optional
        The column(s) to be considered the target(s).
        If None, will assume they are the 4 main predictive class types
        (splitting, merging, continuation, death), by default None
    train_fraction : float, optional
        Fraction of `data` that should be in the training set. Note that this 
        value is not guaranteed, given the constraint that a time window may 
        not be broken up, by default 0.6
    time_variable : str, optional
        Name of column in `data` to use for grouping by time window, 
        by default 'Year'

    Returns
    -------
    4-tuple of pandas DataFrames of the form (X_train, X_test, y_train, y_test)
        The training and testing features (X) and targets (y)
    '''
    if target_columns is None:
        target_columns = [
            'SPLITS_INTO',
            'MERGES_INTO',
            'CONTINUES_AS',
            'DIES'
        ]
    
    window_grouping = data.groupby(time_variable)
    cluster_fractions_by_window = \
        window_grouping.count().iloc[:,0] / window_grouping.count().iloc[:,0].sum()
        
    max_train_year = cluster_fractions_by_window[
        cluster_fractions_by_window.cumsum() <= train_fraction
        ].index.max()
    
    test_start_index = data[data[time_variable] > max_train_year].index.min()
    X_train = data.loc[:test_start_index - 1, feature_columns]
    y_train = data.loc[:test_start_index - 1, target_columns]
    X_test = data.loc[test_start_index:, feature_columns]
    y_test = data.loc[test_start_index:, target_columns]
    
    # Report on how well the shapes match to our goals
    goal_train_percent = train_fraction * 100
    realized_train_percent = round(X_train.shape[0] / len(data) * 100, 2)
    logger.info("As a result of trying to stay as close to "
                f"{goal_train_percent}% of training data as possible without "
                "splitting data within a time window, "
                f"{realized_train_percent}% of the observations are in "
                "the training set")
    
    return X_train, X_test, y_train, y_test

def model_binary_events(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    beta=1.0,
    return_averaged_score=False,
    model=RandomForestClassifier,
    **model_kwargs
):
    '''
    Models cluster evolution events as binary 0/1 (AKA split(s) did/did not 
    occur) vectors and reports on f1-scores, feature importances, and other 
    useful model evaluation items.

    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        Features to train the model on
    X_test : numpy array or pandas DataFrame
        Held-out features for model testing
    y_train : numpy array or pandas DataFrame
        Target(s) to train the model on
    y_test : numpy array or pandas DataFrame
        Target(s) for model testing
    beta : float between 0.0 and inf, optional
        Beta value to use for f-beta score calculations
    return_averaged_score : bool, optional
        If True, return a 2-tuple of form (model, f-beta-score)
    model : sklearn-compatible estimator, optional
        Callable estimator for model training and testing, 
        by default RandomForestClassifier

    Returns
    -------
    Same type as that of the `model` input or optionally 2-tuple as dictated 
    by the `return_average_score` param
        Trained estimator object of the type defined in `model` parameter
        and optionally also a float f-beta-score
    '''
    y_train_binarized = (y_train > 0).astype(int).reset_index(drop=True)
    y_test_binarized = (y_test > 0).astype(int).reset_index(drop=True)
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns
    )
    
    clf = model(**model_kwargs)
    clf.fit(X_train_scaled, y_train_binarized)
    
    predictions = pd.DataFrame(clf.predict(X_test_scaled), columns=y_test.columns)
    logger.info(f"y_train events summary: \n{y_train_binarized.sum()}\n")
    logger.info(f"y_test events summary: \n{y_test_binarized.sum()}]\n")
    logger.info(f"predictions events summary: \n{predictions.sum()}\n")
    
    f1_score = fbeta_score(y_test_binarized, predictions, beta=1.0, average='micro')
    f1_scores = pd.DataFrame([
        fbeta_score(y_test_binarized, predictions, beta=1.0, average=None)
        ], columns=y_test.columns)
    logger.info(f"f1_scores: \n{f1_scores}")
    logger.info(f"Micro-average f1-score = {f1_score}")
    
    f_beta_score = fbeta_score(
        y_test_binarized, predictions, beta=beta, average='micro'
    )
    f_beta_scores = pd.DataFrame([
        fbeta_score(y_test_binarized, predictions, beta=beta, average=None)
        ], columns=y_test.columns)
    logger.info(f"f_{beta}-scores: \n{f_beta_scores}")
    logger.info(f"Micro-average f_{beta}-score = {f_beta_score}")
    
    feature_importances = pd.DataFrame({
        'name': clf.feature_names_in_,
        'importance': clf.feature_importances_
    })
    logger.info(f"feature_importances: \n{feature_importances}")
    
    # How often were these predictions off from the logic we know?
    default_target_columns = [
        'SPLITS_INTO',
        'MERGES_INTO',
        'CONTINUES_AS',
        'DIES'
    ]
    
    # Make sure we have all the columns needed
    if predictions.columns.isin(default_target_columns).sum() == len(default_target_columns):
        split_cont = ((predictions['SPLITS_INTO'] > 0) & (predictions['CONTINUES_AS'] > 0)).sum()
        split_die = ((predictions['SPLITS_INTO'] > 0) & (predictions['DIES'] > 0)).sum()
        merge_cont = ((predictions['MERGES_INTO'] > 0) & (predictions['CONTINUES_AS'] > 0)).sum()
        merge_die = ((predictions['MERGES_INTO'] > 0) & (predictions['DIES'] > 0)).sum()
        cont_die = ((predictions['CONTINUES_AS'] > 0) & (predictions['DIES'] > 0)).sum()

        logic_violations = pd.DataFrame({
            "split_continue": [split_cont],
            "split_die": [split_die],
            "merge_continue": [merge_cont],
            "merge_die": [merge_die],
            "continue_die": [cont_die]
        })
        logger.info("Counts of events that shouldn't co-occur but are predicted "
                    f"to do so anyway: \n\n{logic_violations}")
        
    else:
        logger.info("Skipping logical event violations checking as not all "
                    "expected target columns are present")
    
    if return_averaged_score:
        return clf, f_beta_score
    else:
        return clf

def model_continuous_events(
    X_train, 
    X_test, 
    y_train, 
    y_test, 
    return_averaged_score=False,
    model=RandomForestRegressor,
    **model_kwargs
):
    '''
    Models cluster evolution events as count vectors
    (AKA there were 10 split(s)) and reports on RMSEs, feature importances, 
    and other useful model evaluation items.

    Parameters
    ----------
    X_train : numpy array or pandas DataFrame
        Features to train the model on
    X_test : numpy array or pandas DataFrame
        Held-out features for model testing
    y_train : numpy array or pandas DataFrame
        Target(s) to train the model on
    y_test : numpy array or pandas DataFrame
        Target(s) for model testing
    return_averaged_score : bool, optional
        If True, return a 2-tuple of form (model, variance-averaged R^2)
    model : sklearn-compatible estimator, optional
        Callable estimator for model training and testing, 
        by default RandomForestClassifier

    Returns
    -------
    Same type as that of the `model` input or a 2-tuple
        Either returns the trained estimator object of the type defined in 
        `model` parameter or returns a tuple as defined by the 
        `return_averaged_score` parameter.
    '''
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns
    )
    
    regressor = model(**model_kwargs)
    
    # Chaining allows us to account for correlations between targets
    chained_regressor = RegressorChain(regressor)
    chained_regressor.fit(X_train_scaled, y_train)
    
    predictions = pd.DataFrame(chained_regressor.predict(X_test_scaled), columns=y_test.columns)
    logger.info(f"y_train events summary: \n{y_train.describe()}\n")
    logger.info(f"y_test events summary: \n{y_test.describe()}]n")
    logger.info(f"predictions events summary: \n{predictions.describe()}\n")
    
    rmse = mean_squared_error(
        y_test, 
        predictions, 
        multioutput='uniform_average', 
        squared=False
    )
    rmses = pd.DataFrame(
        [
            mean_squared_error(
                y_test, 
                predictions, 
                multioutput='raw_values', 
                squared=False
            )
            ],
        columns=y_test.columns
    )
    logger.info(f"RMSEs across targets: \n{rmses}\n")
    logger.info(f"Uniform-average RMSE = {rmse}")
    
    r2_uniform = r2_score(
        y_test, 
        predictions, 
        multioutput='uniform_average'
    )
    r2_variance_weighted = r2_score(
        y_test, 
        predictions, 
        multioutput='variance_weighted'
    )
    r2s = pd.DataFrame(
        [
            r2_score(
                y_test, 
                predictions, 
                multioutput='raw_values'
            )
            ],
        columns=y_test.columns
    )
    logger.info(f"R^2s across targets: \n{r2s}\n")
    logger.info(f"Uniform-average R^2 = {r2_uniform}")
    logger.info(f"Target-variance-weighted-average R^2 = {r2_variance_weighted}")
    
    if return_averaged_score:
        return chained_regressor, r2_variance_weighted
    else:
        return chained_regressor

def predict_multilabel_proba_formatted(model, features_test):
    '''
    Converts the native shape of multi-label predict_proba() output from
    sklearn into one that is a tad more intuitive.

    Parameters
    ----------
    model : scikit-learn estimator
        Trained estimator that must have the `predict_proba()` method
    features_test : numpy array or pandas DataFrame
        Held-out input features for the model to predict on. Must have the
        same number, scaling style, etc. of the featuers used for model
        training.

    Returns
    -------
    numpy array of shape (n_samples, n_labels, n_classes)
        Provides probability values for every class (e.g. in the case of 
        binary classification, 0/1 AKA n_classes = 2) and for every
        label/output in a multi-label modeling setup. 
        
        So, for a dataset with 100 observations, modeling a binary outcome, 
        across 3 outputs (e.g. wind speed, temperature, and humidity), the 
        shape of the output would be (100, 3, 2).
    '''
    if model.n_outputs_ < 2:
        logger.warn(f"This model only outputs {model.n_outputs_} labels, so "
                    "just returning the raw output of predict_proba()...")
        return model.predict_proba(features_test)
    else:
        return np.swapaxes(np.array(model.predict_proba(features_test)), 0, 1)