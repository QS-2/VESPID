# -*- coding: utf-8 -*-
from typing import Mapping
from pathlib import PurePosixPath
import os
from pprint import pprint

import py2neo.errors
from tqdm import tqdm
from datetime import datetime as dt
import pathlib

from py2neo import Graph, DatabaseError, ConnectionBroken
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from vespid.data.neo4j_tools import neo4j_arrow
from vespid.data.neo4j_tools.utils import export_df, transform_timestamps, convert_types
from vespid import setup_logger

# from vespid import set_global_log_level

logger = setup_logger(module_name=__name__)
# set_global_log_level(logging.INFO)

# Timestamp we have to use to represent NULL
INVALID_TIMESTAMP = pd.Timedelta(25, 'days') + pd.Timestamp(dt.today())

IMPORT_DIR = PurePosixPath("/var/lib/neo4j/import")  # assumes a linux remote instance
EXPORT_FORMAT_CHOICES = ("graphml", "csv", "json", "cypher")
NEO4J_PASSWORD_ENV_KEY = "NEO4J_PASSWORD"  # check os.environ for this. CBB but good enough for now

MAX_CONCURRENT_REQUESTS = 100
REQUEST_INTERVAL_SECONDS = 1

ALLOWED_NEO4J_DATA_TYPES = [
    'int', 'long', 'float', 'double', 'boolean', 'byte', 'short',
    'char', 'string', 'point',
    'date', 'time', 'localtime', 'datetime', 'localdatetime', 'duration'
]

# Add array variants
ALLOWED_NEO4J_DATA_TYPES.extend([t + '[]' for t in ALLOWED_NEO4J_DATA_TYPES])
BOLT_PORT = '7687'


def get_allowed_neo4j_types():
    """
    Simple helper function to provide the list of data types that Neo4j is
    expecting. Also provides some useful guidance on quirks of data formatting
    needed for successful Neo4j ingestion.


    Parameters
    ----------
    None.


    Returns
    -------
    List of str data types.
    """

    logger.info("Types ending with '[]' are arrays of that type.")
    logger.info("'boolean' types must have all values converted to \
strings 'true' and 'false' (note that they are all lowercase).")
    return ALLOWED_NEO4J_DATA_TYPES[:]


def format_property_columns_for_ingest(
        df,
        renaming_rules=None,
        return_only_renamed=True
):
    """
    Renames pandas DataFrame columns according to the needs of neo4j's ingest
    engine, to provide it with properly-formatted property, label, etc.
    keys.


    Parameters
    ----------
    df: pandas DataFrame containing the data you intend to prepare for Neo4j
        ingest.

    renaming_rules: pandas DataFrame or list of dicts with columns/keys
        'old', 'new', and 'type'. These are the existing column names,
        new column names, and Neo4j data types that they should be associated
        with, resp.

        Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, the column is simply renamed to the value of
        'new' and is not given explicit typing (e.g. this is useful for the
        ":ID" column usually)

    return_only_renamed: bool. If True, subsets ``df`` to only include columns
        named in ``renaming_rules``. Otherwise returns all columns, including
        those never renamed.


    Returns
    -------
    A copy of ``df`` with columns renamed accordingly.
    """

    if isinstance(renaming_rules, pd.DataFrame):
        column_conversions = renaming_rules.copy()

    elif isinstance(renaming_rules, dict):
        column_conversions = pd.DataFrame(renaming_rules)

    else:
        raise ValueError(f"renaming_rules must be of type dict or \
pandas.DataFrame. Got {type(renaming_rules)} instead.")

    # Check that column_conversions only contains allowed neo4j data types
    types = column_conversions['type'].fillna("string")
    if types.isin(ALLOWED_NEO4J_DATA_TYPES).sum() < len(column_conversions):
        raise ValueError("At least one 'type' specified is not in the \
list of allowed Neo4j data types. Please run ``get_allowed_neo4j_types()`` to \
see what types may be used or follow the link provided in the docstring of \
this function for the most up-to-date information.")

    column_conversions['new_with_types'] = \
        column_conversions['new'] + ':' + column_conversions['type']

    # Make sure any that had data type NaN will not differ from 'new' column
    column_conversions['new_with_types'] = \
        column_conversions['new_with_types'].fillna(column_conversions['new'])

    columns_mapping = column_conversions.set_index('old')['new_with_types']

    if return_only_renamed:
        return df[column_conversions['old']].rename(columns=columns_mapping)

    else:
        return df.rename(columns=columns_mapping)


def graph_call_procedure(graph, procedure, *procedure_args):
    """wrapper for calling procedures on the graph"""
    logger.debug(f"calling procedure: {procedure} with args: {procedure_args}")
    graph.call[procedure](*procedure_args)


class Neo4jConnectionHandler:
    NEO4J_QUERY_DRIVERS = ('py2neo', 'neo4j-arrow', 'native')
    
    def __init__(
            self,
            db_ip='host.docker.internal',
            bolt_port='7687',
            arrow_port=9999,
            database='neo4j',
            db_username='neo4j',
            db_password=None,
            secure_connection=True
    ):
        """
        Creates a direct connection to a single Neo4j graph
        (not the full DBMS, but just one of its databases).


        Parameters
        ----------
        db_ip: str. IP address or domain of the DBMS being used.
            The default provided is the equivalent of using "localhost"
            or "127.0.0.1", but when inside a docker container.

        bolt_port: str. Indicates the port through which
            bolt-database-connections are pushed.

        database: str. Name of the database to access.
            Use the default if accessing Neo4j Community Edition
            (as this only allows a single database).

        db_username. str. Username to use for authentication purposes when
            accessing the database.

        db_password: str. Password to use for authentication purposes when
            accessing the database.

        secure_connection: bool. If True, will assume that protocols to be
            used are those associated with SSL-secured instances (e.g. bolt+s).
        """
        self.db_ip = db_ip
        self.bolt_port = bolt_port
        self.database_name = database
        self.user = db_username
        self.stats = None
        self.procedures = None
        self.plugins = None
        self.data_splitting_ids = None
        self.schema = None
        self.secure_connection = secure_connection

        if secure_connection:
            self.uri = f"neo4j+s://{db_ip}:{bolt_port}"
        else:
            self.uri = f"neo4j://{db_ip}:{bolt_port}"

        # Setup the drivers
        self._native_driver = None
        self._py2neo_driver = None
        self._neo4j_arrow_driver = None
        try:
            logger.debug(f"initializing native driver graph connection: "
                         f"{self.uri}")
            self._native_driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, db_password),
                max_connection_lifetime=10_000  # seconds, default is 3600
            )
            #self._native_driver.verify_connectivity()
        except Exception as e:
            logger.error("Failed to create the native driver!")
            raise e

        try:
            logger.debug(f"initializing py2neo driver graph connection: "
                         f"{self.db_ip}:{self.bolt_port}")
            self._py2neo_driver = Graph(
                self.uri,
                name=self.database_name,
                user=self.user,
                password=db_password
            )
        except Exception as e:
            logger.error("Failed to create the py2neo driver!")
            raise e

        try:
            logger.debug(f"initializing neo4j-arrow driver graph connection: "
                         f"{self.db_ip}:{self.bolt_port}")
            self._neo4j_arrow_driver = neo4j_arrow.Neo4jArrow(
                user=self.user,
                password=db_password,
                location=(self.db_ip, arrow_port),
                tls=True,
                verify_tls=True
            )  # Equivalent of what we normally call 'graph'
            # TODO do we want to test this? why is this slow?
            # # and test it
            # test_query = "MATCH (p:Publication) RETURN p.id LIMIT 5"
            # ticket = self._neo4j_arrow_driver.cypher(test_query)
            # buf = StringIO()
            # _ = self._neo4j_arrow_driver.stream(ticket).read_all().to_pandas().info(buf=buf)
            # logger.debug(buf.getvalue())
            # )
        except Exception as e:
            logger.error("Failed to create the neo4j-arrow driver!")
            raise e

        logger.debug(f"initialized {self.__class__.__name__}: "
                     f"db {self.db_ip}:{self.bolt_port}")

    @property
    def graph(self):
        return self._py2neo_driver

    def close(self):
        if self._native_driver is not None:
            self._native_driver.close()

    def __str__(self):
        output = f"Database: {self.database_name}"
        output += f"\nLocation: {self.db_ip}"
        output += f"\nLogged in as user '{self.user}'"

        if self._native_driver is not None:
            output += "\nNative Neo4j python driver connected"
        if self._py2neo_driver is not None:
            output += "\npy2neo driver connected"
        if self._neo4j_arrow_driver is not None:
            output += "\nneo4j-arrow driver connected"

        return output

    def __repr__(self):
        return self.__str__()

    def get_schema(self):
        """
        Returns a tabular representation of the graph's schema, including
        things like properties associated with certain node labels/relationship
        types, etc.


        Returns
        -------
        pandas DataFrame.
        """
        if not self._check_for_plugin('apoc'):
            raise DatabaseError("APOC plugin not found")
        query = "CALL apoc.meta.schema"
        return pd.DataFrame(self.cypher_query_to_dataframe(query).iloc[0, 0])

    def _check_for_plugin(self, name):
        """
        Checks if a plugin is installed in the DBMS and returns a bool.
        """

        if self.plugins is None:
            self.procedures = self.cypher_query_to_dataframe("CALL dbms.procedures()")

            # Procedure families provided by default with core Neo4j install
            defaults = ['db', 'dbms', 'jwt', 'tx']
            self.plugins = pd.Series(self.procedures['name'].str \
                                     .split('.', expand=True)[0].unique()) \
                .replace({d: np.nan for d in defaults}).dropna().tolist()

        return name in self.plugins

    def _identify_graph_projections(self):
        projections = self.cypher_query_to_dataframe("CALL gds.graph.list()")
        return projections['graphName'].tolist()

    @classmethod
    def _query_is_directed(cls, query):
        return '<-' in query or '->' in query

    def get_subgraph_queries(self, subgraph):
        """
        Given a GDS graph projection, determine the Cypher queries
        that were used to construct it, so you can mimic those queries
        as needed.


        Parameters
        ----------
        subgraph: str. Name of the graph projection in the GDS graph catalog.


        Returns
        -------
        2-tuple of strings of the form (node_query, relationship_query).
        """

        if not self._check_for_plugin('gds'):
            raise DatabaseError("GDS plugin missing")

        elif subgraph not in self._identify_graph_projections():
            raise ValueError(f"Subgraph '{subgraph}' not found in the graph catalog")

        query = f"""
            CALL gds.graph.list('{subgraph}')
        """

        results = self.cypher_query_to_dataframe(query)
        node_query, relationship_query = results.loc[0, ['nodeQuery', 'relationshipQuery']]

        return node_query, relationship_query

    def describe(self, subgraph=None, simple=True):
        """
        Runs summary statistics on graph, including things
        like node and relationship counts, as well as more
        advanced measures of connectedness.

        See https://transportgeography.org/contents/methods/graph-theory-measures-indices/
        for details on these measures.

        This can be a computationally expensive item to run.
        As such, this should only be run as needed and the results should be saved
        by the user. Some of the most basic stats (node counts, relationship counts)
        are stored in `self.stats`. Benchmark timings:
        - 130K nodes and 475K relationships: ~3 seconds
        - 57K nodes and 819K relationships: 35 seconds


        Parameters
        ----------
        subgraph: str. If not None, this is the name of the in-memory
            graph projection of interest (e.g. 'citations'). This
            graph will be analyzed instead of the full graph, assuming
            it exists.

        simple: bool. If True, spits out the bare minimum counts-based stats
            of the graph, skipping over more computationally-intense items like
            graph diameter calculations.


        Returns
        -------
        None, only prints the stats to stdout.
        """
        for plugin in ['apoc', 'gds']:
            if not self._check_for_plugin(plugin):
                raise DatabaseError(f'{plugin} plugin not found')

        if subgraph is None:
            subgraph = 'full_graph'

        else:
            if subgraph not in self._identify_graph_projections():
                raise ValueError(f"Graph projection '{subgraph}' does not exist")

        already_projected = self.cypher_query_to_dataframe(f"CALL gds.graph.exists('{subgraph}')",
                                                           verbose=False).loc[0, 'exists']
        if not already_projected and subgraph != 'full_graph':
            raise ValueError(f"Graph projection '{subgraph}' not found")

        elif not already_projected and subgraph == 'full_graph' and not simple:
            logger.warn("Full graph projection not found, generating now...")
            query = """
            CALL gds.graph.create.cypher(
                'full_graph',
                'MATCH (n) RETURN id(n) as id, labels(n) as labels',
                'MATCH (n)-[rel]->(m) RETURN id(n) AS source, id(m) AS target'
            )
            """
            _ = self.cypher_query_to_dataframe(query, verbose=False)
            logger.info("Full graph projection complete")

        logger.debug(f"Using graph projection {subgraph}")

        # Get basic counts and such
        if subgraph != 'full_graph':
            label_counts = None
            relationship_counts = None
            stats = self.cypher_query_to_dataframe(f"CALL gds.graph.list('{subgraph}')",
                                                   verbose=False)
            stats = stats[stats['database'] == self.database_name] \
                .rename(columns={'relationshipCount': 'relCount'})

        else:
            stats = self.cypher_query_to_dataframe("CALL apoc.meta.stats()")
            label_counts = stats.loc[0, 'labels']
            relationship_counts = stats.loc[0, 'relTypesCount']

        # Number of nodes/vertices
        v = stats.loc[0, 'nodeCount']

        # Number of relationships/edges
        e = stats.loc[0, 'relCount']

        # Get various measures of connectedness
        # Beta = e/v; e = relationships, v = nodes
        beta = e / v

        # Alpha = (e - v) / (0.5v**2 - 1.5v + 1)
        alpha = (e - v) / (0.5 * v ** 2 - 1.5 * v + 1)

        # Gamma = e / (0.5v**2 - 0.5v)
        gamma = e / (0.5 * v ** 2 - 0.5 * v)

        if simple:
            stats_temp = {
                'node_count': v,
                'relationship_count': e,
                'label_counts': label_counts,
                'relationship_counts': relationship_counts,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma
            }
            pprint(stats_temp)

            # Do some simple rules-based flagging of values
            if beta < 1.0:
                logger.info(f"beta value of {beta} indicates a sparse graph, "
                            "with a low number of links relative to nodes")

            if alpha < 0.5:
                logger.info(f"Alpha value of {alpha} is fairly low, suggesting a sparse "
                            "network due to low number of observed vs. possible cycles")

            if gamma < 0.5:
                logger.info(f"Gamma value of {gamma} is fairly low, meaning that the "
                            "amount of observed links is low relative to the max possible link count")

        else:
            # Get orphan node counts by label
            query = f"""
                CALL gds.degree.stream('{subgraph}')
                YIELD nodeId, score
                WHERE score = 0.0
                WITH gds.util.asNode(nodeId) AS n
                RETURN labels(n) AS NodeLabels, count(n) AS NumIsolates
            """
            self.isolates = self.cypher_query_to_dataframe(query, verbose=False)
            if self.isolates is None or self.isolates.empty:
                orphan_count = 0

            else:
                orphan_count = self.isolates['NumIsolates'].sum()

            # Transitivity via LCC
            query = f"""
                CALL gds.localClusteringCoefficient.stream('{subgraph}')
                YIELD nodeId, localClusteringCoefficient
                WHERE gds.util.isFinite(localClusteringCoefficient)
                RETURN avg(localClusteringCoefficient)
                """

            result = self.cypher_query_to_dataframe(query, verbose=False)
            transitivity = result.iloc[0, 0]

            # Diameter = length(longest shortest path)
            # Efficiency = average of all shortest path lengths
            query = f"""
                CALL gds.alpha.allShortestPaths.stream('{subgraph}')
                YIELD sourceNodeId, targetNodeId, distance
                WHERE gds.util.isFinite(distance) = true AND distance > 0.0
                RETURN max(distance) AS diameter, avg(distance) AS efficiency
                """

            diameter, efficiency = self.cypher_query_to_dataframe(query,
                                                                  verbose=False).loc[0]

            if subgraph == 'full_graph':
                self.properties = self.cypher_query_to_dataframe("CALL apoc.meta.data()",
                                                                 verbose=False)

                self.stats = {
                    'node_count': v,
                    'orphan_node_count': orphan_count,
                    'orphan_node_fraction': orphan_count / v,
                    'relationship_count': e,
                    'label_counts': label_counts,
                    'relationship_counts': relationship_counts,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'transitivity': transitivity,
                    'diameter': diameter,
                    'efficiency': efficiency
                }

                logger.info(
                    "Attributes now available/updated: "
                    "[self.procedures, self.plugins, self.stats, self.properties, self.isolates]")
                pprint(self.stats)

            else:
                logger.warn("As this is a graph projection, stats will not "
                            "be cached to Neo4jConnectionHandler object, but rather just displayed here:")
                stats_temp = {
                    'node_count': v,
                    'orphan_node_count': orphan_count,
                    'orphan_node_fraction': orphan_count / v,
                    'relationship_count': e,
                    'label_counts': label_counts,
                    'relationship_counts': relationship_counts,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'transitivity': transitivity,
                    'diameter': diameter,
                    'efficiency': efficiency
                }
                pprint(stats_temp)

            # Do some simple rules-based flagging of values
            if beta < 1.0:
                logger.info(f"beta value of {beta} indicates a sparse graph, "
                            "with a low number of links relative to nodes")

            if diameter > 5:
                logger.info(f"Diameter of {int(diameter)} is greater than 5 hops, "
                            "suggesting a fairly sparse graph.")

            if efficiency > 2:
                logger.info(f"Efficiency of {efficiency} may be a bit high, indicating "
                            "a sparse network that takes a long time to get from A to B.")

            if alpha < 0.5:
                logger.info(f"Alpha value of {alpha} is fairly low, suggesting a sparse "
                            "network due to low number of observed vs. possible cycles")

            if gamma < 0.5:
                logger.info(f"Gamma value of {gamma} is fairly low, meaning that the "
                            "amount of observed links is low relative to the max possible link count")

            if transitivity < 0.5:
                logger.info(f"Transitivity is {transitivity}, which is fairly low and indicates "
                            "that many nearest neighbor nodes in the graph are not connected "
                            "(low average local clustering coefficient)")

            logger.info(f"Transitivity is {transitivity} and diameter is {diameter}. "
                        "Recall that complex, small-world graphs tend to have high transitivity "
                        "and low diameter...")

    def train_test_split(
            self,
            query,
            train_fraction=0.6,
            test_fraction=None,
            n_splits=1
    ):
        """
        Allows us to split nodes into train/test groups for modeling building. If there is a temporal
        element to the data (e.g. publish dates), this will abide by those restrictions such
        that no future data are used in training when they shouldn't be (e.g. no shuffling).

        WARNING: this method will assume the data are pre-sorted in order of increasing time
        via the query passed. See the ``query`` example for how to do this.


        Parameters
        ----------
        query: str. Cypher query to be used for generating node IDs that can be
            split into different datasets by this method. Make sure you sort by
            a temporal property if needed. Note that Neo4j doesn't like doing
            DISTINCT calls on IDs, so the WITH clause shown in the example below
            is needed. Make sure only IDs are returned (as a single column).

            TODO: a more performant approach may be to use SKIP and LIMIT to get the datasets, instead of passing IDs.

            Example:
                MATCH (n:Publication)-[:CITES]->(m:Publication)
                WITH DISTINCT n AS paper, n.publicationDate AS date
                ORDER BY date ASC
                RETURN ID(paper) as paperNodeID


        train_fraction: float between 0.0 and 0.99. Indicates the maximum amount of data
            allowed to be used for training in the final fold (if doing multi-fold).

        test_fraction: float between 0.0 and 0.99. Indicates the minimum viable dataset that
            will be heldout for final model testing purposes. If None, will be assumed to be
            n_samples * (1 - train_fraction).

        n_splits: int. Indicates how many train/test splits you want to return.


        Returns
        -------
        Either a single tuple of the form (train_indices_array, test_indices_array)
        if ``n_splits`` = 1, else a generator of such tuples that follows
        Walk-Forward Chaining for time series cross-validation (e.g. each new training dataset
        includes the old training data + more new points that were previously in the test dataset).
        """

        logger.info("Querying for node IDs...")
        all_ids = self.cypher_query_to_dataframe(query).iloc[:, 0]
        self.data_splitting_ids = all_ids
        logger.info("Query complete!")

        train_size = round(train_fraction * len(all_ids))
        if test_fraction is None:
            test_size = len(all_ids) - train_size

        else:
            test_size = round(test_fraction * len(all_ids))

        if n_splits == 1:
            splitter = TimeSeriesSplit(
                n_splits=2,  # trust me on this
                # gap=0,
                test_size=test_size
            )

            # Loop through until the last values are all you have,
            # the full split
            for train_idx, test_idx in splitter.split(all_ids):
                pass

            output = (train_idx, test_idx)

        elif n_splits > 1:
            logger.info("As n_splits > 1, train and test sizes are calculated not user-defined")
            splitter = TimeSeriesSplit(
                n_splits=n_splits,
                # gap=0
            )
            output = splitter.split(all_ids)

        else:
            raise ValueError("n_splits must be an integer >= 1")
        return output

    def get_train_test_data(self, node_properties, train_indices, test_indices, id_query=None):
        """
        Uses pre-determined train/test indices to index node IDs values from
        the graph and get the training/testing data for a single fold/split.


        Parameters
        ----------
        node_properties: list of str. Node properties that will be used to form
            the datasets.

        train_indices: list of int that can be used for indexing an array of node IDs
            to get the nodes that are part of the training dataset.

        test_indices: list of int that can be used for indexing an array of node IDs
            to get the nodes that are part of the test dataset.

        id_query: str. Cypher query used, if one is needed, to draw down the properly-sorted
            node IDs for indexing via ``train_indices`` and ``test_indices``. As it's easy
            to improperly sort these node IDs and thus improperly index them, it is recommended
            that this be left as None such that the graph attribute self.data_splitting_ids
            can be used instead. This attribute only exists, however, if self.train_test_split()
            was run to generate ``train_indices`` and ``test_indices``.


        Returns
        -------
        2-tuple of pandas DataFrames of the form (train_data, test_data).
        """

        if id_query is None and self.data_splitting_ids is None:
            raise ValueError("self.data_splitting_ids is not set, "
                             "please pass a Cypher query for the id_query parameter")

        elif id_query is None:
            all_ids = self.data_splitting_ids

        else:
            all_ids = self.cypher_query_to_dataframe(id_query).iloc[:, 0]

        training_ids = all_ids[train_indices].tolist()
        test_ids = all_ids[test_indices].tolist()

        properties_clause = ', '.join([f"n.{p} AS {p}" for p in node_properties])
        query = f"""
            MATCH (n) 
            WHERE ID(n) IN $ids
            RETURN {properties_clause}
        """
        df_train = self.cypher_query_to_dataframe(query,
                                                  parameters={'ids': training_ids})

        df_test = self.cypher_query_to_dataframe(query,
                                                 parameters={'ids': test_ids})

        return df_train, df_test

    @classmethod
    def _validate_export_format(cls, export_format):
        if export_format not in EXPORT_FORMAT_CHOICES:
            raise ValueError(f"export format {export_format} invalid, i know about {EXPORT_FORMAT_CHOICES}")

    def apoc_export_all(self, export_filename, config: Mapping = None,
                        export_format="graphml"):
        """call `apoc.export.FORMAT.all`, saving to provided filename"""
        self._validate_export_format(export_format)
        procedure = f"apoc.export.{export_format}.all"
        self.call_graph_procedure_assemble_arguments(self.graph, procedure, file=export_filename, config=config)

    def apoc_export_query(self, query, export_filename, config: Mapping = None,
                          export_format="graphml"):
        """call `apoc.export.FORMAT.query` with desired query, saving to provided filename"""
        self._validate_export_format(export_format)
        procedure = f"apoc.export.{export_format}.query"
        self.call_graph_procedure_assemble_arguments(self.graph, procedure,
                                                     query=query, file=export_filename, config=config)

    @classmethod
    def call_graph_procedure_assemble_arguments(cls, graph, procedure,
                                                ordered_procedure_args=('query', 'file', 'config'),
                                                empty_default_args=('config',),
                                                **kwargs):
        """
        Wrapper to call a graph procedure after assembling and ordering arguments, defaulting to empty dict as specified
        For example: CALL apoc.export.graphml.all(file, config), default to empty config on apoc export procedures

        :param graph: py2neo neo4j connection
        :param procedure: name of procedure to call
        :param ordered_procedure_args: keys to extract and assemble as arguments, if present in kwargs
        :param empty_default_args: required keys to default to None if not in kwargs
        :param kwargs: arguments and data for the procedure call
        """
        for a in empty_default_args:
            kwargs[a] = kwargs.get(a, None)
        logger.debug(f"call apoc export, kwargs {kwargs}")
        arg_list = [kwargs[arg] for arg in kwargs if arg in ordered_procedure_args]
        logger.debug(f"became arg list {arg_list}")
        graph_call_procedure(graph, procedure, *arg_list)

    def _query_with_native_driver(
            self,
            query,
            parameters=None,
            db=None,
            verbose=True
    ):
        """
        Uses a Cypher query to manipulate data in a Neo4j instance
        via the native driver.

        Parameters
        ----------
        query : str
            Cypher query
        parameters : dict of form {'str': value(s)}, optional
            Value(s) to feed into the query by means of
            parameter insertion. Useful for sending over large
            amounts of data in a query efficiently, by default None
        db : str, optional
            Name of the database to query, if more than one
            are supported by the target Neo4j instance (usually
            only available for Enterprise Edition), by default None
        verbose : bool, optional
            If True, will provide logger messages
            when the query starts and finishes, by default True

        Returns
        -------
        pandas DataFrame
            The results of the query. If no results are returned,
            will be an empty DataFrame.

        Raises
        ------
        neo_exc
            Generic Neo4j error that indicates a problem that can't
            be solved simply through retrying. Usually raised due to
            a bad Cypher query.
        value_exc
            Indicates a problem with the received/sent data, although
            usually a deque error indicating that the Neo4j data queue wasn't
            properly cleared before a new query was run.
        """
        assert self._native_driver is not None, "Driver not initialized!"
        session = None
        response = None
        q = query.replace("\n", " ").strip()

        while True:
            try:
                if verbose:
                    logger.info("Neo4j query started...")
                if db is not None:
                    session = self._native_driver.session(database=db)
                else:
                    session = self._native_driver.session()
                results = session.run(q, parameters)
                response = pd.DataFrame(results, columns=results.keys())
                if verbose:
                    logger.info("Neo4j query complete!")
                break

            except ServiceUnavailable:
                logger.error("Not able to communicate with Neo4j due to "
                             "ServiceUnavailable, retrying...")

            except Neo4jError as neo_exc:
                raise neo_exc

            except ValueError as value_exc:
                if value_exc.args[0] == 'deque.remove(x): x not in deque':
                    logger.error("deque error detected, retrying query...")
                else:
                    raise value_exc

            # Guarantee that the session is closed when query is done
            finally:
                if session is not None:
                    session.close()

        return response

    def _neo4j_arrow_process_whole_ticket(self, ticket, return_df=True):
        """Take an arrow ticket, stream it, read all into a pyarrow table,
            optionally return as a pandas dataframe, else a pyarrow table"""
        result = self._neo4j_arrow_driver.stream(ticket).read_all()
        if return_df:
            return result.to_pandas()
        return result

    def _query_cypher_arrow(self, query: str, params=None, return_df=True):
        """
        Sends a cypher query via neo4j-arrow driver,
        optionally return as a pandas dataframe, else a pyarrow table

        Args:
            query (str): cypher query
            params (Optional[Dict[str, Any]], optional): [description]. Defaults to None.
            return_df (bool, optional): whether to call .to_pandas() on the resulting pyarrow table. Defaults to True
        """
        ticket = self._neo4j_arrow_driver.cypher(cypher=query, database=self.database_name, params=params)
        return self._neo4j_arrow_process_whole_ticket(ticket, return_df=return_df)

    def _query_gds_nodes_arrow(self, graph_name, properties=None, filters=None, node_id: str = '', extra=None,
                               return_df=True):
        """
        Submit a GDS arrow job for Node properties, optionally return as a pandas dataframe, else a pyarrow table

        Args:
            graph_name (str): graph projection name
            filters (Optional[List[str]], optional): filters for GDS read job. Defaults to None.
            properties (Optional[List[str]], optional): parameters for GDS read job. Defaults to None.
            node_id (str, optional): [description]. node_id for GDS read job. Defaults to ''.
            extra (Optional[Dict[str, Any]], optional): extra dict params for GDS read job. Defaults to None.
            return_df (bool, optional): whether to call .to_pandas() on the resulting pyarrow table. Defaults to True
        """
        ticket = self._neo4j_arrow_driver.gds_nodes(graph=graph_name, database=self.database_name,
                                                    properties=properties, filters=filters, node_id=node_id,
                                                    extra=extra)
        return self._neo4j_arrow_process_whole_ticket(ticket, return_df=return_df)

    def _query_gds_relationships_arrow(self, graph_name, properties=None, filters=None, node_id: str = '', extra=None,
                                       return_df=True):
        """
        Submit a GDS arrow job for Relationship properties,
        optionally return as a pandas dataframe, else a pyarrow table

        Args:
            graph_name (str): graph projection name
            filters (Optional[List[str]], optional): filters for GDS read job. Defaults to None.
            properties (Optional[List[str]], optional): parameters for GDS read job. Defaults to None.
            node_id (str, optional): [description]. node_id for GDS read job. Defaults to ''.
            extra (Optional[Dict[str, Any]], optional): extra dict params for GDS read job. Defaults to None.
            return_df (bool, optional): whether to call .to_pandas() on the resulting pyarrow table. Defaults to True
        """
        ticket = self._neo4j_arrow_driver.gds_relationships(graph=graph_name, database=self.database_name,
                                                            properties=properties, filters=filters, node_id=node_id,
                                                            extra=extra)
        return self._neo4j_arrow_process_whole_ticket(ticket, return_df=return_df)

    # TODO write neo4j_arrow wrapper for WRITE: bulk_import, gds_write_nodes, gds_write_relationships

    def insert_data(self, query, rows, batch_size=1_000):
        """
        Inserts data in batches to a Neo4j instance.

        Parameters
        ----------
        query : str
            Cypher query for doing the data insertion.
            Should include 'UNWIND $rows AS row' near
            the top to ensure the data are pushed appropriately, but
            will be added at beginning of query if missing. Note that it's
            also usually best practice to have any MATCH clause be of the form
            "MATCH (n:Label {id: row.id})".
        rows : iterable of data, often pandas DataFrame
            Tabular data to be inserted in batches
        batch_size : int, optional
            Number of rows to insert per batch, by default 10000

        Returns
        -------
        Nothing.
        """
        if 'unwind $rows as row' not in query.lower():
            logger.warning("Query is missing UNWIND statement for ``rows``, "
                           "inserting it at the beginning of query now...")
            query = 'UNWIND $rows AS row\n' + query
            logger.info(f"New query: {query}")

        for start_index in tqdm(
                range(0, len(rows), batch_size),
                desc='Inserting data into Neo4j',
                unit_scale=batch_size
        ):
            end_index = start_index + batch_size
            if end_index > len(rows):
                end_index = len(rows)

            _ = self._query_with_native_driver(
                query,
                verbose=False,
                parameters={
                    'rows': rows.iloc[start_index: end_index].replace({np.nan: None}).to_dict('records')
                }
            )

    def cypher_query_to_dataframe(self, query, parameters=None, verbose=True, max_num_retries=5,
                                  drivers=None):
        """
        Runs a Cypher query against this graph and returns the results as a
        pandas DataFrame. Tries arrow, then py2neo, then native driver for speed's sake.

        Note that this is most useful for queries that return truly tabular data
        (e.g. only return node and relationship properties, not the nodes and/or
        relationships themselves).

        Parameters
        ----------
        query: Cypher query represented as a string.

        parameters: dict of str:value pairs. Sets parameters that can be
            used in the query being sent along. Useful for ensuring that complex
            or frequently-used parameters can be reference performantly.

            Example: cypher_query_to_dataframe("MATCH (n) WHERE n.name = $name",
                        parameters={'name': 'John'})

        verbose: bool. Indicates if logging messages should be enabled at INFO
            level.

        max_num_retries: int. retry at maximum this number of times (handles potential endless loops)

        drivers: str, None, or List<str>. Ordered list of choices of which driver(s) should be used under the hood.
            If a str, tries that single driver. If a list, provides fallback order in the case of errors.
            Defaults to ('neo4j-arrow', 'py2neo', 'native'), the order the code is structured
        Returns
        -------
        pandas DataFrame.
        """
        q = query.replace("\n", " ").strip()
        num_retries = 0

        def _init_drivers_to_try(d):
            # convert to list for popping when we try stuff
            if not d:
                d = Neo4jConnectionHandler.NEO4J_QUERY_DRIVERS
            elif isinstance(d, str):
                d = (d, )
            logger.debug(f"trying drivers: {d}")
            return list(d)

        class SkipError(Exception):
            """ Error to catch when skipping """
            pass

        drivers_to_try = _init_drivers_to_try(drivers)
        if verbose:
            logger.info(f"Neo4j query starting...")
            logger.debug(f"DB IP:`{self.db_ip}` query: `{q}`")
        while True:
            output = None  # try, in succession, neo4j-arrow, py2neo, native driver...
            # TODO: any way to incorporate tqdm into this for progress bar on data download?
            #  e.g. by paging through results instead of allowing arbitrary return size?
            try:
                if drivers_to_try[0] != 'neo4j-arrow':
                    raise SkipError("skipping neo4j-arrow driver")
                drivers_to_try.remove('neo4j-arrow')
                if verbose:
                    logger.debug("trying neo4j-arrow...")
                output = self._query_cypher_arrow(query=q, params=parameters)
            except (neo4j_arrow.flight.FlightError, neo4j_arrow.pa.ArrowException, IndexError, SkipError) as e:
                if isinstance(e, SkipError):
                    if verbose:
                        logger.debug(f"Skipping to py2neo driver... {e}")
                else:
                    logger.warning(f"neo4j-arrow experienced an error. Falling back to py2neo... error: {e}")
                try:
                    if drivers_to_try[0] != 'py2neo':
                        raise SkipError("skipping py2neo driver")
                    drivers_to_try.remove('py2neo')
                    if verbose:
                        logger.debug("trying py2neo...")
                    output = self._py2neo_driver.run(q, parameters).to_data_frame()
                except ConnectionBroken as e:
                    # pass to a retry in this specific instance...
                    logger.error(f"Query experienced a server communication issue via ConnectionBroken {e}. "
                                 f"Retrying...")
                except (py2neo.errors.Neo4jError, ValueError, IndexError, SkipError) as f:
                    # retry with native driver
                    if isinstance(f, SkipError):
                        if verbose:
                            logger.debug(f"Skipping to native driver... {f}")
                    else:
                        logger.warning(f"py2neo experienced an error. Falling back to native driver... error: {f}")
                    # and raise all errors at this point... we're SOL if this fails
                    try:
                        if drivers_to_try[0] != 'native':
                            raise SkipError("skipping native driver")
                    except IndexError as g:
                        logger.error(f"No drivers left to try. We have left {drivers_to_try}, "
                                     f"when you originally passed: {drivers}")
                        raise g
                    drivers_to_try.remove('native')
                    if verbose:
                        logger.debug("trying native...")
                    logger.warning("The native Neo4j driver is designed to be "
                                   "setup with more granular query "
                                   "transaction logic, so is not usually "
                                   "ideal for READ-type queries")
                    output = self._query_with_native_driver(query=q, parameters=parameters, verbose=False)
            # if output is set at this point, then we succeeded somehow. otherwise, retry...
            if output is not None:
                if verbose:
                    logger.info("Neo4j query complete!")
                break
            else:
                num_retries += 1
                if num_retries >= max_num_retries:
                    logger.error(f"retried {num_retries} times... breaking")
                    raise ValueError(f"max retries {max_num_retries} reached in cypher_query_to_dataframe! "
                                     f"query `{query}` parameters `{parameters}`")
                else:
                    if verbose:
                        logger.info(f"on retry {num_retries} of {max_num_retries} ... ")
                    drivers_to_try = _init_drivers_to_try(drivers)

        return output.replace({None: np.nan})


class Nodes:
    """
    A representation of a whole class of Neo4j nodes (e.g. nodes that all
    share the same label(s)) based off of a DataFrame that provides information
    at the individual node level for properties and such.
    """

    def __init__(
            self,
            parent_label,
            data,
            id_column,
            reference,
            additional_labels=None,
            properties=None
    ):
        """
        Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, that column is assumed to be of the 'string'
        data type.

        Parameters
        ----------
        parent_label: str. Indicates the highest-level descriptive node label
            to apply. For example, when describing a node for a paper author,
            "Person" would be the parent label and ["Author"] would be the
            additional labels.

        data: pandas DataFrame. Must have at least one column named the same
            as the value passed to ``id_column``. Defines instances of
            individual nodes. Extra columns should include values of properties
            each node will have, one column per property.

        id_column: str. Name of the column containing the unique IDs for the
            nodes.

        reference: str. An alias to use for the type of node when referring
            to it in other Nodes objects. E.g. 'paper'.

        additional_labels: list of str. Indicates what label(s) you want associated
            with the nodes. For example, when describing a node for a paper author,
            "Person" would be the parent label and ["Author"] would be the
            additional labels.

        properties: pandas DataFrame or list of dicts with columns/keys
        'old', 'new', and 'type'. These are the existing column names,
        new column names, and Neo4j data types that each property should be
        associated with, resp.
        """

        assert isinstance(additional_labels,
                          list) or additional_labels is None, "``additional_labels`` not list or None"

        if additional_labels is None:
            additional_labels = []

        self.parent_label = parent_label
        self.additional_labels = additional_labels
        self.labels = [parent_label] + additional_labels
        self.reference = reference

        self.id = f'id:ID({reference}-ref)'

        if id_column not in data.columns:
            raise ValueError(f"id_column value of '{id_column}' not found \
in data.columns")

        if isinstance(properties, pd.DataFrame):
            self.column_conversions = properties.copy()

        elif isinstance(properties, dict):
            self.column_conversions = pd.DataFrame(properties)

        elif properties is not None:
            raise ValueError(f"``properties`` must be of type dict or \
    pandas.DataFrame. Got {type(properties)} instead.")

        else:
            self.properties = None

        if properties is not None:
            self.properties = self.column_conversions['new'].tolist()
            self.data = format_property_columns_for_ingest(
                data,
                renaming_rules=self.column_conversions,
                return_only_renamed=True
            )

        else:
            logger.warn(f"No properties identified, so {self.reference}-type nodes will only have an ID and a label!")
            self.data = pd.DataFrame(data[id_column])

        if id_column in self.data.columns:
            self.data = self.data.rename(columns={id_column: self.id})

        else:
            self.data[self.id] = data[id_column]

        # TODO: how to make this capable of accounting for multiple possible
        # labels varying across rows?
        self.data[':LABEL'] = ';'.join(self.labels)

        # Make sure we have IDs for every node, otherwise drop
        num_null_ids = self.data[self.id].isnull().sum()
        if num_null_ids > 0:
            logger.warning(f"Found {num_null_ids} null node IDs! Dropping...")
            self.data.dropna(subset=[self.id], inplace=True)

        # Make sure we don't have any duplicate nodes
        # Drop all but the first duplicate if we do
        num_duplicate_nodes = self.data.duplicated(subset=self.id, keep='first').sum()
        if num_duplicate_nodes > 0:
            logger.warn(f"Found {num_duplicate_nodes} duplicate {self.reference} node IDs! \
Removing all but the first...")
            self.data.drop_duplicates(subset=self.id, keep='first', inplace=True)

    # TODO: incorporate concept of addition as a Nodes method to allow for combinations of Nodes
    #  into a single object like we do for funders + institutions

    def __str__(self):
        output = f"Nodes object with {len(self.data):,} unique nodes and with"

        if len(self.labels) > 1:
            output += f" labels {self.labels}."

        else:
            output += f" label '{self.labels[0]}'."

        if self.properties:
            output += f" Most of these nodes have properties {self.properties}"

        return output

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)

    def export_to_csv(self, filepath):
        """
        Given everything we know about these objects, export the results
        into a CSV that ``neo4j-admin load`` can ingest.


        Parameters
        ----------
        filepath: str. Indicates where to save the exported CSV. Should be of
            the form 'path/to/file.csv'.


        Returns
        -------
        Nothing.
        """
        self.filepath = filepath
        filepath_directories_only = os.path.split(filepath)[0]

        if not os.path.exists(filepath_directories_only) and filepath_directories_only != '':
            logger.warn(f"Filepath {filepath_directories_only} not found, creating it now...")
            pathlib.Path(filepath_directories_only).mkdir(parents=True, exist_ok=True)

        export_df(self.data, filepath, neo4j_object_type='nodes')

    def export_to_neo4j(self, graph, batch_size=1_000):
        """
        Exports Nodes data to running Neo4j instance

        Parameters
        ----------
        graph : Neo4jConnectionHandler object
            Graph connection to the target Neo4j instance
        batch_size : int
            The maximum number of nodes and their data to send at once to Neo4j.
            Note that this number should be adjusted downward if you have a large
            number of properties per node.

        Raises
        ------
        ValueError
            Checks if ``graph`` is of the proper class
        """
        if not isinstance(graph, Neo4jConnectionHandler):
            raise ValueError("``graph`` must be of type Neo4jConnectionHandler")

        data = self.data.rename(
            columns={
                self.id: 'id',
            }
        )
        # Get as many Neo4j-compatible dtypes as possible
        data = convert_types(data)

        # Find out which are datetime columns, if any
        data, datetime_columns = transform_timestamps(data)

        # Need to treat datetime columns as special when sending over
        if datetime_columns is not None:
            node_properties = ", ".join([f"n.{p} = row.{p}" for p in self.properties if p not in datetime_columns])
            node_properties += ", "
            node_properties += ", ".join([f"n.{c} = datetime(row.{c})" for c in datetime_columns])
            properties_clause = f"""
                SET
                    {node_properties}
                """

        elif self.properties is not None:
            node_properties = ", ".join([f"n.{p} = row.{p}" for p in self.properties])
            # Check that there are properties to set
            if len(node_properties) > 0 and self.properties:
                properties_clause = f"""
                SET
                    {node_properties}
                """
            else:
                raise ValueError(f"node properties not successfully set from self.properties: {self.properties}")
        else:
            properties_clause = ""

        # Check if there are labels to set beyond parent
        if self.additional_labels is not None and len(self.additional_labels) > 0:
            if len(properties_clause) > 0:
                properties_clause += ", " + f"n:{':'.join(self.additional_labels)}"
            else:
                properties_clause = f"n:{':'.join(self.additional_labels)}"

        query = f"""
        UNWIND $rows AS row
        MERGE (n:{self.parent_label} {{id: row.id}})
        {properties_clause}
        """

        logger.debug(f"Insertion is using the query \n{query}")

        graph.insert_data(
            query,
            data,
            batch_size=batch_size
        )


class Relationships:
    """
    A representation of a whole class of Neo4j nodes (e.g. nodes that all
    share the same label(s)) based off of a DataFrame that provides information
    at the individual node level for properties and such.
    """

    def __init__(
            self,
            type,
            data,
            start_node,
            end_node,
            id_column_start=None,
            id_column_end=None,
            allow_unknown_nodes=False,
            properties=None
    ):
        """
         Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, that column is assumed to be of the 'string'
        data type.

        Parameters
        ----------
        type: str. Indicates what the relationship type is (e.g.
            'AFFILIATED_WITH'). Can only be one value.

        data: pandas DataFrame. Must have at least one column named the
            equivalent of ``start_node.id`` and one named the equivalent of
            ``end_node.id``. Defines the desired connections from individual
            start nodes to individual end nodes. Extra columns should include
            values of properties each relationships will have, one column
            per property.

        start_node: Nodes object. This will be used to derive the node
            reference to be used at the start of the relationship, check that
            all relationships provided by ``data`` are comprised only of
            existing nodes, etc.

        end_node: Nodes object. This will be used to derive the node
            reference to be used at the end of the relationship, check that
            all relationships provided by ``data`` are comprised only of
            existing nodes, etc.

        id_column_start: str. If, for some reason, the column needed from
            ``data`` for the starting nodes mapping is named something
            different than what is provided by ``start_node.id``, provide it
            here. For example, papers citing other papers would result in two
            ID columns that refer to the same set of nodes, but both columns
            in ``data`` should not be called the same thing (start_node.id).

        id_column_end: str. If, for some reason, the column needed from
            ``data`` for the ending nodes mapping is named something
            different than what is provided by ``end_node.id``, provide it
            here.

        allow_unknown_nodes: bool. If True, relationships may be defined in
            ``data`` that do not have corresponding node IDs present in either
            ``start_node`` or ``end_node``.

        properties: pandas DataFrame or list of dicts with columns/keys
            'old', 'new', and 'type'. These are the existing column names,
            new column names, and Neo4j data types that each property should
            be associated with, resp.
        """

        self.type = type

        self.start_id = f':START_ID({start_node.reference}-ref)'
        self.start_node_labels = start_node.labels

        self.end_id = f':END_ID({end_node.reference}-ref)'
        self.end_node_labels = end_node.labels

        self.start_reference = start_node.reference
        if id_column_start is None:
            self._start_id_input = start_node.id

        else:
            self._start_id_input = id_column_start

        self.end_reference = end_node.reference
        if id_column_end is None:
            self._end_id_input = end_node.id

        else:
            self._end_id_input = id_column_end

        if self._start_id_input not in data.columns:
            raise ValueError(f"Start node ID column '{self._start_id_input}' not found \
in data.columns")

        elif self._end_id_input not in data.columns:
            raise ValueError(f"End node ID column '{self._end_id_input}' not found \
in data.columns")

        if isinstance(properties, pd.DataFrame):
            self.column_conversions = properties.copy()

        elif isinstance(properties, dict):
            self.column_conversions = pd.DataFrame(properties)

        elif properties is not None:
            raise ValueError(f"``properties`` must be of type dict or \
    pandas.DataFrame. Got {type(properties)} instead.")

        else:
            self.properties = None

        if properties is not None:
            self.properties = self.column_conversions['new'].tolist()
            self.data = format_property_columns_for_ingest(
                data,
                renaming_rules=self.column_conversions,
                return_only_renamed=True
            )

            self.data[self.start_id] = \
                data[self._start_id_input]

            self.data[self.end_id] = \
                data[self._end_id_input]

        else:
            self.data = data[[self._start_id_input, self._end_id_input]].rename(
                columns={
                    self._start_id_input: self.start_id,
                    self._end_id_input: self.end_id
                })

        id_columns = [self.start_id, self.end_id]
        num_duplicates = self.data.duplicated(subset=id_columns).sum()
        num_null_ids = self.data[id_columns].isnull().sum().sum()

        if num_null_ids > 0:
            logger.warning(f"Dropping {num_null_ids} relationships with at "
                           "least one null node ID...")
            self.data.dropna(subset=id_columns, inplace=True)

        if num_duplicates > 0:
            logger.warn(f"Dropping {num_duplicates} relationships that are \
duplicative.")
            self.data.drop_duplicates(subset=id_columns, inplace=True)

        # Make sure we aren't connecting nodes that we weren't given
        # unless we should!
        if not allow_unknown_nodes:
            bad_nodes_start = (
                ~self.data[self.start_id].isin(start_node.data[start_node.id])
            )
            num_bad_nodes_start = bad_nodes_start.sum()

            if num_bad_nodes_start > 0:
                logger.warn(f"Dropping {num_bad_nodes_start} relationship \
mappings for {start_node.reference}-type start nodes as they don't exist in \
the Nodes data provided...")

                self.data = self.data[~bad_nodes_start]

            bad_nodes_end = (
                ~self.data[self.end_id].isin(end_node.data[end_node.id])
            )
            num_bad_nodes_end = bad_nodes_end.sum()

            if num_bad_nodes_end > 0:
                logger.warn(f"Dropping {num_bad_nodes_end} relationship \
mappings for {end_node.reference}-type end nodes as they don't exist in \
the Nodes data provided...")

                self.data = self.data[~bad_nodes_end]

        # Check that we don't have any papers citing themselves!
        if self.start_reference == self.end_reference:
            self_connecting = (self.data[self.start_id] == self.data[self.end_id])
            num_self_connecting = self_connecting.sum()

            if num_self_connecting > 0:
                logger.warn(f"Dropping {num_self_connecting} relationships that \
start and end at the same node. How did that happen?!")
                self.data = self.data[~self_connecting]

        self.data[':TYPE'] = self.type

    def __str__(self):
        output = f"Relationships object structured as \
({self.start_reference})-[:{self.type}]->({self.end_reference}) with \
{len(self.data):,} unique relationships."

        if self.properties is not None:
            output += f" Most of these relationships have properties {self.properties}"

        return output

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)

    def export_to_csv(self, filepath):
        """
        Given everything we know about these objects, export the results
        into a CSV that ``neo4j-admin load`` can ingest.

        Parameters
        ----------
        filepath: str. Indicates where to save the exported CSV. Should be of
            the form 'path/to/file.csv'.

        Returns
        -------
        Nothing.
        """
        self.filepath = filepath
        filepath_directories_only = os.path.split(filepath)[0]

        if not os.path.exists(filepath_directories_only) and filepath_directories_only != '':
            logger.warn(f"Filepath {filepath_directories_only} not found, creating it now...")
            pathlib.Path(filepath_directories_only).mkdir(parents=True, exist_ok=True)

        export_df(self.data, filepath, neo4j_object_type='relationships')

    def export_to_neo4j(self, graph, batch_size=1_000):
        """
        Exports Relationships data to running Neo4j instance

        Parameters
        ----------
        graph : Neo4jConnectionHandler object
            Graph connection to the target Neo4j instance
        batch_size : int
            The maximum number of edges and their data to send at once to Neo4j.
            Note that this number should be adjusted downward if you have a large
            number of properties per node.

        Raises
        ------
        ValueError
            Checks if ``graph`` is of the proper class
        """
        if not isinstance(graph, Neo4jConnectionHandler):
            raise ValueError("``graph`` must be of type Neo4jConnectionHandler")

        data = self.data.rename(
            columns={
                self.start_id: 'source_node_id',
                self.end_id: 'target_node_id'
            }
        )
        # Find out which are datetime columns, if any
        data, datetime_columns = transform_timestamps(data)

        # Need to treat datetime columns as special
        if datetime_columns is not None:
            relationship_properties = ", ".join(
                [f"r.{p} = row.{p}" for p in self.properties if p not in datetime_columns])
            if len(relationship_properties) > 0:
                relationship_properties += ", "
            relationship_properties += ", ".join([f"r.{c} = datetime(row.{c})" for c in datetime_columns])

            properties_clause = f"""
                SET
                    {relationship_properties}
                """

        elif self.properties is not None:
            relationship_properties = ", ".join([f"r.{p} = row.{p}" for p in self.properties])
            # Check that there are properties to set
            if len(relationship_properties) > 0 and self.properties:
                properties_clause = f"""
                SET
                    {relationship_properties}
                """
            else:
                raise ValueError(f"relationship_properties not successfully set from self.properties: "
                                 f"{self.properties}")
        else:
            properties_clause = ""

        query = f"""
        UNWIND $rows AS row
        MATCH
        (source:{':'.join(self.start_node_labels)} {{id: row.source_node_id}}),
        (target:{':'.join(self.end_node_labels)} {{id: row.target_node_id}})

        MERGE (source)-[r:{self.type}]->(target)
        {properties_clause}
        """

        graph.insert_data(
            query,
            data,
            batch_size=batch_size
        )


def test_graph_connectivity(db_ip, db_password, db_username='neo4j'):
    """
    Tries to set up a Neo4j connection to make sure all credentials, etc. are
    good to go.

    Parameters
    ----------
    db_ip : str
        IP address/URL of the DBMS to test
    db_password : str
        Password for the given username
    db_username : str, optional
        Username to use for DBMS authentication, by default 'neo4j'
    """
    logger.debug("Testing Neo4j database connectivity...")
    Neo4jConnectionHandler(
        db_ip=db_ip,
        db_username=db_username,
        db_password=db_password
    )
