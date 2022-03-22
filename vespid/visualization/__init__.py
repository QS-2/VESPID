import numpy as np
import pandas as pd
import dask.dataframe as dd
import warnings
import os
import plotly.express as px

import datashader as ds
import datashader.transfer_functions as tf
from datashader.layout import random_layout, forceatlas2_layout
from datashader.bundling import connect_edges, hammer_bundle
import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import (
    datashade, dynspread, bundle_graph
)
from numpy.random import randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from vespid import setup_logger
from vespid.data.neo4j_tools import Neo4jConnectionHandler
from vespid.data.neo4j_tools.utils import transform_timestamps
from vespid.data import find_columns_by_class
from vespid.features.interdisciplinarity import calculate_interdisciplinarity_score
from umap import UMAP
logger = setup_logger(__name__)

def visualize_graph(
    graph,
    query,
    node_properties=None,
    edge_properties=None,
    limit=None,
    cpu_limit=None,
    bundled=False,
    savepath=None,
    width=600,
    height=600,
    classbreak=None,
    node_size=10,
    layout=forceatlas2_layout,
    **layout_kwargs
):        
    '''
    Using a Neo4j connection, builds an interactive visualization of the
    resulting graph from a Cypher query, optionally including edge
    bundling.

    Parameters
    ----------
    graph : Neo4jConnectionHandler
        The connection to the Neo4j database supplying the data
    query : str
        Cypher query that indicates what data to return. Should have a single
        path pattern to be most effective, with the following naming scheme
        for nodes and edges:
        
        ```
        MATCH (n:Publication)-[e:CITED_BY]->(n2:Publication)
        RETURN ...
        ```
    node_properties : list of str, optional
        Names of the node properties to populate in the visualization, 
        by default None
    edge_properties : list of str, optional
        Names of the edge properties to populate in the visualization, 
        by default None
    limit : int, optional
        Max number of paths to return, by default None
    cpu_limit : int, optional
        Indicates how many cores to parallelize the computations over. 
        If None or -1, will use all available CPUs, by default None
    bundled : bool, optional
        If True, generates an edge-bundled graph of the data, by default False
    savepath : str, optional
        Filepath to use for saving graph viz HTML(s), should be of the form
        'path/to/file.html'. If None, will not save, by default None
    width : int, optional
        Width, in pixels, of the graphs
    height : int, optional
        Height, in pixels, of the graphs
    classbreak : str, optional
        Node property to use for categorical coloring. If None, all nodes will 
        be the same color, by default None
    node_size : int, optional
        Size of nodes in force-directed graph
    layout : callable, optional
        Graph layout algorithm for generating XY coordinates of nodes based
        on their edges (usually). Can be found, for example, in 
        `datashader.layout`, by default forceatlas2_layout

    Returns
    -------
    2-tuple of holoviews.element.graphs.Graph objects
        Returns tuple of the form (force-directed graph, edge-bundled graph), 
        wherein the latter is None if bundled is False.
    '''
    if "return" in query:
        query_start = query.split("return")[0]
    elif "RETURN" in query:
        query_start = query.split("RETURN")[0]
    else:
        warnings.warn(RuntimeWarning(
            "No RETURN clause found in query, assuming there isn't one..."
        ))
        query_start = query

    query_end = f"""
    RETURN ID(e) AS edgeID,
    ID(n) AS source, 
    ID(n2) AS target
    """

    query_paths = query_start + query_end

    if limit is not None:
        query_paths += f" LIMIT {limit}"
        
    data = graph.cypher_query_to_dataframe(query_paths)
    
    # Gather the nodes-only data
    node_ids = pd.DataFrame(
        np.unique(np.concatenate(
            [data['source'].values, data['target'].values]
            )), 
        columns=['nodeID']
        ).to_dict('records')
    
    if node_properties is not None:
        node_properties_clause = ', ' + ', '.join([f"n.{p} AS {p}" for p in node_properties])
    else:
        node_properties_clause = ''
        
    query_nodes = f"""
    UNWIND $rows AS row
    MATCH (n)
    WHERE ID(n) = row.nodeID
    RETURN DISTINCT ID(n) AS nodeID,
    labels(n) AS labels
    {node_properties_clause}
    """

    logger.info("Querying for unique nodes data...")
    df_nodes = graph.cypher_query_to_dataframe(
        query_nodes, 
        parameters={'rows': node_ids},
        verbose=True
    )
    
    # Transform datetime data into strings
    df_nodes, _ = transform_timestamps(
        df_nodes
    )
    list_columns = find_columns_by_class(df_nodes, [list])
    if len(list_columns) > 0:
        df_nodes[list_columns] = df_nodes[list_columns].astype(str)
    
    # Gather the edges-only data
    edge_ids = pd.DataFrame(
        np.unique(data['edgeID'].values), 
        columns=['edgeID']
        ).to_dict('records')
    
    if edge_properties is not None:
        edge_properties_clause = ', ' + ', '.join([f"e.{p} AS {p}" for p in edge_properties])
    else:
        edge_properties_clause = ''
        
    query_arrow = '>' if Neo4jConnectionHandler._query_is_directed(query) else ''
    query_edges = f"""
    UNWIND $rows AS row
    MATCH (source)-[e]-{query_arrow}(target)
    WHERE ID(e) = row.edgeID
    RETURN ID(source) AS source,
    ID(e) AS edgeID,
    ID(target) AS target
    {edge_properties_clause}
    """

    logger.info("Querying for unique edges data...")
    df_edges = graph.cypher_query_to_dataframe(
        query_edges, 
        parameters={'rows': edge_ids},
        verbose=True
    )
    
    # Transform datetime data into strings
    df_edges, _ = transform_timestamps(
        df_edges
    )
    
    # Transform list objects to strings
    list_columns = find_columns_by_class(df_edges, [list])
    if len(list_columns) > 0:
        df_edges[list_columns] = df_edges[list_columns].astype(str)
    
    # Get the laid-out version of the nodal data    
    logger.info("Applying chosen node layout...")
    df_nodes = layout(
        df_nodes,
        df_edges,
        **layout_kwargs
    )
    
    logger.info("Dask-ifying nodes and edges data for efficient parallel processing...")
    if cpu_limit is None or cpu_limit == -1:
        cpu_limit = os.cpu_count()
    elif cpu_limit < -1:
        raise ValueError("`cpu_limit` arg should be -1 or greater")
    df_nodes = dd.from_pandas(df_nodes, npartitions=cpu_limit)
    df_edges = dd.from_pandas(df_edges, npartitions=cpu_limit)
    
    # Do the plotting!
    # Activate notebook extension
    hv.extension('bokeh')
    
    # Make sure the node columns are ordered as expected by HoloViews
    ordered_columns = [c for c in df_nodes.columns if c not in ['x', 'y', 'nodeID']]
    ordered_columns = ['x', 'y', 'nodeID'] + ordered_columns
    hv_nodes = hv.Nodes(df_nodes[ordered_columns])
    
    color_map = get_best_holoviews_colormap(bg='dark', data_type='Categorical')
    
    ordered_columns = [c for c in df_edges.columns if c not in ['source', 'target']]
    ordered_columns = ['source', 'target'] + ordered_columns
    graph = hv.Graph((df_edges[ordered_columns], hv_nodes), label=f'Graph of {len(df_edges):,} edges')\
        .options(cmap=color_map, colorbar=True, node_size=node_size, edge_line_width=1,
              node_line_color='gray', node_color=classbreak, bgcolor='black',
              xaxis=None, yaxis=None, show_grid=False,
              height=height, width=width, edge_line_color='white')
    
    if savepath is not None:
        # Prep layout params for use in graphics/file saving
        layout_param_string = '_'.join([f"{k}={v}" for k,v in layout_kwargs.items()])
        hv.save(graph, savepath)
    
    if bundled:
        bundled_layout = bundle_graph(graph, split=False)
        bundled_graph = dynspread(datashade(bundled_layout,cmap=color_map))\
            .options(plot=dict(height=height, width=width, colormap=True, 
                               xaxis=None, yaxis=None, show_grid=False, 
                               bgcolor="black"))
            
        if savepath is not None:
            path_and_filename, extension = os.path.splitext(savepath)
            hv.save(bundled_graph, f'{path_and_filename}_bundled{extension}')
            
        return graph, bundled_graph

    else:
        return graph, None
    
def get_best_holoviews_colormap(bg=None, data_type=None, return_all=False):
    '''
    Returns the first viable string representing a good node color mapping
    based on your parameters.
    
    More information can be found at 
    https://holoviews.org/user_guide/Colormaps.html, 
    including visualizations of available colorbars.

    Parameters
    ----------
    bg : str, optional
        Indicates coloring of plot background, which can impact best color set
        to use. Can be one of ['dark', 'light'], by default None
        
    data_type : str, optional
        Indicates the type of data being mapped. Allowed values are
        ['Categorical', 'Uniform Sequential', 'Diverging', 'Rainbow', 
        'Mono Sequential', 'Other Sequential', 'Miscellaneous']
        A value of None will return the first of all colormap options. 
        
    return_all : bool, optional
        If True, will not select just the first but return the full list of
        available relevant colormaps.

    Returns
    -------
    str or list of str
        The HoloViews-compatible colormap reference string(s). Whether it's
        a str or a list of str is dictated by `return_all`.
    '''
    
    allowed_background_types = ['light', 'dark']
    if bg is not None and bg not in allowed_background_types:
        raise ValueError(f"`bg` must be one of {allowed_background_types}")
    
    allowed_data_types = [
        'Categorical', 
        'Uniform Sequential', 
        'Diverging', 
        'Rainbow', 
        'Mono Sequential', 
        'Other Sequential', 
        'Miscellaneous'
    ]
    if data_type is not None and data_type not in allowed_data_types:
        raise ValueError(f"`data_type` must be one of {allowed_data_types}")
    
    if return_all:
        return hv.plotting.list_cmaps(category=data_type, bg=bg)
    else:
        return hv.plotting.list_cmaps(category=data_type, bg=bg)[0]

def make_interactive_bundled_graph(
    force_directed_graph, 
    bundled_graph, 
    double_width=True, 
    node_size=5
):
    '''
    Overlays the nodes from `force_directed_graph` (which are assumed to 
    have interactive tooltips already) on an edge-bundled variant of the same
    graph, allowing for interactive exploration of the edge-bundled graph.

    Parameters
    ----------
    force_directed_graph : holoviews.Graph object
        The force-directed graph at the core of it all
    bundled_graph : holoviews.DynamicMap object
        The edge-bundled version of `force_directed_graph`
    double_width : bool, optional
        If True, doubles the width of the resulting bundled graph
        relative to the original width of `force_directed_graph`, 
        allowing more space to engage with the graph and space for the legend, 
        by default True
    node_size : int, optional
        Sizing of the nodes. Values of 5 to 7 tend to be good for easily 
        interacting with node tooltips, minimizing node overlap, and still 
        being able to see bundled edges, by default 5

    Returns
    -------
    holoviews.DynamicMap object
        An overlay of the nodes from `force_directed_graph` on top of 
        `bundled_graph`
    '''
    graph_options_dict = force_directed_graph.opts.get().kwargs
    output = (
        bundled_graph * force_directed_graph.nodes.options(
            size=node_size, color=graph_options_dict['node_color'], 
            cmap=graph_options_dict['cmap']
            )
        )
    if double_width:        
        return output.options(width=graph_options_dict['width'] * 2)


def visualize_random_forest(random_forest, write_graphviz, out_file=None, overwrite=False,
                            tree_suffix='tree', get_trees_attribute='estimators_', samples=1.0,
                            **kwargs):
    """
    Visualize a random forest, optionally export, and optionally convert w/graphviz
    :param random_forest: the random forest to visualize as successive images
    :param write_graphviz: True/False whether or not to export to .png
    :param out_file: None if want a string result, else where save, e.g., 'path/to/tree.dot' or 'path/to/tree.png'
    :param overwrite: True/False whether to raise an error or warning if `out_file` exists
    :param samples: integer n_samples or float percent of trees to visualize
    :param tree_suffix: suffix of each `out_file` to append
    :param get_trees_attribute: attribute of the forest to get the trained trees
    :param kwargs: params to pass to `sklearn.tree.export_graphviz`, e.g., max_depth=None,
                            feature_names=None, class_names=None, label='all',
                            filled=False, leaves_parallel=False, impurity=True,
                            node_ids=False, proportion=False, rotate=False,
                            rounded=False, special_characters=False, precision=3,
    :return: nothing, or the string result of the conversion depending on `out_file`
    """
    trees = getattr(random_forest, get_trees_attribute)
    num_trees = len(trees)
    if isinstance(samples, float):
        from math import floor
        k = floor(num_trees * samples)
    elif isinstance(samples, int):
        k = min(samples, num_trees)
    else:
        raise ValueError("samples must be float or int; you passed %s" % samples)
    from random import sample
    trees = sample(trees, k=k)
    for idx, tree in enumerate(trees):
        if out_file is not None:
            from os.path import splitext
            o_file, extension = splitext(out_file)
            current_outfile = f"{o_file}_{tree_suffix}_{idx}.{extension}"
        else:
            current_outfile = None
        visualize_decision_tree(tree, write_graphviz=write_graphviz, out_file=current_outfile,
                                overwrite=overwrite, **kwargs)


def visualize_decision_tree(
    model, 
    write_graphviz=True, 
    out_file=None, 
    overwrite=False, 
    **kwargs
):
    """
    Visualize a random forest, optionally export, and optionally convert w/graphviz
    :param model: the tree/forest to visualize. Note that, if provided a RandomForest, a tree will be chosen at random
    :param write_graphviz: True/False whether or not to export to .png ; requires `graphviz`
    :param out_file: None if want a string result, else where save, e.g., 'path/to/tree.dot' or 'path/to/tree.png'
    :param overwrite: True/False whether to raise an error or warning if `out_file` exists
    :param kwargs: params to pass to `sklearn.tree.export_graphviz`, e.g., max_depth=None,
                            feature_names=None, class_names=None, label='all',
                            filled=False, leaves_parallel=False, impurity=True,
                            node_ids=False, proportion=False, rotate=False,
                            rounded=False, special_characters=False, precision=3,
    :return: nothing, or the string result of the conversion depending on `out_file`
    """
    from sklearn.tree import export_graphviz
    if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
        estimator = model
    elif isinstance(model, (RandomForestClassifier, RandomForestRegressor)):
        logger.warning("Detected model type of Random Forest, selecting a "
                       "decision tree at random to visualize...")
        estimator = model.estimators_[randint(0, len(model.estimators_))]
    else:
        raise ValueError(f"Models of type {type(model)} are not supported, "
                         "please provide a decision tree or random forest")
    
    if out_file is not None:
        # does the path exist / should we write to it?
        from vespid import check_filepath_exists
        check_filepath_exists(out_file, overwrite=overwrite)
        from os.path import splitext
        # verify appropriate extensions for provided file path
        o_file, extension = splitext(out_file)
        if write_graphviz:  # ensure it's .png
            if extension != '.png':
                logger.warning("switching extension to `png` for export")
            export_outfile = f"{o_file}.dot"  # but need intermediate .dot file
            out_file = f"{o_file}.png"
        else:  # ensure it's .dot
            if extension != 'dot':
                logger.warning("switching extension to `dot` for export")
            export_outfile = f"{o_file}.dot"
    else:
        export_outfile = None
    # run export command
    return_val = export_graphviz(estimator, out_file=export_outfile, **kwargs)
    if return_val:  # return value only if out_file was None; mutually exclusive from ability to convert to png
        return return_val
    elif write_graphviz:  # Convert to png using system command (requires Graphviz)
        from subprocess import run
        run(['dot', '-Tpng', export_outfile, '-o', out_file, '-Gdpi=600'])
        
def visualize_language_clusters(
    X,
    cluster_pipeline=None, 
    umap_model=None, 
    cluster_model=None
):
    if cluster_pipeline is not None:
        umap = cluster_pipeline.named_steps['umap']
        clusterer = cluster_pipeline.named_steps['hdbscan']
        logger.debug(f"{clusterer.score()=}")
        
    elif umap_model is not None and cluster_model is not None:
        umap = umap_model
        clusterer = cluster_model
    
    else:
        raise ValueError("`cluster_pipeline` must not be None or `umap_model` "
                         "and `cluster_model` must both not be None")
        
    params = umap.get_params(deep=False)
    params['n_components'] = 2 # for 2D viz
    umap_2d = UMAP(**params)
    logger.info("Fitting 2D UMAP projection...")
    X_2d = pd.DataFrame(umap_2d.fit_transform(X), columns=['x_0', 'x_1'])
    
    # Get cluster labels in there for coloring
    X_2d['cluster'] = pd.Series(clusterer.labels_).astype(str)
    
    # Size noise points smaller than rest
    X_2d['size'] = 0.2
    X_2d.loc[X_2d['cluster'] == '-1', 'size'] = 0.1
    
    logger.info("Calculating interdisciplinarity scores...")
    X_2d['interdisciplinarity_score'] = calculate_interdisciplinarity_score(
        clusterer.soft_cluster_probabilities.values
        #clusterer.soft_cluster(use_original_method=True).values
    )
    
    logger.info("Visualizing 2D projection...")
    # Show just the 2D plot with cluster labels as colors 
    # (should be same for both soft clustering approaches of course)
    fig = px.scatter(
        data_frame=X_2d,
        x='x_0',
        y='x_1',
        # width=1000, 
        # height=667, 
        color='cluster',
        size='interdisciplinarity_score',
        symbol='cluster'
    )
    
    # Add dark outline to markers so we can see the really tiny ones
    fig.update_traces(marker=dict(line=dict(width=2,
                                            color='DarkSlateGrey')))
    
    fig.write_html("figures/2011_SCORE_tuning_test_solution.html")
    fig.show()
    
    return X_2d, umap_2d