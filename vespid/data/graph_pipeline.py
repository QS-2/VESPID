import enum
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import argparse
import gc
from tqdm import tqdm
import sys
import os
from glob import glob
import shutil
import joblib
import pathlib

from pandas.core.indexes import multi
#from guppy import hpy # for memory leak detecting
import ray

from vespid.data.nsf_biblio import make_wos_author_dicts, make_wos_grant_dicts
from vespid.data.semantic_scholar import find_paper_ids
from vespid.data.make_dataset import (
    add_semantic_scholar_to_wos, 
    add_author_ids, 
    make_author_nodes, 
    make_publication_nodes,
    make_institution_nodes,
    make_citation_relationships,
    make_authorship_relationships,
    make_affiliations,
    make_funding_relationships
)

from vespid.data.neo4j_tools import (
    Neo4jConnectionHandler, 
    test_graph_connectivity
)
from vespid.pipeline import Stage, Pipeline, ParallelPipeline
from vespid.data.aws import (
    test_s3_access,
    upload_file,
    upload_object, 
    yield_object_keys, 
    count_s3_objects, 
    get_s3_file
)

from vespid import setup_logger, get_secure_key, get_current_datetime

SEMANTIC_SCHOLAR_API_KEY = get_secure_key(
    'semantic_scholar_API_key',
    bypass_safety_check=True,
    aws_secret=True
)

# Setup new logger that reports to stderr AND records to file
logger = setup_logger(__name__)

def setup_stages(
    input,
    db_ip,
    db_password,
    db_username='neo4j',
    s3_bucket='vespid',
    geocode=False,
    n_jobs=None,
    graph_batch_size=2_000,
    semantic_scholar_max_concurrency=None,
    starting_stage=None
):
    '''
    Generates lists of Stage objects for running parallel and serial processes
    to generate and transmit graph data. Specifically, the Stages are:
    
    *Parallel-Executed*
    1) Ingest pickled WoS records (batches of 25,000)
    2) Transform author information from WoS to be S2-compatible dictionaries
    3) Do something similar as in #2, but for funder data
    4) For those WoS records lacking a DOI, use text queries on S2 to find them
    5) With the set of known paper IDs/DOIs, pull down all S2 record data and 
        merge with WoS records
    6) Leverage S2's disambiguated author information and match author names 
        from WoS with S2 author data (including unique author IDs)
        
    *Serial-Executed*
    7) Pull in the output of the parallel Stages as a single big DataFrame
    8) Setup a fresh connection to our target Neo4j graph
    9) Create unique author node records and push to Neo4j
    10) Create unique publication node records and push to Neo4j
    11) Create unique employer/funder org node records and push to Neo4j
    12) Create and push citation edges
    13) Create and push authorship edges
    14) Create and push employer/affiliation edges
    15) Create and push funder edges
    

    Parameters
    ----------
    input : str or Any
        If `starting_stage` is None, S3 key indicating the input file to use 
        for the first parallel `Stage`. Otherwise, `input` should align with 
        whatever is expected as input for the first `Stage` specified.
    db_ip : str
        IP address of the target graph that will receive the processed data
    db_password : str
        Password for the target graph
    db_username : str, optional
        Username for the target graph, by default 'neo4j'
    s3_bucket : str, optional
        Name of the S3 bucket providing the input files, by default 'vespid'
    geocode : bool, optional
        If True, applies (SLOW) geocoding to Insitution nodes to assign 
        them lat/long coordinates, by default False
    n_jobs : int, optional
        Indicates how many parallel processes to use for CPU-bound 
        calculations, by default None. If None, 0, and 1 all have the same
        effect (only a single process is run). -1 will use all available CPU 
        cores except one. All other values must be positive.
    graph_batch_size : int, optional
        Number of records to send in a batch to the target graph. Larger
        values result in faster uploads, but too high of a value can overwhelm 
        Neo4j and force the connection to be terminated.
    semantic_scholar_max_concurrency : int, optional
        Indicates how many concurrent query workers are allowed across all 
        jobs. This sets the upper limit on how fast we can query that API.
    starting_stage: str, optional
        Name of the `Stage` to start from. Can be any `Stage`, in the parallel 
        or serial `Pipeline`s, so long as `input` conforms to what the `Stage` 
        is expecting. If None, starts from the first `Stage`. 

    Returns
    -------
    2-tuple of lists of Stages
         Of the form (parallel_stages, serial_stages)
    '''
    # Get the Stages set up    
    def _load_pickled_s3_object(bucket, key):
        output = pickle.loads(get_s3_file(bucket, key))
        
        # Track the input filename for forensic purposes
        input_filename = os.path.split(key)[-1]
        output['source'] = input_filename
        
        # Correct a weird issue with bad DOI columns in some input files
        if 'DOI' not in output.columns:
            logger.warning("'DOI' not found in columns for input file "
                           f"{input_filename}, extracting it now...")
            output['DOI'] = output['doi'].str.get(-1)
            output.drop(columns=['doi'], inplace=True)
            
        return output
    
    wos_pull = Stage(
        name='load_wos_api_results',
        function=_load_pickled_s3_object,
        cache_stage_output=False,
        bucket=(s3_bucket, False),
        key=(input, False)
    )

    wos_authors_transform = Stage(
        name='dictionary-ize_WoS_authors',
        function=make_wos_author_dicts,
        cache_stage_output=False,
        #df=('load_wos_api_results', True),
        inplace=(True, False)
    )

    wos_funding_transform = Stage(
        name='dictionary-ize_WoS_funding',
        function=make_wos_grant_dicts,
        cache_stage_output=False,
        #df=('dictionary-ize_WoS_authors', True),
        inplace=(True, False)
    )

    augment_paper_ids = Stage(
        name='augment_paper_ids_from_s2',
        function=find_paper_ids,
        cache_stage_output=False,
        #df=('dictionary-ize_WoS_funding', True),
        api_key=(SEMANTIC_SCHOLAR_API_KEY, False),
        score_threshold=(0.80, False),
        max_concurrent_requests=(semantic_scholar_max_concurrency, False),
        n_jobs=(n_jobs, False)
    )

    merge_wos_and_ss = Stage(
        name='augment_with_semantic_scholar',
        function=add_semantic_scholar_to_wos,
        cache_stage_output=False,
        #df=('augment_dois_from_crossref', True),
        api_key=(SEMANTIC_SCHOLAR_API_KEY, False),
        max_concurrent_requests=(semantic_scholar_max_concurrency, False),
        n_jobs=(n_jobs, False)
    )

    merge_ss_wos_authors = Stage(
        name='merge_ss_and_wos_author_data',
        function=add_author_ids,
        cache_stage_output=False,
        #df=('augment_with_semantic_scholar', True),
        inplace=(True, False)
    )
    
    def _pass_through_data(data):
        '''Simple function for holding data in a Stage to be used in a pipe'''
        return data
    
    serial_input_data = Stage(
        'serial_input_data',
        function=_pass_through_data,
        cache_stage_output=True,
        data=(None, False) # As we haven't executed the prior stages yet
    )
    
    setup_target_graph = Stage(
        'setup_target_graph',
        function=Neo4jConnectionHandler,
        cache_stage_output=True,
        db_ip=(db_ip, False),
        db_password=(db_password, False),
        db_username=(db_username, False)
    )

    author_nodes_stage = Stage(
        name='create_and_save_author_nodes',
        function=make_author_nodes,
        cache_stage_output=True,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True)
    )

    publication_nodes_stage = Stage(
        name='create_and_save_publication_nodes',
        function=make_publication_nodes,
        cache_stage_output=True,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True)
    )

    institution_nodes_stage = Stage(
        name='create_and_save_institution_nodes',
        function=make_institution_nodes,
        cache_stage_output=True,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True),
        geocode=(geocode, False)
    )

    make_citation_rels = Stage(
        name='create_and_save_citation_relationships',
        function=make_citation_relationships,
        cache_stage_output=False,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True),
        paper_nodes=('create_and_save_publication_nodes', True)
    )

    make_authorship_rels = Stage(
        name='create_and_save_authorship_relationships',
        function=make_authorship_relationships,
        cache_stage_output=False,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True),
        paper_nodes=('create_and_save_publication_nodes', True),
        author_nodes=('create_and_save_author_nodes', True)
    )

    make_affiliation_rels = Stage(
        name='create_and_save_affiliation_relationships',
        function=make_affiliations,
        cache_stage_output=False,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True),
        institution_nodes=('create_and_save_institution_nodes', True),
        author_nodes=('create_and_save_author_nodes', True)
    )

    make_funder_rels = Stage(
        name='create_and_save_funding_relationships',
        function=make_funding_relationships,
        cache_stage_output=False,
        df=(serial_input_data.name, True),
        batch_size=(graph_batch_size, False),
        graph=(setup_target_graph.name, True),
        institution_nodes=('create_and_save_institution_nodes', True),
        paper_nodes=('create_and_save_publication_nodes', True)
    )
    
    parallel_stages = [
        wos_pull,
        wos_authors_transform.use_preceding_input(),
        wos_funding_transform.use_preceding_input(),
        augment_paper_ids.use_preceding_input(),
        merge_wos_and_ss.use_preceding_input(),
        merge_ss_wos_authors.use_preceding_input()
    ]
    
    serial_stages = [
        serial_input_data,
        setup_target_graph,
        author_nodes_stage,
        publication_nodes_stage,
        institution_nodes_stage,
        make_citation_rels,
        make_authorship_rels,
        make_affiliation_rels,
        make_funder_rels
    ]
    
    if starting_stage is not None:
        all_stage_names = pd.Series(
            [s.name for s in parallel_stages] + [s.name for s in serial_stages]
        )
        if starting_stage not in all_stage_names:
            raise ValueError("`starting_stage` is not a name of any Stages")
        
        # Find the index corresponding to the start Stage and slice
        # Note that this assumes all Stage names are unique
        start_index = all_stage_names[all_stage_names == starting_stage].index[0]
        
        # Check which list needs updating
        if start_index < len(parallel_stages):
            parallel_stages = parallel_stages[start_index:]
        else:
            # Adjust for length of serial stages
            start_index -= len(parallel_stages)
            parallel_stages = None
            serial_stages = serial_stages[start_index:]
             
    
    return parallel_stages, serial_stages

def copy_ray_logs(
    datetime_identifier,
    target_path,
    ray_base_directory='/tmp/ray/',
    s3_bucket=None,
    s3_prefix=None
):
    
    ray_datetime_str = datetime_identifier.strftime("%Y-%m-%d_%H-%M-%S")
    candidate_directories = sorted(glob(
        ray_base_directory + 'session_' + ray_datetime_str + '*'
    ))
    num_candidates = len(candidate_directories)
    if num_candidates > 1:
        logger.warning(f"Found {num_candidates} possible ray directories to "
                       "choose from, picking the latest one...")
        ray_session_directory = candidate_directories[-1]
    elif num_candidates == 0:
        logger.error(f"Found no ray directories to copy logs from..."
                     "what happened?")
        return False
    else:
        ray_session_directory = candidate_directories[-1]
    
    # Copy them!
    if not os.path.exists(target_path):
        pathlib.Path(target_path).mkdir(parents=True, exist_ok=True) 
        
    ray_log_paths = glob(ray_session_directory + '/logs/worker-*.err')
    for log in tqdm(
        ray_log_paths,
        desc="Copy ray worker logs to disk and optionally S3"
    ):
        shutil.copy(log, target_path)
        if s3_bucket is not None:
            filename = os.path.split(log)[-1]
            upload_file(
                filename,
                s3_bucket,
                object_prefix=s3_prefix
            )
        
    return True
   
def main(
    db_ip, 
    db_username,
    db_password_secret,
    s3_prefix,
    s3_bucket='vespid',
    geocode=False,
    semantic_scholar_max_concurrency=None,
    last_processed_input=None,
    n_jobs=-1,
    graph_batch_size=5_000,
    log_path=None
):
    
    logger = setup_logger(__name__, filepath=log_path, 
                          align_all_loggers=True)
    db_password = get_secure_key(
        db_password_secret,
        aws_secret=True,
        bypass_safety_check=True
    )[db_username]
    
    logger.info("Running some pre-pipeline tests...")
    test_s3_access(s3_bucket)
    test_graph_connectivity(
        db_ip,
        db_password,
        db_username=db_username
    )
    
    pipeline_start_time = get_current_datetime()
    
    
    # Need this for easily matching to other differently-formatted datetimes
    pipeline_start_datetime = get_current_datetime(
        date_delimiter=None, 
        time_delimiter=None
    )
    log_path_no_filename = os.path.split(log_path)[0]
    logger.info("Spinning up code...")
    
    if n_jobs == 0 or n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1       
        logger.info(f"Using {n_jobs=}") 
    elif n_jobs < 0:
        raise ValueError("`n_jobs` less than 0 and not -1 are not supported")
    
    # Make sure we don't have any running ray processes
    ray.shutdown()
    ray.init(
        num_cpus=n_jobs,
        num_gpus=None,
        #include_dashboard=True, 
        #dashboard_port=8265,
        #_temp_dir='/home/jovyan/work/ray/' # doesn't seem to work well
    )

    file_iterator = yield_object_keys(
        bucket=s3_bucket,
        prefix=s3_prefix
    )
    num_input_files = count_s3_objects(s3_bucket, s3_prefix)
    
    if last_processed_input is not None:
        for i, f in enumerate(file_iterator):
            if f == s3_prefix + last_processed_input:
                # Adjust for the number we're skipping
                num_input_files -= i + 1
                break
        
    
    # Iterate through n_jobs-sized batches so we have one fully parallelized 
    # group of data (but no more due to memory constraints) before serial 
    # graph saving occurs
    serial_pipe_reports = []
    for batch_index in tqdm(
        range(0, num_input_files, n_jobs),
        desc='Processing n-jobs-sized batches'
    ):
        # Clean it all up to avoid memory leakage
        logger.info("Garbage collecting before running the batch...")
        gc.collect()
        
        parallel_stages, serial_stages = setup_stages(
            None,
            db_ip=db_ip,
            db_password=db_password,
            db_username=db_username,
            s3_bucket=s3_bucket,
            geocode=geocode,
            n_jobs=n_jobs,
            graph_batch_size=graph_batch_size,
            semantic_scholar_max_concurrency=semantic_scholar_max_concurrency
        )
        
        parallel_pipelines = []
        for f in file_iterator:
            try:
                logger.info(f"Pulled input file {f}")
                parallel_pipelines.append(
                    ParallelPipeline.remote(
                        parallel_stages,
                        save_stages=False,
                        key=(f, False)
                    )
                )
                # Have we filled up the batch?
                if len(parallel_pipelines) == n_jobs:
                    break
            except StopIteration:
                logger.info("Intput file iterator exhausted, almost done!")
                break
    
        # Run and concatenate results of parallel pipelines
        if len(parallel_pipelines) > 1:
            parallel_results = pd.concat(
                ray.get([p.run.remote() for p in parallel_pipelines]),
                ignore_index=True
            )
        elif len(parallel_pipelines) == 0:
            logger.warning("No new input data found, exiting batch loop...")
            break
        else:
            logger.info("Only one parallel pipeline produced")
            parallel_results = ray.get(parallel_pipelines.run.remote())
        
        logger.info("Garbage collecting parallel pipelines and stages "
                    "before pushing results to graph...")
        del parallel_pipelines, parallel_stages
        gc.collect()
        
        # Run the serial stuff - no need to capture output, goes to graph
        serial_pipe_reports.append(
            Pipeline(
                serial_stages,
                save_stages=False,
                data=(parallel_results, False)
            ).run(return_performance_report=True)
        )
        
        upload_object(
            serial_pipe_reports,
            f'{log_path_no_filename}/serial_pipe_reports.pkl',
            bucket=s3_bucket,
            auto_pickle=True
        )
        
        del serial_stages, parallel_results
        
    # Copy the ray worker logs to disk
    # that is volume-mounted before exiting
    if log_path is not None:
        copy_ray_logs(
            pipeline_start_datetime,
            log_path_no_filename
        )
        
        # Find all logs in the pipeline logs directory
        all_logs = glob(os.path.join(log_path_no_filename, '*'))
        for log in tqdm(all_logs, desc='Uploading logs to S3'):
            upload_file(
                log,
                s3_bucket
            )


if __name__ == '__main__':
    # Pull information from command line execution
    parser = argparse.ArgumentParser(
        description="Runs the Neo4j graph data pipeline, writing results "
        "directly into a Neo4j instance"
    )

    parser.add_argument(
        's3_prefix',
        type=str,
        help='path/to/pickle/files/ S3 bucket prefix for finding *.pkl files'
    )
    parser.add_argument(
        'db_ip',
        type=str,
        help='IP address of Neo4j database to use for storing Pipeline results'
    )
    parser.add_argument(
        'db_password_secret',
        type=str,
        help='Name of AWS Secrets Manager secret holding the database '
        'password. Note that it uses the value provided to `--db_username` '
        'to determine which password to grab.'
    )
    parser.add_argument(
        '--s3_bucket',
        type=str,
        default='vespid',
        help='Name of S3 bucket containing input files'
    )
    parser.add_argument(
        '--db_username',
        type=str,
        default='neo4j',
        help='Neo4j username to use'
    )
    parser.add_argument(
        '--geocode',
        default=False,
        action='store_true',
        help='Determines if geocoding should be done on Institution nodes ' \
            + 'such that they can be mapped later via lat/long. ' \
                + 'As geocoding can be very slow, not recommended usually.'
    )
    parser.add_argument(
        '--n_jobs',
        type=int,
        default=1,
        help='Number of parallel processes to use. '\
            + '-1 will use all available CPU cores.'
    )
    parser.add_argument(
        '--graph_batch_size',
        type=int,
        default=5000,
        help='Number of records per batch sent to the target graph for '
        'saving. Larger values tend to speed up the upload process, but can '
        'overwhelm Neo4j and break the connection, so be careful'
    )
    parser.add_argument(
        '--log_path',
        type=str,
        default=f'logs/{get_current_datetime()}_graph_pipeline/main.log'
    )
    parser.add_argument(
        '--semantic_scholar_max_concurrency',
        type=int,
        default=30,
        help='Max number of requests per second (NOT per job) allowed for '
        'querying Semantic Scholar'
    )
    parser.add_argument(
        '--last_processed_input',
        type=str,
        default=None,
        help='If starting the pipeline mid-way through processing of files, '
        'this allows you to indicate the last fully processed file such that '
        'it is skipped and only new files are processed from there. Should '
        'be of the form "file.pkl", no prefix (s3_prefix will be added '
        'automatically)'
    )

    args = vars(parser.parse_args())
    main(**args)