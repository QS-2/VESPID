import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time, sleep
import requests
from glob import iglob, glob
import os

from vespid.data.async_requests import async_request_url_list
from vespid.data import replace_special_solr_characters, decompress_gzip
from vespid.data.aws import upload_file

import multiprocessing


LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s) - %(message)s'
CROSSREF_API_URL = 'https://api.crossref.org/works?query='

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def build_urls(queries, email, batch_size=20, bibliographic_query=True):
    '''
    Builds all the URLs needed for doing RESTful GET requests for CrossRef.


    Parameters
    ----------
    queries: pandas Series of str. The bibliographic information you want to 
        provide. Can include DOI value, title of paper, author name(s), etc. 
        Can also just be a Series of titles.

    email: str. Email address of the user sending this query. CrossRef asks
        that we do this to help them suss out any problems with scripts that
        may be bogging down the API. Providing this gives access to the query
        for running it on dedicated (lower latency presumably) machines.

    batch_size: int. Number of records to return for each query.

    bibliographic_query: bool. If True, will indicate to CrossRef API that 
        this is a strict citation-format-only query. This will limit access
        to some of the other metadata CrossRef could query against (e.g. 
        funder information) but will reduce false positives in match results.


    Returns
    -------
    pandas Series of URL strings.
    '''
    # Deal with special characters

    df = pd.DataFrame(replace_special_solr_characters(queries).values,
                      columns=['biblio_info'], index=queries.index)

    url = CROSSREF_API_URL

    if bibliographic_query:
        url = url[:-1] + '.bibliographic='

    df['url_prefix'] = url
    df['url_suffix'] = f"&mailto={email}&rows={batch_size}"

    df['full_query'] = df['url_prefix'] + df['biblio_info'] + df['url_suffix']

    return df['full_query']


def get_query_limits(email):
    '''
    CrossRef's API query rate limits are set dynamically depending upon 
    server strain at the time of the query. This function will let us do 
    a quick check at the start of any batch of queries to make sure we have 
    the right rate info (assuming our queries don't change this information!).


    Parameters
    ----------
    email: str. Your user email information. For good internet hygiene.


    Returns
    -------
    dict of the form {'max_concurrent_requests': int, 
    'request_time_interval_seconds': int}
    '''
    url = f'{CROSSREF_API_URL}test&mailto={email}&rows=2'

    r = requests.get(url)

    max_queries_per_interval = int(r.headers['x-rate-limit-limit'])

    # Lower the concurrent requests limit if we can, to avoid erroring out
    if max_queries_per_interval > 10:
        max_queries_per_interval -= 10

    interval_size_seconds = \
        pd.to_timedelta(r.headers['x-rate-limit-interval']).seconds

    return {'max_concurrent_requests': max_queries_per_interval,
            'request_time_interval_seconds': interval_size_seconds}


def get_best_text_score(df, score_difference_threshold=0.20,
                        score_column='score', goal_column=None):
    '''
    Checks the text matching scores for a group of records and
    returns only the records that are within the highest-scoring group.

    The idea is that scores on text matching (e.g. from BM25 or Levenshtein 
    distance calculations) are likely to be split into two groups typically:

        (1) High-scorers that are all very close to being the correct match
            for our query
        (2) The rest of the results that are not very good matches and 
            are expected to thus be low-scoring relative to the scores of 
            group 1.

    The issue is that these scores aren't upper-bounded, so we need a way to
    discern which ones are good and which bad (assuming we have ANY good ones).
    So we look for the percent-difference change in score for each consecutive
    record and see if there's a sharp drop in score at any point. If so, we 
    assume that that is the cutoff point for the two groups and return
    just group 1.


    Parameters
    ----------
    df: pandas DataFrame containing the text match scores (one per record)
        and the rest of the data of interest for each record as other columns.

    score_difference_threshold: float between 0.0 and 1.0. Indicates how much
        of an absolute change in score percent difference from one record to 
        another is required to determine that the high-scoring group was 
        found.

    score_column: str. Indicates the name of the column to use for the text
        match score values.

    goal_column: str. If not None, will use this as a null check when picking
        the result to return (e.g. if goal_column='DOI', only a good match
        with a non-null DOI value will be returned)


    Returns
    -------
    If a high-scoring group is found, returns a pandas Series containing the 
    best match. If not, returns np.nan.
    '''

    # Make sure top scorer is first
    df_in = df.sort_values(score_column, ascending=False)

    # Find percent change from one record to the next
    # Calculated as ( score[0] - score[1] ) / score[1]
    pct_diff_consecutive_scores = \
        (df_in['score'] - df_in['score'].shift(-1)) / df_in['score'].shift(-1)

    # Are there any that exceed our threshold?
    high_scorers = \
        df_in[pct_diff_consecutive_scores > score_difference_threshold]

    # Return only the ones with non-null DOI values
    if goal_column:
        output = high_scorers[high_scorers[goal_column].notnull()]

    else:
        output = high_scorers

    # Return the highest scorer
    if not output.empty:
        return output.iloc[0]

    else:
        return pd.Series(np.nan, output.columns)


def build_full_citation(
    df,
    authors_column='author_names',
    publication_date_column='date',
    title_column='title',
    venue_column='source'
):
    '''
    Provided certain parameters of a bibliographic record, builds
    a bibliographic string representing the record that can be used for 
    querying.


    Parameters
    ----------
    df: Pandas DataFrame representing multiple bibliographic records. 

    authors_column: str. Indicates the column for ``df`` that represents
        the authors' names.

    publication_date_column: str. Indicates the column for ``df`` that 
        represents the date of publication.

    title_column: str. Indicates the column for ``df`` that represents
        the publication title.

    venue_column: str. Indicates the column for ``df`` that represents
        the name of the journal/venue.


    Returns
    -------
    A string version of the bibliographic record for each entry in ``df``.
    '''

    output = pd.DataFrame()

    # TODO: how to remove duplicates in performant way?
    # Avoid overwhelming search by limiting to first 200 characters
    output['authors'] = df[authors_column].str.join(
        ', ').str.slice(stop=200).str.lower()

    output['year'] = pd.to_datetime(
        df[publication_date_column]).dt.year.astype(str)

    output['title'] = df[title_column].str.lower()
    output['venue'] = df[venue_column].str.lower()
    # Don't have volume num, issue num, or page start/end values

    # Replace any NaN values with empty strings
    output.fillna('', inplace=True)

    return output['authors'] + " " + output['year'] + ', "' \
        + output['title'] + '" ' + output['venue']


def add_publication_dois(
    df, 
    email, 
    score_difference_threshold=0.20,
    n_jobs=None
):
    '''
    Using mostly-complete bibliometric records (e.g. from Web of Science),
    find the DOIs for any records that lack them in order to use the DOIs
    as a primary key, linking multiple bibliometric databases together
    (e.g. Web of Science to Scopus or Semantic Scholar).

    Primary mechanism to do this is by using the CrossRef RESTful API
    (powered by Apache Solr) to do fuzzy string matching. Initially this 
    function will query using the title only and only do a full bibliographic
    search (which tends to be slower to return results) if the title-only 
    search does not produce conclusive results.

    Example usage: 
    --------------
    df.loc[df['DOI'].isnull(), 'DOI'] = add_publication_dois(df, email)

    # Also: why don't exact title matches seem to happen hardly ever, if ever? 
            Might be some problem in that logic


    Parameters
    ----------
    df: pandas DataFrame. Should at the very least contain a 'title' column.

    email: str. Email address of the user sending this query. CrossRef asks
        that we do this to help them suss out any problems with scripts that
        may be bogging down the API. Providing this gives access to the query
        for running it on dedicated (lower latency presumably) machines.

    score_difference_threshold: float between 0.0 and 1.0. Indicates how much
        of an absolute change in score percent difference from one record to 
        another is required to determine that the high-scoring group was 
        found.
        
    n_jobs: int or None. If n_jobs is 0 or None, no parallelization is assumed.
        If n_jobs is -1, uses all but one available CPU core.

    Returns
    -------
    A copy of ``df`` with the 'DOI' column's null values filled in as best as
    CrossRef can.
    '''
    
    if n_jobs == 0 or n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1        
    elif n_jobs < 0:
        raise ValueError("`n_jobs` less than 0 and not -1 are not supported")

    # Filter out for only records missing DOIs
    df_in = df[df['DOI'].isnull()].copy()

    if len(df_in) == 0:
        logger.info("No null DOIs found, skipping CrossRef augmentation!")
        return df

    # Get the query rate limits
    crossref_query_rate_info = get_query_limits(email)
    max_workers = int(crossref_query_rate_info['max_concurrent_requests'] / n_jobs)
    rate_limit_interval_secs = \
        crossref_query_rate_info['request_time_interval_seconds']

    logger.info(
        f"CrossRef API is currently allowing {max_workers} concurrent requests \
every {rate_limit_interval_secs} seconds for each of {n_jobs} jobs.")
    # max_workers = max(max_workers - 2, 1)  # slightly reduce just in case crossref is having issues
    # rate_limit_interval_secs += 1
    # logger.info(f"Slightly reducing to {max_workers} workers and increasing to {rate_limit_interval_secs} seconds. ")

    # First find the DOIs for ones that can be done with title queries only
    percent_null_dois = round(len(df_in) / len(df) * 100, 2)
    logger.info(
        f"Performing title-only queries for {len(df_in)} ({percent_null_dois}%) missing DOIs")

    urls = build_urls(df_in['title'].str.lower(), email)

    idx_name = 'index'
    matches = async_request_url_list(urls,
                                     process_response,
                                     max_workers=max_workers,
                                     rate_limit_interval_secs=rate_limit_interval_secs,
                                     index=df_in.index, return_results_ordered=True,
                                     flatten_result_dict_include_idx=True,
                                     flattened_idx_field_name=idx_name,
                                     use_tqdm=True, 
                                     score_difference_threshold=score_difference_threshold
                                     )

    df_out_1 = pd.DataFrame(matches).set_index(idx_name)
    df_out_1.index = df_out_1.index.astype(df_in.index.dtype)
    df_out_1.index.name = ''

    # Now look at the ones that we couldn't get results for with titles alone
    # Scores should never be null unless we couldn't find a good match
    if df_out_1['score'].isnull().sum() > 0:
        unmatched_index = df_out_1[df_out_1['score'].isnull()].index
        df_in = df.loc[unmatched_index].copy()

        # Get rid of the nulls so we aren't carrying those rows in memory
        df_out_1.dropna(how='all', inplace=True)

        percent_null_dois = round(len(df_in) / len(df) * 100, 2)
        logger.info(f"Performing full-citation queries for {len(df_in)} \
({percent_null_dois}%) missing DOIs")

        biblio = build_full_citation(df_in)
        urls = build_urls(biblio, email)

        matches = async_request_url_list(urls,
                                         process_response,
                                         max_workers=max_workers,
                                         rate_limit_interval_secs=rate_limit_interval_secs,
                                         index=df_in.index, return_results_ordered=True,
                                         flatten_result_dict_include_idx=True,
                                         flattened_idx_field_name=idx_name,
                                         use_tqdm=True)

        df_out_2 = pd.DataFrame(matches).set_index(idx_name)
        df_out_2.index = df_out_2.index.astype(df_in.index.dtype)
        df_out_2.index.name = ''

        num_null_dois = df_out_2['DOI'].isnull().sum()
        percent_null_dois = round(num_null_dois / len(df_in) * 100, 2)

        logger.info(f"After augmenting the DOIs with what we could find in \
CrossRef, we now have {num_null_dois} ({percent_null_dois}%) DOIs missing.")

        # Get rid of the nulls so we aren't carrying those rows in memory
        df_out_2.dropna(how='all', inplace=True)

        remaining_dois = pd.concat([df_out_1, df_out_2])

    else:
        remaining_dois = df_out_1

    output = df.copy()

    output.loc[output['DOI'].isnull(), 'DOI'] = remaining_dois['DOI']

    return output



def process_response(response, score_difference_threshold=0.2):
    '''
    Processes a single CrossRef response to determine which of the top-scoring
    search results returned from a query string best matches the thing we were
    looking for and returns its record.


    Parameters
    ----------
    response: Response object generated by an API call such as request.get()


    Returns
    -------
    pandas Series corresponding to the best-matching search result.
    '''
    from json import JSONDecodeError
    try:
        try:
            message_items = response.json()['message']['items']
        except (JSONDecodeError, KeyError) as _:
            msg = f"error extracting JSON -> message -> items from response. "
            logger.error(msg, exc_info=True)
            try:
                message_items = response.json()['message']
            except (JSONDecodeError, KeyError) as ex2:
                msg = f"error extracting JSON -> message from response. "
                logger.error(msg, exc_info=True)
                raise KeyError(msg) from ex2

        output = pd.json_normalize(message_items)

    except (Exception, KeyError, JSONDecodeError):  # TODO what will this generate? catch more appropriate exceptions
        logger.error(f"Something went wrong with our query using URL {response.url}"
            f"\nHere's the bad response text: \n\n{response.text}", exc_info=True)
        return pd.Series()

    try:
        assert 'score' in output.columns
    except AssertionError:
        output.info()
        logger.error(f"Something went wrong with our query using URL {response.url}"
                     f"\nHere's what's in output: {output}"
                     f"\nHere's the bad response text: \n\n{response.text}", exc_info=True)
        return pd.Series()
    # Find the best match, if there is one
    try:
        output = get_best_text_score(
            output, 
            goal_column='DOI',
            score_difference_threshold=score_difference_threshold
        )
    except (Exception, KeyError, JSONDecodeError):
        logger.error(f"Something went wrong with get_best_text_score from url {response.url}"
                     f"\nHere's what's in output: {output}"
                     f"\nHere's the bad response text: \n\n{response.text}", exc_info=True)
        return pd.Series()

    return output


def cloud_transfer_snapshot(filepath, prefix=None):
    '''
    Unzips and transfers a CrossRef snapshot dataset into S3.
    '''

    logger.info("Transfer commencing...")
    #TODO: build in a way to check if any of these files are already objects
    for filename in tqdm(glob(filepath + '*.gz'), desc='Unzipping and uploading...'):
        decompressed_filename = decompress_gzip(filename, remove_original=False)
        successful_upload = upload_file(decompressed_filename, 'vespid', object_prefix=prefix)
        if not successful_upload:
            logger.error(f"Filename {filename} could not upload to S3")
        os.remove(decompressed_filename)
        
    logger.info("Transfer complete!")
