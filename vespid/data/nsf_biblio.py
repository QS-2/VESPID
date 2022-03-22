# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from time import time, sleep
import re
import os
import pickle
import requests
import re
import math
import psutil

from vespid.data.async_requests import async_request_url_list
from vespid.data import (
    extract_named_value_from_lists_of_dicts, 
    replace_special_solr_characters
)
from vespid.data.aws import upload_object
from vespid import get_current_datetime


LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s) - %(message)s'
HOST = "http://dis-checker-a01/solr/"
#MAX_CONCURRENT_REQUESTS = min(1000, os.cpu_count() + 4)
MAX_CONCURRENT_REQUESTS = min(1000, os.cpu_count())
REQUEST_INTERVAL_SECONDS = 1

# How many records we can safely fit into memory
MAX_RECORDS_ALLOWED = 25_000


logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def build_urls(queries, collection_names='wos',
               sort_orders=None, filters=None,
               fields=None, offsets=0, limits=50):
    '''
    Using a variety of parameters to define the Solr query, 
    returns the URL that can be used to extract the result 
    (e.g. by inputting it into a browser).


    Parameters
    ----------
    queries: iterable of str. "*" returns all documents, otherwise it uses 
        boolean operators to do text searches on the title and abstract of 
        documents.

    collection_names: iterable of str or str. Name of the collection being used for 
        each query. Allowed values are ['wos', 'scopus', 'proposals', 
        'patents', 'reports', 'arxiv', 'awards', 'covid']. If str, will use
        this collection name for all queries.

    sort_orders: list of 2-tuple of strings of the form (fieldname, ordering) where 
        ``ordering`` can be one of ['asc', 'desc'] for "ascending" order or
        "descending" order. Used to sort the results if needed.

    filters: list of lists of 2-tuples of strings of the form 
        (field_name, condition). Used to filter the result documents in some 
        fashion. For example, [('popularity', '10'), ('section', '0')] will 
        add two AND-like filters to the query that returns only documents 
        with a popularity of 10 and a section of 0.

    fields: list of lists of str. Indicates which collection-specific fields 
        you want returned. If None, will return all possible fields.

    offsets: int or list of int. Indicates which result you want to be the first one returned.
        Most often used when paging through results (e.g. offset = i * limit, where
        i is the page being requested from the total result set).

        If int, will assume that all query URLs should have the same offset.

    limits: int or list of int. Indicates how many results to include in a single response. The NSF
        system defaults to 15 results per request, but this can be set to any value
        desired. However, it may be more efficient to page through results and avoid
        bogging down the system. This has yet to be tested, but likely values around
        50 are ideal.

        If int, will assume that all query URLs should have the same limit.


    Returns
    -------
    List of URL strings that can be used for extracting results via the NSF VDI 
    browser.
    '''
    # {HOST}{collection_name}/select?q={query}&fq=filter&start=offset&rows=limit&wt=json
    if isinstance(queries, str):
        queries = pd.Series([queries])
    urls = "/select?q=" + pd.DataFrame(replace_special_solr_characters(queries), columns=['query'])
    urls['prefix'] = HOST
    urls['collection_name'] = collection_names

    # Build up optional pieces of the query URL
    if filters is not None:
        temp = pd.DataFrame(filters, columns=['field', 'criterion'])
        urls['filters'] = "&fq=" + temp['field'] + ":" + temp['criterion']

    if sort_orders is not None:
        sort_strings = sort_orders.str.join(' ')
        urls['sort_order'] = "&sort=" + sort_strings

    # Join fields list into a single string
    if fields is not None:
        urls['fields'] = "&fl=" + pd.Series([fields] * len(queries)).str.join(sep='+')

    # Set output fields to be all that are available + the document score
    else:
        urls['fields'] = "&fl=*,score"

    if offsets and isinstance(offsets, list):
        urls['offset'] = "&start=" + pd.Series(offsets).astype(str)

    elif offsets and isinstance(offsets, int):
        urls['offset'] = "&start=" + str(offsets)

    if limits and isinstance(limits, list):
        urls['limit'] = "&rows=" + pd.Series(limits).astype(str)

    elif limits and isinstance(limits, int):
        urls['limit'] = "&rows=" + str(limits)

    elif limits and isinstance(limits, float):
        raise ValueError("``limits`` should be of type int, not float")

    # Fields for sorting columns to do quick concatenation
    # No prefix column since it kicks of the concatenating
    url_field_order = ['collection_name', 'query', 'filters', 'sort_order',
                       'fields', 'offset', 'limit', 'facet']

    # Remove columns we don't have
    url_field_order = [u for u in url_field_order if u in urls.columns]

    # Suffix:
    # Make q.op=AND if you want smaller result set that is more relevant to search phrases
    # change to OR if you want larger but likely less relevant result set
    # get JSON output
    output = urls['prefix'].str.cat(urls[url_field_order], sep='') + "&q.op=AND&wt=json"

    # Check that we're not overwhelming memory
    new_urls = []
    for i, query in enumerate(tqdm(queries, desc='Checking if any queries will return too many records to handle in memory...')):
        #print(f"i = {i}, query = {query}, original url = {output[i]}")
        num_results_no_limit = get_num_total_results(query, warn=False)
        if num_results_no_limit > MAX_RECORDS_ALLOWED:
            logger.warn(f"Query '{query}' will exceed max records allowed in memory "
            f"({MAX_RECORDS_ALLOWED:,}) by {(num_results_no_limit - MAX_RECORDS_ALLOWED):,}. "
            "Automatically paging through records to mitigate...")

            # Replace original URL with first page of results
            output[i] = re.sub(r'&rows=\d*', '', output[i]) + f'&rows={MAX_RECORDS_ALLOWED}'
            output[i] = re.sub(r'&start=\d*', '', output[i]) + f'&start=0'

            # Append URLs that page through rest of results
            # Get remaining number of batches needed
            num_batches = math.ceil(num_results_no_limit / MAX_RECORDS_ALLOWED)

            # Generate the new start=X values for each new page
            new_pages = pd.Series([f"{(i+1) * MAX_RECORDS_ALLOWED}" for i in range(num_batches - 1)])

            # Add the old URL + new page start values as copies into the list of URLs to be appended
            new_urls.append(pd.Series([output[i]] * (num_batches - 1)).str.replace(r'&start=\d*', '&start=', regex=True) + new_pages)
            
    if len(new_urls) > 0:        
        output = output.append(new_urls, ignore_index=True)

        
        
    #logger.info(f"Last URL is {output.iloc[-1]}")
    return output

def get_num_total_results(query='*', collection='wos', warn=True):
    '''
    Using a given Solr query, ask the database how many
    results it will ultimately be able to send back if not
    given any limits on response length.


    Parameters
    ----------
    query: str. Solr query desired. Use '*' to get the number
        of records in the entire database collection.

    collection: str. Name of the Solr collection to be queried.

    warn: bool. If True, will warn if the query will return
        more results than can safely fit into memory.


    Returns
    -------
    Integer number of records that could be returned for 
    that query.
    '''
    q = replace_special_solr_characters(pd.Series([query]))[0]
    url = f'http://dis-checker-a01/solr/wos/select?q={q}&rows=1&q.op=AND&wt=json'
    num_results = requests.get(url).json()['response']['numFound']

    if num_results > MAX_RECORDS_ALLOWED and warn:
        logger.warn(f"This query will return {num_results:,} results if not limited, "
        f"exceeding the recommended result return value of {MAX_RECORDS_ALLOWED:,}")
    return num_results


def process_response(response, credentials_path=None, file_iterator=None):
    '''
    Reads the response JSON from a solr instance and returns a pandas 
    DataFrame with one row per document. Used in asynchronous mode.


    Parameters
    ----------
    response: Response object generated by an API call such as request.get()

    credentials_path: str. If not None, will assume you want to upload
        results to an S3 bucket instead of returning them.

    file_iterator: int. If not None, will use this number to assign
        the filename of the uploaded pickle object. Only used if
        credentials_path is not None.


    Returns
    -------
    pandas DataFrame if credentials_path is None, or None otherwise.
    '''
    results = response.json()
    
    try:
        total_possible_results = results['response']['numFound']

    except KeyError:
        logger.error(f"KeyError! Response JSON: {results}")

    logger.debug(f"There were {total_possible_results:,} documents found \
that matched the search criteria")

    output = pd.DataFrame(results['response']['docs'])

    # Format date column
    output['date'] = pd.to_datetime(output['date'])

    # Extract the last DOI in the list as the "official" DOI
    output.rename(columns={'doi': 'DOI'}, inplace=True)
    output['DOI'] = output['DOI'].str.get(-1)

    columns_to_drop = [
        'author_first',
        # 'author_last',
        'author_street',
        'author_and_inst',
        'author_dais',
        'author_all',
        'inst',  # This seems to be non-dupe version of author_inst. DROP
        'inst_address',  # Duplicative of what's in author_address. DROP
        'inst_street',  # Likely to be empty, DROP
        'inst_city',  # can drop in favor of author_* variant
        'inst_state',  # can drop in favor of author_* variant
        'inst_zip',  # can drop in favor of author_* variant
        'inst_country'  # can drop in favor of author_* variant
    ]

    #logger.debug(f"Dataset ranges from {earliest_date} to {latest_date}")

    # Make sure we track the URL of what we've queried so response and query can be matched
    # Assumes that the query string of interest comes after 'select?q=' and before '&fq=
    query = re.search('select\?q=(.*?)\&fq=', response.url)
    query = query.group(1).replace('%20', " ")
    output['query'] = query
    output.drop(columns=columns_to_drop, errors='ignore', inplace=True)



    #return output
    if credentials_path is not None:
        path_prefix = "data/raw/nsf_biblio/WOS/full_corpus/"

        if file_iterator is not None:
            object_path = path_prefix + f'{file_iterator}.pkl'
        else:
            earliest_date = output['date'].min().strftime('%Y-%m-%d')
            latest_date = output['date'].max().strftime('%Y-%m-%d')
            object_path = path_prefix + f"{earliest_date}_to_{latest_date}" + '.pkl'

        upload_object(pickle.dumps(output), object_path, credentials_path, bucket='vespid')

        # Minimize the amount of data in memory
        return None

    else:
        return output


def solr_to_dataframe(filepath=None, queries=None,
                      collection_name='wos',
                      sort=None, filters=None,
                      fields=None, offset=0, limit=50,
                      upload_credentials_path=None):
    '''
    Uses either a set of URLs to concurrently query an Apache Solr instance
    or simply reads the JSON results saved to file. In either case, returns
    a DataFrame for your trouble.


    Parameters
    ----------
    filepath: str indicating where the text input is located. May not be
        None if ``queries`` is None.

    queries: list of str that represents individual Solr queries. May not be
        None if ``filepath`` is None.

    host: host URL string. Should be of the format 'http[s]://domain/solr/'

    collection_name: str. Name of the collection being used. 
        Allowed values are ['wos', 'scopus', 'proposals', 'patents', 
        'reports', 'arxiv', 'awards', 'covid'].

    query: str. "*" returns all documents, otherwise it uses boolean operators
        to do text searches on the title and abstract of documents.

    sort: 2-tuple of strings of the form (fieldname, ordering) where 
        ``ordering`` can be one of ['asc', 'desc'] for "ascending" order or
        "descending" order. Used to sort the results if needed.

    filters: list of 2-tuples of strings of the form (field_name, condition).
        Used to filter the result documents in some fashion. For example,
        [('popularity', '10'), ('section', '0')] will add two AND-like
        filters to the query that returns only documents with a popularity of
        10 and a section of 0.

    fields: list of str. Indicates which collection-specific fields you want 
        returned. If None, will return all possible fields.

    offset: int. Indicates which result you want to be the first one returned.
        Most often used when paging through results (e.g. offset = i * limit, where
        i is the page being requested from the total result set).

    limit: int. Indicates how many results to include in a single response. The NSF
        system defaults to 15 results per request, but this can be set to any value
        desired. However, it may be more efficient to page through results and avoid
        bogging down the system. This has yet to be tested, but likely values around
        50 are ideal.

    upload_credentials_path: str. If not None, provides filepath to CSV file
        representing the S3 credentials needed for saving individual query
        results as pickle objects to the cloud.


    Returns
    -------
    pandas DataFrame with all of the results from all of the queries.
    '''

    if filepath:
        with open(filepath, mode='r') as f:
            results = json.load(f)

        output = pd.DataFrame(results['response']['docs'])

        # Format date column
        output['date'] = pd.to_datetime(output['date'], utc=True)
        earliest_date = output['date'].min()
        latest_date = output['date'].max()
        logger.info(f"Dataset ranges from {earliest_date} to {latest_date}")

    else:
        urls = build_urls(queries, collection_name, filters=filters,
                          fields=fields, offsets=offset, limits=limit)

        idx_name = 'index'
        results = async_request_url_list(urls,
                                         process_response,
                                         max_workers=MAX_CONCURRENT_REQUESTS,
                                         rate_limit_interval_secs=REQUEST_INTERVAL_SECONDS,
                                         index=urls.index,
                                         return_results_ordered=False,
                                         flatten_result_dict_include_idx=True,
                                         flattened_idx_field_name=idx_name,
                                         use_tqdm=True,
                                         credentials_path=upload_credentials_path)

        output = pd.concat(results, ignore_index=True)  # .set_index(idx_name)
        #output.index = output.index.astype(urls.index.dtype)
        #output.index.name = ''

    if collection_name == 'wos':
        # Extract the last DOI in the list as the "official" DOI
        output['date'] = pd.to_datetime(output['date'])

    logger.info(
        f"{round(output['DOI'].isnull().sum() / len(output) * 100, 2)}% \
of the DOIs are missing"
    )

    return output

def solr_to_dataframe_serial(queries=None,
                      collection_name='wos',
                      sort=None, filters=None,
                      fields=None, offset=0, limit=50,
                      upload_credentials_path=None,
                      start_index=0):
    '''
    Uses a set of URLs to serially (not concurrently) query an Apache Solr instance
    and return a DataFrame of the results.


    Parameters
    ----------
    queries: list of str that represents individual Solr queries. May not be
        None if ``filepath`` is None.

    host: host URL string. Should be of the format 'http[s]://domain/solr/'

    collection_name: str. Name of the collection being used. 
        Allowed values are ['wos', 'scopus', 'proposals', 'patents', 
        'reports', 'arxiv', 'awards', 'covid'].

    sort: 2-tuple of strings of the form (fieldname, ordering) where 
        ``ordering`` can be one of ['asc', 'desc'] for "ascending" order or
        "descending" order. Used to sort the results if needed.

    filters: list of 2-tuples of strings of the form (field_name, condition).
        Used to filter the result documents in some fashion. For example,
        [('popularity', '10'), ('section', '0')] will add two AND-like
        filters to the query that returns only documents with a popularity of
        10 and a section of 0.

    fields: list of str. Indicates which collection-specific fields you want 
        returned. If None, will return all possible fields.

    offset: int. Indicates which result you want to be the first one returned.
        Most often used when paging through results (e.g. offset = i * limit, where
        i is the page being requested from the total result set).

    limit: int. Indicates how many results to include in a single response. The NSF
        system defaults to 15 results per request, but this can be set to any value
        desired. However, it may be more efficient to page through results and avoid
        bogging down the system. This has yet to be tested, but likely values around
        50 are ideal.

    upload_credentials_path: str. If not None, provides filepath to CSV file
        representing the S3 credentials needed for saving individual query
        results as pickle objects to the cloud.

    start_index: int. If doing a non-date-based set of queries (e.g. paging
        through the results of a single large query), use this argument
        to start at a non-zero point in the paging process. Often used
        when query parsing fails after persisting results a few times.


    Returns
    -------
    pandas DataFrame with all of the results from all of the queries.
    '''

    urls = build_urls(queries, collection_name, filters=filters,
                        fields=fields, offsets=offset, limits=limit, sort_orders=sort)


    # Don't spit out anything as this will only incur memory increases over time!
    for i, url in enumerate(tqdm(urls[start_index:], desc='Running each query and returning the data')):
        # Getting % usage of virtual_memory ( 3rd field)
        logger.info(f'Memory used before data pull = {psutil.virtual_memory()[2]}%')
        _ = process_response(requests.get(url), credentials_path=upload_credentials_path, file_iterator=i + start_index)
        logger.info(f'Memory used after data pull = {psutil.virtual_memory()[2]}%')

    return None


def make_wos_author_dicts(df, inplace=False):
    '''
    Harmonizes the Web of Science author records with the data model used
    in Semantic Scholar by combining multiple DataFrame columns of lists
    (e.g. author names, institution addresses, etc.) and makes them all into
    a single list of dicts, one dict per author. Each author dict also comes
    with institutional affiliation data.


    Parameters
    ----------
    df: pandas DataFrame containing authors and institutional affiliations.
    
    inplace: bool. If True, indicates that `df` should be changed in-memory
        instead of creating a copy.


    Returns
    -------
    pandas DataFrame with a new column 'authors_wos' containing lists of 
    dicts that fully describe the authors.
    '''

    def make_single_author_dict(series):
        '''
        For a single paper record, give the authors and affiliations as a
        list of dicts, one dict per unique author name.

        NOTE: this assumes that there are no duplicate names among the authors
        for a given paper.
        '''

        # Table of author names, addresses, etc.
        # Authors can be duplicated in this format, but author +
        # institution is assumed to be unique
        author_table = pd.DataFrame.from_dict(dict(zip(series.index,
                                                       series.values)))

        author_dicts = author_table.groupby('name').agg(list)\
            .reset_index().to_dict(orient='records')

        # For each author, extract out their affiliations
        # and save each institution as a dict in a list nested within
        # the author dict

        author_dicts_reformatted = []

        for author in author_dicts:
            # Extract the institution information which is currently
            # in list form for each parameter
            d = {k: v for k, v in author.items() if k not in [
                'name', 'last_name']}

            # Take dict of lists and make into list of dicts
            # So going from {author: name, inst: [a, b], address: [1,2]}
            # to {author: name, inst: [{name: a, address: 1}, {name: b, address:2}]}
            author_dicts_reformatted.append({
                'name': author['name'],
                'last_name': author['last_name'][0],
                'institutions': [dict(zip(d, i)) for i in zip(*d.values())]
            })

        return author_dicts_reformatted

    author_columns = {
        'author_name': 'name',
        'author_last': 'last_name',

        # There is one of these per author, regardless of duplicate insts
        'author_inst': 'institution',

        # Often info like department + university name
        # (maybe campus building too).
        # Also can contain city, state, etc.
        'author_address': 'address',

        'author_city': 'city',
        'author_state': 'state',
        'author_zip': 'postal_code',
        'author_country': 'country',
    }
    
    extraneous_columns = [col for col in author_columns]
    extraneous_columns.extend([
        # 'author_first',
        'corpus',
        'author_count',
        # 'inst_country',
        # 'inst_city',
        # 'author_dais', # TODO: might need to reinstitute if these end up useful
        # 'author_all',
        'cat_heading',
        'cat_subheading',
        'ref_num',
        # 'inst_state',
        # 'inst_address',
        # 'author_street',
        'issn',
        'ref_year',
        # 'inst',
        # 'inst_street',
        # 'inst_zip',
        'DOIs',
        # 'author_and_inst',
        '_version_'
    ])

    tqdm.pandas(desc='Making author dictionaries')

    if inplace:
        df['authors_wos'] = df[author_columns.keys()]\
            .dropna(subset=['author_name'])\
                .rename(columns=author_columns)\
                    .progress_apply(make_single_author_dict,
                                                    axis=1)

        df['author_names'] = \
            extract_named_value_from_lists_of_dicts(
                df['authors_wos'], key='name')

        df.drop(columns=extraneous_columns, inplace=True, errors='ignore')

        # For each column that's a list, make empty lists null
        columns_of_lists = df.columns[
            df.applymap(lambda t: isinstance(t, list)).sum() > 0
        ]

        logger.info(
            f"Setting empty lists in columns {list(columns_of_lists)} to null...")
        for column in columns_of_lists:
            df.loc[df[column].str.len() == 0, column] = np.nan

        return df
        
    else:
        df_in = df[author_columns.keys()].dropna(subset=['author_name'])
        df_in.rename(columns=author_columns, inplace=True)

        output = df.copy()
        output['authors_wos'] = df_in.progress_apply(make_single_author_dict,
                                                    axis=1)

        output['author_names'] = \
            extract_named_value_from_lists_of_dicts(
                output['authors_wos'], key='name')        

        output.drop(columns=extraneous_columns, inplace=True, errors='ignore')

        # For each column that's a list, make empty lists null
        columns_of_lists = output.columns[
            output.applymap(lambda t: isinstance(t, list)).sum() > 0
        ]

        logger.info(
            f"Setting empty lists in columns {list(columns_of_lists)} to null...")
        for column in columns_of_lists:
            output.loc[output[column].str.len() == 0, column] = np.nan

        return output


def make_wos_grant_dicts(df, inplace=False):
    '''
    Given information from the WoS dataset regarding funding entity
    and funding identifier, build a dict that represents the funding provided
    by each funder.


    Parameters
    ----------
    df: pandas DataFrame, where each row is a paper. Must include at least
        the columns ['grant_id','grant_agency'] (which are made up of lists).
        
    inplace: bool. If True, indicates that `df` should be changed in-memory
        instead of creating a copy.


    Returns
    -------
    Copy of ``df`` with a new column `funding` that contains lists of 
    dictionaries describing the funding entities and contracts.
    '''

    def _make_single_paper_grant_dict(row):
        input = row.to_dict()
        grant_dicts = [dict(zip(input, i)) for i in zip(*input.values())]

        output = [{'grantId': d['grant_id'],
                   'funder': d['grant_agency']} for d in grant_dicts]

        return output

    tqdm.pandas(desc='Making funding dictionaries')

    if inplace:
        df['funding'] = df[['grant_id', 'grant_agency']].dropna().progress_apply(
            _make_single_paper_grant_dict, axis=1
        )

        return df
        
    else:
        output = df.copy()

        output['funding'] = df[['grant_id', 'grant_agency']].dropna().progress_apply(
            _make_single_paper_grant_dict, axis=1
        )

        return output
