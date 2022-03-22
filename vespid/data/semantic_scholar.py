# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from thefuzz import fuzz
from urllib.parse import unquote

from vespid.data.async_requests import async_request_url_list
from vespid.data import replace_special_solr_characters


LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s) - %(message)s'
SEMANTIC_SCHOLAR_API_URL = \
    'https://partner.semanticscholar.org/graph/v1/'

# API key constraints based on experimentation
MAX_CONCURRENT_REQUESTS = 30
REQUEST_INTERVAL_SECONDS = 1

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def build_urls(
    identifiers, 
    query_type='DOI', 
    exact_match=True,
    fields_of_interest=None
):
    '''
    Using an iterable of identifiers (e.g. DOIs or Author IDs), 
    returns a list of URLs that can be used for querying the API.


    Parameters
    ----------
    identifiers: pandas Series of str. Used to identify the paper/author
        being searched on. Corresponds directly to the value identified via 
        ``query_type``.

    query_type: str. Can be one of ['DOI', 'S2 Paper ID', 'ArXiv ID',
        'MAG ID', 'ACL ID', 'PubMed ID', 'Corpus ID', 'Author ID', 'keyword'], 
        although 'Corpus ID' is likely to be deprecated soon. 'keyword' does a 
        Lucene-like text search of title, abstract, and venue.
        
    exact_match: bool. If True, will assume that queries must look for exact
        matches to the identifiers, no fuzzy matches allowed. This only really 
        applies to `query_type='keyword'` and results in putting "" around 
        each identifier.
        
    fields_of_interest: list of str. Indicates what fields you want the query 
        response to include. NOTE: different query types have different return 
        fields available. Check the API documentation before changing to a new 
        query type: https://api.semanticscholar.org/graph/v1.
        

    Returns
    -------
    pandas Series of URLs to use for querying.
    '''
    # ALWAYS CHECK https://www.semanticscholar.org/product/api FOR ANY UPDATES TO API!

    urls = pd.DataFrame(identifiers.values,
                        columns=['identifier'],
                        index=identifiers.index)

    allowed_query_types = {
        'DOI': 'paper/DOI:',
        'S2 Paper ID': 'paper/',
        'ArXiv ID': 'paper/arXiv:',
        'MAG ID': 'paper/MAG:',
        'ACL ID': 'paper/ACL:',
        'PubMed ID': 'paper/PMID:',
        'Medline ID': 'paper/PMID:',
        'PubMed Central ID': 'paper/PMCID:',
        'Corpus ID': 'paper/CorpusID:',
        'Author ID': 'author/',
        'keyword': 'paper/search?query='# Defaults to top 10 results
    }
    
    default_fields_of_interest = [
        'externalIds',
        'url',
        'title',
        'abstract',
        'venue',
        'year',
        'referenceCount',
        'citationCount',
        'influentialCitationCount',
        'isOpenAccess',
        'fieldsOfStudy',
        'authors.name',
        'authors.authorId', #Always included
        'authors.externalIds',
        'authors.url',
        'authors.name' #Included if no fields are specified
        #'authors.aliases' # CAN cause issues, should usually drop
        'authors.affiliations',
        'authors.homepage',
        'authors.paperCount',
        'authors.citationCount',
        'authors.hIndex',
        'citations.paperId', #Always included
        'citations.url',
        'citations.title', #Included if no fields are specified
        'citations.venue',
        'citations.year',
        'citations.authors', #Will include: authorId & name
        'references.paperId',
        'references.url',
        'references.title', #Included if no fields are specified
        'references.venue',
        'references.year',
        'references.authors', #Will include: authorId & name
        'embedding', #Vector embedding of paper content from the SPECTER model
        'tldr' #Auto-generated short summary of the paper from the SciTLDR model
    ]
    
    allowed_output_fields = {
        k:default_fields_of_interest for k in allowed_query_types.keys()
    }
    
    # Adjust for specific query types
    allowed_output_fields['keyword'] = [
        'externalIds',
        'url',
        'title',
        'abstract',
        'venue',
        'year',
        'referenceCount',
        'citationCount',
        'influentialCitationCount',
        'isOpenAccess',
        'fieldsOfStudy',
        'authors.name'
    ]

    if query_type not in allowed_query_types.keys():
        raise ValueError(f"query_type value of {query_type} not an allowed \
value. Please choose one of {list(allowed_query_types.keys())}")
        
    # Need to indicate what we want returned in response
    # paperId (SS-corpus-specific) always included
    if fields_of_interest is None and query_type != 'keyword':
        fields_of_interest = [
            'title',
            'abstract',
            'authors.name',
            'authors.url',
            'influentialCitationCount',
            'url',
            'fieldsOfStudy',
            'isOpenAccess',
            'embedding'
        ]
        
    # Expected that we're just looking for best-matching paper to query deeper from there
    elif fields_of_interest is None and query_type == 'keyword':
        fields_of_interest = ['title', 'externalIds']
    elif pd.Series(fields_of_interest).isin(
        allowed_output_fields[query_type]
        ).sum() < len(fields_of_interest):
        raise ValueError("At least one of `fields_of_interest` not valid for "
                         f"query_type of '{query_type}'")
    
    # Why change the field parameter leading token? 
    # ...because they can, apparently!
    if query_type == 'keyword':
        urls['response_fields'] = '&fields=' + ",".join(fields_of_interest)
        urls['identifier'] = replace_special_solr_characters(urls['identifier'])
        if exact_match:
            urls['identifier'] = '"' + urls['identifier'] + '"'
    else:
        urls['response_fields'] = '?fields=' + ",".join(fields_of_interest)

    urls['query'] = SEMANTIC_SCHOLAR_API_URL \
        + allowed_query_types[query_type] \
        + urls['identifier'] \
        + urls['response_fields'] + "&limit=20"
        # Comparison of 10 vs. 20 vs. 50 records per search didn't improve 
        # anything, but 10 vs. 20 didn't slow anything down either, so no harm

    return urls['query']


def process_response(
    response, 
    query_type, 
    score_threshold=0.85
):
    '''
    Processes a single API response.


    Parameters
    ----------
    response: Response object generated by an API call such as request.get()
    
    query_type: str. Indicates what kind of Semantic Scholar query was used 
        and thus the types of data processing required, if any. Note that this 
        assumes `response.identifier` exists when query_type == 'keyword',  
        with that attribute providing the keyword query string used to generate
        this response URL.
        
    score_threshold: float in the range [0.0, 1.0]. Indicates how well 
        `identifier` must match the content of the response. Only really needed
        for `query_type='keyword'`. 
        
    query_fields: list of str. Names of columns in the query response to be 
        used for comparison to the query strings. If your query was a 
        concatenation of the titles and abstracts, this should be 
        `['title', 'abstract']`.


    Returns
    -------
    pandas Series containing the record relevant to your search. 
    If multiple records are somehow returned, throws a warning but returns
    only the first record.
    '''
    try:
        # No guarantees that the JSON is any good...
        if response.text != '':
            if query_type == 'keyword':   
                total_record_count = response.json()['total']
                
                # Pull out keyword query from URL
                query = unquote(
                    response.url.split("query=")[-1].split("&fields=")[0]
                )          
                
                if total_record_count == 0:
                    logger.debug(f"No records found for query "
                                f"'{query}' and URL {response.url}")
                    return pd.Series()
                
                output = pd.json_normalize(response.json()['data']).replace(
                    {None: np.nan}
                )
                
                # Find the best match through fuzzy string matching
                def _compare_strings(row, identifier):
                    full_string = unquote(row['title']).lower()
                    return fuzz.ratio(full_string, identifier.lower())
                
                if 'data' in response.json().keys() and len(response.json()['data']) == 0:
                    logger.error("Got a response but there's no data!")
                    return pd.Series()
                else:
                    output['score'] = output.apply(
                        _compare_strings,
                        identifier=query,
                        axis=1
                    ) / 100
                
                if (output['score'] >= score_threshold).sum() > 0:
                    output = output.sort_values('score', ascending=False)
                    return output.iloc[0]
                else:
                    logger.debug(f"No good matches found for query "
                                f"'{query}' and URL {response.url} - "
                                "best was a score of "
                                f"{output['score'].max()}")
                    return pd.Series()                
            else:
                output = pd.json_normalize(response.json()).replace(
                    {None: np.nan}
                )

        # Sometimes responses aren't returned...old author data?
        else:
            output = pd.json_normalize({})
            
        if len(output) > 1:
            logger.warn(f"Returned {len(output)} records. \
    Parsing out only the first one.")
            return output.iloc[0]

        elif len(output) == 0:
            logger.warn("No record(s) returned that could be parsed!")
            output = pd.json_normalize({})
            
        # If server returns an error, make sure we log it!
        else:
            if 'error' in output.columns:
                if 'not found' in output.iloc[0]['error']:
                    logger.error(f"Paper at {response.url} not found in Semantic "
                                f"Scholar; error is {output.iloc[0]['error']}")
                else:
                    logger.error(f"Unknown error for paper at {response.url}: "
                                f"'{output.iloc[0]['error']}'")
                return pd.Series()
            
            if 'message' in output.columns:
                logger.error(f"Paper at {response.url} exists but not getting "
                                "pulled for some reason; message from server: "
                                f"'{output.iloc[0]['message']}'")
                return pd.Series()

            return output.iloc[0]
            
    except KeyError as e:
        try:
            if response.json()['message'] == 'Internal Server Error':
                logger.error(f"Internal server error for URL {response.url}, "
                            f"with message '{response.text}'")
                return pd.Series()
            
            # Timeouts are bad enough that we want to break the system 
            # so we restart more slowly
            elif response.json()['message'] == 'Endpoint request timed out':
                logger.error("Looks like we're querying Semantic Scholar too quickly!")
                raise e
        # Catch the error in case we can't access the JSON *at all*
        except KeyError:
            logger.error("Got an unknown KeyError from response "
                        f"at URL {response.url}, message: "
                        f"'{response.text}'", exc_info=True)
            return pd.Series()        

    # I give up
    except Exception as e:
        logger.error(f"Something went wrong with our query using URL "
                        f"{response.url}. Here's the bad response text: "
                        f"{response.text}", exc_info=True)
        
        #raise e
        return pd.Series()


def parse_author_dicts(series):
    '''
    Each Semantic Scholar paper record comes with a list of dictionaries
    that represent information about each author on the paper. This parses
    that information and enriches it with further data, such as the author's 
    position in the paper byline and whether the author list is alphabetized
    or not.


    Parameters
    ----------
    series: pandas Series of lists of author dictionaries. Should automatically
        come whenever a single paper record is returned from the Semantic 
        Scholar API.


    Returns
    -------
    A copy of ``series`` but with each dictionary enriched with extra info.
    '''

    def _parse_single_list_of_dicts(l):
        output = pd.DataFrame(l).reset_index().rename(
            columns={'index': 'authorListPosition'}
            )


        # Make author position 1-based, instead of 0-based
        output['authorListPosition'] += 1

        # Given position in reverse order, with last author position == -1
        output['authorListPositionReversed'] = output['authorListPosition'] - \
            (output['authorListPosition'].max() + 1)

        # Check if SS authors' ranks are alphabetically-derived using SS names 
        # Compare element n-1 to n, if earlier alphabetically comparison is False
        # Comparing final element to NaN will always be False
        # So sum of 0 means they are alphabetically sorted
        # Storing this at author-level (even though it's really a paper-level 
        # concept) as it's easier to see/undersatnd from 
        # user-traversing-the-graph perspective

        # Note that we have to check that author list is longer than one
        # so we don't say something was alphabetized when it was actually
        # single-authored
        output['authorListAlphabetized'] = \
        (output['name'] > output['name'].shift(-1)).sum() == 0 and \
        len(output) > 1
        
        return output.to_dict(orient='records')

    tqdm.pandas(desc="Parsing Semantic Scholar author dictionaries")
    return series.progress_apply(_parse_single_list_of_dicts)


def query_semantic_scholar(
    identifiers, 
    query_type='DOI', 
    api_key=None, 
    n_jobs=None,
    max_concurrent_requests=None,
    fields_of_interest=None,
    score_threshold=0.85
):
    '''
    Sends a query to Semantic Scholar REST API to get back paper- or 
    author-specific information.


    Parameters
    ----------
    identifiers: pandas Series of str. Used to identify the paper/author
        being searched on. Corresponds directly to the value identified via 
        ``query_type``. Note that keyword queries only work well when they are
        title-only, not title + abstract.

    query_type: str. Can be one of ['DOI', 'S2 Paper ID', 'ArXiv ID',
        'MAG ID', 'ACL ID', 'PubMed ID', 'Corpus ID', 'Author ID', 'keyword'], 
        although 'Corpus ID' is likely to be deprecated soon. 'keyword' does a 
        Lucene-like text search of title, abstract, and venue.

    api_key: str. Semantic Scholar API key, if one is available. API keys can 
        be requested directly from Semantic Scholar (usually only granted for 
        research purposes). If None, rate limit is 100 queries every 5 
        minutes. If there is a key, rates are decided at the time of key 
        generation and that information should be provided by the relevant
        Semantic Scholar contact providing the key.
        
    n_jobs: int or None. If n_jobs is 0 or None, no parallelization is assumed.
        If n_jobs is -1, uses all but one available CPU core.
        
    max_concurrent_requests: int. If not None, this value will be used to limit
        how many concurrent Semantic Scholar requests are allowed per second. 
        Otherwise, will default to the hard-coded max (usually 30).
        
    fields_of_interest: list of str. Indicates desired response fields from 
        the API.
        
    score_threshold: float in the range [0.0,1.0]. Indicates the minimum 
        acceptable similarity score between the query string and the top 
        search results. If no result scores at or above this value, the record
        will be skipped.


    Returns
    -------
    pandas DataFrame containing the record(s) relevant to your search, one
    per identifier provided as input.
    '''
    
    if n_jobs == 0 or n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = multiprocessing.cpu_count() - 1        
    elif n_jobs < 0:
        raise ValueError("`n_jobs` less than 0 and not -1 are not supported")

    urls = build_urls(
        identifiers, 
        query_type=query_type, 
        fields_of_interest=fields_of_interest
    )

    headers = {'x-api-key': api_key}
    
    if max_concurrent_requests is None:
        max_concurrent_requests = MAX_CONCURRENT_REQUESTS
    
    max_workers = int(max_concurrent_requests / n_jobs)
    logger.info(f"Using {max_workers} concurrent query workers per job")

    idx_name = 'index'
    if query_type == 'keyword':
        request_query_iterables = {'identifier': identifiers}
    else:
        request_query_iterables = None
    results = async_request_url_list(urls,
                                     process_response,
                                     max_workers=max_workers,
                                     rate_limit_interval_secs=REQUEST_INTERVAL_SECONDS,
                                     index=urls.index,
                                     return_results_ordered=True,
                                     flatten_result_dict_include_idx=True,
                                     flattened_idx_field_name=idx_name,
                                     use_tqdm=True,
                                     request_verb_kwargs={'headers': headers},
                                     request_query_iterables=request_query_iterables,
                                     query_type=query_type,
                                     score_threshold=score_threshold
                                     )

    output = pd.DataFrame(results).set_index(idx_name)
    
    if output.empty:
        logger.warning(f"S2 query of type {query_type} returned no results, "
                       "returning empty DataFrame")
        return pd.DataFrame()
    
    output.index = output.index.astype(urls.index.dtype)
    output.index.name = ''

    num_records_requested = len(urls)

    # Drop records from output if they didn't return something useful
    unique_id = 'authorId' if query_type == 'Author ID' else 'paperId'
    logger.info(f"Parsing out unique S2 {unique_id}s")
    try:
        num_null_ids = output[unique_id].isnull().sum()
        num_records_found = output[unique_id].notnull().sum()
    except KeyError as e:
        logger.info(f"{output.head()=}")
        output.info()
        logger.error("Something wrong with the unique S2 id")
        raise e

    record_return_percentage = \
        round(num_records_found / num_records_requested * 100, 2)

    logger.info(f"Of the {num_records_requested} {query_type}s \
provided, {num_records_found} records were found \
({record_return_percentage}% return rate)")

    logger.info(f"Dropping {num_null_ids} records with null {unique_id} values...")
    output.dropna(subset=[unique_id], inplace=True)

    # Drop duplicates
    num_duplicates = output[[unique_id, 'title']].duplicated().sum()
    logger.info(f"Dropping {num_duplicates} records with duplicate {unique_id} values...")
    output.drop_duplicates(subset=[unique_id, 'title'], inplace=True)


    # For each column that's a list, make empty lists null
    columns_of_lists = output.columns[
        output.applymap(lambda t: isinstance(t, list)).sum() > 0
    ]

    logger.info(f"Setting empty lists in columns {list(columns_of_lists)} to null...")
    for column in columns_of_lists:
        output.loc[output[column].str.len() == 0, column] = np.nan

    if output.columns.str.contains('authors').sum() > 0:
        output['authors'] = parse_author_dicts(output['authors'].dropna())
        
    # This is a common one we'll want to use regularly, so make it pretty
    output.rename(
        columns={'externalIds.DOI': 'DOI'}, 
        errors='ignore', 
        inplace=True
    )
    
    # Ignore all-null columns
    output.dropna(how='all', axis=1, inplace=True)
    
    # Redundant info
    output.drop(
        columns=['externalIds.CorpusId'], 
        errors='ignore', 
        inplace=True
    )

    return output


def find_paper_ids(
    df, 
    api_key, 
    n_jobs=None, 
    max_concurrent_requests=300,
    score_threshold=0.80
):
    '''
    Given titles, find the S2 paper ID and, if available, DOI of each paper.

    Parameters
    ----------
    df : pandas DataFrame
        Information to be augmented with further identifiers to aid in 
        successful merging with Semantic Scholar records.
    api_key : str
        Key for accessing S2 API.
    n_jobs : int, optional
        Number of parallel jobs across which this will be executed. 
        Providing this allows for proper calculation of the per-job max 
        concurrent query workers allowed, by default None
    max_concurrent_requests : int, optional
        Across all parallel jobs, the maximum number of concurrent query 
        workers that can be used by the application, by default 300
    score_threshold : float, optional
        In the range of [0.0,1.0], dictates how good of a string match must 
        exist between S2 matches and the title being sought, by default 0.80

    Returns
    -------
    pandas Dataframe
        A copy of the input DataFrame with an `id_ss` column and some (or all!) 
        of the previously-missing DOIs filled out.
    '''
    df_in = df[df['DOI'].isnull()].copy()
    
    if len(df_in) == 0:
        logger.info("No null DOIs found, skipping Semantic Scholar augmentation!")
        return df
    
    percent_null_dois = round(len(df_in) / len(df) * 100, 2)
    logger.info(f"Performing title-only queries for {len(df_in):,} "
                f"({percent_null_dois}%) missing DOIs")
    
    # See if we can pull down Semantic Scholar ID and, possibly, DOI
    matches = query_semantic_scholar(
        df_in['title'],
        query_type='keyword',
        api_key=api_key,
        n_jobs=n_jobs,
        max_concurrent_requests=max_concurrent_requests,
        fields_of_interest=['title', 'externalIds', 'abstract'],
        score_threshold=score_threshold
    )
    
    # Write over the missing data from input with new identifiers
    output = df.copy()
    
    if not matches.empty:
        output.loc[output['DOI'].isnull(), 'DOI'] = matches['DOI']
        output['id_ss'] = matches['paperId']
        
    else:
        logger.warning("No keyword-based matches found, returning copy of input...")
        output.loc[output['DOI'].isnull(), 'DOI'] = np.nan
        output['id_ss'] = np.nan
    
    # Can't identify ones that neither have a DOI nor an SS ID
    num_unidentifiable = len(output[
        (output['id_ss'].isnull())
        & (output['DOI'].isnull())
    ])
    
    percent_null_ids = round((num_unidentifiable) / len(df) * 100, 2)
    logger.info("After searching for papers missing identifiers in Semantic "
                f"Scholar, we now have only {num_unidentifiable:,} "
                f"({percent_null_ids}%) paper identifiers missing.")

    return output