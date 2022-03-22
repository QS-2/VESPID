"""
building off of an online asynchronous requests example,
implement it in a way that works with requests-futures
because asyncio is confusing!
"""
import logging
from logging import NullHandler
from concurrent.futures import as_completed
from time import sleep
# from asyncio import sleep
from typing import Sequence, Hashable
from functools import partial, wraps

from requests import RequestException
from requests_futures.sessions import FuturesSession
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


__author__ = "Michael C. Smith (michael.smith@qs-2.com)"


logging.getLogger(__name__).addHandler(NullHandler())


def add_stderr_logger(level=logging.DEBUG):
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    # This method needs to be in this __init__.py to get the __name__ correct
    # even if urllib3 is vendored within another package.
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.debug("Added a stderr logging handler to logger: %s", __name__)
    return handler


# ... Clean up.
del NullHandler


# def _retry_partial(func, num_retries=5, max_wait_seconds=900, logger=logging.getLogger(__name__),
#                    log_level_on_retry='warning'):
#     """
#     Taken from searchtweets_v2, augmented with https://stackoverflow.com/a/25827070

#     Decorator to handle API retries and exceptions. Defaults to five retries.
#     Rate-limit (429) and server-side errors (5XX) implement a retry design.
#     Other 4XX errors are a 'one and done' type error.
#     Retries implement an exponential backoff...
#     :param func: function for decoration
#     :param max_wait_seconds:
#     :param num_retries:
#     :param logger: if given, use this logger
#     :param log_level_on_retry: use this level of the logger on retrying

#     Returns:
#         decorated function
#     """
#     assert hasattr(logger, log_level_on_retry)  # sanity check

#     @wraps(func)  # https://stackoverflow.com/a/25827070 cool!!
#     def retried_func(*args, **kwargs):
#         max_tries = num_retries
#         tries = 0
#         total_sleep_seconds = 0
#         while True:
#             try:
#                 resp = func(*args, **kwargs)
#             except r_exep.ConnectionError as exc:
#                 exc.msg = "Connection error for session; exiting"
#                 raise exc
#             except r_exep.HTTPError as exc:
#                 exc.msg = "HTTP error for session; exiting"
#                 raise exc
#             if resp.status_code != 200 and tries < max_tries:
#                 tries += 1
#                 getattr(logger, log_level_on_retry)(f" HTTP Error code: {resp.status_code}: "
#                                                     f"{resp.text} | {resp.reason}")
#                 getattr(logger, log_level_on_retry)(f" Request payload: args {args} kwargs {kwargs}")
#                 if resp.status_code == 429:
#                     getattr(logger, log_level_on_retry)("Rate limit hit... Will retry...")
#                     # Exponential backoff, but within a 15-minute (900 seconds) period.
#                     # No sense in backing off for more than 15 minutes.
#                     sleep_seconds = min(((tries * 2) ** 2), max(max_wait_seconds - total_sleep_seconds, 30))
#                     total_sleep_seconds = total_sleep_seconds + sleep_seconds
#                 elif resp.status_code >= 500:
#                     getattr(logger, log_level_on_retry)("Server-side error... Will retry...")
#                     sleep_seconds = 30
#                 else:
#                     # Other errors are a "one and done", no use in retrying error...
#                     logger.error('Quitting... ')
#                     raise r_exep.HTTPError
#                 getattr(logger, log_level_on_retry)(f"Will retry in {sleep_seconds} seconds...")
#                 sleep(sleep_seconds)
#                 continue
#             break
#         return resp
#     return retried_func


# retry = partial(_retry_partial, num_retries=10, max_wait_seconds=900)

def async_request_url_list(urls: Sequence[str], 
                           data_processing_fn=None,
                           index: Sequence[Hashable] = None,
                           max_workers: int = 3, 
                           rate_limit_interval_secs: int = 0,
                           requests_verb='GET',
                           logger=logging.getLogger(__name__),
                           session_args=tuple(), 
                           session_kwargs=None,
                           request_verb_kwargs=None,
                           raise_exc=False, 
                           exc_info=False,
                           flatten_result_dict_include_idx=True, 
                           flattened_idx_field_name='idx',
                           return_verbose_json=False,
                           return_results_ordered=False, 
                           use_tqdm=False,
                           num_retries=0, 
                           retry_statuses=tuple(range(400, 600)),
                           request_query_iterables=None,
                           **process_kwargs):
    """
    Make async requests of a URL list, aggregating the results

    :param urls: list of urls to hit
    :param data_processing_fn: function to execute in parallel on response. defaults to calling .json().
                               Should take only a Requests object as input.
    :param max_workers: max concurrency allowed to spawn (e.g. 100 if 100 requests per time period allowed)
    :param rate_limit_interval_secs: how long to wait after every request. calculate this based on any rate limits
    :param requests_verb: 'GET', 'POST', etc.
    :param logger: python logger object, defaults to one from this module
    :param session_args: other arguments for session initialization
    :param session_kwargs: other keyword-arguments for session initialization
    :param request_verb_kwargs: other arguments for requests calls
    :param raise_exc: True if want to raise encountered exceptions, default False
    :param exc_info: True if want to log exception info, False default
    :param return_results_ordered: True if want to present results ordered by idx, False default.
                                   Note! The function does its best, but there is
                                   NO guarantee this will work with your data_processing_fn!
    :param use_tqdm: True if want to display tqdm progress bar, False default
    :param index: sequence of ordered labels, of same length as urls; order implies 1:1 mapping with provided urls
                  (e.g., ['label 0', 'label 1' ...] or pandas Index)
    :param return_verbose_json: True if want verbose returned output, e.g.:
                                [{'idx': 0, 'url': 'www.ok.com', 'response': <200>, 'result': ... }, ]
                                False (default) returns [ {idx:result}, ...]
                                mutually exclusive with flatten_result_dict_include_idx
    :param flatten_result_dict_include_idx: if True add flattened_idx_field_name as a key in each result datum,
                                            to return [ result,  ] instead of [ {idx:result}, ...]
                                            this assumes result datum is dict-like!
                                            mutually exclusive with return_verbose_json
                                            False, returns [ {idx:result}, ...]
                                            default True
    :param flattened_idx_field_name: name of flattened idx field, default 'idx'
    :param num_retries: if greater than zero, retry that many times on status codes in retry_statuses arg
    :param retry_statuses: list of status codes on which to retry. defaults to all 400-599. in particular crossref often throws 408, 503
    :param request_query_iterables: dict indicating any iterables you want to 
        pass along to the processing function with the same sort order as the 
        query URLs, allowing it to associate each element of the iterable 
        with a corresponding query URL. Form should be 
        {'future_attribute_name': iterable}.
    :param process_kwargs: kwargs to pass to data_processing_fn. Note that, 
        if you want information from the input URLs available to process the 
        output (e.g. keyword query that the URL was built from), you can pass
        that here as a RequestQuery object.

    :return: list of results
    """
    if return_verbose_json and flatten_result_dict_include_idx:
        raise ValueError(f"return_verbose_json mutually exclusive with flatten_result_dict_include_idx")
    if session_kwargs is None:
        session_kwargs = dict()
    if request_verb_kwargs is None:
        request_verb_kwargs = dict()

    session = FuturesSession(max_workers=max_workers, *session_args, **session_kwargs)
    # TODO https://stackoverflow.com/a/58257232

    if int(num_retries) > 0:
        retry = Retry(
            total=num_retries,
            respect_retry_after_header=True,
            status_forcelist=retry_statuses,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)

    # noinspection PyUnusedLocal
    #FIXME: this shouldn't be assuming it has access to the parent function args/kwargs
    def response_hook(resp, *args, **kwargs):
        """declare a custom response hook that post-processes and addresses rate limits"""
        # Check if there are any values from iterables to assign to the 
        # result for later processing
        # if request_query_iterables is not None:
        #     for k, v in request_query_iterables.items():
        #         setattr(resp, k, v[resp.idx]) #TODO: how to index properly? Have to use URL instead of integer index? Not unique enough...
        
        if data_processing_fn is not None:
            resp.data = data_processing_fn(resp, **process_kwargs) if process_kwargs else data_processing_fn(resp)
        else:  # default to json
            resp.data = resp.json()
        sleep(rate_limit_interval_secs)
        
    if 'hooks' in request_verb_kwargs:
        raise ValueError(f"hooks parameter already in provided kwargs: {request_verb_kwargs['hooks']}")
    request_verb_kwargs['hooks'] = {'response': response_hook}

    futures = []
    if index is not None:  # do a few sanity checks...
        _l_index = len(index)
        _l_urls = len(urls)
        if _l_index != _l_urls:  # assert same length!
            raise ValueError(f"len(index) {_l_index} != len(urls) {_l_urls}")
        _l_set_index = len(set(index))
        if _l_index != _l_set_index:  # warn if index isn't unique!
            logger.warning(f"index not unique! len(index) {_l_index} != len(set(index)) {_l_set_index}")
            
    for idx, u in enumerate(urls):
        try:
            f = session.request(requests_verb, url=u, **request_verb_kwargs)
            if index is not None:  # Use the provided index to keep track of records
                f.idx = index[idx]
            else:  # use the enumerated index
                f.idx = idx
            f.url = u
            
            
            logger.debug(f"assembling future {idx}: {requests_verb} for url={u}")
            futures.append(f)
        except AttributeError as e:
            logger.error(f"encountered AttributeError {e} for verb {requests_verb}", exc_info=exc_info)
            raise e

    results = []
    if use_tqdm:
        from tqdm import tqdm
        future_iter = tqdm(as_completed(futures), desc='Query workers deployed', total=len(futures))
    else:
        future_iter = as_completed(futures)
    for f in future_iter:
        try:
            response = f.result()
            logger.debug(f"future {f.idx} response:{response}")
            # ensure there's a .data attribute
            try:
                result_datum = response.data
                logger.debug(f"result: {result_datum}")
                
            except AttributeError as a:  # for some reason the .data attribute wasn't set?
                logger.error(f"future {f.idx} -> {f.url} had exception {a}", exc_info=exc_info)
                raise a
            
            # format as desired
            if return_verbose_json:
                datum = {'idx': f.idx, 'url': f.url, 'response': response, 'result': result_datum}
            elif flatten_result_dict_include_idx:
                try:
                    if not hasattr(result_datum, "__contains__"):
                        logger.warning(f"are you sure this is correct? "
                                       f"result_datum has no __contains__ operator"
                                       f"{type(result_datum)} -> {result_datum}")
                    if flattened_idx_field_name in result_datum:
                        raise ValueError(f"flattened idx field name {flattened_idx_field_name} in "
                                        f"result datum already! {result_datum}")
                    if not hasattr(result_datum, "__setitem__") or hasattr(result_datum, "__index__"):
                        logger.warning(f"are you sure this is correct? "
                                       f"result_datum has no __setitem__ operator, or has __index__ : "
                                       f"{type(result_datum)} -> {result_datum}")
                    result_datum[flattened_idx_field_name] = f.idx
                except (AttributeError, TypeError, ValueError):
                    logger.error(f"error when setting {flattened_idx_field_name} in result datum. is it dict-like? "
                                 f"I have: {result_datum}. "
                                 f"Query URL causing the problem: {f.url}", exc_info=exc_info)
                    result_datum = None
                datum = result_datum
            else:
                datum = {f.idx: result_datum}
                
                    
            # and save
            results.append(datum)
        except RequestException as e:
            logger.warning(f"future {f.idx} -> {f.url} had exception {e}", exc_info=exc_info)
            if raise_exc:
                raise IOError from e
            
    if return_results_ordered:
        try:
            if return_verbose_json:
                results = sorted(results, key=lambda x: x['idx'])
            elif flatten_result_dict_include_idx:
                results = sorted(results, key=lambda x: x[flattened_idx_field_name])
            else:
                results = sorted(results, key=lambda x: list(x)[0])  # sort by idx, the key
        except (ValueError, TypeError) as e:
            logger.warning(f"sorting problem! {e}", exc_info=exc_info)
    logger.debug(results[:10])
    return results
