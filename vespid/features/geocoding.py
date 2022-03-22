# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from geopy.exc import GeopyError
from time import time, sleep
from tqdm import tqdm

from vespid.data.async_requests import async_request_url_list
from vespid.data import extract_named_value_from_lists_of_dicts, replace_special_solr_characters


LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s) - %(message)s'
MAX_CONCURRENT_REQUESTS = 100
REQUEST_INTERVAL_SECONDS = 2


logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)

def get_from_openstreetmap(
    df, 
    **kwargs
):
    '''
    Given a list of address strings, iterate through them and find the lat/long,
    using as much address info as is possible.

    Note: this currently only has accuracy at the city level, as OSM results get
    pretty wonky/error-prone when you start querying at the street address level
    (especially since a lot of our addresses are weird, being universities).


    Parameters
    ----------
    df: pandas DataFrame with some location data included as columns.

    street: str. Indicates which column in ``df`` has the full address.

    city: str. Indicates which column in ``df`` has the city name.

    state: str. Indicates which column in ``df`` has the state name, if there is one.

    postalcode: str. Indicates which column in ``df`` has the ZIP/postal code, if there is one.

    country: str. Indicates which column in ``df`` has the country name.


    Returns
    -------
    Copy of ``df`` with columns 'address', 'latitude', and 'longitude'.
    '''

    def _build_urls(
        full_address_strings=None,
        location_data=None,
        **kwargs
    ):
        '''
        Given some address data, build OSM URLs to query for them.


        Parameters
        ----------
        full_address_strings: pandas Series of addresses with a format like "Cleveland, OH, USA".
            These will be used to construct free-text queries. All other parameters will be ignored
            if this isn't None.

        location_data: pandas DataFrame. If not using ``full_address_strings``, this DataFrame should
            contain structured data to be used in an address-segment-specific query (e.g. city=X, country=Y),
            with each unique address segment/entity type (e.g. city or state) contained in separate columns.

        street: str. If using ``location_data``, this is the name of the column in which the 
            street address strings can be found. If None, this will not be included in the query.
            Format of this should be "<house number> <street name>".

        city: str. If using ``location_data``, this is the name of the column in which the 
            city name strings can be found. If None, this will not be included in the query.

        state: str. If using ``location_data``, this is the name of the column in which the 
            state name strings can be found. If None, this will not be included in the query.

        postalcode: str. If using ``location_data``, this is the name of the column in which the 
            postal code strings can be found. If None, this will not be included in the query.

        country: str. If using ``location_data``, this is the name of the column in which the 
            country name strings can be found. If None, this will not be included in the query.


        Returns
        ----------
        pandas Series of query URL strings.
        '''
        base_url = 'https://nominatim.openstreetmap.org/search?'
        url_suffix = '&format=json&limit=1'
        segments = {**kwargs}

        if full_address_strings is not None:
            queries = 'q=' + replace_special_solr_characters(full_address_strings)
            idx = full_address_strings.index

        elif location_data is not None and len(segments) > 0:
            idx = location_data.index
            for i, (field, value) in enumerate(segments.items()):
                if i == 0:
                    queries = field + "=" + replace_special_solr_characters(location_data[value]).fillna('')
                
                else:
                    queries += "&" + field + "=" + replace_special_solr_characters(location_data[value]).fillna('')
            
        else:
            raise ValueError("No data passed for ``location_data`` or no columns identified to use from it.")

        output = base_url + queries + url_suffix

        # Make sure our input data index is maintained
        output.index = idx
        
        return output


    def _geocode(urls):
        '''
        Handles the geocode query buildings, sending, and error catching.
        '''

        # Allows us to re-try the server if we error out every once in a while
        idx_name = 'index'
        

        def _process_response(response):
            '''
            Assumes that we are only doing limit = 1 for geocode queries
            '''
            try:
                # No guarantees that the JSON is any good...
                data = response.json()
                if len(data) == 0: # empty list? Will return as all-null row
                    return pd.Series(dtype='object')
                
                elif response.text != '':
                    output = pd.json_normalize(data).replace({None: np.nan})
                    
                    column_name_dict = {'display_name': 'address', 'lat': 'latitude', 'lon': 'longitude'}
                    output.rename(columns=column_name_dict,
                                inplace=True)
                    
                    # Return only the columns we bothered renaming
                    output = output[column_name_dict.values()].iloc[0]

                else:
                    output = pd.Series(dtype='object')

            except Exception:
                if response.status_code == 200:
                    logger.error(f"Something went wrong with our query using URL {response.url}"
                                f"\nHere's the bad response JSON: \n\n{response.json()}",
                                exc_info=False)
                else:
                    logger.error(f"Something went wrong with our query using URL {response.url}"
                        f"\nHere's the bad response text: \n\n{response.text}",
                                exc_info=False)

                output = pd.Series(dtype='object')

            output['url'] = response.url
            return output

        idx_name = 'index'
        headers = {'User-Agent': 'vespid'}

        output = async_request_url_list(
            urls,
            data_processing_fn=_process_response,
            max_workers=1,
            rate_limit_interval_secs=1,
            return_results_ordered=True,
            flatten_result_dict_include_idx=True,
            flattened_idx_field_name=idx_name,
            index=urls.index,
            use_tqdm=True,
            request_verb_kwargs={'headers': headers}
        )

        output = pd.DataFrame(output).set_index(idx_name)
        output.index = output.index.astype(urls.index)
        output.index.name = ''

        # For some reason, they come back as strings
        if 'latitude' in output.columns and 'longitude' in output.columns:
            output[['latitude', 'longitude']] = output[['latitude', 'longitude']].astype(float)

        return output

    # Build out URLs using structured data (e.g. say 'city=CITY' instead of 'query=CITY,STATE,ETC.')
    # Note that OSM is pretty rigid about its tag/address segment values with this type of query
    # so we will likely have a number of null results we'll need to fill in another way (see below)
    location_dict = {**kwargs}

    urls = _build_urls(
        location_data=df,
        city=location_dict['city'],
        state=location_dict['state'],
        postalcode=location_dict['postalcode'],
        country=location_dict['country']
    )
        
    results = df.copy()
    location_columns = ['address', 'latitude', 'longitude', 'url']

    results[location_columns] = \
        _geocode(urls)

    # Deal with failed responses
    num_null_results = results['address'].isnull().sum()
    address_separator_string = ', '

    # It all errored out!
    if results.empty:
        logger.warn("No geocoding results found! Retrying with the simplest free-text search: city + country...")
        queries = df[location_dict['city']].fillna('') + address_separator_string +\
            df[location_dict['country']].fillna('')

        urls = _build_urls(full_address_strings=queries)
        results[location_columns] = \
            _geocode(urls)

    # Fill in those that couldn't be processed via structured query
    elif num_null_results > 0:
        logger.info(f"{num_null_results} locations didn't get geocoded, retrying as full-text address search...")
        remaining_index = results[results['address'].isnull()].index

        columns = [
            location_dict['city'],
            location_dict['state'],
            location_dict['postalcode'],
            location_dict['country']
        ]
        for i,column in enumerate(columns):
            if i == 0:
                queries = df.loc[remaining_index, column].fillna('').str.strip()

            else:
                queries += address_separator_string + df.loc[remaining_index, column].fillna('').str.strip()

        urls = _build_urls(full_address_strings=queries)
        results.loc[remaining_index, location_columns] = \
            _geocode(urls)
        
    
    num_null_results = results['address'].isnull().sum()
    # We have some results, but not fully filled out
    if num_null_results > 0:
        logger.info(f"{num_null_results} locations still not geocoded, "
        "retrying as full-text address search but with only city + country...")
        remaining_index = results[results['address'].isnull()].index

        columns = [
            location_dict['city'],
            location_dict['country']
        ]
        for i,column in enumerate(columns):
            if i == 0:
                queries = df.loc[remaining_index, column].fillna('').str.strip()

            else:
                queries += address_separator_string + df.loc[remaining_index, column].fillna('').str.strip()

        urls = _build_urls(full_address_strings=queries)
        results.loc[remaining_index, location_columns] = \
            _geocode(urls)


    return results