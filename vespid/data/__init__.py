# -*- coding: utf-8 -*-
import pandas as pd
import re
from urllib.parse import quote_plus
from vespid import setup_logger
import gzip
import os
import shutil

from vespid import setup_logger, set_global_log_level
logger = setup_logger(module_name=__name__)


def replace_special_solr_characters(data, columns=None):
    '''
    Apache Solr interprets certain characters as special if it sees them anywhere in a query.
    In order to avoid accidentally querying for something you don't intend to,
    you need to escape these characters properly.


    Parameters
    ----------
    data: either a pandas Series or DataFrame with the string data needing 
        substring replacements.

    columns: iterable of str. Indicates, if the input is a pandas DataFrame,
        which columns to apply substring replacement on. If None and ``data``
        is a DataFrame, will assume all columns are to be used.


    Returns
    -------
    The input data with substring replacements made. If a pandas DataFrame
    is provided as the input, will return the entire DataFrame, not just 
    ``columns``.
    '''

    output = data.copy()

    special_characters = [
        r'+',
        r'-',
        r'&', # single & must be manually converted
        r'&&', 
        r'|', # single | must be manually converted
        r'||', 
        r'!',
        r'(',
        r')',
        r'{',
        r'}',
        r'[',
        r']',
        r'^',
        r'"',
        r'~',
        r'*',
        r'?',
        r':',
        r'/',
        r'#'
    ]

    special_characters_dict = {}
    for c in special_characters:
        # escaped = re.escape(c)

        # # Deal with special non-escapable cases
        # if c in ['#', '&', '|', '?', '!']:
        #     escaped = quote_plus(c)

        # # # Check that we actually escaped it!
        # # elif len(escaped) != 2 * len(c):
        # #     escaped = "\\" + c        

        # special_characters_dict[c] = escaped
        special_characters_dict[c] = ''

    def _replace_characters(series, replacement_dict):
        output = series.copy()
        for char, char_escaped in replacement_dict.items():
            output = output.str.replace(
                    char, 
                    char_escaped, 
                    regex=False
                )

        return output

    if isinstance(data, pd.core.series.Series):
        output = _replace_characters(output, 
            special_characters_dict)

    elif isinstance(data, pd.core.frame.DataFrame) and columns is not None:
        for column in columns:
            output[column] = _replace_characters(output[column], 
                special_characters_dict)

    elif isinstance(data, pd.core.frame.DataFrame) and columns is None:
        for column in data.columns:
            output[column] = _replace_characters(output[column], 
                special_characters_dict)

    else:
        raise ValueError(f"``data`` must a pandas Series or DataFrame, not {type(data)}")
    

    return output

def expand_dict_lists(lists, unique_ids=None, drop_duplicates=True,
    keep_index=False):
    '''
    Given an iterable (e.g. pandas Series) containing lists of dicts that
    define unique entities and their metadata, return the unique entities + 
    metadata as a DataFrame, with duplicates dropped.


    Parameters
    ----------
    lists: list-like iterable. Should contain data about each entity as
        a list of dicts, one list per element. Dicts are expected to be of the
        form {'authorId': int, 'name': str, 'url': str}, although the only
        real requirement is that each dict have the same keys.

    unique_ids: str or list of str. Indicates the key(s) from each dict that 
        should be considered unique identifier(s) for the dict and can be 
        used to de-duplicate the records.

    drop_duplicates: bool. If True, will drop duplicate records using the
        ``unique_ids`` provided or simply use all columns to determine which
        rows to drop. If False, will skip duplicate dropping.

    keep_index: bool. If True, retains the index from the original ``lists``.
        Otherwise, drops it in favor of a new continuous non-duplicative index.


    Returns
    -------
    pandas DataFrame containing the dictionary entries and their metadata.
    '''

    if not isinstance(lists, pd.Series):
        lists_in = pd.Series(lists)

    else:
        lists_in = lists

    # Make the lists of dicts into a single list of dicts
    exploded_lists = lists_in.dropna().explode().dropna()
    single_list_of_dicts = exploded_lists.tolist()
    single_list_index = exploded_lists.index

    # Transform list of dicts into a flat DataFrame
    if drop_duplicates:
        output = pd.DataFrame(
            single_list_of_dicts, 
            index=single_list_index).drop_duplicates(subset=unique_ids)

    else:
        output = pd.DataFrame(
            single_list_of_dicts, 
            index=single_list_index)

    if keep_index:
        return output

    else:
        return output.reset_index(drop=True)


def extract_named_value_from_lists_of_dicts(l, key='name'):
    '''
    Using a known key, extracts every value found for that key in lists of 
    dicts (e.g. a pandas Series of lists of dicts) and returns the input
    iterable but with lists of only the values found for that key.


    Parameters
    ----------
    l: iterable of lists of dicts. The data you want to extract values from.

    key: str. Key used in each dict across the iterables the values of which
        you want to extract.


    Returns
    -------
    A pandas Series with only the values of ``key`` as lists.
    '''
    
    output = expand_dict_lists(l, drop_duplicates=False, keep_index=True)

    return output.groupby(output.index)[key].agg(list)


def decompress_gzip(filepath, remove_original=True):
    '''
    Decompresses a *.gz file located at ``filepath`` and saves
    a new copy with the same filename (except the *.gz is removed)
    in the source directory, optionally removing the old file if desired.


    Parameters
    ----------
    filepath: str. Should be of the form 'path/to/*.gz'

    remove_original: bool. If True, deletes the zipped file
        from the source directory, leaving only the decompressed
        file.


    Returns
    -------
    Filepath of the newly-decompressed file.
    '''
    with gzip.open(filepath, 'rb') as f_in:
        with open(filepath[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    if remove_original:
        os.remove(filepath)

    return filepath[:-3]

def find_columns_by_class(df, classes_to_find):
    '''
    Given a pandas DataFrame, find the columns that have at least one object 
    of the the same type as any of the `classes_to_find` provided.

    Parameters
    ----------
    df : pandas DataFrame
        The DataFrame with all your data
        
    class_to_find : list of object classes
        The type of class being sought

    Returns
    -------
    list of str
        The names of the columns with at least one object of the specified 
        class
    '''
    columns_found = set(
        df.columns[df.applymap(
            lambda t: isinstance(t, tuple(classes_to_find))
            ).sum() > 0]
    )
    
    return list(columns_found)