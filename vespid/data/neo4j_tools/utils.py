import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_pandas_datetime
import numpy as np
import logging
from vespid import get_current_datetime
from tqdm import tqdm
import os
from typing import Iterable
from interchange.time import DateTime as py2neo_datetime
from neo4j.time import DateTime as neo4j_datetime
from vespid.data import find_columns_by_class


LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s) - %(message)s'
MAX_CONCURRENT_REQUESTS = 100
REQUEST_INTERVAL_SECONDS = 1

logging.basicConfig(
    format=LOG_FORMAT,
    level=logging.INFO,
    datefmt='%m/%d/%Y %H:%M:%S')

logger = logging.getLogger(__name__)

def df_convert_cols_to_semicolon_delimited_str(df, *column_names, delimiter=';'):
    """for each column in column_names, convert each item to str and then concatenate with delimiter"""
    for column in column_names:
        logger.debug(f"Making lists in column {column} into {delimiter}-delimited strings...")
        if isinstance(df[column], Iterable):
            df[column] = [delimiter.join(map(str, _l)) for _l in df[column]]
    return df

def _check_column_types(df, neo4j_type, pandas_dtype=None, csv_export=True):
    '''
    Checks all the columns in a pandas DataFrame to determine if they
    are using the proper types for Neo4j CSV export. It does so by first inferring
    each column's pandas dtype and checking it against the dtype that
    user expects to be relevant for their use case. It then looks to see
    if any columns have been given neo4j data type's in their names that
    are relevant as well.

    Parameters
    ----------
    df : pandas DataFrame
        The table containing columns we wish to check
    neo4j_type : str
        If not None, this will be used as an indicator
        at the end of a column name that it is prepared
        for CSV export already.
    pandas_dtype : pd.core.dtypes.dtypes object, optional
        Indicates the dtype we're searching for. If None,
        only columns found to align with ``neo4j_type`` 
        naming are returned, by default None
    csv_export : bool, optional
        If True, indicates that the data should already be
        prepared for neo4j-admin import loading and
        thus stricter rules to column naming are applied, by default True

    Raises
    ------
    ValueError
        Raised if a column that should have ``neo4j_type`` in its name
        is missing it unexpectedly.
        
    Returns
    -------
    list of column names or None
    '''

    columns = set()

    # Check if any columns exist that have the expected neo4j type string
    # in their name
    if neo4j_type is not None:
        columns.update(df.columns[df.columns.str.endswith(neo4j_type)])

    # Check if a particular pandas dtype exists that we should use too
    if pandas_dtype:
        for column_name, column_dtype in df.dtypes.iteritems():
            if isinstance(column_dtype, pandas_dtype):
                if neo4j_type not in column_name and csv_export:
                    raise ValueError(f"Column {column_name} inferred to \
be a {pandas_dtype} column but does not have the expected Neo4j data type \
({neo4j_type}) in column name")

                else:
                    columns.add(column_name)

    columns = list(columns)
    return columns if len(columns) > 0 else None

def convert_types(df):
    # Infers the data types of pandas DataFrame columns
    # and transforms the typing to be compatible with
    # Neo4j.
    data = df.copy()
    
    #TODO: handle datetimestamps here too
    
    # Mainly need to worry about 'object' type
    numpy_array_columns = data.columns[
        data.apply(lambda col: col.agg(type).unique()).iloc[0] == np.ndarray
    ]
    
    # Transform numpy arrays to simple lists
    for column in numpy_array_columns:
        data[column] = data[column].apply(lambda array: array.tolist())
        
    return data

def transform_timestamps(df):
    '''
    Checks if any datetime/timestamp columns are
    present in a DataFrame and reformats to strings
    per Neo4j requirements.

    Parameters
    ----------
    df : pandas DataFrame
        Table with the datetime column(s)

    Returns
    -------
    2-tuple of the form (pandas DataFrame, [list, of, str])
        DataFrame with the newly-reformatted datetime column(s)
        and the names of the column(s) impacted
    '''
    
    datetime_columns = []
    for column in df.columns:
        if is_pandas_datetime(df[column]) or \
            df[column].apply(
                lambda col: isinstance(
                    col, 
                    (neo4j_datetime, py2neo_datetime)
                )).sum() > 0:
            datetime_columns.append(column)
    
    output = df.copy()

    # Put 'T' between date and time so Neo4j properly assesses
    if len(datetime_columns) > 0:
        for column in datetime_columns:
            output[column] = output[column].astype(str).str.replace(" ", "T")
            
    else: 
        datetime_columns = None
            
    return output, datetime_columns

def export_df(df, filepath, neo4j_object_type='nodes'):
    '''
    Given a pandas DataFrame, convert to a CSV following the various
    rules established for Neo4j imports.


    Parameters
    ----------
    df: pandas DataFrame to export. Column headers should already be formatted
        in the style neo4j requires (e.g. node property columns are of the 
        format 'propertyName:dataType').

        For formatting rules and "gotchas", see the documentation:
        https://neo4j.com/docs/operations-manual/current/tools/neo4j-admin-import/

    filepath: str representation of the filepath to use for the resulting
        CSV file. Should be of the form '/path/to/filename.csv'. Pass None to skip writing to disk.

    neo4j_object_type: str. Can be one of ['nodes', 'relationships']. 
        Indicates what type of Neo4j objects are going to be instantiated 
        using ``df``.


    Returns
    -------
    pandas DataFrame representative of what was saved to disk. Useful for 
    auditing the results.
    '''

    output = df.copy()

    # Check that there is a a :LABEL or :TYPE column, if node or rel, resp.
    if neo4j_object_type in ['nodes', 'node']:
        type_column = ':LABEL'
        if output.columns.str.contains(type_column).sum() == 0:
            raise ValueError(f"df missing '{type_column}' column")

        # Check for an :ID column
        if output.columns.str.contains(':ID').sum() == 0:
            raise ValueError("df missing ':ID' column")

    elif neo4j_object_type in ['relationships', 'relationship']:
        type_column = ':TYPE'
        if output.columns.str.contains(type_column).sum() == 0:
            raise ValueError(f"df missing '{type_column}' column")

        id_columns = [':START_ID', ':END_ID']
        for id_column in id_columns:
            if output.columns.str.contains(id_column).sum() == 0:
                raise ValueError(f"df missing '{id_column}' column")

    else:
        raise ValueError(f"'{neo4j_object_type}' is not a valid \
neo4j_object_type value")


    # Make sure to remove all newline characters from free text columns
    text_columns = _check_column_types(output, 'string', None)

    if text_columns:
        for column in tqdm(text_columns, desc='Replacing newline characters'):
            output[column] = output[column]\
            .str.replace(r'\n',' ')\
            .str.replace(r'\\n', ' ')

    # Check that datetime column is properly labeled
    # E.g. datetime object from pandas will not likely play well with 'date'
    # data type for Neo4j, but may for 'datetime' data type.
    # Note also that datetime values should have date and time portions
    # separated by 'T' e.g. '2021-01-01T00:00:00+00:00'

    output, _ = transform_timestamps(output)


    # Check for columns that contain python lists
    columns_of_lists = find_columns_by_class(output, [list])

    # Check for column names that end with '[]'
    more_list_columns = _check_column_types(output, '[]')
    if more_list_columns is not None:
        columns_of_lists.update(more_list_columns)

    # Lists need to be semicolon-delimited strings
    for column in columns_of_lists:
        if '[]' not in column:
            raise ValueError(f"Column {column} inferred to \
be a list()-type column but does not have the expected Neo4j data type \
([]) in column name")
        #
        # else:
        #     logger.debug(f"Making lists in column {column} into semicolon-delimited strings...")
        #     output[column] = output[column].str.join(";")
    output = df_convert_cols_to_semicolon_delimited_str(output, *columns_of_lists, delimiter=";")

    # Check for boolean columns
    # and convert them to strings of 'true' and 'false' (case sensitive!)
    bool_columns = output.columns[output.dtypes == np.bool]
    if not bool_columns.empty:
        for column in bool_columns:
            output[column] = output[column].astype(str).str.lower()

    if filepath is not None:
        output.to_csv(filepath, index=False)

    return output


def generate_ingest_code(node_files=None, relationship_files=None,
                         path_to_bin=None, dbname='neo4j'):
    '''
    Given a collection of CSV files, this generates a string that can be
    pasted into the terminal for using neo4j-admin import to ingest large
    amounts of data into a NEW database (note that this will NOT WORK for
    updating an existing database with new data/relationships).


    Parameters
    ----------
    node_files: list of str. Each element should be the string form of a 
        filepath to a CSV file that defines unique nodes and their properties.

    relationship_files: list of str. Each element should be the string form 
        of a filepath to a CSV file that defines unique relationships and 
        their properties.

    path_to_bin: Filepath pointing to Neo4j's /bin/ directory, wherein the neo4j-admin 
        executable/script is located. Best way to format this is via, e.g.
        os.path.join("..", "bin"). Defaults to "../bin/" if None.

    db_name: str. Indicates the name of the user database to ingest data into.
        Default is 'neo4j' and that's the only option if you're using Neo4j
        Community Edition.


    Returns
    -------
    A string that contains the command needed to execute the data ingest.
    '''

    def get_filenames(filepaths):
        '''
        Given an iterable of filepath strings, strip out the filenames.

        Useful for generating the import command when assuming your working directory
        already contains the files of interest.
        '''

        return pd.Series(filepaths).str.split('/', expand=True).iloc[:, 
            -1].tolist()

    files = {'nodes': get_filenames(node_files),
             'relationships': get_filenames(relationship_files)}
    if path_to_bin is not None:
        import_snippet = [
            os.path.join(path_to_bin, "neo4j-admin import"),
            # f"--database {db_name}"
        ]

    else:
        import_snippet = [
            os.path.join("..", "bin", "neo4j-admin import"),
            # f"--database {db_name}"
        ]

    import_snippet += [f"--{data_type} {file}" for data_type,
                       fs in files.items() for file in fs if fs is not None]

    current_datetime = get_current_datetime()

    import_snippet += [f"--trim-strings=true > import_{current_datetime}.out"]

    output = " ".join(import_snippet)

    logger.warn("Don't forget to spin down your database and back it up \
before you run this command! All data for the database of interest will be \
lost upon ingest!!")

    return output