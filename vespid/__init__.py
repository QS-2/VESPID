import logging
from datetime import datetime
import pytz
import psutil
import os
import atexit
from io import StringIO
import boto3
from botocore.exceptions import ClientError
import base64
import json
import pickle
import pathlib
import pandas

# Allow for asyncio to be run from within Jupyter notebooks 
# and other parent event loops
import nest_asyncio
nest_asyncio.apply()

LOG_FORMAT = '%(asctime)s: %(levelname)s (%(name)s:%(lineno)d) - %(message)s'
DATETIME_FORMAT = "%m-%d-%Y_T%H_%M_%S"
AWS_REGION = "us-west-2"
LOG_MODULES_WARN_ONLY = ['elasticsearch']

# Need a copy of this to avoid circular import issues
def _upload_object(
    object, 
    path, 
    access_key_path=None, 
    bucket='vespid', 
    session=None,
    auto_pickle=False
):
    """
    Upload an arbitrary object to an S3 bucket, such as 
    a pickled DataFrame.

    Note that, if you send an object as a bytestring (e.g. via pickle.dumps()),
    you will want to read the resulting file when you need it via

        with open('path/to/file.pkl', 'rb') as f:
            obj = pickle.load(f)
            
    or
        pickle.loads('path/to/file.pkl')

    
    Parameters
    ----------
    object: object to upload, usually in a StringIO or ByteStringIO format

    path: str. URI to use for object when in the bucket (e.g. <bucket>/<path>).
    
    access_key_path: str. Indicates filepath to AWS access credentials CSV.
        If None, will assume it's operating in a pre-approved security environment
        (e.g. an AWS instance with proper role permissions)
    
    bucket: str. Target S3 bucket.
    
    session: boto3.Session object. If None, will attempt to upload via 
        boto3.resource
        
    auto_pickle: bool. If True, and `object` is determined to not already be 
        a bytestring, will pickle `object` before sending to S3 via 
        `pickle.dumps(object)'


    Returns
    -------
    Nothing.
    """

    if path[-1] == '/':
        raise ValueError("object name should not end with '/'. "
        "Please ensure a valid object name has been provided.")

    if access_key_path is not None:
        aws_access_key, secret_access_key = pandas.read_csv(access_key_path).loc[0]
    else:
        aws_access_key, secret_access_key = None, None

    # Upload the file
    if session is None:
        s3_resource = boto3.resource(
            's3',
            aws_access_key_id=aws_access_key, 
            aws_secret_access_key=secret_access_key
        )
    else:
        s3_resource = session.resource('s3')

    # Detect if object is not already in proper format, transform accordingly
    if not isinstance(object, bytes) and auto_pickle:
        print(f"WARNING: Detected that object is of type {type(object)}; "
        "pickling it and sending to S3 as a byte string...")
        to_upload = pickle.dumps(object)

    else:
        to_upload = object

    try:
        s3_resource.Object(bucket, path).put(Body=to_upload)

    except Exception as e:
        raise e

def setup_logger(
    module_name='NoModuleIdentified',
    default_level=logging.INFO,
    filepath=None,
    s3_bucket=None,
    align_all_loggers=False
):
    '''
    Sets up logging consistently across modules 
    when imported and run at the top of a module.


    Parameters
    ----------
    module_name: str. Should be set to __name__ by the calling
        module so as to properly identify where it comes from.

    default_level: int, although recommended that you 
        pass logging.<LEVEL> for consistency. If you want 
        functions/classes/etc. within your module to log 
        messages at a level other than the default INFO, 
        set it here.
        
    filepath: str of the form 'path/to/log.log'. 
        If not None, the contents of the log will be output to 
        stderr (as is typical) *and* to the file specified. Note that the 
        log's storage endpoint will be dictated by other parameters in this
        function, but defaults to local disk.
        
    s3_bucket: str. If not None, will assume that the log should be pushed 
        to an AWS S3 bucket in addition to the local disk. `filepath` will be 
        utilized to indicate the key of the object these data will be sent to.
        
        Note that the push to S3 only happens once the program calling the
        logger is terminated, to avoid frequent uploads to S3.
        
    align_all_loggers: bool. If True, will force all loggers called
        from other modules to use the configuration for this one.


    Returns
    -------
    Logger object.
    '''
    # We pretty much always want to write to stdout, just to be safe
    handlers = [logging.StreamHandler()]
    
    # Setup to write to file, if requested, too
    if filepath is not None:
        
        # Check that directory exists and create if it does not
        directory = os.path.split(filepath)[0]
        if not os.path.exists(directory) and directory != '':
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True) 
            
        # Translate potential relative filepath to absolute
        # Should hopefully avoid creating more than one log file when
        # logging from other modules
        absolute_filepath = os.path.abspath(filepath) + '/'
            
        handlers.append(logging.FileHandler(absolute_filepath, mode='a'))
        
        if s3_bucket is not None:
            raise NotImplementedError("This is still buggy")
            log_stringio = StringIO()
            handlers.append(logging.StreamHandler(log_stringio))
            
            atexit.register(
                _upload_object, 
                object=log_stringio.getvalue(),
                path=filepath,
                bucket=s3_bucket
            )
        
    elif filepath is None and s3_bucket is not None:
        raise ValueError("`filepath` must not be None if `s3_bucket` is used")
        
    logging.basicConfig(
        format=LOG_FORMAT,
        level=default_level,
        datefmt=DATETIME_FORMAT,
        handlers=handlers,
        force=align_all_loggers
    )

    return logging.getLogger(module_name)

def set_global_log_level(logger, level=logging.INFO):
    logger.setLevel(level)
    
def _get_aws_secret(
    secret_name, 
    region_name='us-west-2',
    version=None
):
    '''
    Grabs a key-value pair secret from AWS Secrets Manager service for use.

    Parameters
    ----------
    secret_name : str
        The Secret Name used for referencing this secret in AWS Secrets Manager
    region_name : str, optional
        AWS region, by default 'us-west-2'
    version : str, optional
        VersionStage(s) referring to a specific version of the key, by default 
        None. If None, will simply use version 'AWSCURRENT'.

    Returns
    -------
    dict
        {secret_key: secret_value}. For passwords, the format is usually 
        {<username>: <password>}.
    '''
    # Code adapted from snippet by AWS
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.
    
    kwargs = {'SecretId': secret_name}
    if version is not None:
        kwargs['VersionStage'] = version

    try:
        get_secret_value_response = client.get_secret_value(
            **kwargs
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            return json.loads(get_secret_value_response['SecretString'])
        else:
            return base64.b64decode(get_secret_value_response['SecretBinary'])
            

def get_secure_key(
    name, 
    filepath='/home/jovyan/work/secure_keys/', 
    aws_secret=False,
    bypass_safety_check=False
):
    '''
    Using a locally-stored key file, read the key and return it as a string.
    Note that you should NEVER print out a secret key in a log or to stdout.


    Parameters
    ----------
    name: str. Name of the secret key entry/file. 
        Will be appended to ``filepath`` to find the file of aws_secret is 
        False.

    filepath: str. Absolute or relative path to the key file, excluding the
        filename.
        
    aws_secret: bool. If True, assumes you want to query AWS Secrets Manager 
        to get the key and will query for a secret named `filename`. If this is
        True, `filepath` is ignored.

    bypass_safety_check: bool. If True, does not ask for user to confirm
        that they will avoid printing out/logging/etc. the key. Only set
        to True when you know this to be the case!


    Returns
    -------
    If `aws_secret` is False, just the key value. If it's True, returns a dict 
    of the form `{<username>: <password>}` typically.
    '''

    if not bypass_safety_check:
        confirm = input("Please confirm that you are NOT printing the "
        "key to a log/terminal/notebook/anywhere other than a variable "
        "(y/n): ")
    else:
        confirm = 'y'

    if confirm == 'y':
        if not aws_secret:
            full_path = filepath + name
            with open(full_path) as f:
                output = f.readline().strip()
        else:
            output = _get_aws_secret(name)
        return output
    else:
        raise RuntimeError("Don't print out secret keys!")

def get_current_datetime(date_delimiter="-", time_delimiter="_"):
    '''
    Gets the current UTC date and time.


    Parameters
    ----------
    date_delimiter: str. Should be a single character
        like "/" or "-". Indicates what to use as the
        separator character between days, months, and years.

    time_delimiter: str. Should be a single character
        like ":" or "_". Indicates what to use as the
        separator character between hours, minutes, 
        and seconds (e.g. string_delimiter=":" -> "08:00:00").


    Returns
    -------
    datetime.datetime if both delimiter args are None or 
    string object otherwise.
    '''

    current_datetime = datetime.utcnow()

    if date_delimiter is not None or time_delimiter is not None:
        if date_delimiter is not None:
            date_format = f"%m{date_delimiter}%d{date_delimiter}%Y"

        else:
            date_format = "%m-%d-%Y"

        if time_delimiter is not None:
            time_format = f"%H{time_delimiter}%M{time_delimiter}%S"

        else:
            time_format = "%H_%M_%S"

        full_format = date_format + "_T" + time_format
        return current_datetime.strftime(full_format)

    else:
        return current_datetime

def get_current_time(string_delimiter="_"):
    '''
    Gets the current UTC time.


    Parameters
    ----------
    string_delimiter: str. Should be a single character
        like ":" or "_". Indicates what to use as the
        separator character between hours, minutes, 
        and seconds (e.g. string_delimiter=":" -> "08:00:00").


    Returns
    -------
    datetime.time or string object as described above.
    '''

    if string_delimiter is not None:
        time_format = f"%H{string_delimiter}%M{string_delimiter}%S"
        return datetime.utcnow().time().strftime(time_format)

    else:
        return datetime.utcnow().time()
    
def local_datetime_to_utc(local_datetime_str, local_timezone='US/Eastern'):
    
    # Make local time timezone-aware
    timezone = pytz.timezone(local_timezone)
    local_time = timezone.localize(
        datetime.strptime(local_datetime_str, "%m-%d-%Y %H:%M:%S")
    )
    
    return local_time.astimezone(pytz.UTC)


def _df_str_extract_assign_to_colnames(df, column, sep, col_names, regex_strs, prefixes=None, suffixes=None,
                                       global_prefix="^", global_suffix="$"):
    """
    extract column names and assign to the dataframe
    adapted from https://stackoverflow.com/a/45377776
    :param df: dataframe input
    :param column: column in df to which to apply .str.extract(...)
    :param sep: separator regex
    :param col_names: names of capturing groups that correspond to column names in df
    :param regex_strs: regex strs inside of each capturing group, respectively
    :param prefixes: valid regex prepended as prefix for each string match
    :param suffixes: valid regex appended as suffix for each string match
    :param global_prefix: valid regex prepended as overall prefix
    :param global_suffix: valid regex appended as overall suffix
    :return:
    """
    def _extend_list_with_entries(l_extend, l_ref, entry=''):
        if not l_extend:
            l_extend = []
        n = len(l_extend)
        return l_extend + [entry] * (len(l_ref) - n)

    def _assemble_regex(labels, _prefixes, captured_regexes, _suffixes, separator=sep):
        assert len(labels) == len(captured_regexes)
        things = []
        for label, prefix, regex_str, suffix in zip(labels, _extend_list_with_entries(_prefixes, labels),
                                                    captured_regexes,
                                                    _extend_list_with_entries(_suffixes, labels)):
            labeled_capture_group = f'{prefix}(?P<{label}>{regex_str}){suffix}'
            things.append(labeled_capture_group)
        return separator.join(things)

    regex = global_prefix + _assemble_regex(col_names, prefixes, regex_strs, suffixes) + global_suffix
    return df.join(df.loc[:, column].str.extract(regex, expand=True))


def df_str_extract_assign_to_colnames(df, column, sep, col_names, regex_strs, prefixes=None, suffixes=None,
                                      global_prefix="^", global_suffix="$"):
    return df.pipe(_df_str_extract_assign_to_colnames, column, sep, col_names, regex_strs,
                   prefixes=prefixes, suffixes=suffixes,
                   global_prefix=global_prefix, global_suffix=global_suffix)
    

def get_memory_usage(logger):
    '''
    Helper function to log how we're doing 
    on memory consumption.

    Parameters
    ----------
    logger : logging.logger object
        Logger to use for logging memory usage.

    Returns
    -------
    float
        Percentage of memory being used on 
        the [0.0, 100.0] scale.
    '''
    #memory_used = psutil.virtual_memory()[2]
    total_memory, available_memory = psutil.virtual_memory()[0:2]
    memory_used = round(
        (total_memory - available_memory) / total_memory * 100,
        2
    )
    logger.info(f'Memory used: {memory_used}%')
    return memory_used


def check_filepath_exists(out_file, overwrite, logger=logging.getLogger(__name__)):
    """does the path to this file exist / should we write to it or raise an error?"""
    from pathlib import Path
    out_path = Path(out_file)
    if out_path.exists():
        if out_path.is_file():
            if not overwrite:
                logger.error("out_path exists: %s" % out_path)
                raise ValueError("out_path exists: %s" % out_path)
            else:
                logger.warning("out_path exists: %s" % out_path)
        elif out_path.is_dir():
            logger.error("out_path is dir: %s" % out_path)
            raise ValueError("out_path is dir: %s" % out_path)
        else:  # it's either a file a directory... right?
            raise ValueError("unknown path status for %s" % out_path)
