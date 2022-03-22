from json import loads
from os.path import splitext

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pickle
from smart_open import open as s_open

from vespid import setup_logger
logger = setup_logger(__name__)


def yield_object_keys(bucket, prefix, limit=float('inf'),
                      skip_until_suffix=None):
    """yield object keys matching s3://bucket/prefix up to limit, skipping until suffix given"""
    logger.debug(f"yielding {limit} object keys from {bucket} / {prefix}")
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    count = 0
    skipping = False
    if skip_until_suffix:
        skipping = True
    for page in pages:
        for obj in page['Contents']:
            if count >= limit:
                return
            logger.debug(f"{count} key found: {obj}")
            if skip_until_suffix:
                key = obj['Key']
                basename = splitext(key)[0]
                if basename.endswith(skip_until_suffix):
                    skipping = False
                    logger.debug(f"stop skipping {obj} vs {skip_until_suffix}")
                else:
                    logger.debug(f"keep skipping! {obj} vs {skip_until_suffix}")
            if skipping:
                logger.debug(f"{count} key skipped: {obj}")
                continue
            logger.debug(f"{count} yielding: {obj}")
            yield obj['Key']
            count += 1


def yield_json_from_s3_obj(key, s3_bucket, encoding='utf-8', limit=float('inf')):
    """yield lines of loaded JSON from s3://bucket/key up to limit, 
    assuming open access to the S3 bucket"""
    logger.info(f"yielding {limit} json rows from {s3_bucket} / {key} with encoding {encoding}")
    with s_open(f"s3://{s3_bucket}/{key}", encoding=encoding) as thing:
        count = 0
        for line in thing:
            if count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            json_paper = loads(line)
            logger.debug(f"{count} json found: {json_paper}")
            yield json_paper
            count += 1

def get_s3_file(bucket, key, resource=None):
    '''
    Pulls the contents of a given object in S3 into memory.

    Parameters
    ----------
    bucket : str
        Name of the S3 bucket containing the object
    key : str
        Object prefix + name. E.g. if the bucket is called "vespid",
        and the file of interest is s3://vespid/data/a.json, the key
        would be "data/a.json"
    resource : boto3.resource object, optional
        Result of calling `boto3.resource('s3')`, by default None

    Returns
    -------
    bytestring
        The file contents
    '''
    if resource is None:
        s3 = boto3.resource('s3')
    else:
        s3 = resource
        
    return s3.Bucket(bucket).Object(key).get()['Body'].read()

def count_s3_objects(bucket, prefix):
    '''
    Given a bucket name and file prefix to match, counts how many qualifying
    objects there are. 

    Parameters
    ----------
    bucket : str
        Name of S3 bucket to inspect
    prefix : str
        Prefix of matching objects (e.g. 'data/processed/')

    Returns
    -------
    int
        The number of matching objects found
    '''
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket)
    
    num_s3_objects = sum(
        1 for _ in bucket.objects.filter(
            Prefix=prefix
        )
    )
    return num_s3_objects

def upload_file(file_name, bucket, object_prefix=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_prefix: prefix to add to ``file_name``, e.g. '<data/external/>filename'
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_prefix is None:
        object_name = file_name

    else:
        object_name = object_prefix + file_name

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logger.error(e)
        return False
    return True

def upload_object(
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
        aws_access_key, secret_access_key = pd.read_csv(access_key_path).loc[0]
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
        logger.debug(f"Detected that object is of type {type(object)}; "
        "pickling it and sending to S3 as a byte string...")
        to_upload = pickle.dumps(object)

    else:
        to_upload = object

    try:
        s3_resource.Object(bucket, path).put(Body=to_upload)

    except Exception as e:
        raise e
    
def test_s3_access(bucket):
    '''
    Tests S3 ingress and egress. Useful for code run in different environments 
    (e.g. AWS Batch) to run this before doing any heavy lifting to make sure 
    access credentialing, etc. is good to go.

    Parameters
    ----------
    bucket : str
        Name of S3 bucket to test
    '''
    # Test uploading
    test_object = pd.DataFrame({'a': [1,2,3], 'b': [4,5,6]})
    logger.debug("Testing upload of a pickled DataFrame to S3...")
    upload_object(
        test_object, 
        'upload_test.pkl',
        auto_pickle=True, 
        bucket=bucket
    )
    
    # Testing downloading
    assert get_s3_file(bucket, 'upload_test.pkl') is not None, \
    "S3 file download failed"
    
    logger.info("S3 upload test successful!")