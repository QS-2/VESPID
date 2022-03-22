import numpy as np
import pandas as pd
import logging
from vespid.data.nsf_biblio import solr_to_dataframe, solr_to_dataframe_serial
from vespid import setup_logger
from datetime import date
from dateutil.relativedelta import relativedelta
from argparse import ArgumentParser

logger = setup_logger()


def generate_date_buckets(start_date=None, frequency='monthly'):
    '''
    Given either a start date or a number of years in the past
    to start from, generates start and end dates representing 
    intervals of length ``frequency`` that continue until 
    they reach the present day.

    Note that, because the intervals are assumed to be inclusive,
    start dates won't be exactly one month before end dates, but
    rather they'll be (one month - one day) before it so we don't
    have overlapping samples.


    Parameters
    ----------
    start_date: str of the form 'YYYY-MM-DD'. Indicates the first date to use
        as a seed for the interval generator.
        
    frequency: str. Can be one of ['monthly', 'weekly', 'daily'].
        Dictates the size of each interval.


    Returns
    -------
    pandas DataFrame of start and end dates with column name
    'start_date' and 'end_date'.
    '''

    # Make date object out of start date
    year, month, day = [int(d) for d in start_date.split('-')]
    start = date(year=year, month=month, day=day)
    delta = relativedelta(date.today(), start)

    if frequency == 'monthly':    	
        time_back = delta.years * 12 + delta.months
        interval_size = relativedelta(months=1)
        
    elif frequency == 'weekly':
        time_back = delta.years * 52 + delta.weeks
        interval_size = relativedelta(weeks=1)
        
    elif frequency == 'daily':
        time_back = (date.today() - start).days
        interval_size = relativedelta(days=1)
        
    else:
        raise ValueError(f"`frequency` value of {frequency} not supported")
    
    intervals = []
    for i in range(time_back):
        # Make sure we don't iterate into the future!
        if start + interval_size * i < date.today():
            intervals.append(start + interval_size * i)
        else:
            break
    start_dates = pd.Series(intervals)

    # Subtract a day to each end date and shift by one month 
    # so we end up with 6-23 to 7-22, 7-23 to 8-22, etc. 
    # (since date ranges are inclusive in Solr usually)
    if frequency != 'daily':
        end_dates = start_dates.apply(lambda d: d + relativedelta(days=-1)).shift(-1)
    else:
        end_dates = start_dates.shift(-1)

    date_buckets = pd.DataFrame(
        {
            "start_date": start_dates, 
            "end_date": end_dates
        }
    ).dropna().sort_values('start_date', ascending=True)

    return date_buckets


def intervals_to_filters(intervals):
    '''
    Given a set of time intervals defined by their start and end dates,
    generate Solr-query-friendly filters for using them.


    Parameters
    ----------
    intervals: tuples of the form (start_date, end_date) wherein the 
        start and end dates are datetime.date objects. If using 
        generate_monthly_date_buckets, recommended input form is:
        
            `[(s, e) for _, s, e in intervals.itertuples()]`


    Returns
    -------
    List of 2-tuples of strings of the form [('date', '<start_date> TO <end_date>')]
    '''

    return [('date', f'[{str(s)} TO {str(e)}]') for s, e in intervals]


def query_solr_batch_dates(start_date, aws_credentials_path, frequency='monthly', start_index=0):
    '''
    Given a set of start and end dates, batch-query Solr to pull down records
    specific to those dates. 
    
    Note that this function currently assumes
    that we want to push the results into S3 under the 'vespid' bucket,
    with results being pickled pandas DataFrames with filenames
    indicating the maximum date represented in them (roughly)
    one month or less after the start date for the query).

    Note also that this function only pulls down the necessary
    fields required for our data pipeline as of 7/26/21. This
    is done to reduce the data payload.

    Only the Web of Science corpus is currently supported.


    Parameters
    ----------
    start_date: str of the form 'YYYY-MM-DD'. Indicates the first date to use
        as a seed for the interval generator. 
        All interval queries will assume batches of one month
        in length that terminate with today's date.

    frequency: str. Can be one of ['monthly', 'weekly', 'daily'].
        Dictates the size of each interval.

    aws_credentials_path: str. If not None, provides filepath to CSV file
        representing the S3 credentials needed for saving individual 
        query results as pickle objects to the cloud.

    start_index: int. If doing a non-date-based set of queries (e.g. paging
        through the results of a single large query), use this argument
        to start at a non-zero point in the paging process. Often used
        when query parsing fails after persisting results a few times.


    Returns
    -------
    Only the first two rows of each query interval period, for
    API consistency with our async querying module.
    '''
    
    # Columns we absolutely need for the data pipeline
    # Querying for only these will significantly reduce
    # data payloads and thus query times.
    columns_to_keep = [
        'date',
        'ref_id',
        'author_name',
        'cat_subject',
        'author_inst',
        'source',
        'pubtype',
        'author_address',
        'title',
        'author_email',
        'id',
        'page_count',
        'author_state',
        'author_city',
        'author_zip',
        'cat_heading',
        'author_country',
        'abstract',
        'cat_subheading',
        'doctype',
        'author_last',
        'doi',
        'fund_text',
        'grant_agency',
        'grant_id',
        'grant_source'
    ]

    if start_date != '':
        date_buckets = generate_date_buckets(start_date, frequency=frequency)
        final_date = date_buckets['end_date'].max()
        logger.info(f"Queries will be between {str(start_date)} and {str(final_date)}")
        filters = intervals_to_filters([(s, e) for _, s, e in date_buckets.itertuples()])

        minimal_results = solr_to_dataframe_serial(
            queries=pd.Series(['*'] * len(filters)), 
            collection_name='wos', 
            filters=filters,
            fields=columns_to_keep,
            limit=40_000_000, # full corpus size - that should be safe!
            upload_credentials_path=aws_credentials_path
        )

    else:
        minimal_results = solr_to_dataframe_serial(
            queries=pd.Series(['*']), 
            collection_name='wos', 
            filters=[('date', '[1990-01-01 TO NOW]')],
            fields=columns_to_keep,
            limit=40_000_000, # full corpus size - that should be safe!
            upload_credentials_path=aws_credentials_path,
            start_index=start_index
        )

    

    return minimal_results


if __name__ == '__main__':
    parser = ArgumentParser(description='NSF WoS corpus date-based queries batch pulling')

    parser.add_argument('aws_credentials_path', type=str, 
    help="str. If not None, provides filepath to CSV file "
    "representing the S3 credentials needed for saving individual "
    "query results as pickle objects to the cloud.")

    parser.add_argument('--start_date', type=str, default='',
    help="str of the form 'YYYY-MM-DD'. Indicates the first date to use "
    "as a seed for the interval generator. "
    "All interval queries will assume batches of one month "
    "in length that terminate with today's date.")

    parser.add_argument('--start_index', type=int, default=0,
    help="Index to start parsing queries at. Usually used if parsing was interrupted "
    "after persisting the results of the completed portion.")

    parser.add_argument('--frequency', type=str, default='',
    help="str. Can be one of ['monthly', 'weekly', 'daily']. "
    "Dictates the size of each query interval.")
    
    args = parser.parse_args()

    _ = query_solr_batch_dates(
        args.start_date, 
        args.aws_credentials_path,
        frequency=args.frequency,
        start_index=args.start_index
    )