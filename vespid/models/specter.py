# builtins
import os
from argparse import ArgumentParser
from io import StringIO
import gc
import time
# then 3rdparty
from transformers import AutoModel, AutoTokenizer
import torch
import pandas as pd
import numpy as np
import boto3
# then from this package
from vespid.data.neo4j_tools import Neo4jConnectionHandler
from vespid.data.neo4j_tools.utils import df_convert_cols_to_semicolon_delimited_str
from vespid import setup_logger

logger = setup_logger(module_name=__name__)


class Dataset:

    def __init__(self, data, max_length, batch_size, cuda_flag, id_field='paper_id', data_fields=('title', 'abstract')):
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.max_length = max_length
        self.batch_size = batch_size
        # data is assumed to be a dataframe with 
        # columns: unique ID for paper, paper title, and abstract
        self.data = data
        self.cuda_flag = cuda_flag
        self.id_field = id_field
        self.data_fields = data_fields

    def batches(self):
        # create batches
        batch = []
        batch_ids = []
        batch_size = self.batch_size
        i = 0
        for index, row in self.data.iterrows():
            # current data is used by both cases
            field_data = [row[x] or '' for x in self.data_fields]
            logger.debug(field_data)
            data_str = self.tokenizer.sep_token.join(field_data)
            # either add to a batch
            if i % batch_size != 0 or i == 0:
                batch_ids.append(row[self.id_field])
                batch.append(data_str)
            else:  # or process one and reset
                input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                           return_tensors="pt", max_length=self.max_length)
                if self.cuda_flag:
                    yield input_ids.to('cuda'), batch_ids
                else:
                    yield input_ids, batch_ids
                batch_ids = [row[self.id_field]]
                batch = [data_str]
            i += 1
        if len(batch) > 0:
            input_ids = self.tokenizer(batch, padding=True, truncation=True,
                                       return_tensors="pt", max_length=self.max_length)
            if self.cuda_flag:
                input_ids = input_ids.to('cuda')
            yield input_ids, batch_ids


class Model:

    def __init__(self, cuda_flag):
        self.model = AutoModel.from_pretrained('allenai/specter')
        self.cuda_flag = cuda_flag
        if self.cuda_flag:
            self.model.to('cuda')
        self.model.eval()

    def __call__(self, input_ids):
        output = self.model(**input_ids)
        return output.last_hidden_state[:, 0, :]


def one_piece_processing(data, id_field='paper_id', data_fields=('title', 'abstract'), embedding_field='embedding'):
    """ 
    Function takes in dataframe with columns ['paper_id', 'title', 'abstract'] and returns specter embeddings
        for the data. Processing is done in single step rather than in batches

    Parameters
    ----------
    data: dataframe containing publication information especially title, abstract and unique ID for each publication.
          Make sure the data in dataframe is complete without any none types
    id_field: field in each row that is an ID, e.g., `paper_id`
    data_fields: fields in each row to concatenate and embed. defaults to `title` and `abstract`
    embedding_field: e.g., 'embedding'

    Returns
    -------
    df: dataframe containing publication unique ID as an index and specter embeddings, title, abstract, topics
        as its columns
    """
    start = time.time()
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')
    data_fields = list(data_fields)
    df = data.dropna(subset=data_fields).reset_index(drop=True)

    # create a list with concatenated title and abstract
    title_abs = df[data_fields].agg(tokenizer.sep_token.join, axis=1).to_list()

    # creating inputs to the model by tokenizing concatenated data
    inputs = [tokenizer(paper, padding=True, truncation=True, return_tensors="pt", max_length=512) for paper in
              title_abs]

    # embeddings
    embeddings = [model(**i).last_hidden_state[0, 0, :].detach().numpy().tolist() for i in inputs]

    df[embedding_field] = embeddings
    df = df.set_index(id_field)
    end = time.time()
    time_taken = end - start
    logger.info(f"Took {time_taken} seconds for processing by cpu")
    if len(data) != len(df):
        logger.warning(f"Number of data points with none type: {len(data) - len(df)}")
    return df


def batch_processing(batch_size, data, cuda_flag, max_length=512,
                     id_field='paper_id', data_fields=('title', 'abstract'), embedding_field='embedding'):
    """
    Given a dataframe containing coulumns ['paper_id', 'title', 'abstract'], process it in batches of
        given batch size and return their specter embeddings.
        
        
    Parameters
    ----------
    batch_size: batch sizes to process data (commonly 1, 2, 4, 8, 16, 32)
    data: dataframe with columns ['paper_id', 'title', 'abstract']
    max_length: max length of tokenized text (default 512)
    cuda_flag: tells the prorgam to use GPU or not
    id_field: field in each row that is an ID, e.g., `paper_id`
    data_fields: fields in each row to concatenate and embed. defaults to `title` and `abstract`
    embedding_field: e.g., 'embedding'
    
    
    Returns
    -------
    df: a dataframe containing `id_field` and `embedding_field` as its columns
    """
    gc.collect()
    torch.cuda.empty_cache()
    start = time.time()
    batch_size = batch_size
    dataset = Dataset(data=data, max_length=max_length, batch_size=batch_size, cuda_flag=cuda_flag,
                      id_field=id_field, data_fields=data_fields)
    model = Model(cuda_flag=cuda_flag)
    results = {}
    batches = []
    try:
        if cuda_flag:
            # raise Exception("Sorry, CUDA implementation not yet available")
            for batch, batch_ids in dataset.batches():
                batches.append(batch) # Why is this here? 
                emb = model(batch)
                #TODO: can we speed this up by assigning everything directly to a DataFrame, instead of looping?
                for paper_id, embedding in zip(batch_ids, emb.unbind()):
                    results[paper_id] = {id_field: paper_id, embedding_field: embedding.detach().cpu().numpy().tolist()}
            gc.collect()
            torch.cuda.empty_cache()
            del model
            del dataset
            del batch
            end = time.time()
            time_taken = end - start
            print(f"Took {time_taken} seconds for a batch size of {batch_size} and cuda")
        else:
            for batch, batch_ids in dataset.batches():
                batches.append(batch) # Why is this here? 
                emb = model(batch)
                #TODO: can we speed this up by assigning everything directly to a DataFrame, instead of looping?
                for paper_id, embedding in zip(batch_ids, emb.unbind()):
                    results[paper_id] = {id_field: paper_id, embedding_field: embedding.detach().numpy().tolist()}

            end = time.time()
            time_taken = end - start
            print(f"Took {time_taken} seconds for a batch size of {batch_size} and without cuda")
        df = pd.DataFrame(columns=[id_field, embedding_field])
        #FIXME: pd.concat([list, of, dfs]) is far faster than appending in a loop, according to docs
        for key in results.keys():
            df = df.append(results[key], ignore_index=True)
        return df

    except RuntimeError as error:
        print(repr(error) + "\n" + "CUDA out of memory try reducing batch size or increasing GPU memory")


# TODO: this will probably work well for the generic string embedding algo, just need to rename variables and do other cleanup
def get_keyphrase_embeddings(phrases):
    """ 
    Takes keyphrase lists (one row per document/cluster) and returns SPECTER embeddings
        for the data. Processing is done in single step rather than in batches.

    Parameters
    ----------
    phrases: numpy array of strings containing keyphrases to embed. 
        Should have no null values.

    Returns
    -------
    numpy array of embedding vectors.
    """
    # TODO put this efficiency gains into batch processing and one piece - numpy array accept vs for loop over list
    if isinstance(phrases, pd.Series):
        keyphrases = phrases.values
    elif isinstance(phrases, np.ndarray):
        keyphrases = phrases
    elif isinstance(phrases, list):
        keyphrases = np.array(phrases)
    else:
        raise ValueError("``phrases`` must be of type pd.Series, list, or np.ndarray. "
                         f"Got {type(phrases)} instead")

    start = time.time()

    # load model and tokenizer
    logger.debug("Instantiating tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')

    logger.debug("Instantiating SPECTER model...")
    model = AutoModel.from_pretrained('allenai/specter')

    num_nulls = pd.Series(keyphrases).isnull().sum()
    if num_nulls > 0:
        raise ValueError(f"{num_nulls} elements in ``keyphrases`` have null values. "
                         "Please drop nulls and re-run.")

    # creating inputs to the model by tokenizing concatenated data
    logger.debug("Tokenizing inputs...")
    inputs = tokenizer(
        keyphrases.tolist(),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )

    logger.debug("Generating embeddings...")
    embeddings = model(**inputs).last_hidden_state[:, 0, :].detach().numpy()

    end = time.time()
    time_taken = end - start
    logger.debug(f"Took {time_taken} seconds for processing by CPU")

    return embeddings

#FIXME: topics aren't used in this function so far as I can see and are likely to significantly slow down data retrieval - remove
def get_data(ip_address, db_password=os.environ.get("NEO4J_PASSWORD"), limit=None,
             query="MATCH (a:Publication) RETURN a.id AS paper_id, a.title AS title, "
                   "a.abstract AS abstract, a.topics AS topics"):
    logger.info("getting data from neo4j...")
    if not db_password:
        logger.info(f"db_password is {db_password}, attempting to read `NEO4J_PASSWORD` from os.environ...")
        db_password = os.environ["NEO4J_PASSWORD"]
    graph = Neo4jConnectionHandler(db_ip=ip_address, db_password=db_password)  # 'Vespid!')
    # query = "MATCH (a:Publication) " \
    #         "RETURN a.id AS paper_id, a.title AS title, a.abstract AS abstract, a.topics AS topics"
    if limit and int(limit) > 0:
        query += f" limit {int(limit)}"
    df_papers = graph.cypher_query_to_dataframe(query)
    return df_papers


def main(db_ip, db_password, db_query_limit=None, query=None, s3_path=None, return_embeddings=False,
         processing_type='gpu', process='batch', batch_size=8, id_field='paper_id', data_fields=('title', 'abstract'),
         s3_bucket='vespid', local_path=None, output_neo4j_embeddings=False):

    """
:auto USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM
'file:///score_specter_embeddings_gpu.csv' as row
MATCH (p:Publication {id: row.paper_id})
SET p.embedding = [x IN split(row.embedding, ';') | toFloat(x) ]
    """
    # double-check options at start of code -- accept only one or the other
    num_options_specified = sum(bool(x) for x in (s3_path, return_embeddings, local_path))
    if num_options_specified != 1:
        raise ValueError(f"invalid combination of arguments: "
                         f"s3_path `{s3_path}` and return embeddings `{return_embeddings}`. "
                         f"Specify one or the other.")
    df_papers = get_data(ip_address=db_ip, db_password=db_password, limit=db_query_limit,
                         query=query)
    embedding_col = 'embedding'
    logger.info("processing...")
    if process == 'batch':
        if processing_type == 'gpu':
            using_cuda = torch.cuda.is_available()
            if using_cuda:
                results = batch_processing(batch_size=batch_size, data=df_papers, cuda_flag=True,
                                           id_field=id_field, data_fields=data_fields, embedding_field=embedding_col)
            else:
                logger.error('cuda not available try different method')
                raise ValueError('cuda not available try different method')
        elif processing_type == 'cpu':
            results = batch_processing(batch_size=batch_size, data=df_papers, cuda_flag=False,
                                       id_field=id_field, data_fields=data_fields, embedding_field=embedding_col)
        else:  # sanity check to avoid warnings or bugs
            raise ValueError(f"passed processing type: {processing_type}")
    elif process == 'onePiece':
        results = one_piece_processing(data=df_papers, id_field=id_field, data_fields=data_fields,
                                       embedding_field=embedding_col)
    else:  # sanity check to avoid warnings or bugs
        raise ValueError(f"passed invalid process: {process}")
    if output_neo4j_embeddings:
        results = df_convert_cols_to_semicolon_delimited_str(results, embedding_col)
    if s3_path:
        csv_buffer = StringIO()
        results.to_csv(csv_buffer)
        s3 = boto3.client('s3')
        s3.put_object(Bucket=s3_bucket, Key=s3_path, Body=csv_buffer.getvalue())
        return None
    elif return_embeddings:
        return results
    elif local_path:
        results.to_csv(local_path)
        return None
    else:  # sanity check to avoid warnings or bugs
        raise ValueError(f"invalid combination of arguments: "
                         f"s3_path `{s3_path}` and return embeddings `{return_embeddings}`. "
                         f"Specify one or the other.")


if __name__ == '__main__':
    parser = ArgumentParser(description='SPECTER inference. do the following with the resulting file:' + 
                            """:auto USING PERIODIC COMMIT 10000
LOAD CSV WITH HEADERS FROM
'file:///YOUR_FILE_PATH_ON_NEO4J_SERVER.csv' as row
MATCH (p:Publication {id: row.paper_id})
SET p.embedding = [x IN split(row.embedding, ';') | toFloat(x) ]""")
    # you can add an initial flag for shorthand
    parser.add_argument('-p', '--process', type=str, default='batch', help='batch or onePiece')
    # convention is you should stick with hyphenated words only all lowercase, no underscores either
    parser.add_argument('-t', '--processing-type', type=str, default='gpu', help='gpu or cpu')
    parser.add_argument("-b", '--batch-size', type=int, default=8,
                        help='What batch size to be used for batch processing')
    parser.add_argument('--db-ip', default=DB_IP, help="IP address of database. "
                                                       "defaults to SME+ at `35.164.169.229`. "
                                                       "SCORE is at `54.202.173.127`")
    parser.add_argument('--db-password', default=None,
                        help="password for database. if not given load 'NEO4J_PASSWORD' from os.environ")
    parser.add_argument("--s3-bucket", default='vespid', help="bucket to which to save results in S3")
    # address a to-do in this file
    g = parser.add_mutually_exclusive_group(required=True)
    # require this to be explicitly given now that we have multiple datasets and multiple embeddings
    g.add_argument("--s3-path", help="where save results in --s3-bucket, e.g., `data/processed/specter_embeddings.csv`")
    g.add_argument("--local-path", help="where save results locally " "e.g., `data/test_embeddings.csv`")
    g.add_argument("--return-embeddings", action='store_true',
                   help="return the embeddings instead of uploading to s3")
    # helpful for testing
    parser.add_argument("-l", "--db-query-limit", default=None, type=int,
                        help="pass this limit to the query that returns initial data")
    parser.add_argument("-q", "--query", default="MATCH (a:Publication) RETURN a.id AS paper_id, a.title AS title, "
                                                 "a.abstract AS abstract, a.topics AS topics",
                        help="send this query to neo4j for data")
    # address a to-do earlier
    parser.add_argument("--id-field", default="paper_id", help="id field to use")
    parser.add_argument("--data-fields", nargs="+", default=('title', 'abstract'),
                        help="string data fields to use to embed")
    # for easier import into neo4j...
    parser.add_argument("--output-neo4j-embeddings", action='store_true',
                        help="instead of outputting a json array, e.g., `[1.0,-3.0,...]`, "
                             "give `1.0;-3.0;...` semicolon-delimited for a `split` call")
    args = parser.parse_args()
    # if args.return_embeddings and args.s3_path:
    #     parser.error("specify only one of --s3-path or --return-embeddings")
    # elif not args.return_embeddings and not args.s3_path:
    #     parser.error("specify at least one of --s3-path or --return-embeddings")
    main_results = main(db_ip=args.db_ip, db_password=args.db_password, db_query_limit=args.db_query_limit,
                        query=args.query, return_embeddings=args.return_embeddings,
                        s3_path=args.s3_path, s3_bucket=args.s3_bucket, local_path=args.local_path,
                        processing_type=args.processing_type, process=args.process, batch_size=args.batch_size,
                        id_field=args.id_field, data_fields=args.data_fields,
                        output_neo4j_embeddings=args.output_neo4j_embeddings)
    if main_results is not None:
        logger.info(main_results)
