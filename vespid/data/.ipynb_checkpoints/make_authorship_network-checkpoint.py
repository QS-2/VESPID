from argparse import ArgumentParser
from itertools import combinations
from csv import QUOTE_ALL
import logging
from os.path import splitext
from collections import Counter

from pandas import read_csv
from pandas import DataFrame
from pandas import option_context as pandas_option_context


def _list_append_key_value_with_fn(dict_, key, value, fn=lambda x: x):
    """create and append value to list stored at key, applying fn to value, return dict"""
    _l = dict_.get(key, [])
    _l.append(fn(value))
    dict_[key] = _l
    return dict_


def main(first_csv, author_id_col, author_split_char, author_split_col_suffix,
         first_id_col=None, author_data_cols=tuple(),
         output_csv=None, file_suffix_authordata="", file_suffix_edgelist=""):
    # read in data
    one = read_csv(first_csv, index_col=first_id_col, quoting=QUOTE_ALL)
    if first_id_col is not None and 'Unnamed: 0' in one.columns:
        one = one.drop(columns=['Unnamed: 0'])
    else:
        logger.debug(f"one.columns: {one.columns}")
    one = one.replace(r"&apos;", "'", regex=True)
    logger.info(one.head())
    logger.info(one.shape)
    # ensure we have necessary structure
    if author_id_col not in one.columns:
        raise ValueError(f"--author-id-col {author_id_col} not in columns of {first_csv}: {one.columns}")
    for c in one.columns:
        if c.endswith(author_split_col_suffix):
            raise ValueError(f"--author-split-col-suffix {author_split_col_suffix} ends column {c}")
    cols_to_split = [author_id_col] + list(author_data_cols)
    logger.debug(f"cols_to_split: {cols_to_split}")
    for col in cols_to_split:
        if not any(one[col].str.contains(author_split_char)):
            raise ValueError(f"--author-split-char {author_split_char} not in {col}")
    if file_suffix_authordata == file_suffix_edgelist:
        raise ValueError(f"authordata suffix == edgelist suffix {file_suffix_authordata} vs {file_suffix_edgelist}")
    # actually do the split
    split_cols = []
    split_id_col = author_id_col + author_split_col_suffix  # keep separate
    logger.debug(f"one.columns: {one.columns}")
    one = one.dropna(subset=cols_to_split)
    for col in cols_to_split:
        logger.debug(f"splitting col: {col}")
        new_col = col + author_split_col_suffix
        split_cols.append(new_col)
        logger.debug(one[col].str.split(author_split_char))
        one[new_col] = one[col].str.split(author_split_char)
    logger.debug(f"split cols: {split_cols}")
    assert split_id_col in split_cols  # sanity_check
    # and now do what we care about...
    author_data = {}  # id -> {col -> info}
    edge_counter = Counter()  # and populate the edge list from the split columns
    i = 0
    for name, row in one.iterrows():
        # if i > 10:
        #     break
        logger.debug(f"on row:{name} i:{i}")
        i += 1
        # author data
        curr_author_data = row[split_cols]
        logger.debug(f"author data:{curr_author_data}")
        # expand them... they all should be same length
        assert len(curr_author_data.apply(len).unique()) == 1
        num_authors_in_row = curr_author_data.apply(len).max()
        logger.debug(f"num_authors_in_row:{num_authors_in_row}")
        author_processed = DataFrame(
            curr_author_data.to_list(), columns=[f"author{idx}" for idx in
                                                 range(num_authors_in_row)],
            index=[c.replace(author_split_col_suffix, "").rstrip("_") for c in curr_author_data.index]
        ).transpose()
        logger.debug(f"author processsed:{author_processed}")
        # and assign
        for idx, author_row in author_processed.iterrows():
            curr_id = author_row[author_id_col]
            curr_entry = author_data.get(curr_id, {})
            # id col -> str
            # everything else, lists -- though expect things like name to all be same
            for k, v in author_row.items():
                if k in [author_id_col, "authname"]:
                    curr_entry[k] = v
                else:
                    curr_entry = _list_append_key_value_with_fn(curr_entry, k, v)
            # nth author
            curr_entry = _list_append_key_value_with_fn(curr_entry, "nth_author", int(idx.replace("author", "")) + 1)
            # num authors
            curr_entry = _list_append_key_value_with_fn(curr_entry, "num_authors", num_authors_in_row)
            # paper id for linking
            curr_entry = _list_append_key_value_with_fn(curr_entry, "eid", name)
            # date
            curr_entry = _list_append_key_value_with_fn(curr_entry, "coverDate", row["coverDate"])
            logger.debug(f"{curr_id} -> ({idx}) {curr_entry}")
            author_data[curr_id] = curr_entry
        # logger.debug(f"author_data:{author_data}")
        # edge list: undirected
        for p in combinations(row[split_id_col], r=2):
            edge_counter[tuple(p)] += 1
    edge_list = [(pair[0], pair[1], edge_counter[pair]) for pair in edge_counter]
    if output_csv is not None:
        with pandas_option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
            author_csv = splitext(output_csv)[0] + file_suffix_authordata + splitext(output_csv)[1]
            logger.info(f"writing authordata to {author_csv}")
            DataFrame.from_dict(author_data, orient='index').to_csv(author_csv, quoting=QUOTE_ALL)
            edge_csv = splitext(output_csv)[0] + file_suffix_edgelist + splitext(output_csv)[1]
            logger.info(f"writing edgelist to {edge_csv}")
            DataFrame(edge_list, columns=["Author1", "Author2", "Count"]).to_csv(edge_csv, quoting=QUOTE_ALL)
    else:
        logger.info("not writing...")


if __name__ == "__main__":
    parser = ArgumentParser(description="hotfix: create authorship network")
    parser.add_argument("paper_list_csv")
    parser.add_argument("-p", "--paper-id-col", required=True)
    parser.add_argument("-a", "--author-id-col", required=True)
    parser.add_argument("--author-data-cols", nargs="*", default=tuple())
    parser.add_argument("--author-split-char", default=";")
    parser.add_argument("--author-split-col-suffix", default="_split")
    parser.add_argument("-o", "--output-csv", help="prefix for --edgelist-suffix and --authordata-suffix output files")
    parser.add_argument("--edgelist-suffix", default="_edgelist", help="append to -o/--output-data")
    parser.add_argument("--authordata-suffix", default="_authordata", help="append to -o/--output-data")
    parser.add_argument("-v", "--verbose", action='count', default=0)
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    logger.debug(f'args: {args}')
    log_val = logging.ERROR - args.verbose * 10
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_fmt, level=max(log_val, logging.DEBUG))
    main(args.paper_list_csv, args.author_id_col, first_id_col=args.paper_id_col,
         author_data_cols=args.author_data_cols, author_split_char=args.author_split_char,
         output_csv=args.output_csv, author_split_col_suffix=args.author_split_col_suffix,
         file_suffix_authordata=args.authordata_suffix, file_suffix_edgelist=args.edgelist_suffix)
