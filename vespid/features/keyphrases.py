from logging import log
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.utils import check_array
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer
)
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
from nltk.corpus import stopwords as stopwords_nltk
#from spacy.lang.en.stop_words import STOP_WORDS as stopwords_spacy
from vespid.features.preprocessing import preprocess_text
from vespid.models.specter import get_keyphrase_embeddings

import hdbscan
from vespid.models.clustering import HdbscanEstimator

from vespid import setup_logger
logger = setup_logger(__name__)

# Determines how best to tell BERT that we've connected two
# strings together that aren't always like that.
# Use whenever aggregating/joining documents
tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
CONCATENATING_TOKEN = tokenizer.sep_token

class ClusterTfidf(TfidfTransformer):
    """
    A Cluster/Class-based TF-IDF procedure using scikit-learn's TfidfTransformer as a base. 
    Adapted from the BERTopic project: https://github.com/MaartenGr/BERTopic.

    C-TF-IDF can best be explained as a TF-IDF formula adopted for multiple classes
    by joining all documents per class. Thus, each class is converted to a single document
    instead of set of documents. Then, the frequency of words **t** are extracted for
    each class **i** and divided by the total number of words **w**.
    Next, the total, unjoined, number of documents across all classes **m** is divided by the total
    sum of word **i** across all classes.
    """

    def __init__(
        self, 
        cluster_label='cluster_label', 
        embedding_aggregation_type='mean',
        embedding_weights=None,
        top_n=30,
        ngram_range=(2,3),
        stop_words='nltk',
        **kwargs
    ):
        '''
        Parameters
        ----------
        cluster_label: str. Indicates column in DataFrames provided
            that provides cluster membership label for a given document.

        embedding_aggregation_type: str. See 
            self._aggregate_embeddings() for 
            list of allowed values.

        embedding_weights: numpy array of float of shape 
            (n_documents,) or and HdbscanEstimator object. 

            If passed a numpy array, the values will be
            used to weight the contribution of each document
            to the aggregate embedding vector of the
            relevant cluster.

            If passed an HdbscanEstimator object, it must 
            be the trained model used to generate the cluster 
            labels. The model's soft clustering probabilities
            will be used to provide weights for embedding 
            aggregation.

            If None, an unweighted aggregation will be performed.

        top_n: int. Indicates how many candidate keyphrases to generate
            per cluster. Recommended values are between 10 and 30.

        ngram_range: 2-tuple of ints of the form (min, max). Indicates the
            minimum and maximum keyphrase length to allow.

        stop_words: str. Can be one of ['nltk', 'sklearn', 'spacy']. Indicates what
            the stop words should be that are used for preprocessing, if
            any. If None, stopwords will not be used.

            * 'nltk': generates 179 stopwords
            * 'spacy': generates 326 stopwords
            * 'sklearn': TBD
        
        kwargs: keyword arguments for the sklearn TfidfTransformer class
        '''
        super().__init__(**kwargs)
        self.cluster_label = cluster_label
        self.embedding_aggregation_type = embedding_aggregation_type
        self.embedding_weights = embedding_weights
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def _prepare_text_for_keyphrase_extraction(
        self,
        data
    ):
        '''
        Prepares the text data for keyphrase extraction at 
        the document cluster level.


        Parameters
        ----------
        df: pandas DataFrame. Should contain at least the 
            columns ['<cluster_label>', 'title', 'abstract'].


        Returns
        -------
        Numpy array of preprocessed text for each document 
        with shape (n_documents,).
        '''
        df = data.copy()

        # Just in case
        df[self.cluster_label] = df[self.cluster_label].astype(int)

        titles = preprocess_text(df['title'])
        abstracts = preprocess_text(df['abstract'])

        return titles + f' {CONCATENATING_TOKEN} ' + abstracts

    def _aggregate_embeddings(self, embeddings):
        '''
        Aggregate vectorized embeddings, possibly in a weighted fashion.
        Note that aggregation will default to unweighted calculations
        if 


        Parameters
        ----------
        embeddings: pandas Series or numpy array (if unweighted) 
            of embedding vectors or pandas DataFrame (if weighted) 
            with columns ['embedding', 'weights']


        Returns
        -------
        numpy array of shape (embedding_size,) that 
        represents the aggregate vector
        '''
        # Make sure we get arrays even if we're given lists
        if isinstance(embeddings, (pd.Series, np.ndarray)):
            e = np.array(embeddings.tolist())
            weights = None
        elif isinstance(embeddings, pd.DataFrame):
            e = np.array(embeddings['embedding'].tolist())
            if 'weights' in embeddings.columns:
                weights = embeddings['weights'].values
            else:
                weights = None
        else:
            raise ValueError(f"``embeddings`` is of type {type(embeddings)} "
                            "which is not supported")

        if self.embedding_aggregation_type == 'mean':
            return np.average(e, axis=0, weights=weights)

        else:
            raise ValueError(f"``agg_type`` of '{self.embedding_aggregation_type}' not supported")

    def _prepare_data_for_keyphrase_extraction(
        self,
        df
    ):
        '''
        Takes document-level information (e.g. text and cluster labels) 
        and returns cluster-level 
        data ready for keyphrase extraction.


        Parameters
        ----------
        df: pandas DataFrame. Should contain at least the 
            columns ['<cluster_label>', 'text', 'embedding'] wherein
            'text' is preprocessed and concatenated
            titles and abstracts and 'embedding'
            is a language vector embedding for the document
            (e.g. from SPECTER).


        Returns
        -------
        pandas DataFrame with columns ['<cluster_label>', 'text', 'embedding'],
        one row per cluster. The text is a concatenation of all documents
        for a given cluster and the embedding is an aggregation of the 
        embeddings for each document in a given cluster.
        '''

        data = df.copy()

        allowed_aggregation_types = ['mean']
        if self.embedding_aggregation_type not in allowed_aggregation_types:
            raise ValueError("embedding_aggregation_type of type "
                            f"{type(self.embedding_aggregation_type)} not supported")

        if self.embedding_weights is None or isinstance(self.embedding_weights, np.ndarray):
            if self.embedding_weights is None:
                pass # do nothing, aggregation function knows what to do
            else:
                data['weights'] = self.embedding_weights

        elif isinstance(self.embedding_weights, HdbscanEstimator):
            data['weights'] = self.embedding_weights.soft_cluster_probabilities

        else:
            raise ValueError("``embedding_weights`` type of "
                            f"{type(self.embedding_weights)} not supported")

        # Pre-process text elements and concatenate titles + abstracts
        data['text'] = self._prepare_text_for_keyphrase_extraction(data)

        # Concatenate all documents together on a cluster level
        tqdm.pandas(desc='Concatenating all documents per cluster')
        output = pd.DataFrame(
            data.groupby(self.cluster_label)['text'].progress_apply(
                lambda t: t.str.cat(sep=f" {CONCATENATING_TOKEN} ")
            )
        )

        # Aggregate document embeddings on a cluster level
        #TODO: this is currently the biggest time suck, any way to speed up?
        tqdm.pandas(
            desc='Aggregating all document embeddings into one embedding per cluster')
        cluster_embeddings = pd.DataFrame(data.groupby(self.cluster_label)
                                            .progress_apply(
            self._aggregate_embeddings
        ), columns=['embedding'])

        output['embedding'] = output.join(
            cluster_embeddings, how='left')['embedding']
        output.reset_index(drop=False, inplace=True)

        return output

    def extract_keyphrase_candidates(
        self,
        df
    ):
        '''
        Using cluster-based tf-idf, extract a bunch of candiate keyphrases
        from each cluster to be used as the full set of possible keyphrases
        per cluster in downstream modeling.


        Parameters
        ----------
        pandas DataFrame. Should contain at least the
            columns ['<cluster_label>', 'title', 'abstract', 'embedding'].
            Each row is a document.


        Returns
        -------
        pandas DataFrame with columns ['<cluster_label>', 'keyphrases'],
        with the latter being lists of strings.
        '''
        num_documents = len(df)
        num_candidates = 30

        # List of stopwords to exclude that we know we want to avoid
        additional_stopwords = [
            'elsevier'
        ]

        if self.stop_words == 'nltk':
            stops = set(stopwords_nltk.words('english'))

        elif self.stop_words == 'spacy':
            # Tokenizer removes punctuation so this must too
            #stops = set(t.replace("'", "") for t in stopwords_spacy)
            logger.warning("spacy stopwords currently having issues, switching to nltk stopwords...")
            stops = set(stopwords_nltk.words('english'))

        elif self.stop_words == 'sklearn':
            stops = 'english'

        elif self.stop_words is not None:
            raise ValueError(f"``stop_words`` value of {self.stop_words} is not supported")

        stops.update(additional_stopwords)
        
        # Pre-process document-level text
        df['text'] = self._prepare_text_for_keyphrase_extraction(df)

        # Transform document-level data to cluster-level
        df_clusters = self._prepare_data_for_keyphrase_extraction(df)

        logger.info("Counting terms across the corpus...")
        # Make sure we leave the [SEP] token alone for BERT
        #TODO: find a way to use CONCATENATING_TOKEN here without losing regex escapes
        tokenizing_pattern = r'(?u)\b\w\w+\b|\[SEP\]'
        
        #TODO: make this part of init()
        vectorizer = CountVectorizer(
            ngram_range=self.ngram_range, 
            stop_words=stops, 
            token_pattern=tokenizing_pattern, 
            lowercase=False,
            #max_features=100_000,
            #dtype=np.int32 # default is np.int64
        )

        #TODO: Make this part of fit()
        X = vectorizer.fit_transform(df_clusters['text'])
        words = vectorizer.get_feature_names_out()
        logger.debug(f"len(words) = {len(words)}")

        logger.info("Calculating tf-idf scores...")
        tfidf_matrix = self.fit_transform(X, n_samples=num_documents)

        # Get words with highest scores for each cluster
        # Remember that, for tfidf_matrix[i][j], i == document, j == word
        #TODO: find a way to get value-sorted sparse matrix to save on memory
        logger.warning("This algorithm can't handle cluster counts in the 20,000 range!")
        tfidf_matrix_dense = tfidf_matrix.toarray()

        # Get indices for the top_n highest-scoring words for each cluster
        # Returns indices sorted in ascending order based on the values they refer to
        top_indices = tfidf_matrix_dense.argsort()[:, -num_candidates:]
        logger.debug(f"top_indices.shape = {top_indices.shape}")

        # Return word-score pairs for top_n of each cluster label integer
        # Form is {cluster_num: [(phrase1, score1), (phrase2, score2), etc.]}
        logger.info("Extracting topics...")
        #TODO: find an efficient numpy-driven way to do this
        #topics = {label: [(words[j], tfidf_matrix_dense[i][j]) for j in top_indices[i]][::-1] for i, label in enumerate(df[self.cluster_label])}
        topics = {}
        for i, label in enumerate(df_clusters[self.cluster_label]):
            topics[label] = [words[j] for j in top_indices[i]][::-1]

        # Build out a DataFrame for quick merging
        topics = pd.DataFrame(pd.Series(topics, name='keyphrases'))\
            .reset_index(drop=False).rename(columns={'index': self.cluster_label})
        df_clusters = df_clusters.merge(topics, on=self.cluster_label)

        return df_clusters
    

    def fit(self, X: sp.csr_matrix, n_samples: int):
        """
        Learn the idf vector (global term weights).


        Parameters
        ----------
        X: A matrix of term/token counts.

        n_samples: Number of total documents prior to
            cluster-wise document contatenation.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'))
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        dtype = np.float64

        if self.use_idf:
            _, n_features = X.shape
            df = np.squeeze(np.asarray(X.sum(axis=0)))
            avg_nr_samples = int(X.sum(axis=1).mean())
            idf = np.log(avg_nr_samples / df)
            self._idf_diag = sp.diags(idf, offsets=0,
                                      shape=(n_features, n_features),
                                      format='csr',
                                      dtype=dtype)

        return self

def mmr(doc_embedding,
        word_embeddings,
        words,
        top_n=5,
        diversity=0.8):
    """ 
    Calculate Maximal Marginal Relevance (MMR)
    between candidate keywords and the document.
    
    MMR considers the similarity of keywords/keyphrases with the
    document, along with the similarity of already selected
    keywords and keyphrases. This results in a selection of keywords
    that maximize their within diversity with respect to the document.

    Note that this is copied from the BERTopic project implementation:
    https://github.com/MaartenGr/BERTopic/blob/1ffc4569a40bf845d2083bfd63c30c8d648a3772/bertopic/_mmr.py
    
    
    Parameters
    ----------
    doc_embeddings: numpy array. A single document embedding.
    
    word_embeddings: numpy array. The embeddings of the selected 
        candidate keywords/phrases
    
    words: iterable of str. The selected candidate keywords/keyphrases
        that are represented in ``word_embeddings``.
        
    top_n: int. The number of keywords/keyhprases to return.
    
    diversity: float in the range [0.0, 1.0] . Indicates how diverse 
        the selected keywords/keyphrases are. The higher the value,
        the more priority diversity is given over similarity
        to the document.
    
    
    Returns
    -------
    numpy array of str - the selected keywords/keyphrases.
    """

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding.reshape(1, -1))
    word_similarity = cosine_similarity(word_embeddings)

    # Initialize candidates and already choose best keyword/keyphras
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)
        
    results = pd.Series([words[idx] for idx in keywords_idx])
    
    # Drop BERT concatenating tokens from results
    # And remove any leading/trailing whitespace
    results = results.str.replace(CONCATENATING_TOKEN, '', regex=False).str.strip().values

    return results

#TODO: convert this and all related functions into methods of a Keyphrases class or some such
def extract_cluster_keyphrases(
    documents_data,
    cluster_label='cluster_label',
    top_n=5,
    embedding_aggregation_type='mean',
    embedding_weights=None,
    ngram_range=(2,3),
    diversity=0.5
):
    '''
    Using Cluster-based Tf-idf (as implemented in the BERTopic project), 
    extracts keyphrases representative of each
    generated cluster that balance similarity
    with their cluster contents with diversity
    of keyphrases.

    Sources: https://github.com/MaartenGr/BERTopic


    Parameters
    ----------
    documents_data: pandas DataFrame containing
        document-level information about all
        documents in the cluster solution.
        Must at least include columns 
        ['<cluster_label>', 'title', 'abstract', 'embedding'] wherein
        the embedding is a SPECTER language embedding.

    cluster_label: str. Name of the column in ``documents_data``
        that indicates cluster membership.

    top_n: int. Indicates how many keyphrases per cluster
        are desired. Note that this cannot be larger than
        the number of candidate keyphrases generated prior
        to de-duplication and document similarity checks
        (currently set to 30).

    embedding_aggregation_type: str. Can be one of ['mean'].

    embedding_weights: numpy array of float. 
        Indicates how to weight the document-level 
        embeddings when aggregating them into a cluster-level embedding.

        If a model, will use the soft clustering probabilities
        from ``clusterer`` as the weights, with documents having
        stronger membership in a cluster being counted more
        heavily.

        If a numpy array, the values from the array will be
        used as is.

        If None, no weights will be applied.

    ngram_range: 2-tuple of ints of the form (min, max). Indicates the
        minimum and maximum keyphrase length to allow.

    diversity: float between 0.0 and 1.0. Indicates how much diversity 
        in keyphrases is desired. Values closer to 1.0 indicate greater
        diversity (at the cost of how similar the keyphrases are to
        the cluster itself).


    Returns
    -------
    pandas DataFrame with columns ``cluster_label`` and ``top_keyphrases`` of 
    length n_clusters.
    '''
    if not isinstance(embedding_weights, np.ndarray) \
        and embedding_weights is not None:
        raise ValueError("`embedding_weights` must be a numpy array")
    
    num_candidates_per_cluster = 30

    if top_n >= num_candidates_per_cluster:
        raise ValueError(f"``top_n`` must be less than {num_candidates_per_cluster}")

    # Account for potential naming scheme elsewhere in the project
    if 'labels' in documents_data.columns:
        df = documents_data.rename(columns={'labels': cluster_label})
    else:
        df = documents_data.copy()

    keyphrase_model = ClusterTfidf(
        cluster_label=cluster_label,
        embedding_aggregation_type=embedding_aggregation_type,
        embedding_weights=embedding_weights,
        top_n=top_n,
        ngram_range=ngram_range,
        stop_words='nltk'
    )

    df_clusters = keyphrase_model.extract_keyphrase_candidates(
        df
    )
    
    tqdm.pandas(desc='Generating SPECTER embeddings for each candidate keyphrase')
    df_clusters['keyphrase_embeddings'] = df_clusters['keyphrases'].progress_apply(get_keyphrase_embeddings)

    tqdm.pandas(desc='Calculating the most similar and most diverse keyphrases for each cluster')
    df_clusters['top_keyphrases'] = df_clusters.progress_apply(lambda row: mmr(
        row['embedding'], 
        row['keyphrase_embeddings'], 
        row['keyphrases'], 
        top_n=top_n, 
        diversity=diversity
    ), 
    axis=1)
    
    df_clusters = df_clusters.loc[
        df_clusters[cluster_label] > -1, 
        [cluster_label, 'top_keyphrases', 'embedding']]

    return df_clusters