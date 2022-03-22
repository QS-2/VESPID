from multiprocessing.sharedctypes import Value
import hdbscan
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
import pickle
from tqdm import tqdm
import umap
import optuna
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import mlflow
from mlflow.tracking import MlflowClient
from vespid.models.mlflow_tools import load_model as load_mlflow_model
import cloudpickle
from smart_open import open

from vespid.features.interdisciplinarity import calculate_interdisciplinarity_score
from vespid import setup_logger


logger = setup_logger(module_name=__name__)

RANDOM_STATE = 42  # for reproducibility


class HdbscanEstimator(hdbscan.HDBSCAN, mlflow.pyfunc.PythonModel):
    '''
    Simple wrapper class for HDBSCAN that allows us to
    add some scikit-learn-like scoring functionality
    for doing parameter tuning with sklearn tools easily.
    '''

    # Note that __init__ should not be overwritten as it is difficult to abide
    # by the sklearn introspection techniques hdbscan.HDBSCAN uses under the
    # hood when doing so

    def fit(self, X, y=None, soft_cluster=False):
        '''
        Modified version of the original hdbscan fit method from 
        https://github.com/scikit-learn-contrib/hdbscan/blob/2179c24a31742aab459c75ac4823acad2dca38cf/hdbscan/hdbscan_.py#L1133.

        This version allows for soft clustering to be done at the same time as 
        the main clustering task, for efficiency's sake.

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or 
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.
        y : ignored
        soft_cluster : bool, optional
            Indicates if soft clustering should occur along with fitting 
            via the main algorithm, by default False
        '''
        super().fit(X)
        self.n_features_ = X.shape[1]
        self._is_fit = True

        if soft_cluster:
            logger.debug("Fitting secondary soft clustering model...")
            self.soft_cluster(X, train_model=True)

        return self

    def soft_cluster(self, data=None, train_model=False, use_original_method=False):
        '''
        Performs a soft-clustering with the given trained
        model to generate probabilities of membership
        for each datum relative to all clusters in the solution
        of the trained model. Serves as a way to determine
        the strength of membership of a given datum to every
        cluster in the solution.


        Parameters
        ----------
        data: numpy array or pandas DataFrame. Data that you want
            to assess for cluster membership. Note that, if this is 
            run post-clustering, the cluster
            solution will *not* be re-solved, but rather the new
            points will simply be assessed relative to the 
            existing solution. Should be of shape 
            (n_samples, n_features_original), 
            where n_features_original refers to the feature 
            dimensionality present in the training data.

            If None, results will be returned for the training
            data and assigned as the attribute self.soft_clusters.

        train_model: bool. If True, assumes you don't want to use the 
            existing trained soft clustering model (if there is one). Ignored
            if self.soft_cluster_model is None or `use_original_method=True`.

        use_original_method: bool. Indicates if you want to use the 
            distance- and noise-based approach to soft clustering from the 
            hdbscan library (True) or the new approach using a secondary model 
            predicting probabilities of class membership (False). 

            Note that, with `use_original_method=True`, the probabilities
            for each sample are not guaranteed to sum to 1.0.


        Returns
        -------
        pandas DataFrame that is of shape (n_samples, n_clusters).
        '''

        if not self.prediction_data and use_original_method:
            raise ValueError("Model didn't set `prediction_data=True` "
                             "during training so we probably can't get soft "
                             "cluster probabilities...")
        if use_original_method:
            self.soft_cluster_model = None
            if data is None:
                soft_clusters = hdbscan.all_points_membership_vectors(self)
            else:
                soft_clusters = hdbscan.prediction.membership_vector(
                    self, data)

        else:
            num_labels = len(np.unique(self.labels_))
            if num_labels <= 1:
                logger.warning(f"Clustering resulted in only {num_labels} "
                               "labels, which means soft clustering won't "
                               "work!")
                raise optuna.TrialPruned()
                
            elif train_model or not hasattr(self, 'soft_cluster_model'):
                if isinstance(data, pd.DataFrame):
                    data = data.values

                # Make data into DataFrame to facilitate tracking shuffled samples
                df = pd.DataFrame(
                    data,
                    columns=[f'x_{i}' for i in range(data.shape[1])]
                )
                feature_columns = df.columns
                df['label'] = self.labels_
                df_no_noise = df[df['label'] > -1]
                
                if len(df_no_noise) == 0:
                    raise RuntimeError("df_no_noise is length zero")

                X_train, X_test, y_train, y_test = train_test_split(
                    df_no_noise[feature_columns],
                    df_no_noise['label'],
                    train_size=0.6,
                    shuffle=True,
                    stratify=df_no_noise['label'],
                    random_state=RANDOM_STATE
                )

                scaler = StandardScaler()
                logreg = LogisticRegressionCV(
                    Cs=25,
                    cv=5,  # 80/20 train/test split
                    scoring='f1_micro',
                    multi_class='multinomial',
                    max_iter=100_000,
                    random_state=RANDOM_STATE
                )

                self.soft_cluster_model = Pipeline([
                    ('scaler', scaler),
                    ('logreg', logreg)
                ])

                self.soft_cluster_model.fit(X_train, y_train)
            soft_clusters = self.soft_cluster_model.predict_proba(data)

        # Put everything into one view, labels and probabilities
        df = pd.DataFrame(
            soft_clusters,
            columns=[f"prob_cluster{n}" for n in range(soft_clusters.shape[1])]
        )

        # Save for later reference and usage
        if train_model:
            self.soft_cluster_probabilities = df

        self._test_soft_clustering()

        return df

    def predict_proba(self, X=None):
        '''
        Generates probability vectors for all points provided,
        indicating each point's probability of belonging to one of the 
        clusters identified during fitting.

        Similar to self.soft_cluster(), but this works in pure inference mode
        and assumes no knowledge of the cluster labels (as this can be used
        for "clustering" new points without re-fitting the entire algorithm).

        Parameters
        ----------
        X : pandas DataFrame, optional
            The input features for soft clustering. Should be of shape 
            self.n_features_. If None, will just return the pre-computed 
            probability vectors for the training data, by default None

        Returns
        -------
        pandas DataFrame with columns "prob_cluster{i}"
            Membership vectors (one cluster per column) for each data point.

        Raises
        ------
        RuntimeError
            If no data are provided but no soft clustering results are 
            available either.
        '''
        if X is None and hasattr(self, 'soft_cluster_probabilities'):
            return self.soft_cluster_probabilities
        elif not hasattr(self, 'soft_cluster_probabilities'):
            raise RuntimeError("No soft clustering model available")
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        return pd.DataFrame(
            self.soft_cluster_model.predict_proba(X),
            columns=self.soft_cluster_probabilities.columns,
            index=X.index
        )

    def predict(self, context, X=None):
        '''
        Provides cluster labels for all points provided. If no data are 
        provided, simply returns the labels for the training data. Uses the 
        pre-trained soft clustering model to find the cluster for each point
        with the highest predicted probability and returns that label.

        Parameters
        ----------
        context : mlflow.pyfunc.PythonModelContext object
            Provides contextual information (usually data from mlflow 
            artifact store needed to do predictions). Can be None and often is.

        X : pandas DataFrame, optional
            Input features. X.shape[1] == self.n_features_. If None, 
            pre-computed labels for all training data will be returned, 
            by default None

        Returns
        -------
        pandas DataFrame with an index matching `X.index`
            Cluster labels for each point.
        '''
        if X is None:
            logger.info("No data provided for inference so returning "
                        "pre-calcluated labels")
            return self.labels_

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        # Use the soft clustering model to predict new data cluster membership
        return pd.DataFrame(
            self.predict_proba(X).values.argmax(axis=1),
            index=X.index
        )

    # def load_context(self, context):
    # '''Used for loading data to be used at prediction time mainly'''
    #     with open(context.artifacts["hdb_model"], 'rb') as f:
    #         self = cloudpickle.load(f)

    def info(self):
        '''
        Prints to the log a bunch of useful metrics about
        the cluster solution, like DBCV, mean persistence, 
        etc.
        '''

        num_noise_points = len(self.labels_[self.labels_ == -1])
        num_all_points = len(self.labels_)
        pct_noise_points = round(num_noise_points / num_all_points * 100, 2)
        num_clusters = pd.Series(self.labels_[self.labels_ > -1]).nunique()
        persistence = self.cluster_persistence_

        soft_model_presence = 'is' if hasattr(
            self, 'soft_cluster_model') else 'is not'

        logger.info(f"Number of clusters in this solution is {num_clusters} and "
                    f"there are {num_noise_points} ({pct_noise_points})% noise points. "
                    f"DBCV score is {self.relative_validity_}. "
                    "Additionally, there is a mean cluster persistence of "
                    f"{persistence.mean()} and a standard deviation of {persistence.std()}. "
                    f"Average cluster membership probability is {self.probabilities_.mean()} "
                    f"and average outlier score is {self.outlier_scores_.mean()}. "
                    f'Soft cluster model {soft_model_presence} present')

    def score(self, X=None, y=None):
        '''
        Provides the DBCV value of the clustering solution found
        after fitting.


        Parameters
        ----------
        X: ignored, as it's passed during the fit() phase.

        y: ignored, as this is unsupervised learning!


        Returns
        -------
        Dict of various trained-model-specific scores/stats.
        '''
        # Give some useful info for reporting during tuning
        self.info()

        num_clusters = pd.Series(self.labels_[self.labels_ > -1]).nunique()
        num_noise_points = len(self.labels_[self.labels_ == -1])
        scores = {
            'DBCV': self.relative_validity_,
            'num_clusters': num_clusters,
            'num_noise_points': num_noise_points,
            'persistence_mean': self.cluster_persistence_.mean(),
            'persistence_std': self.cluster_persistence_.std(),
            'probability_mean': self.probabilities_.mean(),
            'probability_std': self.probabilities_.std(),
            'outlier_score_mean': self.outlier_scores_.mean(),
            'outlier_score_std': self.outlier_scores_.std(),
        }

        return scores

    # @classmethod
    # def load_model(cls, filepath):
    #     '''
    #     Loads a trained HDBSCAN model, checks it for some valuable
    #     attributes we'll need downstream, and spits out the trained
    #     model object as well as a DataFrame with useful data.

    #     Parameters
    #     ----------
    #     filepath: str. Path to pickled HDBSCAN object.

    #     Returns
    #     -------
    #     HdbscanEstimator object with DataFrame
    #     self.soft_cluster_probabilities included, showing the probability of each
    #     datum belonging to each possible cluster in this model's solution.
    #     '''

    #     with open(filepath, 'rb') as f:
    #         model = pickle.load(f)

    #     # Check that model has everything we want to play around with
    #     target_attributes = [
    #         'cluster_persistence_',
    #         'probabilities_',
    #         'outlier_scores_',
    #         'exemplars_'
    #     ]

    #     model_contents = dir(model)

    #     for a in target_attributes:
    #         if a not in model_contents:
    #             raise ValueError(f"We're missing the attribute '{a}'!!")

    #     num_clusters = np.unique(model.labels_).shape[0]  # includes noise
    #     logger.info(f"This HDBSCAN solution has {num_clusters} clusters, "
    #                 "including a noise cluster, if noise is present")

    #     # For backwards compatibility
    #     # Check if model has each method; if not, bind the method to it for use
    #     if 'soft_cluster' not in model_contents or model.soft_cluster is None:
    #         model.soft_cluster = cls.soft_cluster.__get__(model)

    #     if 'num_clusters_' not in model_contents:
    #         model.num_clusters_ = len([l for l in np.unique(model.labels_) if l != -1])

    #     model.soft_cluster()

    #     return model

    @classmethod
    def load_model(
        cls,
        run_id=None,
        model_name=None,
        model_version=None
    ):
        '''
        Loads an mlflow-logged model from its serialized (AKA pickled) form.

        NOTE: this is not expected to work unless you are loading it into an 
        identical environment as the one described by the model's logged 
        conda.yaml file. If this isn't possible, consider loading it via 
        the `mlflow.pyfunc.load_model()` functionality described in the UI 
        page for the model. This will, however, not provide you direct access
        to all of the original model attributes/methods/etc., whereas this
        method will.

        Parameters
        ----------
        run_id : str, optional
            Unique ID of a model training run. May only be used if `model_name`
            and `model_version` are not used, by default None
        model_name : str, optional
            Unique name of a registered model. May only be used if `run_id` is 
            not used, by default None
        model_version : int, optional
            Version number of the registered model identified by `model_name`. 
            May not be used if `run_id` is used, by default None
        model_stage : str, optional
            Model stage of registered model, if there is one. Allowed values 
            can be found in vespid.models.mlflow_tools.MODEL_STAGES, 
            by default None

        Returns
        -------
        HdbscanEstimator object
            The logged model object in its original state as of the logging 
            event.

        Raises
        ------
        ValueError
            Errors out if a run ID is specified at the same time as registered 
            model information
        '''
        # TODO: complete URI generation code in vespid.models.mlflow to handle run_ids, registered model names, etc.
        return load_mlflow_model(run_id=run_id)
        if run_id is not None:
            if model_name is not None or model_version is not None:
                raise ValueError("`run_id` must be the only parameter "
                                 "specified if it is not None")
            model_uri = client.get_run(run_id).info.artifact_uri + '/model'
        elif model_name is not None:
            if run_id is not None:
                raise ValueError("`run_id` and `model_name` cannot both be "
                                 "specified")
            model_uri = client.get_model_version_download_uri(
                model_name,
                model_version
            )

    def _test_soft_clustering(self):

        if isinstance(self.soft_cluster_probabilities, pd.DataFrame):
            membership_vectors = self.soft_cluster_probabilities.values
        else:
            membership_vectors = self.soft_cluster_probabilities

        # Check the membership_vectors shape
        # Should be (num_points, num_clusters)
        vector_shape = membership_vectors.shape
        if -1 in self.labels_:
            num_clusters = len(np.unique(self.labels_)) - \
                1  # account for noise
        else:
            num_clusters = len(np.unique(self.labels_))
        assert len(vector_shape) == 2, f"membership_vectors should be 2-d, got {len(vector_shape)} dimensions"

        assertion_str = f"membership_vectors should be of shape " + \
            f"(num_points, num_clusters), but got {vector_shape}"
        assert vector_shape[1] == num_clusters, assertion_str

        # Each vector between 0.0 and 1.0? Noise can allow less than 1.0 with some fuzziness for rounding error
        assert np.logical_and(membership_vectors.sum(axis=1) < 1.0000001, membership_vectors.sum(axis=1) > 0).sum(
        ) / membership_vectors.shape[0] == 1, f"Some probability vectors sum up to less than 0.0 or more than 1.0"

        df_clusters = pd.DataFrame({
            'label': self.labels_,
            'probability': self.probabilities_,
            'max_soft_probability': membership_vectors.max(axis=1),
            'max_soft_p_label': membership_vectors.argmax(axis=1)
        })

        # Does the max soft cluster probility match the original cluster assignment?
        df_clusters = df_clusters[df_clusters['label'] > -1]

        labels_match_rate = (
            df_clusters['label'] == df_clusters['max_soft_p_label']
        ).sum() / len(df_clusters)
        labels_match_pct = round(labels_match_rate * 100, 2)

        assertion_str = f"Max soft cluster assignment only " + \
            f"matches main algorithm assignment for non-noise points " + \
            f"{labels_match_pct}% of the time"
        if labels_match_rate < 0.9:
            assert labels_match_rate == 1, assertion_str
        else:
            logger.info("Soft-clustering-derived labels and main "
                           f"algorithm labels agree {labels_match_pct}% "
                           "of the time.")
        return labels_match_rate


def build_cluster_pipeline(
    umap_n_components=100,
    umap_n_neighbors=30,
    umap_min_dist=0.0,
    umap_metric='euclidean',
    hdbscan_min_samples=25,
    hdbscan_min_cluster_size=75,
    hdbscan_cluster_selection_method='eom'
):
    '''
    Given a set of hyperparameters,
    builds an un-trained UMAP + HdbscanEstimator pipeline
    that can be trained and then used for other downstream tasks.


    Parameters
    ----------
    umap_n_components: int. Desired dimensionality of the output
        UMAP embedding. Must be smaller than embeddings.shape[1].

    umap_n_neighbors: int. Used in determining the balance
        between local (low values) and global (high values)
        structure in the UMAP reduced-dimensionality embedding.

    umap_min_dist: float in the range [0.0, 0.99]. Controls how tightly 
        UMAP is allowed to pack points together, with low values
        causing "clumpier" embeddings and thus being useful for 
        clustering.

    umap_metric: str. Indicates the distance metric to 
        be used when generating UMAP embeddings.

    hdbscan_min_samples: int. Number of neighbors required by HDBSCAN
        for a point to be considered a "core point". Typically
        higher values lead to more points labeled as noise
        and denser + fewer clusters.

    hdbscan_min_cluster_size: int. Number of points required for
        HDBSCAN to consider a cluster valid.
        
    hdbscan_cluster_selection_method: str. Type of method to use for 
        determining final flat cluster structure. Options are ['eom', 'leaf'].


    Returns
    -------
    sklearn.Pipeline object with a UMAP component 
    (accessed via pipe.named_steps['umap']) and an 
    HdbscanEstimator component (accessed via 
    pipe.named_steps['hdbscan']).
    '''

    um = umap.UMAP(
        n_jobs=1,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        metric=umap_metric,
        n_components=umap_n_components
    )

    hdb = HdbscanEstimator(
        gen_min_span_tree=True,
        prediction_data=False,
        algorithm='boruvka_kdtree',
        core_dist_n_jobs=1,
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
        cluster_selection_method=hdbscan_cluster_selection_method
    )

    return Pipeline(steps=[('umap', um), ('hdbscan', hdb)])
