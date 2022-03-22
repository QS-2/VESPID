from collections.abc import Iterable
import pandas as pd
import numpy as np
import optuna
import mlflow
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import math
from plotly import graph_objects as go
import plotly.express as px

from copy import deepcopy

from vespid import setup_logger
logger = setup_logger(__name__)

class Hyperparameter():    
    def __init__(
        self, 
        name, 
        constant=None,
        min=None, 
        max=None, 
        categories=None
    ):
        '''
        A class for tracking a hyperparameter's allowable range/options for 
        tuning.

        Parameters
        ----------
        name : str
            Hyperparameter name used for reference purposes primarily
        constant : Any, optional
            If set, this is assumed to be a constant value, by default None
        min : float or int, optional
            If set, indicates that the value is continuous and sets the 
            minimum allowable value. If used, ``max`` must also be set, by 
            default None
        max : float or int, optional
            If set, indicates that the value is continuous and sets the 
            maximum allowable value. If used, ``min`` must also be set, by 
            default None
        categories : iterable of Any, optional
            If set, indicates that this is a categorical variable, with 
            levels defined by the values in the iterable provided, 
            by default None

        Raises
        ------
        ValueError
            Checks that the expected value types are used and that min and max
            are set together, if one is set.
        '''
        self.name = name
        
        if constant is not None:
            self.type = 'constant'
            self.constant = constant
        
        elif min is not None and max is not None \
        and isinstance(min, (int, float)) \
        and isinstance(max, (int, float)):
            self.type = 'continuous'
            self.min = min
            self.max = max
            
        elif categories is not None and isinstance(categories, Iterable):
            self.type = 'categorical'
            self.categories = categories
            
        else:
            raise ValueError("You must either provide arguments for both "
                             "`min` and `max` OR provide values for "
                             "`categories` OR provide a single value for "
                             "`constant`.")
            
    def __str__(self):
        output = f"Hyperparameter '{self.name}' "
        if self.type == 'constant':
            output += f"with value {self.constant}"
        elif self.type == 'continuous':
            output += f"with tuning range [{self.min},{self.max}]"
        elif self.type == 'categorical':
            output += f"with category levels {self.categories}"
        return output
        
    def __repr__(self):
        return str(self)

#TODO: add in ability to define metric calculation via callable that takes features, targets (optional), and a trained model maybe?
class Criterion():    
    def __init__(self, name, direction, range=(0,1)):
        '''
        A class for tracking a single optimization criterion 
        (e.g. "minimize RMSE").

        Parameters
        ----------
        name : str
            Name of the metric being used for this criterion
        direction : str
            Indicates how to optimize the criterion. Allowed values are 
            ['minimize', 'maximize']
        range : 2-tuple of float or int
            Indicates the min and max (inclusive) possible values for this 
            criterion. Note that, if a value has no real upper/lower bound, 
            +/- np.inf should be used, by default (0,1)

        Raises
        ------
        ValueError
            Raised if ``direction`` value is invalid
        '''
        allowed_directions = ['minimize', 'maximize'] 
        if direction not in allowed_directions:
            raise ValueError("`direction` must be one of "
                             f"{allowed_directions}, got '{direction}' "
                             "instead")
        
        self.name = name
        self.direction = direction
        self._range_min, self._range_max = range
        
        self._is_unbounded = self._range_min == -np.inf or self._range_max == np.inf
        
    def __str__(self):            
        output = f"Criterion '{self.name}' is set to {self.direction} and " + \
        f"with a possible value range of [{self._range_min, self._range_max}]"
        
        return output
        
    def __repr__(self):
        return str(self)
    
    
class Objectives(object): 
    
    def __init__(
        self, 
        hyperparameters, 
        criteria, 
        objective_function,
        features, 
        targets=None,
        mlflow=(None, None)
    ):
        '''
        A class that defines the hyperparameter bounds (min/max) to be used
        for tuning a model or set of models and the objective function to 
        optimize. Allows for a single object to be created and then provided
        to `optuna` without needing to track separate but related values 
        throughout the `optuna` calls (e.g. the ordered names of metrics
        being optimized as well as the optimization direction).

        Parameters
        ----------
        hyperparameters : iterable of Hyperparameter objects
            The information about hyperparameter tuning ranges, etc.
        criteria : iterable of Criterion objects
            The information about criteria that need to be optimized (e.g. 
            name and optimization direction of metrics)
        objective_function : callable
            Function used to train and evaluate a given hyperparameter 
            permutations
        features : numpy array or pandas DataFrame
            Input features for model training and testing
        targets : numpy array or pandas DataFrame, optional
            If a supervised problem, the targets/labels to used, 
            by default None
        mlflow : 2-tuple of form (Experiment, MlflowClient), optional
            MLFlow objects used for connecting to the MLFlow tracking server 
            and artifact store for model logging, run querying, and model 
            rehydration, by default (None, None)

        Raises
        ------
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        ValueError
            _description_
        '''
        # Arg checking for hyperparams
        names = set()
        for h in hyperparameters:
            if not isinstance(h, Hyperparameter):
                raise ValueError("All values in `hyperparameters` must be "
                                 "of type `Hyperparameter`")
            names.add(h.name)
            
        if len(names) < len(hyperparameters):
            raise ValueError("All names of objects in `hyperparameters` must "
                    
                             "be unique")
            
        # Arg checking for criteria
        names = set()
        possible_minima = []
        possible_maxima = []
        self._has_unbounded_criteria = False
        for c in criteria:
            if not isinstance(c, Criterion):
                raise ValueError("All values in `criteria` must be "
                                 "of type `Criterion`")
            names.add(c.name)
            possible_minima.append(c._range_min)
            possible_maxima.append(c._range_max)
            
            if c._is_unbounded:
                self._has_unbounded_criteria = True
            
        self._range_min = min(possible_minima)
        self._range_max = max(possible_maxima)
        
        if self._range_min > self._range_max:
            raise ValueError("Minimium possible value of all Criterion "
                             "objects cannot be greater than the maximum "
                             "possible value")
            
        elif self._range_min == self._range_max:
            raise ValueError("Minimum possible value of all Criterion objects "
                             "cannot be equal to the maximum possible value "
                             f"({self._range_max})")
        
        if len(names) < len(criteria):
            raise ValueError("All names of objects in `criteria` must "
                             "be unique")
                
        # Arg checking for optimizer
        if not callable(objective_function):
            raise ValueError("`optimizer` must be a function")
                
        self.hyperparameters = hyperparameters
        self._hyperparameters_dict = {h.name:h for h in  hyperparameters}
        self.criteria = criteria
        self._criteria_dict = {c.name:c for c in criteria}
        self._objective_function = objective_function
        self._features = features
        self._targets = targets
        
        if mlflow[0] is None or mlflow[1] is None:
            raise ValueError("Neither `mlflow` element may be None")
        self.experiment, self.client = mlflow
        
        # Track whether tuning has occurred
        self._tuned = False
        
    def get_hyperparameter(self, name):
        '''
        Get a single Hyperparameter object by name

        Parameters
        ----------
        name : str
            Name of the Hyperparameter object

        Returns
        -------
        Hyperparameter object
            The Hyperparameter object with that name.
        '''
        return self._hyperparameters_dict[name]
    
    def get_hyperparameter_names(self, skip_constants=False):
        '''
        Gets all Hyperparameter names, in order.

        Parameters
        ----------
        skip_constants : bool, optional
            If True, Hyperparameters that are not tuned but constant
            are skipped, by default False

        Returns
        -------
        list of str
            Names of Hyperparameters
        '''
        if not skip_constants:
            return [h.name for h in self.hyperparameters]
        else:
            return [h.name for h in self.hyperparameters if h.type != 'constant']
    
    def get_criterion(self, name):
        '''
        Get a single Criterion object by name

        Parameters
        ----------
        name : str
            Name of the Criterion object

        Returns
        -------
        Criterion object
            The Criterion object with that name.
        '''
        return self._criteria_dict[name]
    
    def get_criteria_directions(self):
        '''
        Returns the directions of all criteria

        Returns
        -------
        list of str
            List of values 'maximize' or 'minimize'
        '''
        if isinstance(self.criteria, Iterable):
            return [c.direction for c in self.criteria]
        else:
            return self.criteria.direction
        
    def search_criteria(
        self, 
        range_min=-np.inf, 
        range_max=np.inf, 
        names=None,
        negate=False
    ):
        '''
        Returns only the criteria meeting search filter rules. Useful for 
        identifying which criteria exist in a certain range of possible values.

        Parameters
        ----------
        range_min : float, optional
            Minimum allowable range. Only criteria with a minimum at or above 
            this value will not be returned, by default -np.inf
        range_max : float, optional
            Maximum allowable range. Only criteria with a maximum at or below 
            this value will be returned, by default +np.inf
        names : list of str, optionanl
            Indicates the criteria to be included in the search. If None, all
            will be included.
        negate : bool, optional
            If True, indicates that only criteria that *don't* match the search
            parameters should be returned.

        Returns
        -------
        List of Criterion objects
            The criteria meeting the search parameters. If no matches are 
            found, returns None.
        '''
        if names is None:
            names = self.get_criteria_names()
        
        output = []
        for c in self.criteria:
            if c.name in names:
                if c._range_min >= range_min and c._range_max <= range_max:
                    output.append(c)
        
        if negate:
            old_output = deepcopy(output)
            output = []
            for c in self.criteria:
                if c.name not in [e.name for e in old_output]:
                    output.append(c)
                    
        if len(output) == 0:
            return None
        
        return output
        
    def get_criteria_names(self):
        if isinstance(self.criteria, Iterable):
            return [c.name for c in self.criteria]
        else:
            return self.criteria.name
    
    def __call__(self, trial):
        # Expected form of objective function is objective(
        #     trial,
        #     features,
        #     targets, # optional
        #     criteria,
        #     mlflow_experiment, # optional
        #     mlflow_client, # optional but must come with mlflow_experiment
        #     **Hyperparameter_kwargs
        # )
        self._tuned = True
        
        if self._targets is None:
            if self.experiment is None:
                return self._objective_function(
                    trial,
                    self._features,
                    self.get_criteria_names(),
                    **self._hyperparameters_dict
                )
            else:
                return self._objective_function(
                    trial,
                    self._features,
                    self.get_criteria_names(),
                    self.experiment,
                    self.client,
                    **self._hyperparameters_dict
                )
        else:
            if self.experiment is None:
                return self._objective_function(
                    trial,
                    self._features,
                    self._targets,
                    self.get_criteria_names(),
                    **self._hyperparameters_dict
                )
            else:
                return self._objective_function(
                    trial,
                    self._features,
                    self._targets,
                    self.get_criteria_names(),
                    self.experiment,
                    self.client,
                    **self._hyperparameters_dict
                )
                
    def find_best_multiobj_solution(
        self,
        study=None,
        criteria_subset=None,
        scale_all=True,
        return_type='best_hyperparameters'
    ):
        '''
        Given the results of a multi-objective optuna Bayesian
        hyperparameter tuning study, find the best possible
        hyperparameter set by finding the solution most optimized
        for our purposes.

        Note that this assumes your objectives that you tuned to
        are ["dbcv", "num_clusters", "mean_cluster_persistence", "std_cluster_persistence"],
        in that order. This transforms the results of these metrics such that
        1.0 is representative of a perfect run for each and then
        measures the Euclidean distance between the best runs and
        the vector [1,1,1,1] to find the solution closest to it.


        Parameters
        ----------
        study: pre-run optuna Study object. If None, assumes MLFlow experiment 
            has the information.
            
        criteria_subset: list of str. Names of Criterion objects to include in 
            the analysis. If None, uses them all.
            
        scale_all: bool. If True, indicates that every Criterion should be 
            min-max-scaled when calculating how close to optimal the trial 
            values got. This has the advantage of essentially leveling the 
            playing field for all metrics (e.g. a metric that has a max value 
            of 58 and one that has a max value of 0.25 will both scale to 1.0),
            but only works well when comparing trials that are *part of the 
            same experiment.* 
            
            When trying to compare across experiments, using an 
            absolute scale is preferred, so this should be set to False and 
            `criteria_subset` should be used to identify the Criterion objects 
            that are naturally on a [0,1] scale.
            
            If set to False, only Criterion objects with a range other than 
            [0,1] will be min-max-scaled.
        
        return_type: str. Can be one of 
            ['best_hyperparameters', 'all_best_trials', 'full_best_trial']. 
            Indicates what you want returned.

            best_hyperparameters: returns a dict of the form 
                {'hyperparameter_name': value}. Useful for automated selection 
                and training of the final model.

            full_best_trial: returns a pandas Series with all reported values
                from optuna corresponding to the best trial, not just its
                hyperparameters. Useful for manual inspection of the solution.

            all_best_trials: returns a pandas DataFrame containing information
                derived in this function, including scaled hyperparameters
                and the distance AKA "goodness score" (smaller is better)
                of each one. Useful for determining if the optuna optimization 
                strategy seems to be working as intended.


        Returns
        -------
        See discussion of `return_type` parameter above.
        '''
        if study is not None:
            df_results = optuna_best_trials_to_dataframe(
                study, 
                self.get_criteria_names()
            )
            output = df_results[self.get_criteria_names()].copy()
            
        # MLFlow approach
        else:
            # Grab all trials/runs
            df_results = mlflow.search_runs(
                self.experiment.experiment_id, 
                filter_string='', 
                run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
            )
            
            # Synchronize with criteria names
            if criteria_subset is None:
                criteria_columns = self.get_criteria_names()
                criteria_to_use = self.criteria
            else:
                criteria_columns = criteria_subset
                criteria_to_use = self.search_criteria(names=criteria_subset)
                for c in criteria_columns:
                    if c not in self.get_criteria_names():
                        raise ValueError(f"Criterion '{c}' not a valid "
                                         "Criterion")
                
            column_mapping = {
                f"metrics.{name}":name for name in criteria_columns
            }
            df_results = df_results[df_results['status'] == 'FINISHED']\
                .rename(columns=column_mapping)
            
            # Grab only the criteria columns
            output = df_results[criteria_columns].copy()
        
        
        # Track which metrics must be scaled and/or have $1 - metric$ applied
        columns_to_scale = {}
        columns_to_track = []
        minimizing_columns = {}
        for c in criteria_to_use:       
            columns_to_track.append(f"{c.name}_scaled")
            
            if not scale_all:
                if c._is_unbounded:
                    columns_to_scale[columns_to_track[-1]] = c.name
                
                # No need to scale it at all, so make sure we don't record the 
                # scaled name!
                else:
                    columns_to_track[-1] = c.name
                
                #FIXME: something about how we're tracking the minimizing columns makes it so they don't get _scaled on their names here. 
                
                # Make sure we properly label based on the fact that scaling is
                # occurring
                if c.direction == 'minimize' and c._is_unbounded:
                    minimizing_columns[columns_to_track[-1]] = columns_to_track[-1]
                elif c.direction == 'minimize' and not c._is_unbounded:
                    minimizing_columns[columns_to_track[-1]] = c.name
                
            else:
                columns_to_scale[columns_to_track[-1]] = c.name
                if c.direction == 'minimize':
                    minimizing_columns[columns_to_track[-1]] = columns_to_track[-1]
                    
        logger.debug(f"{columns_to_track=}")
        logger.debug(f"{columns_to_scale=}")
        logger.debug(f"{minimizing_columns=}")
        
        # Ensure that all metrics are on [0,1] scale 
        if len(columns_to_scale) > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            output[list(columns_to_scale.keys())] = scaler.fit_transform(output[list(columns_to_scale.values())])
            
        if len(minimizing_columns) > 0:
            output[list(minimizing_columns.keys())] = 1 - output[list(minimizing_columns.values())]
            
        output['distance'] = euclidean_distances(
            output[columns_to_track],
            np.ones((1, len(columns_to_track)))
        )
        best_index = np.argmin(output['distance'].values)

        if return_type == 'all_best_trials':
            return output.sort_values('distance', ascending=True)

        elif return_type == 'full_best_trial':
            return df_results.loc[output.iloc[best_index].name]

        elif return_type == 'best_hyperparameters':
            if study is not None:
                return study.best_trials[best_index].params
            
            #MLFlow option
            else:
                # Pare down to just hyperparameters with synchronized names
                column_mapping = {
                    f"params.{h}":h for h in self.get_hyperparameter_names(
                        skip_constants=True
                    )
                }
                df_results.rename(columns=column_mapping, inplace=True)
                result = df_results.iloc[best_index][column_mapping.values()]
                
                # MLFlow returns values as strings even when numeric weirdly
                def _str_to_num(h):
                    try:
                        return pd.to_numeric(h)
                    except ValueError:
                        return h
                return result.apply(_str_to_num).to_dict()

        else:
            raise ValueError(
                f"``return_type`` value of '{return_type}' is not supported")
            
    def analyze_experiment(self):
        '''
        Analyzes a set of hyperparameter tuning trials in an MLFlow experiment
        to show how well the underlying optuna algorithm learned to improve on
        our criteria.

        Returns
        -------
        pandas DataFrame
            All of the MLFlow trial data used to generate the figures/numbers.
        '''
        experiment_id = self.experiment.experiment_id
        
        logger.info("Fetching experiment data...")
        data = mlflow.search_runs(
            experiment_id, 
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY
        ).sort_values('start_time', ascending=True)
        
        num_trials_total = len(data)
        num_trials_complete = len(data[data['status'] == 'FINISHED'])    
        data['duration'] = data['end_time'] - data['start_time']
        
        data = data[data['status'] == 'FINISHED']
        avg_trial_time = data['duration'].mean()
        
        logger.info(f"Experiment data fetch complete, found {num_trials_total} "
                    f"total trials, {num_trials_complete} of which completed. "
                    "Of those that completed, the average time per trial was "
                    f"{avg_trial_time}")
        
        # Visually inspect learning rate
        for criterion in self.criteria:
            fig = px.line(data_frame=data.reset_index(drop=True), x='start_time', y=f'metrics.{criterion.name}', title=criterion.name)
            fig.show()
            
        # Do our 1-vector distance calculation and use linear regression coeff to determine learning rate
        data['error'] = self.find_best_multiobj_solution(
            return_type='all_best_trials',
            scale_all=False
        )['distance']

        plotting_data = data.reset_index(drop=True)

        # Do a simple linear fit
        linreg = LinearRegression()
        linreg.fit(plotting_data.index.values.reshape(-1, 1), plotting_data['error'])
        optimal_trial = math.ceil(-linreg.intercept_/linreg.coef_[0]) # when the error value is 0.0

        # Add the trend line predictions to the learning error visualization
        fig = px.line(data_frame=plotting_data, x=plotting_data.index, y='error', title=f"Error AKA Distance to Optimum (convergence trial = {optimal_trial:,})")
        fig.add_trace(go.Scatter(x=plotting_data.index, y=linreg.predict(plotting_data.index.values.reshape(-1, 1)), mode='lines'))
        fig.show()
        
        return data
            

def optuna_best_trials_to_dataframe(study, objective_names=None):
    '''
    Loads up a completed optuna Study object and provides
    a pandas DataFrame containing only the best trials
    based on a multi-objective optimization.


    Parameters
    ----------
    TBD.


    Returns
    -------
    TBD.
    '''
    if len(study.directions) <= 1:
        raise ValueError("``study`` does not appear to have been tuned "
                         "for multiple objectives. Given this, just run `study.best_trial`.")

    if not isinstance(study, optuna.study.study.Study):
        raise ValueError("``study`` must be an optuna Study object, "
                         f"but detected type is {type(study)}")

    best_trial_indices = [t.number for t in study.best_trials]
    df_results = study.trials_dataframe().loc[best_trial_indices]

    if objective_names is not None:
        df_results.rename(columns={
            f'values_{i}': f'{c}' for i, c in enumerate(objective_names)
        }, inplace=True)

    return df_results
