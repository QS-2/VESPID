# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from time import time
from joblib import dump as dump_obj, load as load_obj
from copy import copy, deepcopy
import os
import ray
import pathlib

from vespid import get_current_datetime
from vespid import setup_logger, get_memory_usage

logger = setup_logger(__name__)


class Stage():
    '''
    Class that defines an instance of a stage of the pipeline. Note that,
    if you want to use the output of stage N in any stage other than 
    N+1, you need to instantiate the Stage outside of a Pipeline so that
    it may be referenced more than when it is first run in the Pipeline.

    When a Stage object is called directly it will return the output of
    the stage. Example: 

        s = Stage(...)
        s.execute(...)
        s()
        #### Returns s.output, if available
    '''

    def __init__(
            self,
            name,
            function,
            cache_stage_output=False,
            **kwargs
        ):
        '''
        Parameters
        ----------
        name: str. Allows for name-based access to this Stage when it is used
            in a Pipeline.

        function: python function. Defines the processing to be done with
            the input(s) and what should be output.

        cache_stage_output: bool. If True, indicates that you want the output of 
            the stage in question to be kept in memory such that it can be 
            accessed by later stages (but not necessarily only the very 
            next stage). Use the `other_inputs` key to provide other 
            stages' outputs saved to memory using `cache_stage` as input 
            into the relevant stage(s).

        kwargs: keyword arguments that get passed to ``function``. Note that
            these should be of the form `<keyword>=(<value>, <is_stage_name>)`,
            wherein <is_stage_name> is a boolean that indicates if <value>
            should be a string that corresponds to another Stage's name 
            attribute (commonly done when a Stage is inside a Pipeline object 
            to reference the output value of that Stage) or if it is simply
            a standard string.
        '''

        self.name = name
        self._function = function
        self._cache_stage_output = cache_stage_output
        
        # Error-check the 2-tuples of kwargs
        for k, (v, is_stage) in kwargs.items():
            # Check all kwargs to ensure that, if they are a Stage name,
            # the value is a string
            if is_stage and not isinstance(v, str):
                raise ValueError("kwargs that refer to other Stages "
                                    "must be names of type str. "
                                    f"Received {type(v)} for kwarg '{k}' "
                                    "instead")
                
        self._kwargs = kwargs

        self._output = None
        self._execution_time = None
        self._initialization_time = get_current_datetime()
        
        self._memory_percentage_used_start = None
        self._memory_percentage_used_end = None    
    
    def is_output_cached(self):
        '''
        Determines if a Stage's output is being/will be stored in memory.

        Returns
        -------
        bool
            True if the output is/will be cached, False otherwise.
        '''
        return self._cache_stage_output

    def execute(self, input=None):
        '''
        Execute the Stage's assigned function given the input provided 
        either as an arg or the kwargs of the Stage itself.
        
        
        Parameters
        ----------
        input : Any
            First argument to be given to `function` in addition to the kwargs.
            Only kwargs are provided if this is None.

        Returns
        -------
        Output type of self._function
            Output of the Stage's assigned function, given its input and/or 
            the Stage's kwargs
        '''
        kwargs = {}
        start_time = time()

        # If any of our kwargs are Stage objects,
        # explicitly use their output in their place
        
        # This is a workaround to ensure we can pass previous Stages
        # to later Stages at instantiation time without throwing an error
        self._memory_percentage_used_start = get_memory_usage(logger)
        for k, (v, is_stage) in self._kwargs.items():
            if isinstance(v, type(self)) and is_stage:
                kwargs[k] = v._output
                
            elif not isinstance(v, type(self)) and is_stage:
                raise ValueError(f"kwarg '{k} expected to be of type Stage, "
                                 f"but got {type(v)} instead")

            else:
                kwargs[k] = v

        if input is not None:
            output = self._function(input, **kwargs)
        else:
            output = self._function(**kwargs)

        if self._cache_stage_output:
            self._output = output
            
        self._execution_time = timedelta(seconds=time() - start_time)
        logger.info(f"Stage {self.name} took {self._execution_time} to execute")
        self._memory_percentage_used_end = get_memory_usage(logger)
        self._executed = True

        return output


    def get_results(self):
        '''
        Returns the output of the Stage, assuming execution has already
        occurred.
        '''
        if self._output is not None:
            return self._output
        
        elif not self._executed and self._cache_stage_output:
            raise RuntimeError("This Stage has no output because it has not "
                               "yet been run. Please run Stage.execute()")

        elif not self._cache_stage_output:
            raise RuntimeError("This Stage has no output because "
                               "cache_stage_output is False")
            
        #FIXME: see if we can do better than this last case
        else:
            raise RuntimeError("This Stage has no output but the reason why"
                               "is unclear")


    def __str__(self):
        d = {**self.__dict__}
        return f"Stage called {self.name} with inputs {str(d)}"

    # This variant throws errors when a Stage object is part of another iterable (e.g. list or dict)
    #def __repr__(self):
        #return str(self)
        
    def use_preceding_input(self):
        '''
        Flags the Stage as being one expecting the output of the immediately
        preceding Stage in a Pipeline as its input.

        Returns
        -------
        PipelineFlag
            PipelineFlag with the Stage stored for providing info to the 
            Pipeline about its execution.
        '''
        return PipelineFlag('use_preceding_input', stage=self)
        
    @classmethod
    def load_stage_results_from_disk(cls, filename):
        logger.info(f"Loading Stage from disk: `{filename}`... ")
        stage_result = load_obj(filename)
        return stage_result
    
class PipelineFlag():
    '''
    A class designed specifically to just flag to Pipelines that something
    is being set that they should pay attention to, e.g. a Stage that has 
    been flagged as needing the Pipeline to provide the output of the preceding
    Stage for it as input.
    '''
    def __init__(self, type, value=None, stage=None):
        '''
        Initializes the flag with basic info.

        Parameters
        ----------
        type : str
            Indicates the type of flag being used. Can be one of the 
            following:
            
            * 'use_preceding_input': tells Pipeline to cache the output of 
                the previous Stage to use as the first input to this Stage. 
                Can be used to save memory during a big Pipeline run.
            * 'cancel_pipeline': tells Pipeline to stop executing and return
                `value` as the result of `Pipeline.run()`
        value : Any, optional
            Indicates what value to associate with this flag, default None
        stage : Stage, optional
            If not None, provides a Stage to be used by the Pipeline, augmented
            with the information provided by the flag, default None
        '''
        allowed_types = ['use_preceding_input', 'cancel_pipeline']
        
        if type not in allowed_types:
            raise ValueError(f"`type` of {type} not supported")
        
        self.type = type
        self.value = value
        self.stage = stage
        
    @property
    def name(self):
        if self.stage is not None:
            return self.stage.name
        else:
            return None
        
    def __str__(self):
        output = f"Flag of type '{self.type}'"
        if self.value is not None:
            output += f" having value {self.value}"
        if self.stage is not None:
            output += f" referring to a Stage called '{self.stage.name}'"
        return output
    
    def __repr__(self):
        return str(self)

class Pipeline():
    '''
    Class for creating linear data pipelines using arbitrary inputs and 
    functions for each step/stage.
    '''

    def __init__(
            self,
            stages,
            save_stages=False,
            **first_stage_kwargs,
        ):
        '''
        Parameters
        ----------
        stages: list of Stage and possibly PipelineFlag objects.
            
            NOTE: when instantiated, the Pipeline will make a deep copy of
            ``stages`` and every Stage in it. As such, please ensure that no 
            large objects (e.g. a large array) are being included as kwargs
            to any Stage, as they will be copied in memory.

        save_stages: bool indicating if a pickled and compressed form of the 
            data output from each stage of the Pipeline is saved. If True, 
            compressed *.joblib files are saved to the working directory
            in a new subdirectory named PipelineStages_<current_datetime>.
            
        first_stage_kwargs: if provided, these kwargs will be used to overwrite
            the identified kwargs of stages[0]. Any kwargs defined in the Stage
            that are not provided here will be left as they are. This is useful
            for providing a different input to copies of the Pipeline without
            re-defining the starting Stage each time (e.g. when parallelizing
            the Pipeline).
        
            Should be 2-tuples provided in the form 
            `<stages[0]_function_kwarg>=(<value>, <is_stage>)`, with the first
            tuple element being the actual value to pass as the kwarg and the 
            second element indicating if the value is actually a string name
            of another Stage in ``stages``, the output of which should be used
            as the value for the kwarg in question.
            
            This format allows the Pipeline to setup inter-Stage dependencies
            when instantiated but before running it.
        '''

        #TODO: do initial inputs checking and error-raising
        #TODO: refactor to stop requiring 2-tuples to indicate if an earlier Stage is being referenced in `stages`
        
        self.stages = {}
        for i,s in enumerate(stages):
            if isinstance(s, Stage):
                self.stages[s.name] = copy(s)
            elif isinstance(s, PipelineFlag) and i == 0:
                raise RuntimeError("PipelineFlag cannot be the first stage")
            elif s.name in self.stages.keys():
                raise ValueError(f"Stage name '{s.name}' used more than once")
            elif isinstance(s, PipelineFlag):
                self.stages[s.stage.name] = copy(s)
            else:
                raise RuntimeError(f"Stage is of unknown type '{type(s)}'")

        self._executed = False
        self._first_stage_kwargs = deepcopy(first_stage_kwargs)
        
        
        # Parse the Stage kwargs, checking for inter-Stage references
        # and replace kwarg values with Stage object from stages where relevant
        self._insert_stage_kwargs()

        self.save_stage_output = save_stages
        self._build_time = time()
        self._build_time_str = get_current_datetime()

        if save_stages:
            self.cache_filepath = f"pipeline_{self._build_time_str}/"
            if not os.path.isdir(self.cache_filepath):
                pathlib.Path(self.cache_filepath).mkdir(parents=True, exist_ok=True)

        else:
            self.cache_filepath = None

    def _insert_stage_kwargs(self):
        '''
        Parses the kwargs provided for Stages (either when the Stages were
        constructed or when the Pipeline was constructed) and 
        inserts/overwrites the kwargs for the first Stage with the values from
        ``first_stage_kwargs``. Then, for any kwarg identified as being 
        a reference to another Stage in the Pipeline, replaces the string
        name identifier of the other Stage with the actual Stage object.

        Raises
        ------
        RuntimeError
            Checks if the Stage being referenced as a kwarg exists in the
            Pipeline's ``stages``. Also checks that the Stage being referenced
            is caching its results in memory.
        '''
        # Parse the Stages
        for i, (_, stage) in enumerate(self.stages.items()): 
            if isinstance(stage, PipelineFlag):
                flag_type = stage.type
                stage = stage.stage
               
            # Parse the kwargs for each Stage        
            for k, (v, is_stage) in deepcopy(stage._kwargs).items():
                # Replace any relevant kwargs in first Stage with what was given 
                # to Pipeline as first_stage_kwargs
                if i == 0:
                    for k,v in self._first_stage_kwargs.items():
                        stage._kwargs[k] = v
                
                # Check that a Stage used as kwarg is actually caching itself 
                # in memory
                if is_stage and v not in self.stages.keys():
                    raise RuntimeError(f"kwarg {k} refers to a Stage that was "
                                    "not provided in ``stages``")
                    
                elif is_stage and not self.stages[v].is_output_cached():
                    raise RuntimeError(f"kwarg {k} uses a Stage ('{v}') that "
                                    "is not being cached in memory when "
                                    "executed. Please re-instantiate the Stage "
                                    "with `cache_output_stages=True`")
                    
                # Grab the Stage needed
                elif is_stage:
                    stage._kwargs[k] = (self.stages[v], is_stage)

    def __str__(self):
        '''
        Print the schema of this Pipeline (source->sink relationships
        and transformations along the way).


        Parameters
        ----------
        None.


        Returns
        -------
        Nothing, prints to stdout.
        '''

        output = ' -> '.join(self.stages.keys())
        if self._executed:
            output += f"\nPipeline took {self.execution_time} to execute fully."

        #print(self.input)
        return output


    def __repr__(self):
        return str(self)

    
    def run(self, verbose=False, return_performance_report=False):
        '''
        Runs the Pipeline.


        Parameters
        ----------
        verbose: bool. If True, elevates status updates for each stage
            from INFO to DEBUG logger level.
            
        return_performance_report: bool. If True, returns a DataFrame 
            reporting how long each Stage took, memory at start and end of 
            Stage, etc.


        Returns
        -------
        If return_performance_report is True, returns pandas DataFrame 
        produced by self.track_stages(). Otherwise returns the output of the 
        final Stage of the Pipeline.
        '''

        reporter = logger.debug if verbose else logger.info
        proceed = False

        if self._executed:
            raise RuntimeError(f"This pipeline has already been run previously.")

        execution_start_time = time()
        for i, (stage_name, stage) in enumerate(self.stages.items()):
            reporter(f"Starting stage {stage_name}...")
            start_time = time()
            
            if isinstance(stage, Stage):
                data_in = stage.execute()
                
            elif isinstance(stage, PipelineFlag) \
                and stage.type == 'use_preceding_input':
                data_in = stage.stage.execute(data_in)

            if self.save_stage_output and i != len(self.stages) - 1:
                dump_obj(data_in, 
                    f"{self.cache_filepath}{stage.name}.joblib",
                    compress=('gzip', 3))

            self._executed = True
            
            # Check if Pipeline cancel signal received and break out of loop if so
            if isinstance(data_in, PipelineFlag) \
                and data_in.type == 'cancel_pipeline':
                # Extract the real return value from the PipelineFlag object
                data_in = data_in.value
                logger.warning("Received a cancellation signal from Stage "
                               f"{stage.name}")
                break

        self.execution_time = timedelta(seconds=time() - execution_start_time)
        reporter(f"Pipeline took {self.execution_time} to execute.")
        
        if return_performance_report:
            return self.track_stages()
        else:
            return data_in
    
    def track_stages(self):
        '''
        Provides metadata about Stages executed.

        Returns
        -------
        pandas DataFrame
            Log of executed Stages. Note that "absolute_percent_memory_change" 
            column is calculated by subtracting percent of memory used at 
            start of Stage excecution from the percent used at the end 
            (e.g. 10% start -> 11% end = 1%).
        '''
        execution_times = []
        build_times = []
        memory_used_start = []
        memory_used_end = []
        for stage in self.stages.values():
            execution_times.append(stage._execution_time)
            build_times.append(stage._initialization_time)
            memory_used_start.append(stage._memory_percentage_used_start),
            memory_used_end.append(stage._memory_percentage_used_end)
            
        output = pd.DataFrame({
            'stage': self.stages.keys(),
            'built_on': build_times,
            'time_to_execute': execution_times,
            'percent_memory_used_start': memory_used_start,
            'percent_memory_used_end': memory_used_end
        })
        output['absolute_percent_memory_change'] = \
            output['percent_memory_used_end'] \
                - output['percent_memory_used_start']
        
        return output


    def load_stage_output(self, stage_name, from_memory=True):
        '''
        Given the name of a stage in the already-executed pipeline, load up
        the cached stage file and return the resulting Python object.


        Parameters
        ----------
        stage_name: str. Name used for the stage in question.

        from_memory: bool. If True, loads the Stage output from a cached version
            of the Stage, instead of trying to load it into memory from disk.


        Returns
        -------
        Object that was generated as the output of the named stage (often
        a DataFrame).
        '''

        if stage_name not in self.stages.keys():
            raise ValueError(f"{stage_name} not a stage from this Pipeline")

        elif from_memory:
            stage_index = self.stages.keys().index(stage_name)
            output = self.stages[stage_index].get_results()

        elif not self.save_stage_output:
            raise ValueError("This Pipeline did not save its stages")

        elif not self._executed:
            raise RuntimeError("This Pipeline has not yet been executed. \
Please use the run() method to execute so that saved stages may be \
inspected")

        else:
            with open(f"{self.cache_filepath}{stage_name}.joblib", 'rb') as f:
                output = load_obj(f)

        return output
    
    @classmethod
    def cancel(cls, return_value=None):
        
        return PipelineFlag('cancel_pipeline', value=return_value)


    def save(self, filepath=None):
        '''
        Archives a copy of the Pipeline so it can be used later/shared. 
        Loading the saved Pipeline can be achieved via:

        from vespid.data import load_pipeline
        load_pipeline(filepath)


        Parameters
        ----------
        filepath: str indicating the destination to which the Pipeline should
            be saved. Should be of the format 'path/to/pipeline.joblib'. If 
            None, will save to the directory used for saving Stages, if available, 
            else will save to the current working directory.


        Returns
        -------
        Nothing.
        '''
        if not filepath and self.cache_filepath is not None:
            output_path = self.cache_filepath + f'Pipeline.joblib'

        elif filepath:
            output_path = filepath

        else:
            output_path = f'Pipeline_{self._build_time}.joblib'

        dump_obj(self, output_path, compress=False)


    @classmethod
    def load(cls, filepath):
        '''
        Given the location of a Pipeline saved on disk, loads it into memory for 
        use.


        Parameters
        ----------
        filepath: str indicating the destination from which the Pipeline should
                be loaded. Should be of the format 'path/to/pipeline.joblib'.


        Returns
        -------
        Pipeline object.
        '''

        return load_obj(filepath)


@ray.remote
class ParallelPipeline(Pipeline):
    '''
    Class for creating linear data pipelines using arbitrary inputs and 
    functions for each step/stage. This class is designed to be identical
    to the Pipeline class, but with ray-enhanced parallelization.
    
    Note that the constructor should not be called via ParallelPipeline(args), 
    but rather via ParallelPipeline.remote(args). Likewise, methods should
    be called via parallel_pipe.method.remote(method_args).
    '''