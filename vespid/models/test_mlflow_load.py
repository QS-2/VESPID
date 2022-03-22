from vespid.models.mlflow_tools import load_model, setup_mlflow
from vespid import setup_logger

logger = setup_logger()

import mlflow

experiment = setup_mlflow('SCORE_tuning_test_10Job_2000Trials_v2 - 2011')
best_runs = mlflow.search_runs(search_all_experiments=True, filter_string="tags.best_model='true'")
for idx, row in best_runs.iterrows():
    logger.debug(row)
    thing = row['run_id']
    date = row['end_time']
    try:
        model = load_model(run_id=thing)
        logger.debug('date: `%s` -> %s' % (date, str(model)))
        logger.info("loaded run id: `%s` end date `%s`" % (date, thing))
    except (RuntimeError, AttributeError) as e:
        logger.error("e: `%s` run id: `%s` end date `%s`" % (e, date, thing))
