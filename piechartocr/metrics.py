from .data_helpers import test_data_percentages, get_upscaled_steph_test_path, load_annotations_from_csv
from .helperfunctions import get_root_path
import os
import json
import logging
from .multiprocess_ocr import METRICS_FILENAME


# load test metrics from the JSON file
def load_test_metrics_json(filename=METRICS_FILENAME):

    path = os.path.join(get_root_path(), 'artifacts', filename)
    logging.debug("Path to JSON test metrics: {0}".format(path))

    with open(path, 'r') as jsonfile:
        test_metrics = json.load(jsonfile)

    logging.debug("test_metrics: {0}".format(test_metrics))

    return test_metrics


# compare loaded test metrics to existing data files
def compare_test_metrics(error_on_diff=True, error_on_miss=True, test_metrics=None, filename=METRICS_FILENAME):

    if test_metrics is None:
        test_metrics = load_test_metrics_json(filename=filename)

    n_list = test_data_percentages()
    n_list = [str(n) for n in n_list]

    missing_metrics = [el for el in n_list if el not in test_metrics.keys()]
    logging.info("missing_metrics: {0}".format(missing_metrics))

    additional_metrics = [el for el in test_metrics.keys() if el not in n_list]
    logging.info("additional_metrics: {0}".format(additional_metrics))

    if error_on_diff:
        if bool(missing_metrics) or bool(additional_metrics):
            raise ValueError("Metrics do not match")

    elif error_on_miss:
        if bool(missing_metrics):
            raise ValueError("Some metrics are missing")

    return missing_metrics, additional_metrics


# load annotations from test dataset
def load_test_annotations(n):

    csvpath, _ = get_upscaled_steph_test_path(n, existence_check=True)

    res_tuples = load_annotations_from_csv(csvpath)

    return res_tuples
