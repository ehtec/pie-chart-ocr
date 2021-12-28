from .data_helpers import test_data_percentages
from .helperfunctions import get_root_path
import os
import json
import logging


# load test metrics from the JSON file
def load_test_metrics_json():

    path = os.path.join(get_root_path(), 'artifacts', 'ocr_test_metrics.json')
    logging.debug("Path to JSON test metrics: {0}".format(path))

    with open(path, 'r') as jsonfile:
        test_metrics = json.load(jsonfile)

    logging.debug("test_metrics: {0}".format(test_metrics))

    return test_metrics
