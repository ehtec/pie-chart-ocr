import copy

from .data_helpers import test_data_percentages, get_upscaled_steph_test_path, load_annotations_from_csv
from .helperfunctions import get_root_path
import os
import json
import logging
from .multiprocess_ocr import METRICS_FILENAME


# matching precision when comparing annotations
MATCHING_PRECISION = 5


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

    res_tuples = load_annotations_from_csv(csvpath, load_reversed=True)

    return res_tuples


# check if annotations match exactly
def check_simple_annotations_match(annotations1, annotations2, ignorecase=True):

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a), MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a), MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    return set(annotations1_copy) == set(annotations2_copy)


# check if annotations match exactly apart from one element
def check_simple_annotations_match_but_one(annotations1, annotations2, ignorecase=True):

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a), MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a), MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    diff1 = set(annotations1_copy) - set(annotations2_copy)
    diff2 = set(annotations2_copy) - set(annotations1_copy)

    return (len(diff1) <= 1) and (len(diff2) <= 1)


# compute all metrics
def compute_metrics(test_metrics=None, filename=METRICS_FILENAME, interactive=False):

    if test_metrics is None:
        test_metrics = load_test_metrics_json(filename=filename)

    total_metrics_count = len(test_metrics)

    metric_functions = [
        check_simple_annotations_match,
        check_simple_annotations_match_but_one
    ]

    res_dict = {func.__name__: [] for func in metric_functions}

    for n_str, data in test_metrics.items():

        n = int(n_str)

        annotations1 = load_test_annotations(n)

        if 'res' not in data.keys():
            logging.warning("res key not found")
            continue

        annotations2 = data['res']

        logging.info("annotations1: {0}".format(annotations1))
        logging.info("annotations2: {0}".format(annotations2))

        for func in metric_functions:
            res = func(annotations1, annotations2)
            logging.info("{0}: {1}".format(func.__name__, res))
            res_dict[func.__name__].append(res)

        if interactive:
            input()

    final_dict = {}

    for func in metric_functions:

        if func.__name__.startswith("check"):
            true_values_count = sum(res_dict[func.__name__])
            true_ratio = true_values_count / total_metrics_count
            final_dict.update({func.__name__: true_ratio})

    logging.info("final_dict: {0}".format(final_dict))

    return final_dict