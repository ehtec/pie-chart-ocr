import copy
from fuzzywuzzy import fuzz
from .data_helpers import test_data_percentages, get_upscaled_steph_test_path, load_annotations_from_csv
from .helperfunctions import get_root_path
import os
import json
import logging
from .multiprocess_ocr import METRICS_FILENAME
import matplotlib.pyplot as plt


# matching precision when comparing annotations
MATCHING_PRECISION = 5

# default fuzzywuzzy minimum score
DEFAULT_FUZZ_MIN_SCORE = 90


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
def check_simple_annotations_match(annotations1, data, ignorecase=True):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    return set(annotations1_copy) == set(annotations2_copy)


# check if annotations match with fuzz ratio
def check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=True, min_score=DEFAULT_FUZZ_MIN_SCORE,
                                       mode='ratio'):

    if mode == 'ratio':
        score_func = fuzz.ratio

    elif mode == 'partial_ratio':
        score_func = fuzz.partial_ratio

    elif mode == 'partial_ratio_reversed':
        score_func = lambda a, b: fuzz.partial_ratio(b, a)

    else:
        raise NotImplementedError("Mode {0} not implemented".format(mode))

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    if len(annotations1) != len(annotations2):
        return False

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    correct_count = 0

    for a, b in annotations1_copy:

        valid = [el[1] for el in annotations2_copy if el[0] == a]

        if not bool(valid):
            continue

        scores = [score_func(b, el) for el in valid]

        max_score = max(scores)

        if max_score >= min_score:
            correct_count += 1

    return correct_count == len(annotations1)


# wrapper for check_fuzz_ratio_annotations_match with min_score=90
def check_fuzz_ratio_annotations_match_90(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=90, mode='ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=80
def check_fuzz_ratio_annotations_match_80(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=80, mode='ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=70
def check_fuzz_ratio_annotations_match_70(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=70, mode='ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=90 and mode='partial_ratio'
def check_fuzz_partial_ratio_annotations_match_90(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=90,
                                              mode='partial_ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=80 and mode='partial_ratio'
def check_fuzz_partial_ratio_annotations_match_80(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=80,
                                              mode='partial_ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=70 and mode='partial_ratio'
def check_fuzz_partial_ratio_annotations_match_70(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=70,
                                              mode='partial_ratio')


# wrapper for check_fuzz_ratio_annotations_match with min_score=90 and mode='partial_ratio_reversed'
def check_fuzz_partial_ratio_reversed_annotations_match_90(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=90,
                                              mode='partial_ratio_reversed')


# wrapper for check_fuzz_ratio_annotations_match with min_score=80 and mode='partial_ratio'
def check_fuzz_partial_ratio_reversed_annotations_match_80(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=80,
                                              mode='partial_ratio_reversed')


# wrapper for check_fuzz_ratio_annotations_match with min_score=70 and mode='partial_ratio'
def check_fuzz_partial_ratio_reversed_annotations_match_70(annotations1, data, ignorecase=True):

    return check_fuzz_ratio_annotations_match(annotations1, data, ignorecase=ignorecase, min_score=70,
                                              mode='partial_ratio_reversed')


# check if annotations match exactly apart from one element
def check_simple_annotations_match_but_one(annotations1, data, ignorecase=True):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    diff1 = set(annotations1_copy) - set(annotations2_copy)
    diff2 = set(annotations2_copy) - set(annotations1_copy)

    return (len(diff1) <= 1) and (len(diff2) <= 1)


# check percent numbers match exactly
def check_percent_numbers_match(annotations1, data):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    percent_numbers_1 = [el[0] for el in annotations1_copy]
    percent_numbers_2 = [el[0] for el in annotations2_copy]

    return set(percent_numbers_1) == set(percent_numbers_2)


# check percent numbers match exactly but one
def check_percent_numbers_match_but_one(annotations1, data):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    percent_numbers_1 = [el[0] for el in annotations1_copy]
    percent_numbers_2 = [el[0] for el in annotations2_copy]

    diff1 = set(percent_numbers_1) - set(percent_numbers_2)
    diff2 = set(percent_numbers_2) - set(percent_numbers_1)

    return (len(diff1) <= 1) and (len(diff2) <= 1)


# check if texts match exactly
def check_simple_texts_match(annotations1, data, ignorecase=True):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    annotations1_copy = copy.deepcopy(annotations1)
    annotations2_copy = copy.deepcopy(annotations2)

    annotations1_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations1_copy]
    annotations2_copy = [(round(float(a) * 10 ** MATCHING_PRECISION), b) for a, b in annotations2_copy]

    if ignorecase:
        annotations1_copy = [(a, b.lower()) for a, b in annotations1_copy]
        annotations2_copy = [(a, b.lower()) for a, b in annotations2_copy]

    texts_1 = [el[1] for el in annotations1_copy]
    texts_2 = [el[1] for el in annotations2_copy]

    return set(texts_1) == set(texts_2)


# check if texts and percentages match (only the order might be wrong)
def check_simple_texts_and_percentages_match(annotations1, data, ignorecase=True):

    return check_simple_texts_match(annotations1, data, ignorecase) and check_percent_numbers_match(annotations1, data)


# check if an error occurred during annotation computation
def check_success(annotations1, data):

    if 'res' not in data.keys():
        return False

    return data['success']


# check if annotations are not empty
def check_not_empty(annotations1, data):

    if not check_success(annotations1, data):
        return False

    return bool(data['res'])


# check if annotations length matches
def check_annotations_len_match(annotations1, data):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    return len(annotations1) == len(annotations2)


# check if annotations length matches or is 1 too high
def check_annotations_len_match_plus_one(annotations1, data):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    return len(annotations1) <= len(annotations2) <= len(annotations1) + 1


# check if annotations length matches or is 1 too low
def check_annotations_len_match_minus_one(annotations1, data):

    if 'res' not in data.keys():
        return False

    annotations2 = data['res']

    return len(annotations1) - 1 <= len(annotations2) <= len(annotations1)


# compute all metrics
def compute_metrics(test_metrics=None, filename=METRICS_FILENAME, interactive=False):

    if test_metrics is None:
        test_metrics = load_test_metrics_json(filename=filename)

    total_metrics_count = len(test_metrics)

    metric_functions = [
        check_simple_annotations_match,
        check_simple_annotations_match_but_one,
        check_percent_numbers_match,
        check_success,
        check_simple_texts_match,
        check_annotations_len_match,
        check_simple_texts_and_percentages_match,
        check_annotations_len_match_minus_one,
        check_annotations_len_match_plus_one,
        check_not_empty,
        check_percent_numbers_match_but_one,
        check_fuzz_ratio_annotations_match_70,
        check_fuzz_ratio_annotations_match_80,
        check_fuzz_ratio_annotations_match_90,
        check_fuzz_partial_ratio_annotations_match_70,
        check_fuzz_partial_ratio_annotations_match_80,
        check_fuzz_partial_ratio_annotations_match_90,
        check_fuzz_partial_ratio_reversed_annotations_match_70,
        check_fuzz_partial_ratio_reversed_annotations_match_80,
        check_fuzz_partial_ratio_reversed_annotations_match_90
    ]

    res_dict = {func.__name__: [] for func in metric_functions}

    for n_str, data in test_metrics.items():

        n = int(n_str)

        annotations1 = load_test_annotations(n)

        # if 'res' not in data.keys():
        #     logging.warning("res key not found")
        #     continue
        #
        # annotations2 = data['res']

        logging.info("annotations1: {0}".format(annotations1))

        if 'res' in data.keys():
            annotations2 = data['res']
            logging.info("annotations2: {0}".format(annotations2))

        for func in metric_functions:
            res = func(annotations1, data)
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

    final_dict = dict(sorted(final_dict.items(), key=lambda x: x[0]))

    final_dict = {k: round(100 * v, 2) for k, v in final_dict.items()}

    logging.info("final_dict: {0}".format(final_dict))

    return final_dict


# create metrics plot
def create_metrics_plot(metrics_dict):

    plt.figure(figsize=(8.4, 4.8))

    plt.barh(range(len(metrics_dict)), metrics_dict.values())

    plt.yticks(range(len(metrics_dict)), metrics_dict.keys())

    plt.xlabel('Percentage')
    plt.ylabel('Metric')

    plt.tight_layout()

    path = os.path.join(get_root_path(), 'artifacts', 'ocr_test_metrics.png')
    plt.savefig(path)
