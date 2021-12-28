import multiprocessing
import pebble
import logging
from . import pie_chart_ocr
from .data_helpers import get_upscaled_steph_test_path, test_data_percentages
import concurrent.futures
from .helperfunctions import delete_keys_from_dict, get_root_path
import os
import json
from datetime import datetime
from tqdm import tqdm


# get the path for upscaled test image n and execute pie_chart_ocr.main() non-interactively
def pie_chart_ocr_wrapper(n):

    logging.info("Executing pie_chart_ocr for test image {0}...".format(n))

    _, path = get_upscaled_steph_test_path(n)

    logging.info("Upscaled image path for chart {0}: {1}".format(n, path))

    ocr_res = pie_chart_ocr.main(path, interactive=False)

    logging.info("Result for chart {0}: {1}".format(n, ocr_res))

    return n, ocr_res


# execute pie_chart_ocr_wrapper(n) for multiple arguments in parallel
def multiprocess_pie_chart_ocr(n_list, worker_count=None, show_progress=True):

    if worker_count is None:
        worker_count = multiprocessing.cpu_count()

    total_fut_count = len(n_list)

    fut_count = 0

    allres = []

    # actually not needed, but suppresses PyCharm warning
    pbar = None

    if show_progress:
        pbar = tqdm(total=total_fut_count)

    with pebble.ProcessPool(max_workers=worker_count, max_tasks=1) as executor:

        jobs = {}

        for n in n_list:
            job = executor.schedule(pie_chart_ocr_wrapper, args=[n])
            jobs[job] = n

        while fut_count < total_fut_count:

            for future in concurrent.futures.as_completed(jobs):
                res = future.result()
                allres.append(res)
                fut_count += 1
                if show_progress:
                    pbar.update(1)

    if show_progress:
        pbar.close()

    allres.sort(key=lambda x: x[0])

    return allres


# convert ocr results to dictionary
def convert_ocr_results_to_dict(ocr_res):

    dictionary = dict(ocr_res)

    # remove the approx key because ndarray is not JSON serializable
    dictionary = delete_keys_from_dict(dictionary, ['approx'])

    return dictionary


# store ocr results as JSON
def store_ocr_results_as_json(ocr_res, filename):

    path = os.path.join(get_root_path(), 'artifacts', filename)

    dictionary = convert_ocr_results_to_dict(ocr_res)

    logging.info("Dumping dictionary to path: {0}".format(path))

    with open(path, 'w') as jsonfile:
        json.dump(dictionary, jsonfile)


# generate test metrics JSON
def generate_test_metrics_json():

    logging.info("Generating test metrics...")

    start_time = datetime.now()
    logging.info("START TIME: {0}".format(start_time))

    logging.debug("Fetching list of usable charts...")
    n_list = test_data_percentages()
    logging.debug("{0} usable charts found.".format(len(n_list)))

    worker_count = multiprocessing.cpu_count() - 2
    logging.debug("Parsing charts using {0} cores...".format(worker_count))
    ocr_res = multiprocess_pie_chart_ocr(n_list)

    logging.debug("Storing ocr results to artifacts/ocr_test_metrics.json...")
    store_ocr_results_as_json(ocr_res, 'ocr_test_metrics.json')

    stop_time = datetime.now()

    logging.info("Test metrics successfully generated!")

    logging.info("STOP TIME: {0}".format(stop_time))
    logging.info("EXECUTION TIME: {0}".format((stop_time - start_time).total_seconds()))
