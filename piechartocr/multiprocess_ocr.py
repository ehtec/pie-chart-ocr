import multiprocessing
import pebble
from pebble.common import ProcessExpired
import logging
from . import pie_chart_ocr
from .data_helpers import get_upscaled_steph_test_path, test_data_percentages
import concurrent.futures
from .helperfunctions import delete_keys_from_dict, get_root_path, NpEncoder
import os
import json
from datetime import datetime
from tqdm import tqdm
import copy


# maximum cap of workers, in case there are more CPUs
MAX_WORKERS_CAP = 24

# maximum number of iterations to repeat failed pebble jobs
MAX_RETRIES = 5

# metrics filename
METRICS_FILENAME = 'ocr_test_metrics.json'


# get the path for upscaled test image n and execute pie_chart_ocr.main() non-interactively
def pie_chart_ocr_wrapper(n):

    try:

        logging.info("Executing pie_chart_ocr for test image {0}...".format(n))

        _, path = get_upscaled_steph_test_path(n)

        logging.info("Upscaled image path for chart {0}: {1}".format(n, path))

        ocr_res = pie_chart_ocr.main(path, interactive=False)

        logging.info("Result for chart {0}: {1}".format(n, ocr_res))

        return n, ocr_res

    except Exception as e:
        logging.exception(e)
        raise e


# execute pie_chart_ocr_wrapper(n) for multiple arguments in parallel
def multiprocess_pie_chart_ocr(n_list, worker_count=None, show_progress=True):

    if worker_count is None:
        worker_count = multiprocessing.cpu_count()

    allres = []
    n_list_done = []
    it_counter = 0

    n_list_copy = copy.deepcopy(n_list)

    # actually not needed, but suppresses PyCharm warning
    pbar = None

    if show_progress:
        pbar = tqdm(total=len(n_list_copy))

    while bool(n_list_copy):

        it_counter += 1

        total_fut_count = len(n_list_copy)

        fut_count = 0

        logging.info("Creating process pool: try {0}".format(it_counter))

        with pebble.ProcessPool(max_workers=worker_count, max_tasks=1) as executor:

            jobs = {}

            for n in n_list_copy:
                job = executor.schedule(pie_chart_ocr_wrapper, args=[n])
                jobs[job] = n

            while fut_count < total_fut_count:

                for future in concurrent.futures.as_completed(jobs):

                    try:
                        res = future.result()
                        allres.append(res)
                        n_list_done.append(res[0])

                    except ProcessExpired:
                        logging.critical("Received ProcessExpired error!")

                    except Exception as e:
                        logging.exception(e)

                    fut_count += 1
                    if show_progress:
                        pbar.update(1)

        n_list_copy = [el for el in n_list_copy if el not in n_list_done]

        logging.critical("n_list_copy after iteration {0}: {1}".format(it_counter, n_list_copy))

        if bool(n_list_copy) and (it_counter > MAX_RETRIES):
            logging.critical("Max retries ({0}) reached but job still failing!".format(MAX_RETRIES))

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
        json.dump(dictionary, jsonfile, cls=NpEncoder)


# generate test metrics JSON
def generate_test_metrics_json(filename=METRICS_FILENAME):  # pragma: no cover

    logging.info("Generating test metrics...")

    start_time = datetime.now()
    logging.info("START TIME: {0}".format(start_time))

    logging.debug("Fetching list of usable charts...")
    n_list = test_data_percentages()
    logging.debug("{0} usable charts found.".format(len(n_list)))

    worker_count = min(multiprocessing.cpu_count() - 2, MAX_WORKERS_CAP)
    logging.debug("Parsing charts using {0} cores...".format(worker_count))
    ocr_res = multiprocess_pie_chart_ocr(n_list, worker_count=worker_count)

    logging.debug("Storing ocr results to artifacts/ocr_test_metrics.json...")
    store_ocr_results_as_json(ocr_res, filename)

    stop_time = datetime.now()

    logging.info("Test metrics successfully generated!")

    logging.info("STOP TIME: {0}".format(stop_time))
    logging.info("EXECUTION TIME: {0}".format((stop_time - start_time).total_seconds()))
