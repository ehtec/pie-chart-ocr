import multiprocessing
import pebble
import logging
from . import pie_chart_ocr
from .data_helpers import get_upscaled_steph_test_path
import concurrent.futures
from .helperfunctions import delete_keys_from_dict


# get the path for upscaled test image n and execute pie_chart_ocr.main() non-interactively
def pie_chart_ocr_wrapper(n):

    logging.info("Executing pie_chart_ocr for test image {0}...".format(n))

    _, path = get_upscaled_steph_test_path(n)

    logging.info("Upscaled image path for chart {0}: {1}".format(n, path))

    ocr_res = pie_chart_ocr.main(path, interactive=False)

    logging.info("Result for chart {0}: {1}".format(n, ocr_res))

    return n, ocr_res


# execute pie_chart_ocr_wrapper(n) for multiple arguments in parallel
def multiprocess_pie_chart_ocr(n_list, worker_count=None):

    if worker_count is None:
        worker_count = multiprocessing.cpu_count()

    total_fut_count = len(n_list)

    fut_count = 0

    allres = []

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

    allres.sort(key=lambda x: x[0])

    return allres
