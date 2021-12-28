import multiprocessing
import pebble
import logging
from piechartocr import pie_chart_ocr
from piechartocr.data_helpers import get_upscaled_steph_test_path


# get the path for upscaled test image n and execute pie_chart_ocr.main() non-interactively
def pie_chart_ocr_wrapper(n):

    logging.info("Executing pie_chart_ocr for test image {0}...".format(n))

    _, path = get_upscaled_steph_test_path(n)

    logging.info("Upscaled image path for chart {0}: {1}".format(n, path))

    ocr_res = pie_chart_ocr.main(path, interactive=False)

    return ocr_res


# execute pie_chart_ocr_wrapper(n) for multiple arguments in parallel
