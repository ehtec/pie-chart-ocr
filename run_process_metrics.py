import logging
logging.basicConfig(level=logging.DEBUG)
from piechartocr.metrics import compare_test_metrics


compare_test_metrics(error_on_diff=True)
