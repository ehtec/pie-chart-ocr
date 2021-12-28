import logging
logging.basicConfig(level=logging.DEBUG)
from piechartocr.metrics import compare_test_metrics, load_test_metrics_json, compute_metrics


test_metrics = load_test_metrics_json()

compare_test_metrics(error_on_diff=False, error_on_miss=False, test_metrics=test_metrics)

compute_metrics(test_metrics=test_metrics, interactive=True)
