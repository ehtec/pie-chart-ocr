import logging
logging.basicConfig(level=logging.WARNING)
from piechartocr.multiprocess_ocr import generate_test_metrics_json


# parse charts and generate JSON with data
generate_test_metrics_json()
