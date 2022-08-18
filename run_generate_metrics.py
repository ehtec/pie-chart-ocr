import logging
logging.basicConfig(level=logging.ERROR)
from piechartocr.multiprocess_ocr import generate_test_metrics_json
import os


# parse charts and generate JSON with data
generate_test_metrics_json(os.getenv("PIECHARTOCR_METRICS_JSON_FILENAME"))

# # metrics for generated_pie_charts_legend folder
# os.environ["PIECHARTOCR_BASE_FOLDERNAME"] = "generated_pie_charts_legend"
# os.environ["PIECHARTOCR_UPSCALED_BASE_FOLDERNAME"] = "generated_pie_charts_legend"
# os.environ["PIECHARTOCR_UPSCALED_FILENAME"] = "image.png"
# generate_test_metrics_json("ocr_test_metric_mock_legend.json")
#
# # metrics for generated_pie_charts_without_legend folder
# os.environ["PIECHARTOCR_BASE_FOLDERNAME"] = "generated_pie_charts_without_legend"
# os.environ["PIECHARTOCR_UPSCALED_BASE_FOLDERNAME"] = "generated_pie_charts_without_legend"
# os.environ["PIECHARTOCR_UPSCALED_FILENAME"] = "image.png"
# generate_test_metrics_json("ocr_test_metric_mock_without_legend.json")
