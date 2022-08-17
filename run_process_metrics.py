import logging
logging.basicConfig(level=logging.DEBUG)
from piechartocr.metrics import compare_test_metrics, load_test_metrics_json, compute_metrics, create_metrics_plot
import os
from piechartocr.helperfunctions import get_root_path


def main():

    artifacts_path = os.path.join(get_root_path(), 'artifacts')

    for filename in os.listdir(artifacts_path):

        if not filename.endswith('.json'):
            continue

        logging.info("Processing {0}...".format(filename))

        filename_without_extension = os.path.splitext(filename)[0]

        json_filename = f"{filename_without_extension}.json"
        png_filename = f"{filename_without_extension}.png"

        test_metrics = load_test_metrics_json(filename=json_filename)

        compare_test_metrics(error_on_diff=False, error_on_miss=False, test_metrics=test_metrics)

        metrics_dict = compute_metrics(test_metrics=test_metrics, interactive=False)

        create_metrics_plot(metrics_dict, filename=png_filename)

        logging.debug("Wrote {0}".format(png_filename))


if __name__ == "__main__":
    main()
