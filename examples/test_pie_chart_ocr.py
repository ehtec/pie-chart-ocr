import logging

logging.basicConfig(level=logging.INFO)

from piechartocr import pie_chart_ocr
from piechartocr import mser_functions
from piechartocr.data_helpers import get_steph_test_path, load_annotations_from_csv, test_data_format,\
    test_data_duplicates, test_data_percentages
from piechartocr.helperfunctions import get_root_path
import os


def main():

    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019_1.png'
    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019.png'
    IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-024.jpg'
    # IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-038.jpg'

    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_1.png"  # 30 misdetected for 20 with 0.5 accuracy setting
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_2.jpg"  # some typos in the text
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_3.jpg"
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_4.jpg"
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_5.png"  # not well detected, a little better with scaling at step 1 and scaling factor 4
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_6.png"
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_7.png"  # percentages detected, but false positive for descriptions
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_8.png"
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_9.png"  # good, but a few false positives
    # IMG_INPUT_PATH = "/home/elias/pie-chart-ocr/pie_charts/test_10.png"

    # i = int(input("Chart number: "))

    # mser_functions.main(IMG_INPUT_PATH)

    # correct_numbers = test_data_percentages()
    #
    # for i in correct_numbers:
    #
    #     print("PROCESSING IMAGE: {0}".format(i))
    #
    #     csvpath, IMG_INPUT_PATH = get_steph_test_path(i)
    #
    #     # pie_chart_ocr.main(IMG_INPUT_PATH)
    #
    #     mser_functions.main(IMG_INPUT_PATH)

    # print(load_annotations_from_csv(csvpath))
    #
    # test_data_format()
    #
    # test_data_duplicates()
    #
    # test_data_percentages()

    n = int(input("Image id: "))

    path = os.path.join(get_root_path(), "temp2", "upscaled{0}.png".format(n))

    pie_chart_ocr.main(path)


if __name__ == "__main__":
    main()
