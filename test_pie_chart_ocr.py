import logging

logging.basicConfig(level=logging.WARNING)

import pie_chart_ocr
import os


# get (csvpath, imagepath) by number from stephs first test dataset
def get_steph_test_path(n):

    basepath = os.path.join("/home/elias/pie-chart-ocr/data", "charts_steph", "Chart_{0}".format(n))

    logging.debug("basepath: {0}".format(basepath))

    l = os.listdir(basepath)

    logging.debug("l: {0}".format(l))

    imagefiles = [el for el in l if "image" in el]

    logging.debug("imagefiles: {0}".format(imagefiles))

    if not bool(imagefiles):
        raise Exception("No image found for chart number {0}".format(n))

    imagepath = os.path.join(basepath, imagefiles[0])

    csvpath = os.path.join(basepath, "annotation.csv")

    if not os.path.isfile(csvpath):
        raise Exception("No annotation found for chart number {0}".format(n))

    if not os.path.isfile(imagepath):
        raise Exception("No image found for chart number {0}".format(n))

    return csvpath, imagepath


# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019_1.png'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-019.png'
# IMG_INPUT_PATH = '/home/elias/pdf_images/saved_images/image-024.jpg'
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

IMG_INPUT_PATH = get_steph_test_path(1)[1]

pie_chart_ocr.main(IMG_INPUT_PATH)


