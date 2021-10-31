import logging

logging.basicConfig(level=logging.WARNING)

import pie_chart_ocr
import os
import csv
from helperfunctions import hash_file


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


# load annotations csv into tuples with percentages as ratios
def load_annotations_from_csv(csvpath):

    res_tuples = []

    with open(csvpath, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')

        for row in spamreader:

            logging.debug(row)

            if not bool(row):
                continue

            res_tuples.append((row[0].strip(), float(row[1].strip()) / 100.0))

    logging.debug("res_tuples: {0}".format(res_tuples))

    return res_tuples


# test for formatting errors in stephs work
def test_data_format():

    correct_count = 0
    wrong_count = 0
    total_count = 0

    correct_numbers = []
    failed_numbers = []

    for i in range(1, 401):

        # total_count += 1

        try:

            try:

                csvpath, IMG_INPUT_PATH = get_steph_test_path(i)

            except Exception as e:
                continue

            total_count += 1

            annotations = load_annotations_from_csv(csvpath)

            correct_count += 1
            correct_numbers.append(i)

        except Exception as e:
            wrong_count += 1
            failed_numbers.append(i)

    correct_percentage = round(100 * correct_count / total_count, 2)
    wrong_percentage = round(100 * wrong_count / total_count, 2)

    print("total_count: {0}".format(total_count))
    print("correct_count: {0} ({1}%)".format(correct_count, correct_percentage))
    print("wrong_count: {0} ({1}%)".format(wrong_count, wrong_percentage))
    print("")
    print("failed_numbers: {0}".format(failed_numbers))

    return correct_numbers


# check if test data contains duplicates
def test_data_duplicates():

    correct_numbers = test_data_format()

    all_annotation_tuples = []
    all_file_tuples = []

    all_annotations = []
    all_files = []

    for i in correct_numbers:

        csvpath, imagepath = get_steph_test_path(i)

        annotations = load_annotations_from_csv(csvpath)

        all_annotation_tuples.append((i, set(annotations)))
        all_annotations.append(set(annotations))

        the_hash = hash_file(imagepath)

        all_file_tuples.append((i, the_hash))
        all_files.append(the_hash)

    annotation_duplicates = [el for el in all_annotation_tuples if all_annotations.count(el[1]) > 1]
    file_duplicates = [el for el in all_file_tuples if all_files.count(el[1]) > 1]

    print("{0} annotation duplicates found!".format(len(annotation_duplicates)))
    print(annotation_duplicates)
    print("")
    print("{0} file duplicates found!".format(len(file_duplicates)))
    print(file_duplicates)


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

# csvpath, IMG_INPUT_PATH = get_steph_test_path(1)

# pie_chart_ocr.main(IMG_INPUT_PATH)

# print(load_annotations_from_csv(csvpath))

# test_data_format()

test_data_duplicates()

