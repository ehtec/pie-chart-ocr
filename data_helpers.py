import logging
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
    print("")

    remaining_annotation_duplicates = []
    remaining_file_duplicates = []

    remaining_annotations = []
    remaining_files = []

    deleted_annotation_duplicates = []
    deleted_file_duplicates = []

    for el in annotation_duplicates:

        if el[1] in remaining_annotations:

            deleted_annotation_duplicates.append(el[0])

        else:

            remaining_annotation_duplicates.append(el[0])
            remaining_annotations.append(el[1])

    for el in file_duplicates:

        if el[1] in remaining_files:

            deleted_file_duplicates.append(el[0])

        else:

            remaining_file_duplicates.append(el[0])
            remaining_files.append(el[1])

    # annotation duplicates are most of the time ok
    print("Remaining annotation duplicates: {0}".format(remaining_annotation_duplicates))
    print("Deleted annotation duplicates: {0}".format(deleted_annotation_duplicates))
    print("")
    print("Remaining file duplicates: {0}".format(remaining_file_duplicates))
    print("Deleted file duplicates: {0}".format(deleted_file_duplicates))
    print("")


# check if percentages add up to 1
def test_data_percentages(approximation_inaccuracy=0.02):

    correct_numbers = test_data_format()

    correct_numbers_2 = []
    wrong_numbers_2 = []

    total_count = 0
    correct_count = 0
    wrong_count = 0

    for i in correct_numbers:

        total_count += 1

        csvpath, imagepath = get_steph_test_path(i)

        annotations = load_annotations_from_csv(csvpath)

        # allow some approximation inaccuracy

        lower_bounds = 1.0 - approximation_inaccuracy
        upper_bounds = 1.0 + approximation_inaccuracy

        if not lower_bounds < sum([el[1] for el in annotations]) < upper_bounds:
            wrong_count += 1
            wrong_numbers_2.append(i)

        else:
            correct_count += 1
            correct_numbers_2.append(i)

    correct_percentage = round(100 * correct_count / total_count, 2)
    wrong_percentage = round(100 * wrong_count / total_count, 2)

    print("total_count: {0}".format(total_count))
    print("correct_count: {0} ({1}%)".format(correct_count, correct_percentage))
    print("wrong_count: {0} ({1}%)".format(wrong_count, wrong_percentage))
    print("")
    print("wrong_numbers_2: {0}".format(wrong_numbers_2))

    return correct_numbers_2