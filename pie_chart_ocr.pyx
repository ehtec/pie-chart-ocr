import pytesseract
from pytesseract import Output
import cv2
from pprint import pprint
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import copy
# from pylab import *
from scipy.ndimage import measurements

import mser_functions
from helperfunctions import group_pairs_to_nested_list, clean_folder_contents
from helperfunctions import pre_rectangle_center, rect_from_pre, detect_percentage, connect_polygon_cloud_2
# import time
import itertools
from statistics import mean
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import re
from polygon_calc_wrapper import PolygonCalc
from color_processer_wrapper import ColorProcesser
import logging
import uuid
from polygon_calc_wrapper import PolygonCalc


# white rgb pixel
WHITE_PIXEL = (255, 255, 255)

# black rgb pixel
BLACK_PIXEL = (0, 0, 0)

# threshold for color replacement
COLOR_DISTANCE_THRESHOLD = 18.0

# color threshold for cluster detection
# CLUSTER_COLOR_DISTANCE_THRESHOLD = 11.0

# minimum ratio of the total pixels to be a large cluster
LARGE_CLUSTER_RATIO = 0.005

# minimum confidence for text to be used
MIN_CONFIDENCE = 50  # 40  # 30

# factor for image up / downscaling before ocr step 2
SCALING_FACTOR = 2  # 2

# border width to improve border text recognition
BORDER_WIDTH = 15

# contrast factor to apply to image
CONTRAST_FACTOR = 1.0  # 1.2

# minimum area ratio to be covered to be counted as double detection of the same word
MIN_INTERSECTION_AREA_RATIO = 0.75

# maximum distance of words to be recognized as belonging to the same paragraph in terms of letter height
MAX_WORD_DISTANCE_RATIO = 0.2  # 0.75

# number of colors to use
COLORS_NUM = 120

# maximum number of concurrent processes to launch
MAX_WORKERS = 10  # 14

# # override scaling factor dynamically
# OVERRIDE_SCALING_FACTOR = True

# target pixel size after scaling
TARGET_PIXEL_SIZE = 8000*800

# maximum area ratio of a single word
MAX_WORD_AREA_RATIO = 0.05

# save temporary images
SAVE_TEMP_IMAGES = True


# def partition(pred, iterable):
#     t1, t2 = itertools.tee(iterable)
#     return itertools.filterfalse(pred, t1), filter(pred, t2)


# check if a color is closer to black or white
# 0: black, 1: white
def white_or_black(pixel):

    luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

    if luminance < 128:
        return 0

    else:
        return 1


def main(path):

    clean_folder_contents('temp1')
    clean_folder_contents('temp')

    start_time = datetime.now()

    print("START TIME: {0}".format(start_time))

    filtered_res_tuples, img = mser_functions.main(path)

    L2 = []

    print("Starting with step 5...")

    # comb = itertools.combinations(filtered_res_tuples, 2)

    pc = PolygonCalc()

    # for elem in comb:
    #
    #     pre_p1 = tuple(elem[0][2:6])
    #     pre_p2 = tuple(elem[1][2:6])
    #
    #     # p1 = ((pre_p1[0], pre_p1[1]), (pre_p1[0], pre_p1[3]), (pre_p1[2], pre_p1[3]), (pre_p1[2], pre_p1[1]))
    #     # p2 = ((pre_p2[0], pre_p2[1]), (pre_p2[0], pre_p2[3]), (pre_p2[2], pre_p2[3]), (pre_p2[2], pre_p2[1]))
    #
    #     p1 = rect_from_pre(pre_p1)
    #     p2 = rect_from_pre(pre_p2)
    #
    #     p1_height = pre_p1[3] - pre_p1[1]
    #     p2_height = pre_p2[3] - pre_p2[1]
    #
    #     p_height = min(p1_height, p2_height)
    #
    #     max_word_dist = MAX_WORD_DISTANCE_RATIO * p_height
    #
    #     # print("p1:")
    #     # print(p1)
    #
    #     # min_dist = min_poly_distance(p1, p2)
    #
    #     # pc = PolygonCalc()
    #
    #     min_dist = pc.min_poly_distance(p1, p2)
    #
    #     # area_ratio = poly_intersection_area_ratio(p1, p2)
    #
    #     # if all([min_dist == 0, area_ratio > MIN_INTERSECTION_AREA_RATIO]):
    #     #
    #     #     L.append(elem)
    #
    #     if min_dist < max_word_dist:
    #
    #         L2.append(elem)

    # del pc

    # for elem in filtered_res_tuples:
    #
    #     L2.append((elem, elem))
    #
    # word_grouped_tuples = group_pairs_to_nested_list(L2)

    word_grouped_tuples = pc.group_elements(filtered_res_tuples, MAX_WORD_DISTANCE_RATIO, -1, start_pos=2)

    # word_grouped_tuples = copy.deepcopy(filtered_res_tuples)

    del pc

    print("word_grouped_tuples:")
    pprint(word_grouped_tuples)

    print("Starting with step 6...")

    all_paragraph_tuples = []

    for paragraph in word_grouped_tuples:

        comb = itertools.permutations(paragraph, 2)

        # for elem in paragraph:
        #
        #     L3.append((elem, elem))

        # list for word row detection
        L3 = []

        for elem in paragraph:

            L3.append((elem, elem))

        for elem in comb:
            pre_p1 = tuple(elem[0][2:6])
            pre_p2 = tuple(elem[1][2:6])

            # p1 = ((pre_p1[0], pre_p1[1]), (pre_p1[0], pre_p1[3]), (pre_p1[2], pre_p1[3]), (pre_p1[2], pre_p1[1]))
            # p2 = ((pre_p2[0], pre_p2[1]), (pre_p2[0], pre_p2[3]), (pre_p2[2], pre_p2[3]), (pre_p2[2], pre_p2[1]))

            p1 = rect_from_pre(pre_p1)
            p2 = rect_from_pre(pre_p2)

            if pre_p1[3] - pre_p1[1] > 0:
                y1 = pre_p1[1]
                y2 = pre_p1[3]

            else:
                y1 = pre_p1[3]
                y2 = pre_p1[1]

            # if pre_p1[-1] == 10023:
            # print(elem)
            # print("YES")

            if any([y1 <= pre_p2[1] <= y2, y1 <= pre_p2[3] <= y2,
                    all([y1 <= pre_p2[1], y1 <= pre_p2[3], y2 >= pre_p2[1], y2 >= pre_p2[3]])]):

                L3.append(elem)

        paragraph_tuples = group_pairs_to_nested_list(L3)

        for elem in paragraph_tuples:

            elem.sort(key=lambda x: x[4])

        paragraph_tuples.sort(key=lambda x: mean([pre_rectangle_center(elem[2:6])[1] for elem in x]), reverse=False)

        all_paragraph_tuples.append(paragraph_tuples)

        print("paragraph_tuples:")
        pprint(paragraph_tuples)

    joined_tuples = []

    print("Starting with step 7...")

    old_all_paragraph_tuples = copy.deepcopy(all_paragraph_tuples)

    all_paragraph_tuples = []

    for paragraph in old_all_paragraph_tuples:

        # if len(paragraph) < 2:
        #     all_paragraph_tuples.append(paragraph)
        #     continue

        remaining_paragraph = []

        for item in paragraph:

            for elem in item:

                print("elem: {0}".format(elem))

                if detect_percentage(elem[1]) is not None:
                    all_paragraph_tuples.append([[elem]])

                else:
                    remaining_paragraph.append(elem)

        if bool(remaining_paragraph):
            all_paragraph_tuples.append([remaining_paragraph])

    for paragraph in all_paragraph_tuples:

        l = [elem[1] for el in paragraph for elem in el]

        s = ' '.join(l).strip()

        s = re.sub(r'^[^A-Za-z0-9]+', '', s)

        s = re.sub(r':$', '', s)

        s = s.strip()

        re_res = re.findall(r'[A-Za-z0-9]', s)

        if not bool(re_res):
            continue

        # s = re.sub(r'^[^A-z0-9]+', '', s)
        #
        # s = re.sub(r':$', '', s)

        x1 = min([elem[2] for el in paragraph for elem in el])

        x2 = max([elem[4] for el in paragraph for elem in el])

        y1 = min([elem[3] for el in paragraph for elem in el])

        y2 = max([elem[5] for el in paragraph for elem in el])

        coord = (x1, y1, x2, y2)

        res_tuple = (coord, s)

        joined_tuples.append(res_tuple)

    print("joined_tuples:")
    pprint(joined_tuples)

    print("Starting with step 8...")

    for res_tuple in joined_tuples:

        t = res_tuple[0]

        x1 = t[0]
        y1 = t[1]

        x2 = t[2]
        y2 = t[3]

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    polygons_percent = []
    polygons_text = []

    polygons_percent_data = []
    polygons_text_data = []

    print("Starting with step 9...")

    for res_tuple in joined_tuples:

        p1 = rect_from_pre(res_tuple[0])

        s = res_tuple[1]

        det_per = detect_percentage(s)

        if det_per is not None:

            polygons_percent_data.append((p1, det_per))

            polygons_percent.append(p1)

        else:

            polygons_text_data.append((p1, s))

            polygons_text.append(p1)

    if any([
        not bool(polygons_percent),
        not bool(polygons_text)
    ]):
        pairs = []

    else:
        pairs = connect_polygon_cloud_2(polygons_percent, polygons_text)

    print("pairs:")
    pprint(pairs)

    res = []

    p = polygons_percent_data + polygons_text_data

    for i1, i2 in zip(*np.nonzero(pairs)):

        if i1 > i2:

            i1, i2 = i2, i1

        res.append((p[i1][1], p[i2][1]))

    print("res:")
    pprint(res)

    percent_sum = sum([elem[0] for elem in res])

    if percent_sum != 1.0:
        print("Percentages sum does not add up to 100%!")

    stop_time = datetime.now()

    seconds_elapsed = (stop_time - start_time).total_seconds()

    print("STOP TIME: {0}".format(stop_time))

    print("SECONDS ELAPSED: {0}".format(seconds_elapsed))

    cv2.imshow('img', img)
    cv2.waitKey(0)

    # img = img[162:184, 633:650]
    # cv2.imwrite('test.png', img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)



