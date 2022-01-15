# import pytesseract
# from pytesseract import Output
import cv2
# from pprint import pprint
# from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import copy
# from pylab import *
# from scipy.ndimage import measurements
from . import mser_functions
from .helperfunctions import clean_folder_contents, get_root_path, calculate_distance_matrix
from .helperfunctions import pre_rectangle_center, rect_from_pre, detect_percentage, connect_polygon_cloud_2
from .helperfunctions import remove_sc_prefix, remove_sc_suffix
# import time
# import itertools
from statistics import mean
from datetime import datetime
# from concurrent.futures import ProcessPoolExecutor
import re
# from polygon_calc_wrapper import PolygonCalc
# from color_processer_wrapper import ColorProcesser
# import logging
# import uuid
from .polygon_calc_wrapper import PolygonCalc
import os
import logging


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
MAX_WORD_DISTANCE_RATIO = 0.60  # 0.20  # 0.75

# number of colors to use
COLORS_NUM = 120

# maximum number of concurrent processes to launch
MAX_WORKERS = 10  # 14

# # override scaling factor dynamically
# OVERRIDE_SCALING_FACTOR = True

# target pixel size after scaling
TARGET_PIXEL_SIZE = 8000 * 800

# maximum area ratio of a single word
MAX_WORD_AREA_RATIO = 0.05

# save temporary images
SAVE_TEMP_IMAGES = True

# threshold distance for paragraph sentence grouping. Not used if everything went well before (because only words near
#   enough each other should have been grouped into the same paragraph)
PARAGRAPH_THRESHOLD_DIST = 0.5  # 0  # 1.0  # 5.0


def main(path, interactive=True):

    clean_folder_contents(os.path.join(get_root_path(), 'temp1'))
    clean_folder_contents(os.path.join(get_root_path(), 'temp'))

    start_time = datetime.now()

    logging.info("START TIME: {0}".format(start_time))

    filtered_res_tuples, img, chart_data = mser_functions.main(path, interactive=interactive)

    logging.debug("Starting with step 5...")

    pc = PolygonCalc()

    word_grouped_tuples = pc.group_elements(filtered_res_tuples, MAX_WORD_DISTANCE_RATIO, -1, start_pos=2,
                                            size_metric_mode=1)

    logging.debug("word_grouped_tuples: {0}".format(word_grouped_tuples))
    # pprint(word_grouped_tuples)

    logging.debug("Starting with step 6...")

    all_paragraph_tuples = []

    for paragraph in word_grouped_tuples:

        paragraph_tuples = pc.group_elements(paragraph, PARAGRAPH_THRESHOLD_DIST, mser_functions.SLOV_RATIO,
                                             start_pos=2, size_metric_mode=1)

        for elem in paragraph_tuples:

            elem.sort(key=lambda x: x[4])

        paragraph_tuples.sort(key=lambda x: mean([pre_rectangle_center(elem2[2:6])[1] for elem2 in x]), reverse=False)

        all_paragraph_tuples.append(paragraph_tuples)

        logging.debug("paragraph_tuples: {0}".format(paragraph_tuples))
        # pprint(paragraph_tuples)

    joined_tuples = []

    logging.debug("Starting with step 7...")

    old_all_paragraph_tuples = copy.deepcopy(all_paragraph_tuples)

    all_paragraph_tuples = []

    for paragraph in old_all_paragraph_tuples:

        # if len(paragraph) < 2:
        #     all_paragraph_tuples.append(paragraph)
        #     continue

        remaining_paragraph = []

        for item in paragraph:

            blocked_elems = []

            for elem_counter in range(len(item)):

                if elem_counter in blocked_elems:
                    continue

                elem = item[elem_counter]

                next_starts_with_percent = False

                if elem_counter + 1 < len(item):

                    next_elem = item[elem_counter + 1]

                    if next_elem[1].strip().startswith('%'):

                        next_starts_with_percent = True

                        blocked_elems.append(elem_counter + 1)

                logging.debug("elem: {0}".format(elem))

                if detect_percentage(elem[1]) is not None:
                    all_paragraph_tuples.append([[elem]])

                elif next_starts_with_percent and detect_percentage(elem[1] + '%') is not None:
                    # referenced before assignment warning false positive
                    all_paragraph_tuples.append([[elem, next_elem]])

                else:

                    remaining_paragraph.append(elem)

        if bool(remaining_paragraph):
            all_paragraph_tuples.append([remaining_paragraph])

    for paragraph in all_paragraph_tuples:

        li = [elem[1] for el in paragraph for elem in el]

        s = ' '.join(li).strip()

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

    logging.debug("joined_tuples: {0}".format(joined_tuples))
    # pprint(joined_tuples)

    logging.debug("Starting with step 8...")

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

    logging.debug("Starting with step 9...")

    for res_tuple in joined_tuples:

        p1 = rect_from_pre(res_tuple[0])

        s = res_tuple[1]

        det_per = detect_percentage(s)

        if det_per is not None:

            polygons_percent_data.append((p1, det_per))

            polygons_percent.append(p1)

        else:

            s = remove_sc_suffix(s)
            s = remove_sc_prefix(s)

            if not bool(s):
                continue

            polygons_text_data.append((p1, s))

            polygons_text.append(p1)

    # actually not needed, but suppresses warning in PyCharm
    res = None
    pairs = None

    # this variable is only set to False in a specific case where results are calculated differently
    calculate_res = True

    # specify if a fallback was used (the results might be wrong / inaccurate)
    fallback_used = False

    if any([
        not bool(polygons_percent),
        not bool(polygons_text)
    ]):
        pairs = []

    else:

        if not chart_data['has_chart_ellipse']:
            logging.warning("No chart ellipse found!")
            logging.warning("Falling back to legacy method. The results might be wrong.")
            chart_data['has_legend'] = False
            fallback_used = True

        # not possible to reach with the current method, as no legend is computed when no chart ellipse is found
        # if (chart_data['has_legend']) and (not chart_data['has_chart_ellipse']):
        #     logging.warning("The chart has a legend, but no chart ellipse was found.")
        #     logging.warning("Falling back to legacy method. The results might be wrong.")
        #     chart_data['has_legend'] = False
        #     fallback_used = True

        if chart_data['has_legend']:
            if len(chart_data['legend_shapes']) != len(polygons_percent):
                logging.warning("Number of legend shapes does not match the number of percent texts: {0} vs {1}".format(
                    len(chart_data['legend_shapes']), len(polygons_percent)
                ))
                logging.warning("Falling back to legacy method. The results might be wrong.")
                chart_data['has_legend'] = False
                fallback_used = True

        if chart_data['has_legend']:
            logging.info("We are dealing with a chart WITH legend.")

            legend_polygons = [el['approx'] for el in chart_data['legend_shapes']]
            sector_centers = chart_data['sector_centers']

            # this should never happen. Would be an implementation error
            assert len(legend_polygons) == len(sector_centers)
            assert len(legend_polygons) == len(polygons_percent)

            # we cannot proceed if there is less text than percent numbers. Probably too many overlaps detected
            # assert len(polygons_percent) <= len(polygons_text)
            if len(polygons_percent) > len(polygons_text):
                logging.warning("Not enough text areas detected. Aborting.")
                data = {"success": False, "error": "Not enough text areas detected"}
                data.update(chart_data)
                logging.info("data: {0}".format(data))
                return data

            # polygons made of a single point
            sector_polygons = [[el] for el in sector_centers]

            sector_centers_percent_dm = calculate_distance_matrix(np.array(sector_polygons + polygons_percent,
                                                                           dtype=object))
            legend_polygons_percent_dm = calculate_distance_matrix(np.array(legend_polygons + polygons_percent,
                                                                            dtype=object))

            logging.debug("sector_centers_percent_dm: {0}".format(sector_centers_percent_dm))
            logging.debug("legend_polygons_percent_dm: {0}".format(legend_polygons_percent_dm))

            sector_centers_percent_pairs = connect_polygon_cloud_2(sector_polygons, polygons_percent,
                                                                   sector_centers_percent_dm)
            legend_polygons_percent_pairs = connect_polygon_cloud_2(legend_polygons, polygons_percent,
                                                                    legend_polygons_percent_dm)

            logging.debug("sector_centers_percent_pairs: {0}".format(sector_centers_percent_pairs))
            logging.debug("legend_polygons_percent_pairs: {0}".format(legend_polygons_percent_pairs))

            used_sector_centers_distance = sector_centers_percent_pairs * sector_centers_percent_dm
            used_legend_polygons_distance = legend_polygons_percent_pairs * legend_polygons_percent_dm

            logging.debug("used_sector_centers_distance: {0}".format(used_sector_centers_distance))
            logging.debug("used_legend_polygons_distance: {0}".format(used_legend_polygons_distance))

            total_sector_centers_distance = used_sector_centers_distance.sum()
            total_legend_polygons_distance = used_legend_polygons_distance.sum()

            logging.debug("total_sector_centers_distance: {0}".format(total_sector_centers_distance))
            logging.debug("total_legend_polygons_distance: {0}".format(total_legend_polygons_distance))

            if total_legend_polygons_distance < total_sector_centers_distance:
                logging.info("The percent numbers seem to be next to the legend.")
                logging.info("Therefore, we can handle this pie chart as if there was no legend.")
                pairs = connect_polygon_cloud_2(polygons_percent, polygons_text)

            else:
                logging.info("The percent numbers seem to be in or next to the sectors.")

                # calculate pairs between legend polygons and text polygons
                legend_text_pairs = connect_polygon_cloud_2(legend_polygons, polygons_text)

                logging.debug("legend_text_pairs: {0}".format(legend_text_pairs))

                # create dictionary from sector_index to percent_index
                sector_percent_dict = {}
                for i1, i2 in zip(*np.nonzero(sector_centers_percent_pairs)):

                    if i1 > i2:
                        i1, i2 = i2, i1

                    sector_index = i1
                    percent_index = i2 - len(sector_centers)

                    sector_percent_dict.update({sector_index: percent_index})

                logging.debug("sector_percent_dict: {0}".format(sector_percent_dict))

                # create dictionary from legend_index to text_index (legend_index == sector_index)
                legend_text_dict = {}
                for i1, i2 in zip(*np.nonzero(legend_text_pairs)):

                    if i1 > i2:
                        i1, i2 = i2, i1

                    legend_index = i1
                    text_index = i2 - len(legend_polygons)

                    legend_text_dict.update({legend_index: text_index})

                logging.debug("legend_text_dict: {0}".format(legend_text_dict))

                calculate_res = False

                res = []

                for i in range(len(sector_centers)):

                    percent_index = sector_percent_dict[i]
                    text_index = legend_text_dict[i]

                    percent_elem = polygons_percent_data[percent_index]
                    text_elem = polygons_text_data[text_index]

                    res.append((percent_elem[1], text_elem[1]))

        else:
            logging.info("We are dealing with a chart WITHOUT legend.")
            pairs = connect_polygon_cloud_2(polygons_percent, polygons_text)

    logging.debug("pairs: {0}".format(pairs))
    # pprint(pairs)

    if calculate_res:
        res = []

        p = polygons_percent_data + polygons_text_data

        for i1, i2 in zip(*np.nonzero(pairs)):

            if i1 > i2:

                i1, i2 = i2, i1

            res.append((p[i1][1], p[i2][1]))

    logging.info("res: {0}".format(res))
    # pprint(res)

    percent_sum = sum([elem[0] for elem in res])

    if percent_sum != 1.0:
        logging.warning("Percentages sum does not add up to 100%!")

    if fallback_used:
        logging.warning("fallback_used: {0}".format(fallback_used))

    else:
        logging.info("fallback_used: {0}".format(fallback_used))

    stop_time = datetime.now()

    seconds_elapsed = (stop_time - start_time).total_seconds()

    logging.info("STOP TIME: {0}".format(stop_time))

    logging.info("SECONDS ELAPSED: {0}".format(seconds_elapsed))

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    if interactive:
        # show resized window with image instead
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        cv2.imshow('img', img)
        cv2.resizeWindow('img', 800, 800)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # img = img[162:184, 633:650]
    # cv2.imwrite('test.png', img)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    data = {
        "success": True,
        "res": res,
        "fallback_used": fallback_used,
        "percent_sum": percent_sum
    }

    # also include data previously computed, like legend type
    data.update(chart_data)

    logging.info("data: {0}".format(data))

    return data
