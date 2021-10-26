import pytesseract
from pytesseract import Output
import cv2
from pprint import pprint
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import copy
# from pylab import *
from scipy.ndimage import measurements
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
MIN_CONFIDENCE = 40  # 30

# factor for image up / downscaling before ocr step 2
SCALING_FACTOR = 2  # 2

# border width to improve border text recognition
BORDER_WIDTH = 15

# contrast factor to apply to image
CONTRAST_FACTOR = 1.0  # 1.2

# minimum area ratio to be covered to be counted as double detection of the same word
MIN_INTERSECTION_AREA_RATIO = 0.75

# maximum distance of words to be recognized as belonging to the same paragraph in terms of letter height
MAX_WORD_DISTANCE_RATIO = 0.75

# number of colors to use
COLORS_NUM = 120

# maximum number of concurrent processes to launch
MAX_WORKERS = 14

# # override scaling factor dynamically
# OVERRIDE_SCALING_FACTOR = True

# target pixel size after scaling
TARGET_PIXEL_SIZE = 8000*800

# maximum area ratio of a single word
MAX_WORD_AREA_RATIO = 0.05


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


def row_1_calc(row, orig_pil_img, k):

    try:

        the_color = row[1]

        distance_dict = {}

        pil_img = copy.deepcopy(orig_pil_img)

        replace_color = WHITE_PIXEL

        should_revert = False

        if white_or_black(the_color):
            replace_color = BLACK_PIXEL
            should_revert = True

        # exit()

        width = pil_img.size[0]
        height = pil_img.size[1]

        cp = ColorProcesser()

        pil_img_array = np.array(pil_img)

        color_distances = cp.array_color_distance(the_color, pil_img_array)

        del cp

        pil_img_array[color_distances > COLOR_DISTANCE_THRESHOLD] = replace_color

        pil_img = Image.fromarray(pil_img_array)

        if should_revert:
            pil_img = ImageOps.invert(pil_img)

        # img = np.array(pil_img)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey(3)
        # time.sleep(3)

        pil_img = pil_img.convert('1', dither=Image.NONE)  # .convert('RGB')

        # invert to remove clusters
        pil_img = ImageOps.invert(pil_img.convert('RGB')).convert('1', dither=Image.NONE).convert('RGB')

        # # add white border
        # new_img = Image.new('RGB', (width + 30, height + 30))
        # new_img.paste(pil_img, (15, 15))
        #
        # pil_img = new_img

        # remove noise
        # pil_img = pil_img.filter(ImageFilter.BLUR)
        # pil_img = pil_img.filter(ImageFilter.MinFilter(3))
        # pil_img = pil_img.filter(ImageFilter.MinFilter)

        pil_img = pil_img.filter(ImageFilter.MedianFilter())
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2)

        pil_img = pil_img.convert('1', dither=Image.NONE)

        # remove large clusters
        pil_img_array = np.array(pil_img)

        lw, num = measurements.label(pil_img_array)

        min_cluster_count = LARGE_CLUSTER_RATIO * height * width

        for elem in list(set(list(lw.flatten()))):

            # elem_count = len([item for item in list(lw.flatten()) if item == elem])

            elem_count = list(lw.flatten()).count(elem)

            # print(elem_count / width / height)

            if elem_count > min_cluster_count:

                pil_img_array[lw == elem] = 0

        pil_img = Image.fromarray(pil_img_array)

        # invert back
        pil_img = ImageOps.invert(pil_img.convert('RGB'))

        pil_img = pil_img.convert('RGB')

        # add white border
        # new_img = Image.new('RGB', (width + 30, height + 30))
        # new_img.paste(pil_img, (15, 15))
        #
        # pil_img = new_img

        pil_img = ImageOps.expand(pil_img, border=BORDER_WIDTH, fill='white')

        # img = np.array(pil_img)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        # d = pytesseract.image_to_data(pil_img, output_type=Output.DICT)

        s = pytesseract.image_to_string(pil_img, lang='eng').strip()

        # img = np.array(pil_img)
        #
        # img = img[:, :, ::-1].copy()
        #
        # # pprint(d)
        # n_boxes = len(d['level'])
        # for i in range(n_boxes):
        #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        if s:
            print(s)

            return True, k

        else:
            return False, k

    except Exception as e:
        print(e)
        logging.exception(e)
        raise e


def row_2_calc(row, orig_pil_img, k):

    try:

        the_color = row[1]

        distance_dict = {}

        pil_img = copy.deepcopy(orig_pil_img)

        width = pil_img.size[0]
        height = pil_img.size[1]

        pil_img = pil_img.resize(SCALING_FACTOR * np.array((width, height)), Image.BICUBIC)

        replace_color = WHITE_PIXEL

        should_revert = False

        if white_or_black(the_color):
            replace_color = BLACK_PIXEL
            should_revert = True

        # exit()

        width = pil_img.size[0]
        height = pil_img.size[1]

        pil_img_array = np.array(pil_img)

        cp = ColorProcesser()

        color_distances = cp.array_color_distance(the_color, pil_img_array)

        del cp

        pil_img_array[color_distances > COLOR_DISTANCE_THRESHOLD] = replace_color

        pil_img = Image.fromarray(pil_img_array)

        if should_revert:
            pil_img = ImageOps.invert(pil_img)

        pil_img.save("temp1/{0}.jpg".format(uuid.uuid4()))

        pil_img = pil_img.convert('1', dither=Image.NONE)  # .convert('RGB')

        # invert to remove clusters
        pil_img = ImageOps.invert(pil_img.convert('RGB')).convert('1', dither=Image.NONE).convert('RGB')

        # # add white border
        # new_img = Image.new('RGB', (width + 30, height + 30))
        # new_img.paste(pil_img, (15, 15))
        #
        # pil_img = new_img

        # remove noise
        # pil_img = pil_img.filter(ImageFilter.BLUR)
        # pil_img = pil_img.filter(ImageFilter.MinFilter(1))
        # pil_img = pil_img.filter(ImageFilter.MinFilter)

        # pil_img = pil_img.filter(ImageFilter.MedianFilter())
        # enhancer = ImageEnhance.Contrast(pil_img)
        # pil_img = enhancer.enhance(2)

        pil_img = pil_img.convert('1', dither=Image.NONE)

        # remove large clusters
        pil_img_array = np.array(pil_img)

        lw, num = measurements.label(pil_img_array)

        min_cluster_count = LARGE_CLUSTER_RATIO * height * width

        pil_img_array = np.array(pil_img)

        for elem in list(set(list(lw.flatten()))):

            # elem_count = len([item for item in list(lw.flatten()) if item == elem])

            # elem_count = list(lw.flatten()).count(elem)

            elem_count = np.count_nonzero(lw == elem)

            # print(elem_count / width / height)

            if elem_count > min_cluster_count:

                pil_img_array[lw == elem] = 0

        pil_img = Image.fromarray(pil_img_array)

        # invert back
        pil_img = ImageOps.invert(pil_img.convert('RGB'))

        pil_img = pil_img.convert('RGB')

        # add white border
        # new_img = Image.new('RGB', (width + 30, height + 30))
        # new_img.paste(pil_img, (15, 15))
        #
        # pil_img = new_img

        # img = np.array(pil_img)
        #
        # cv2.imshow('img', img)
        # cv2.waitKey(1)

        pil_img.save("temp/{0}.jpg".format(uuid.uuid4()))

        pil_img = ImageOps.expand(pil_img, border=15, fill='white')

        d = pytesseract.image_to_data(pil_img, output_type=Output.DICT)

        s = pytesseract.image_to_string(pil_img, lang='eng').strip()

        return d, s, k

    except Exception as e:
        logging.exception(e)
        print(e)
        raise e


def row_1_iterator(color_list, orig_pil_img):

    allres = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for k in range(len(color_list)):

            row = color_list[k]

            # orig_row = copy.deepcopy(row)

            # res = row_1_calc(row, orig_pil_img, k)

            futures.append(executor.submit(row_1_calc, row, orig_pil_img, k))

            # allres.append(res)

        for future in futures:

            res = future.result()

            allres.append(res)

    # for k in range(len(color_list)):
    #
    #     row = color_list[k]
    #
    #     res = row_1_calc(row, orig_pil_img, k)
    #
    #     allres.append(res)

    for res in allres:

        k = res[1]

        orig_row = copy.deepcopy(color_list[k])

        if not res[0]:
            continue

        yield orig_row


def row_2_iterator(new_rows, orig_pil_img, width, height):

    allres = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for k in range(len(new_rows)):

            row = new_rows[k]

            # res = row_2_calc(row, orig_pil_img, k)

            futures.append(executor.submit(row_2_calc, row, orig_pil_img, k))

            # allres.append(res)

        for future in futures:

            res = future.result()

            allres.append(res)

    for res in allres:

        d, s, k = res

        # row = new_rows[k]

        if not s:
            continue

        # print(s)

        # new_rows.append(row)

        # img = np.array(pil_img)

        # img = np.array(orig_pil_img)
        #
        # img = img[:, :, ::-1].copy()

        # pprint(d)
        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > MIN_CONFIDENCE:
                (x, y, w, h) = (d['left'][i] - BORDER_WIDTH, d['top'][i] - BORDER_WIDTH, d['width'][i], d['height'][i])
                # cv2.rectangle(img, (x // SCALING_FACTOR, y // SCALING_FACTOR), ((x + w) // SCALING_FACTOR, (y + h) // SCALING_FACTOR), (0, 255, 0), 2)
                # print(d['text'][i])
                print("{0}             {1} {2} {3} {4}".format(d['text'][i], x // SCALING_FACTOR, y // SCALING_FACTOR, (x + w) // SCALING_FACTOR, (y + h) // SCALING_FACTOR))

                res_tuple = (d['conf'][i], d['text'][i].strip(), x // SCALING_FACTOR, y // SCALING_FACTOR,
                             (x + w) // SCALING_FACTOR, (y + h) // SCALING_FACTOR, 10000 * k + i)

                the_area = w * h / (SCALING_FACTOR**2)

                the_area_ratio = the_area / (width * height)

                if the_area_ratio > MAX_WORD_AREA_RATIO:
                    print("Discarding {0} because of area ratio {1}".format(res_tuple, the_area_ratio))
                    continue

                the_str = d['text'][i].strip()

                if not bool(re.findall(r'[A-z0-9%]+', the_str)):
                    print("Discarding {0} because it does not have at least one needed character.".format(res_tuple))
                    continue

                # res_tuples.append(res_tuple)

                # res_tuple[1] = res_tuple[1].strip()

                if not bool(res_tuple[1]):
                    continue

                yield res_tuple


def step3_calc(elem, pc):

    pre_p1 = tuple(elem[0][2:6])
    pre_p2 = tuple(elem[1][2:6])

    # p1 = ((pre_p1[0], pre_p1[1]), (pre_p1[0], pre_p1[3]), (pre_p1[2], pre_p1[3]), (pre_p1[2], pre_p1[1]))
    # p2 = ((pre_p2[0], pre_p2[1]), (pre_p2[0], pre_p2[3]), (pre_p2[2], pre_p2[3]), (pre_p2[2], pre_p2[1]))

    p1 = rect_from_pre(pre_p1)
    p2 = rect_from_pre(pre_p2)

    # p1_height = pre_p1[3] - pre_p1[1]
    # p2_height = pre_p2[3] - pre_p2[1]
    #
    # p_height = min(p1_height, p2_height)
    #
    # max_word_dist = MAX_WORD_DISTANCE_RATIO * p_height

    # print("p1:")
    # print(p1)

    # pc = PolygonCalc()

    # min_dist = min_poly_distance(p1, p2)
    min_dist = pc.min_poly_distance(p1, p2)

    # area_ratio = poly_intersection_area_ratio(p1, p2)
    area_ratio = pc.poly_intersection_area_ratio(p1, p2)

    # if 'Development:' in [elem[0][1], elem[1][1]]:
    #     if min_dist == 0:
    #         print("min_dist: {0} - area_ratio: {1} - p1: {2} - p2: {3} - t1 {4} - t2 {5}".format(min_dist, area_ratio, p1, p2, elem[0][1], elem[1][1]))

    if all([min_dist == 0, area_ratio > MIN_INTERSECTION_AREA_RATIO]):
        return elem

    else:
        return None


def step3_iterator(comb):

    allres = []

    pc = PolygonCalc()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = []

        for elem in comb:

            futures.append(executor.submit(step3_calc, elem, pc))

        for future in futures:

            res = future.result()

            if res is None:
                continue

            allres.append(res)

    del pc

    for elem in allres:

        yield elem


def main(path):

    clean_folder_contents('temp1')
    clean_folder_contents('temp')

    start_time = datetime.now()

    print("START TIME: {0}".format(start_time))

    # previous_pil_img = Image.open('/home/elias/pdf_images/image-024.jpg')

    previous_pil_img = Image.open(path)

    the_size = previous_pil_img.size

    pix_count = the_size[0] * the_size[1]

    scaling_factor = np.sqrt(TARGET_PIXEL_SIZE / pix_count)

    newsize = (int(the_size[0] * scaling_factor), int(the_size[1] * scaling_factor))

    if scaling_factor < 1.0:

        previous_pil_img = previous_pil_img.resize(newsize, resample=Image.BICUBIC)

    pil_img = previous_pil_img.convert('P', dither=Image.NONE, palette=Image.ADAPTIVE, colors=COLORS_NUM).convert('RGB')

    enhancer = ImageEnhance.Contrast(pil_img)
    pil_img = enhancer.enhance(CONTRAST_FACTOR)

    img = np.array(pil_img)

    cv2.imshow('img', img)
    cv2.waitKey(3)

    orig_pil_img = copy.deepcopy(pil_img)

    color_list = pil_img.getcolors(maxcolors=1000)

    color_list.sort(key=lambda x: x[0], reverse=True)

    print("len: {0}".format(len(color_list)))

    # color_list = color_list[0:20]

    new_rows = []

    print("Starting step 1...")

    new_rows = list(row_1_iterator(color_list, orig_pil_img))

    # img = np.array(pil_img)
    #
    # img = img[:, :, ::-1].copy()
    #
    # # pprint(d)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    print("Starting with step 2...")

    res_tuples = list(row_2_iterator(new_rows, orig_pil_img, width=the_size[0], height=the_size[1]))

    pprint(res_tuples)

    print("Starting with step 3...")

    comb = list(itertools.combinations(res_tuples, 2))

    # list for same word detections
    L = list(step3_iterator(comb))

    for res_tuple in res_tuples:

        L.append((res_tuple, res_tuple))

    # list for same paragraph detections
    L2 = []

    zero_grouped_tuples = group_pairs_to_nested_list(L)

    old_zero_grouped_tuples = copy.deepcopy(zero_grouped_tuples)

    zero_grouped_tuples = []

    for a_group in old_zero_grouped_tuples:

        avg_len = np.median([len(elem[1]) for elem in a_group])

        new_group = [elem for elem in a_group if len(elem[1]) >= avg_len]

        if bool(new_group):

            zero_grouped_tuples.append(new_group)

    # word_grouped_tuples = group_pairs_to_nested_list(L2)

    print("zero_grouped_tuples:")
    pprint(zero_grouped_tuples)

    # print("word_grouped_tuples:")
    # pprint(word_grouped_tuples)

    blocked_i_values = [elem[-1] for el in zero_grouped_tuples for elem in el]

    print("Starting with step 4...")

    filtered_res_tuples = []

    for elem in zero_grouped_tuples:

        # texts_array = [item[1] for item in elem]

        elem.sort(key=lambda x: x[0], reverse=True)

        max_text = max(elem, key=lambda x: sum([el[0]**2 for el in elem if el[1] == x[1]]))

        print("max_text: {0}".format(max_text))

        filtered_res_tuples.append(max_text)

    # for elem in res_tuples:
    #
    #     if elem[-1] not in blocked_i_values:
    #
    #         filtered_res_tuples.append(elem)

    print("filtered_res_tuples:")
    pprint(filtered_res_tuples)

    print("Starting with step 5...")

    comb = itertools.combinations(filtered_res_tuples, 2)

    pc = PolygonCalc()

    for elem in comb:

        pre_p1 = tuple(elem[0][2:6])
        pre_p2 = tuple(elem[1][2:6])

        # p1 = ((pre_p1[0], pre_p1[1]), (pre_p1[0], pre_p1[3]), (pre_p1[2], pre_p1[3]), (pre_p1[2], pre_p1[1]))
        # p2 = ((pre_p2[0], pre_p2[1]), (pre_p2[0], pre_p2[3]), (pre_p2[2], pre_p2[3]), (pre_p2[2], pre_p2[1]))

        p1 = rect_from_pre(pre_p1)
        p2 = rect_from_pre(pre_p2)

        p1_height = pre_p1[3] - pre_p1[1]
        p2_height = pre_p2[3] - pre_p2[1]

        p_height = min(p1_height, p2_height)

        max_word_dist = MAX_WORD_DISTANCE_RATIO * p_height

        # print("p1:")
        # print(p1)

        # min_dist = min_poly_distance(p1, p2)

        # pc = PolygonCalc()

        min_dist = pc.min_poly_distance(p1, p2)

        # area_ratio = poly_intersection_area_ratio(p1, p2)

        # if all([min_dist == 0, area_ratio > MIN_INTERSECTION_AREA_RATIO]):
        #
        #     L.append(elem)

        if min_dist < max_word_dist:

            L2.append(elem)

    del pc

    for elem in filtered_res_tuples:

        L2.append((elem, elem))

    word_grouped_tuples = group_pairs_to_nested_list(L2)

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
                y2 = pre_p1[3]

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

        s = re.sub(r'^[^A-z0-9]+', '', s)

        s = re.sub(r':$', '', s)

        s = s.strip()

        re_res = re.findall(r'[A-z0-9]', s)

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



