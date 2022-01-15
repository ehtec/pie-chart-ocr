import logging
import copy
import numpy as np
import re
import itertools
import cvxpy
from .polygon_calc_wrapper import PolygonCalc
import os
import shutil
import hashlib
from PIL import Image
from colorthief import MMCQ
import scipy
import scipy.misc
import scipy.cluster
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from sklearn.cluster import DBSCAN
import cv2
import operator
from collections.abc import MutableMapping
import json
import math


# checks if a string is a valid float
def isfloat(value):
    try:

        float(value)

        if math.isnan(value):
            return False

        if np.isnan(value):
            return False

        return True

    except (ValueError, TypeError):
        return False


# equivalent to rm -rf
def clean_folder_contents(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            logging.exception(e)


# helper for grouping. Used by pie_chart_ocr.py
def partition(pred, iterable):
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


# group pairs to a nested list. used in pie_chart_ocr.py
def group_pairs_to_nested_list(L):

    # print(L)

    zero_grouped_tuples = []

    for a, b in L:
        unrelated, related = partition(lambda group: any(aa == a or bb == b or aa == b or bb == a for aa, bb in group),
                                       zero_grouped_tuples)
        zero_grouped_tuples = [*unrelated, sum(related, [(a, b)])]

    old_zero_grouped_tuples = copy.deepcopy(zero_grouped_tuples)
    # zero_grouped_tuples = []

    for m in range(len(old_zero_grouped_tuples)):

        elem = old_zero_grouped_tuples[m]

        zero_grouped_tuples[m] = []

        for el in elem:

            for item in el:

                if item not in zero_grouped_tuples[m]:

                    zero_grouped_tuples[m].append(item)

    return zero_grouped_tuples


# calculate center of a rectangle of the form (x, y, x+w, y+h)
def pre_rectangle_center(p1):

    x_0 = (p1[0] + p1[2]) / 2

    y_0 = (p1[1] + p1[3]) / 2

    return x_0, y_0


# calculate distance matrix, used for polygon cloud connections
def calculate_distance_matrix(points):

    d = np.zeros((len(points), len(points)))

    pc = PolygonCalc()

    for i in range(len(points)):

        for j in range(len(points)):

            if i == j:
                d[i, j] = 0.0

            elif d[j, i] != 0.0:
                d[i, j] = d[j, i]

            else:

                p1 = points[i]
                p2 = points[j]

                d[i, j] = pc.min_poly_distance(p1, p2)

    del pc

    return d


# connect polygons to each other, targeting minimum connection length. Used to map percentage numbers to captions.
# You can supply a pre-calculated distance matrix as d.
def connect_polygon_cloud_2(points1, points2, d=None):

    # points = points1 + points2

    points = np.array(list(points1) + list(points2), dtype=object)

    p1_range = range(len(points1))

    # p2_range = range(len(points1), len(points))

    N = points.shape[0]

    # I, J = np.indices((N, N))
    # d = np.sqrt(sum((points[I, i] - points[J, i]) ** 2 for i in range(points.shape[1])))

    if d is None:
        d = calculate_distance_matrix(points)

    use = cvxpy.Variable((N, N), integer=True)
    # each entry use[i,j] indicates that the point i is connected to point j
    # each pair may count 0 or 1 times
    constraints = [use >= 0, use <= 1]
    # point i must be used in at most one connection
    constraints += [sum(use[i, :]) + sum(use[:, i]) <= 1 for i in range(N)]
    # at least floor(N/2) connections must be presented
    constraints += [sum(use[i, j] for i in range(N) for j in range(N)) >= min(len(points1), len(points2))]

    constraints += [sum([use[i, j] for i in range(N) for j in range(N) if ((i in p1_range) == (j in p1_range))]) == 0]

    # let the solver  to handle the problem
    P = cvxpy.Problem(cvxpy.Minimize(sum(use[i, j] * d[i, j] for i in range(N) for j in range(N))), constraints)

    P.solve()

    return use.value


# detect if a number is a percentage and return as ratio. Can be used to remove false ocr characters and spaces after
#   OCR.
def detect_percentage(s):

    res = re.findall(r'[A-Za-z]', s)

    if bool(res):
        logging.debug("Letters found in string {0}".format(s))
        return None

    res = re.findall(r'[0-9]', s)

    if not bool(res):
        logging.debug("No digit found in string {0}".format(s))
        return None

    res = re.findall(r'[%]', s)

    if not bool(res):
        logging.debug("No percent symbol found in string {0}".format(s))
        return None

    s = s.split('%')[0].strip()

    s = s.replace(',', '.')

    res = re.findall(r'\d+\.?\d*', s)

    logging.debug("res: {0}".format(res))

    if not bool(res):
        logging.debug("No float found in string {0}".format(s))
        return None

    fs = str(max(res, key=len))

    return float(fs) / 100.0


# return 4 corner points rectangle from format (x, y, x+w, y+h)
def rect_from_pre(pre_p1):

    p1 = ((pre_p1[0], pre_p1[1]), (pre_p1[0], pre_p1[3]), (pre_p1[2], pre_p1[3]), (pre_p1[2], pre_p1[1]))

    return p1


# compute sha256 hash of file
def hash_file(path):

    with open(path, 'rb') as bytesfile:
        the_bytes = bytesfile.read()

    h = hashlib.sha256()

    h.update(the_bytes)

    return h.hexdigest()


# convert list of floats to integers
def integerize(li):

    return [int(el) for el in li]


# helper to put characters together to words after MSER
def grouper(iterable, interval=2):

    prev = None

    group = []

    for item in iterable:

        if not prev or abs(item[1] - prev[1]) <= interval or abs(item[3] - prev[3] <= interval):
            group.append(item)

        else:
            yield group
            group = [item]

        prev = item

    if group:
        yield group


# !!! ALL THE DOMINANT COLOR FUNCTIONS RETURN BGR VALUES, NOT RGB !!!
# Method number 3 is the best, according to my tests with my use case

# get dominant color from cv2 image by counting
def get_cv2_dominant_color(img, colors_num):

    pil_img = Image.fromarray(img)

    pil_img = pil_img.convert('P', dither=Image.NONE, palette=Image.WEB, colors=colors_num).convert('RGB')
    # pil_img = pil_img.convert('P', dither=Image.NONE, palette=Image.ADAPTIVE, colors=colors_num).convert('RGB')

    color_list = pil_img.getcolors(maxcolors=1000)

    color_list.sort(key=lambda x: x[0], reverse=True)

    dominant_color = color_list[0][1]

    logging.debug("Dominant color: {0}".format(dominant_color))

    return dominant_color


# get dominant color from cv2 image using colorthief
def get_cv2_dominant_color_2(img, colors_num):

    shape = img.shape

    linear_array = img.reshape(shape[0] * shape[1], shape[2])

    valid_pixels = list(linear_array)

    cmap = MMCQ.quantize(valid_pixels, colors_num)

    palette = cmap.palette

    return palette[0]


# get dominant color from cv2 image using K-means clustering
def get_cv2_dominant_color_3(img, colors_num, reshape=True, return_integers=True, return_peak_only=True):

    if reshape:

        shape = img.shape

        img = img.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    try:

        codes, dist = scipy.cluster.vq.kmeans(img, colors_num)

        vecs, dist = scipy.cluster.vq.vq(img, codes)

        counts, bins = scipy.histogram(vecs, len(codes))

    except ValueError as e:
        logging.warning("ValueError encountered when executing get_cv2_dominant_color_3; img: {0}, colors_num: {1}"
                        .format(img, colors_num))
        raise e

    # index_max = scipy.argmax(counts)
    #
    # peak = codes[index_max]
    #
    # if return_integers:
    #     peak = integerize(peak)
    #
    # return peak

    counts_and_codes = list(zip(codes, counts))

    counts_and_codes.sort(reverse=True, key=lambda x: x[1])

    codes_sorted = [el[0] for el in counts_and_codes]

    if return_integers:
        codes_sorted = [integerize(el) for el in codes_sorted]

    if return_peak_only:
        return codes_sorted[0]

    return codes_sorted


# get dominant color from cv2 image using cie lab space
def get_cv2_dominant_color_4(img, colors_num, return_integers=True):

    shape = img.shape

    linear_array = img.reshape(shape[0] * shape[1], shape[2])

    valid_pixels = list(linear_array)

    new_valid_pixels = []

    for c in valid_pixels:

        sc = tuple(np.array(c) / 255.0)

        color_rgb = sRGBColor(*sc)

        color_lab = convert_color(color_rgb, LabColor)

        new_valid_pixels.append([color_lab.lab_l, color_lab.lab_a, color_lab.lab_b])

    new_img = np.array(new_valid_pixels)

    # print(new_valid_pixels)

    # print(new_img.shape)

    dominant_lab_color = get_cv2_dominant_color_3(new_img, colors_num, reshape=False, return_integers=False)

    # return dominant_lab_color

    dominant_srgb_color = convert_color(LabColor(*dominant_lab_color), sRGBColor)

    # sRGBColor.clamped_rgb_b

    dominant_rgb_color = tuple(np.array([dominant_srgb_color.clamped_rgb_r,
                                         dominant_srgb_color.clamped_rgb_g,
                                         dominant_srgb_color.clamped_rgb_b]) * 255.0)

    if return_integers:
        dominant_rgb_color = integerize(dominant_rgb_color)

    return dominant_rgb_color


# calculate average color of cv2 image
def get_cv2_dominant_color_5(img, return_integers=True):

    res = list(img.mean(axis=0).mean(axis=0))

    if return_integers:
        res = integerize(res)

    return res


# get root path
def get_root_path():

    return os.path.dirname(os.path.dirname(__file__))


# cluster 1D array of values by using an absolute deviation
def cluster_abs_1d(input_values, atol):

    if not bool(input_values):
        return []

    sorted_input_values = list(sorted(input_values))

    res_clusters = [[sorted_input_values[0]]]

    for i in range(1, len(sorted_input_values)):

        if sorted_input_values[i] - res_clusters[-1][-1] <= atol:
            res_clusters[-1].append(sorted_input_values[i])

        else:
            res_clusters.append([sorted_input_values[i]])

    res_clusters = list(sorted(res_clusters, key=lambda x: len(x), reverse=True))

    return res_clusters


# cluster 1D array of values by using a relative deviation
def cluster_rel_1d(input_values, rtol):

    if not bool(input_values):
        return []

    sorted_input_values = list(sorted(input_values))

    res_clusters = [[sorted_input_values[0]]]

    for i in range(1, len(sorted_input_values)):

        if (res_clusters[-1][-1] == 0) or (res_clusters[-1][-1] * sorted_input_values[i] < 0):
            res_clusters.append([sorted_input_values[i]])
            continue

        if abs((sorted_input_values[i] / res_clusters[-1][-1]) - 1) <= rtol:
            res_clusters[-1].append(sorted_input_values[i])

        else:
            res_clusters.append([sorted_input_values[i]])

    res_clusters = list(sorted(res_clusters, key=lambda x: len(x), reverse=True))

    return res_clusters


# cluster multidimensional array with the DBSCAN algorithm
def cluster_dbscan(input_array, eps, min_samples=1, input_objects=None):

    empty_return = []
    if input_objects is not None:
        empty_return = [], []

    if input_array is None:
        return empty_return

    if input_objects is not None:
        assert len(input_objects) == len(input_array)

    if isinstance(input_array, np.ndarray):
        parsed_input_array = input_array.tolist()

    else:
        parsed_input_array = input_array

    if not bool(parsed_input_array):
        return empty_return

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(parsed_input_array)

    the_labels = db.labels_.tolist()

    res_clusters = [[] for _ in range(max(the_labels) + 1)]
    obj_clusters = copy.deepcopy(res_clusters)

    for i in range(len(the_labels)):
        j = the_labels[i]
        res_clusters[j].append(parsed_input_array[i])
        if input_objects is not None:
            obj_clusters[j].append(input_objects[i])

    res_clusters = list(sorted(res_clusters, key=lambda x: len(x), reverse=True))
    obj_clusters = list(sorted(obj_clusters, key=lambda x: len(x), reverse=True))

    if input_objects is not None:
        return res_clusters, obj_clusters

    return res_clusters


# get color pixels inside contour of opencv image as 1d array
def get_image_color_pixels(img, contour, erosion_kernel_size, erosion_iterations=1):

    img_mask = np.full(img.shape, 0)
    img_mask = np.ascontiguousarray(img_mask, dtype=np.uint8)

    cv2.drawContours(img_mask, [contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)

    # apply erode filter
    if erosion_iterations > 0:
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=erosion_iterations)

    # vis = img_mask.copy()
    # cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
    # cv2.imshow('vis', vis)
    # cv2.resizeWindow('vis', 800, 800)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    condition = np.where(img_mask == (255, 255, 255))[:-1]

    logging.debug("condition: {0}".format(condition))

    color_pixels = img[condition[0], condition[1]]

    logging.debug("color_pixels: {0}".format(color_pixels))
    logging.info("color_pixels.shape: {0}".format(color_pixels.shape))

    return color_pixels


# check if a color is closer to black or white
# 0: black, 1: white
def white_or_black(pixel):

    luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]

    if luminance < 128:
        return 0

    else:
        return 1


# perform a couple of erosion / dilation operations on an opencv image.
# the operations parameter has the following format: [('erosion' | 'dilation', kernel_size, iterations)]
def erosion_dilation_operations(img, operations):

    img_copy = img.copy()

    for operation_type, kernel_size, iterations in operations:

        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        if operation_type == 'erosion':
            func = cv2.erode

        elif operation_type == 'dilation':
            func = cv2.dilate

        else:
            raise NotImplementedError("Operation {0} is not implemented.".format(operation_type))

        img_copy = func(img_copy, kernel, iterations=iterations)

    return img_copy


# return the middle index from the longest sequence where condition is True
def find_longest_sequence(elements, condition):

    if not bool(elements):
        return None

    r = max((list(y) for (x, y) in itertools.groupby((enumerate(elements)), operator.itemgetter(1)) if condition(x)),
            key=len)

    middle_index = int(sum([el[0] for el in r]) / len(r))

    return middle_index


# remove special characters from beginning of string
def remove_sc_prefix(s):

    sc_regex = r'[A-Za-z0-9].*'

    res = re.findall(sc_regex, s)

    if not bool(res):
        return ''

    return res[0]


# remove special characters from end of string
def remove_sc_suffix(s):

    sc_regex = r'.*[A-Za-z0-9]'

    res = re.findall(sc_regex, s)

    if not bool(res):
        return ''

    return res[0]


# remove a set of keys from a nested dictionary
def delete_keys_from_dict(dictionary, keys):

    if not isinstance(dictionary, MutableMapping):
        return dictionary

    keys_set = set(keys)

    modified_dict = {}

    for key, value in dictionary.items():
        if key not in keys_set:
            if isinstance(value, MutableMapping):
                modified_dict[key] = delete_keys_from_dict(value, keys_set)

            elif isinstance(value, list) or isinstance(value, tuple):
                modified_dict[key] = [delete_keys_from_dict(value[i], keys_set) for i in range(len(value))]

            else:
                modified_dict[key] = value

    return modified_dict


# encoder to dump numpy data types. Use like this: json.dumps(data, cls=NpEncoder)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
