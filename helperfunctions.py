import logging
import copy
import numpy as np
import re
import itertools
import cvxpy
from polygon_calc_wrapper import PolygonCalc
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

    x_0 = (p1[0] + p1[2])/2

    y_0 = (p1[1] + p1[3])/2

    return x_0, y_0


# same as above method, but with polygons
def connect_polygon_cloud_2(points1, points2):

    # points = points1 + points2

    points = np.array(list(points1) + list(points2))

    p1_range = range(len(points1))

    # p2_range = range(len(points1), len(points))

    N = points.shape[0]

    # I, J = np.indices((N, N))
    # d = np.sqrt(sum((points[I, i] - points[J, i]) ** 2 for i in range(points.shape[1])))

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

    dist = P.solve()

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

    print("res: {0}".format(res))

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


# get dominant color from cv2 image
# !!! ALL THE DOMINANT COLOR FUNCTIONS RETURN BGR VALUES, NOT RGB !!!
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
def get_cv2_dominant_color_3(img, colors_num, reshape=True):

    if reshape:

        shape = img.shape

        img = img.reshape(scipy.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(img, colors_num)

    vecs, dist = scipy.cluster.vq.vq(img, codes)

    counts, bins = scipy.histogram(vecs, len(codes))

    index_max = scipy.argmax(counts)

    peak = codes[index_max]

    return peak


# get dominant color from cv2 image using cie lab space
def get_cv2_dominant_color_4(img, colors_num):

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

    dominant_lab_color = get_cv2_dominant_color_3(new_img, colors_num, reshape=False)

    # return dominant_lab_color

    dominant_srgb_color = convert_color(LabColor(*dominant_lab_color), sRGBColor)

    # sRGBColor.clamped_rgb_b

    dominant_rgb_color = tuple(np.array([dominant_srgb_color.clamped_rgb_r,
                                         dominant_srgb_color.clamped_rgb_g,
                                         dominant_srgb_color.clamped_rgb_b]) * 255.0)

    return dominant_rgb_color


# calculate average color of cv2 image
def get_cv2_dominant_color_5(img):

    return list(img.mean(axis=0).mean(axis=0))
