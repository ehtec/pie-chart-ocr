import logging

import cv2
import cv2.cv2
import numpy as np
import re
import pytesseract
from pytesseract import image_to_string
from PIL import Image, ImageOps, ImageEnhance
from pprint import pprint
# from colorthief import ColorThief, MMCQ
from PIL import Image, ImageFilter
from helperfunctions import get_cv2_dominant_color, get_cv2_dominant_color_2, get_cv2_dominant_color_3,\
    get_cv2_dominant_color_4, get_cv2_dominant_color_5
from polygon_helperfunctions import group_words
from polygon_calc_wrapper import PolygonCalc
from pytesseract import Output
import pytesseract
from color_processer_wrapper import ColorProcesser


# maximum ratio of the total area a mser box might take
MAX_MSER_BOX_RATIO = 0.01

# maximum ratio of the total height a mser box might take
MAX_MSER_BOX_HEIGHT = 0.2

# maximum ratio of the total width a mser box might take
MAX_MSER_BOX_WIDTH = 0.1

# absolute minimum mser box height in pixels
ABSOLUTE_MIN_MSER_BOX_HEIGHT = 8  # 8

# absolute minimum mser box width in pixels
ABSOLUTE_MIN_MSER_BOX_WIDTH = 3  # 8

# number of colors for adaptive palette when finding dominant color
COLORS_NUM = 3

# padding height when finding surrounding color. The padding height is applied between the measurement height and the
#   object
PADDING_HEIGHT = 10

# measurement height when finding surrounding color
MEASUREMENT_HEIGHT = 10

# padding width in pixels of border before OCR
BORDER_WIDTH = 15

# outer padding pixels border of full picture. Should be at least MEASUREMENT_HEIGHT + PADDING_HEIGHT
OUTER_BORDER_WIDTH = 22

# color distance difference (CIE2000) for background color check
BG_COLOR_DISTANCE = 13.0

# set minimum confidence threshold
MIN_CONFIDENCE = 20

# maximum ratio between distance and the size_metric in group_elements when grouping letters
MAX_LETTER_DISTANCE_RATIO = 0.8  # 0.8  # 3.0

# same line overlap ratio
SLOV_RATIO = 0.92

# SCALING_FACTOR = 2


# get BGR background color of a cropped cv2 text image (by using space slightly above and below)
def get_text_background_color(img, x, y, w, h, padding_height=PADDING_HEIGHT, measurement_height=MEASUREMENT_HEIGHT):

    # total margin height
    tmh = PADDING_HEIGHT + MEASUREMENT_HEIGHT

    # img2 = img[y - tmh: y + h + tmh, x: x + w]

    img2_1 = img[y - tmh: y - PADDING_HEIGHT, x: x + w]

    img2_2 = img[y + h + PADDING_HEIGHT: y + h + tmh, x: x + w]

    print(img2_1.shape)
    print(img2_2.shape)

    img_sum = np.append(img2_1, img2_2, axis=0)

    print(img_sum.shape)

    dominant_color = get_cv2_dominant_color_3(img_sum, colors_num=COLORS_NUM)

    return dominant_color


# get the BGR background color of a cropped cv2 image (by using the outmost surroundings)
def get_background_color(img, padding_height=PADDING_HEIGHT, measurement_height=MEASUREMENT_HEIGHT):

    # total margin height
    tmh = padding_height + measurement_height

    image_list = []

    height = img.shape[0]
    width = img.shape[1]

    coord_list = [
        (padding_height, padding_height, width - 2 * padding_height, measurement_height),
        (width - tmh, tmh, measurement_height, height - 2 * tmh),
        (padding_height, height - tmh, width - 2 * padding_height, measurement_height),
        (padding_height, tmh, measurement_height, height - 2 * tmh)
    ]

    for x, y, w, h in coord_list:

        cropped_image = img[y: y + h, x: x + w]

        print("Shape before reshape: {0}".format(cropped_image.shape))

        cropped_image = cropped_image.reshape((-1, 1, 3))

        print("Shape after reshape: {0}".format(cropped_image.shape))

        image_list.append(cropped_image)

    img_sum = np.concatenate(image_list, axis=0)

    print("sum shape: {0}".format(img_sum.shape))

    dominant_color = get_cv2_dominant_color_3(img_sum, colors_num=COLORS_NUM)

    return dominant_color


def main(path):

    # color_thief = ColorThief(path)

    # dominant_color = color_thief.get_color(quality=1)
    #
    # print("Dominant color: {0}".format(dominant_color))

    # Your image path i-e receipt path
    img = cv2.imread(path)

    full_background_color = get_background_color(img)

    print("full_background_color: {0}".format(full_background_color))

    img = cv2.copyMakeBorder(img, OUTER_BORDER_WIDTH, OUTER_BORDER_WIDTH, OUTER_BORDER_WIDTH, OUTER_BORDER_WIDTH,
                             cv2.BORDER_CONSTANT, value=full_background_color)

    shape = img.shape

    height, width = shape[0], shape[1]

    # dominant_color = get_cv2_dominant_color(img, colors_num=COLORS_NUM)
    #
    # print("Dominant color: {0}".format(dominant_color))

    # img = cv2.resize(img, (img.shape[1] * SCALING_FACTOR, img.shape[0] * SCALING_FACTOR), interpolation=cv2.INTER_AREA)

    # total_area = img.shape[0] * img.shape[1]

    total_area = height * width

    # print("MAX AREA: {0}".format(total_area * MAX_MSER_BOX_RATIO))

    # Create MSER object
    mser = cv2.MSER_create(max_area=int(MAX_MSER_BOX_RATIO * total_area))  # delta = 20

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    # detect regions in gray scale image
    regions, bounding_boxes = mser.detectRegions(gray)

    print("NUMBER OF BOUNDING BOXES: {0}".format(len(bounding_boxes)))

    print(img.shape)

    filtered_res_tuples = []

    for box in bounding_boxes:

        x, y, w, h = box

        # print(box)
        #
        # # total margin height
        # tmh = PADDING_HEIGHT + MEASUREMENT_HEIGHT
        #
        # img2 = img[y - tmh: y + h + tmh, x: x + w]
        #
        # img2_1 = img[y - tmh: y - PADDING_HEIGHT, x: x + w]
        #
        # img2_2 = img[y + h + PADDING_HEIGHT: y + h + tmh, x: x + w]
        #
        # print(img2_1.shape)
        # print(img2_2.shape)
        #
        # img_sum = np.append(img2_1, img2_2, axis=0)
        #
        # print(img_sum.shape)
        #
        # dominant_color_1 = get_cv2_dominant_color(img_sum, colors_num=COLORS_NUM)
        # dominant_color_2 = get_cv2_dominant_color_2(img_sum, colors_num=COLORS_NUM)
        # dominant_color_3 = get_cv2_dominant_color_3(img_sum, colors_num=COLORS_NUM)
        # dominant_color_4 = get_cv2_dominant_color_4(img_sum, colors_num=COLORS_NUM)
        # dominant_color_5 = get_cv2_dominant_color_5(img_sum)
        #
        # print("Dominant color 1: {0}".format(dominant_color_1))
        # print("Dominant color 2: {0}".format(dominant_color_2))
        # print("Dominant color 3: {0}".format(dominant_color_3))
        # print("Dominant color 4: {0}".format(dominant_color_4))
        # print("Dominant color 5: {0}".format(dominant_color_5))
        #
        # cv2.imshow('img2', img2)
        #
        # # cv2.waitKey(0)
        #
        # cv2.imshow('img_sum', img_sum)
        #
        # cv2.waitKey(0)

        if x < OUTER_BORDER_WIDTH:
            continue

        if y < OUTER_BORDER_WIDTH:
            continue

        if x + w > width - OUTER_BORDER_WIDTH:
            continue

        if y + h > height - OUTER_BORDER_WIDTH:
            continue

        area = w * h

        if area / total_area > MAX_MSER_BOX_RATIO:
            continue

        if w / width > MAX_MSER_BOX_WIDTH:
            continue

        if h / height > MAX_MSER_BOX_HEIGHT:
            continue

        if w < ABSOLUTE_MIN_MSER_BOX_WIDTH:
            continue

        if h < ABSOLUTE_MIN_MSER_BOX_HEIGHT:
            continue

        # print(h/height)
        #
        # print(box)

        # cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        filtered_res_tuples.append((x, y, x + w, y + h))

    # cv2.imshow('vis', vis)
    #
    # cv2.waitKey(0)
    #
    # vis = img.copy()

    # word_grouped_tuples = group_words(filtered_res_tuples)

    pc = PolygonCalc()

    word_grouped_tuples = pc.group_elements(filtered_res_tuples, MAX_LETTER_DISTANCE_RATIO, SLOV_RATIO)

    # pprint(word_grouped_tuples)

    print(len(word_grouped_tuples))

    res_tuples = []

    k = 0

    for word in word_grouped_tuples:

        x1 = min([elem[0] for elem in word])

        x2 = max([elem[2] for elem in word])

        y1 = min([elem[1] for elem in word])

        y2 = max([elem[3] for elem in word])

        # for elem in word:
        #     cv2.rectangle(vis, (elem[0], elem[1]), (elem[2], elem[3]), (0, 255, 0), 1)
        #
        # cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cropped_img = img[y1: y2, x1: x2]

        background_color = get_text_background_color(img, x1, y1, x2 - x1, y2 - y1)

        print("background_color: {0}".format(background_color))

        cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        cp = ColorProcesser()

        the_color = (background_color[2], background_color[1], background_color[0])

        color_distances = cp.array_color_distance(the_color, cropped_img_rgb)

        cropped_img_bin = cropped_img.copy()  # np.zeros(cropped_img.shape)

        cropped_img_bin[color_distances <= BG_COLOR_DISTANCE] = (255, 255, 255)  # set background to white

        cropped_img_bin[color_distances > BG_COLOR_DISTANCE] = (0, 0, 0)  # set foreground to black

        pil_img = Image.fromarray(cropped_img_bin)

        # FILTERS START
        # pil_img = pil_img.filter(ImageFilter.BLUR)
        # pil_img = pil_img.filter(ImageFilter.MinFilter(3))
        # pil_img = pil_img.filter(ImageFilter.MinFilter)

        # pil_img = pil_img.filter(ImageFilter.MedianFilter())
        # enhancer = ImageEnhance.Contrast(pil_img)
        # pil_img = enhancer.enhance(3)
        # FILTERS END

        cropped_img_bin = np.array(pil_img)

        # pprint(cropped_img_bin)

        print("cropped_img.shape: {0}".format(cropped_img.shape))
        print("color_distances.shape: {0}".format(color_distances.shape))
        print("cropped_img_bin.shape: {0}".format(cropped_img_bin.shape))

        im_gray = cv2.cvtColor(cropped_img_bin, cv2.COLOR_BGR2GRAY)

        th, im_gray_th_otsu = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)

        # pprint(im_gray_th_otsu)

        im_gray_th_otsu = cv2.copyMakeBorder(im_gray_th_otsu, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH, BORDER_WIDTH,
                                             cv2.BORDER_CONSTANT, value=(255, 255, 255))

        print("im_gray_th_otsu.shape: {0}".format(im_gray_th_otsu.shape))

        d = pytesseract.image_to_data(im_gray_th_otsu, lang='eng', output_type=Output.DICT, config='--psm 7')

        # res_tuples.append(d)

        # for elem in word:
        #     cv2.rectangle(im_gray_th_otsu, (elem[0] - x1, elem[1] - y1), (elem[2], elem[3]), (0, 255, 0), 1)

        # cv2.rectangle(im_gray_th_otsu, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.imshow('cropped', im_gray_th_otsu)

        cv2.waitKey(0)

        # for box in word:
        #
        #     x, y, a, b = box
        #
        #     w = a - x
        #     h = b - y
        #
        #     # cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

        n_boxes = len(d['text'])
        for i in range(n_boxes):
            if int(d['conf'][i]) > MIN_CONFIDENCE:
                (x, y, w, h) = (d['left'][i] - BORDER_WIDTH + x1,
                                d['top'][i] - BORDER_WIDTH + y1,
                                d['width'][i],
                                d['height'][i])

                # cv2.rectangle(img, (x // SCALING_FACTOR, y // SCALING_FACTOR), ((x + w) // SCALING_FACTOR, (y + h) // SCALING_FACTOR), (0, 255, 0), 2)
                # print(d['text'][i])
                print("{0}             {1} {2} {3} {4}".format(d['text'][i], x, y, (x + w), (y + h)))

                res_tuple = (d['conf'][i], d['text'][i].strip(), x, y, (x + w), (y + h), 10000 * k + i)

                the_str = d['text'][i].strip()

                if not bool(re.findall(r'[A-z0-9%]+', the_str)):
                    print("Discarding {0} because it does not have at least one needed character.".format(res_tuple))
                    continue

                # res_tuples.append(res_tuple)

                # res_tuple[1] = res_tuple[1].strip()

                if not bool(res_tuple[1]):
                    continue

                res_tuples.append(res_tuple)

        k += 1

    vis = img.copy()

    for res_tuple in res_tuples:

        x1 = res_tuple[2]
        y1 = res_tuple[3]
        x2 = res_tuple[4]
        y2 = res_tuple[5]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # cv2.imshow('vis', vis)
    #
    # cv2.waitKey(0)

    pprint(res_tuples)

    # print("")

    # pprint([el['text'] for el in res_tuples if bool(el['text'])])

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]  # regions

    # # cv2.polylines(vis, regions, 1, (0, 255, 0))
    # cv2.polylines(vis, hulls, 1, (0, 255, 0))
    #
    # cv2.imshow('img', vis)
    #
    # cv2.waitKey(0)

    mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

    new_hulls = []

    areas = []

    for contour in hulls:  # hulls

        area = cv2.contourArea(contour)

        # print(area)

        areas.append(area)

        if area > total_area * MAX_MSER_BOX_RATIO:
            continue

        new_hulls.append(contour)

        cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
        # cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

    areas.sort(reverse=True)

    # for i in range(5):
    #     print(areas[i])

    # # cv2.polylines(vis, regions, 1, (0, 255, 0))
    # cv2.polylines(vis, new_hulls, 1, (0, 255, 0))

    cv2.imwrite('temp2/mser_result.png', vis)

    cv2.imshow('img', vis)

    cv2.waitKey(0)

    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    # pprint(mask)

    # pprint(list(set(list(mask.flatten()))))

    # pil_img = Image.fromarray(img)

    # pprint(np.array(mask).shape)

    # mask[mask == 0] = np.array([0, 0, 0])
    # mask[mask == 255] = np.array([255, 255, 255])

    mask = np.repeat(mask, repeats=3, axis=2)

    # pprint(np.array(mask).shape)

    # pil_img_array = np.array(pil_img)

    text_only[mask == 0] = 255  # np.array([255, 255, 255])

    # text_only[mask == (255, 255, 255)] = (255, 255, 255)

    cv2.imwrite('temp2/mser_result_text_only.png', text_only)

    cv2.imshow("text only", text_only)

    cv2.waitKey(0)


def main2(path):

    image_obj = Image.open(path)

    rgb = cv2.imread(path)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    # threshold the image
    _, bw = cv2.threshold(small, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # get horizontal mask of large size since text are horizontal components
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    # find all the contours
    contours, hierarchy, = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Segment the text lines
    counter = 0
    array_of_texts = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        cropped_image = image_obj.crop((x - 10, y, x + w + 10, y + h))
        str_store = re.sub(r'([^\s\w]|_)+', '', image_to_string(cropped_image))
        array_of_texts.append(str_store)
        counter += 1

    print(array_of_texts)


