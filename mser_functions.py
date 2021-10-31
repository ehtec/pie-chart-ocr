import cv2
import numpy as np
import re
import pytesseract
from pytesseract import image_to_string
from PIL import Image
from pprint import pprint


# maximum percentage of the total are a mser box might take
MAX_MSER_BOX_RATIO = 0.01

SCALING_FACTOR = 2


def main(path):

    # Your image path i-e receipt path
    img = cv2.imread(path)

    img = cv2.resize(img, (img.shape[1] * SCALING_FACTOR, img.shape[0] * SCALING_FACTOR), interpolation=cv2.INTER_AREA)

    total_area = img.shape[0] * img.shape[1]

    print("MAX AREA: {0}".format(total_area * MAX_MSER_BOX_RATIO))

    # Create MSER object
    mser = cv2.MSER_create(max_area=int(MAX_MSER_BOX_RATIO * total_area))

    # Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    vis = img.copy()

    # detect regions in gray scale image
    regions, bounding_boxes = mser.detectRegions(gray)

    for box in bounding_boxes:

        x, y, w, h = box

        area = w * h

        if area / total_area > MAX_MSER_BOX_RATIO:
            continue

        # cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)

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

    for i in range(5):
        print(areas[i])

    # cv2.polylines(vis, regions, 1, (0, 255, 0))
    cv2.polylines(vis, new_hulls, 1, (0, 255, 0))

    cv2.imshow('img', vis)

    cv2.waitKey(0)

    # this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

    # pprint(mask)

    pprint(list(set(list(mask.flatten()))))

    pil_img = Image.fromarray(img)

    pprint(np.array(mask).shape)

    # mask[mask == 0] = np.array([0, 0, 0])
    # mask[mask == 255] = np.array([255, 255, 255])

    mask = np.repeat(mask, repeats=3, axis=2)

    pprint(np.array(mask).shape)

    # pil_img_array = np.array(pil_img)

    text_only[mask == 0] = 255  # np.array([255, 255, 255])

    # text_only[mask == (255, 255, 255)] = (255, 255, 255)

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


