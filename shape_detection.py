import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


# max area ratio of a shape
MAX_SHAPE_AREA_RATIO = 0.40

# minimum absolute area of a shape in pixels
MIN_SHAPE_AREA = 15

# maximum deviation for shape detection (square, rectangle,...)
MAX_DEVIATION = 0.05

# accuracy parameter for approxPolyDP
APPROX_POLY_ACCURACY = 0.1


# check if an array of four points and shape (4, 2) is a square (returning 2), a rectangle (returning 1) or neither
def check_rect_or_square(arr, max_deviation=MAX_DEVIATION):

    if arr.shape != (4, 2):
        raise ValueError("Invalid input shape: {0}".format(arr.shape))

    # find diagonal intersection

    mid_1 = (arr[0] + arr[2]) / 2
    mid_2 = (arr[1] + arr[3]) / 2

    mid_dist = np.linalg.norm(mid_2 - mid_1)

    points_lst = list(arr)

    distances_lst = [np.linalg.norm(points_lst[i] - points_lst[i - 1]) for i in range(len(points_lst))]

    md = np.mean(distances_lst)

    if mid_dist > md * MAX_DEVIATION:
        logging.info("Rectangle check negative!")
        return 0

    else:
        logging.info("Rectangle check positive!")

    dist_diff = [abs(md - el) for el in distances_lst]

    if max(dist_diff) < max_deviation * md:
        logging.info("Square check positive!")
        return 2

    return 1


# detect shapes in black-white RGB formatted cv2 image
def detect_shapes(img, approx_poly_accuracy=APPROX_POLY_ACCURACY):

    res_dict = {
        "rectangles": [],
        "squares": []
    }

    vis = img.copy()

    shape = img.shape

    height, width = shape[0], shape[1]

    total_area = height * width

    # Morphological closing: get rid of holes
    # img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Morphological opening: get rid of extensions at the border of the objects
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # cv2.imshow('intermediate', img)
    # cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    logging.info("Number of found contours for shape detection: {0}".format(len(contours)))

    # vis = img.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    cv2.imshow('vis', vis)
    cv2.waitKey(0)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_SHAPE_AREA:
            continue

        if area > MAX_SHAPE_AREA_RATIO * total_area:
            continue

        # find the center of the shape

        M = cv2.moments(contour)

        if M['m00'] == 0.0:
            logging.warning("Unable to compute shape center! Skipping.")
            continue

        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        approx = cv2.approxPolyDP(contour, approx_poly_accuracy * cv2.arcLength(contour, True), True)

        la = len(approx)

        if la < 3:
            logging.warning("Invalid shape detected! Skipping.")
            continue

        if la == 3:
            logging.info("Triangle detected at position {0}".format((x, y)))

        elif la == 4:
            logging.info("Quadrilateral detected at position {0}".format((x, y)))

            if approx.shape != (4, 1, 2):
                raise ValueError("Invalid shape before reshape to (4, 2): {0}".format(approx.shape))

            approx = approx.reshape(4, 2)

            r_check = check_rect_or_square(approx)

            blob_data = {"position": (x, y), "approx": approx}

            if r_check == 2:
                res_dict["squares"].append(blob_data)

            elif r_check == 1:
                res_dict["rectangles"].append(blob_data)

        elif la == 5:
            logging.info("Pentagon detected at position {0}".format((x, y)))

        elif la == 6:
            logging.info("Hexagon detected at position {0}".format((x, y)))

        else:
            logging.info("Circle, ellipse or arbitrary shape detected at position {0}".format((x, y)))

    logging.info("res_dict: {0}".format(res_dict))

    return res_dict
