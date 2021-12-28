import copy
import cv2
import numpy as np
import logging
from ellipse import LsqEllipse
# import matplotlib.pyplot as plt
from .polygon_calc_wrapper import PolygonCalc
# from pprint import pprint
# from hull_computation import concave_hull
from .helperfunctions import cluster_abs_1d, cluster_dbscan, get_image_color_pixels, get_cv2_dominant_color_3
from .helperfunctions import erosion_dilation_operations, find_longest_sequence
from .basefunctions import complex_to_real
from .color_processer_wrapper import ColorProcesser


logging.basicConfig(level=logging.INFO)


# max area ratio of a shape
MAX_SHAPE_AREA_RATIO = 0.70  # 0.40

# min area ratio of the chart ellipse / circle
MIN_CHART_ELLIPSE_AREA_RATIO = 0.03

# max area ratio of a legend shape
MAX_LEGEND_SHAPE_AREA_RATIO = 0.01

# maximum absolute deviation clustering tolerance for legend shapes
MAX_LEGEND_ATOL = 3.0

# minimum length of a legend shape cluster
MIN_LEGEND_SHAPE_CLUSTER_LEN = 3

# minimum absolute area of a shape in pixels
MIN_SHAPE_AREA = 15

# maximum deviation for shape detection (square, rectangle,...)
MAX_DEVIATION = 0.1

# maximum area deviation to count as successful ellipse fit
MAX_AREA_DEVIATION = 0.05

# maximum area deviation for a successful rectangle fit
MAX_RECT_AREA_DEVIATION = 0.20

# maximum deviation to count ellipse as circle
MAX_CIRCLE_DEVIATION = 0.1

# accuracy parameter for approxPolyDP
APPROX_POLY_ACCURACY = 0.02

# color detection erosion kernel size
COLOR_DETECTION_EROSION_KERNEL_SIZE = 7

# maximum CIEDE2000 color distance to be considered the same sector in the chart ellipse
CHART_ELLIPSE_SECTOR_COLOR_DISTANCE = 9.5


# get number of parents of a contour
def get_parents_count(i, hierarchy):

    temp_contour = hierarchy[i]

    parents_count = 0

    while temp_contour[3] > 0:
        parents_count += 1
        temp_contour = hierarchy[temp_contour[3]]

    logging.info("parents_count: {0}".format(parents_count))

    return parents_count


# get the area deviation ratio of two contours
def get_area_deviation_ratio(p1, p2):

    # print("p1:")
    # pprint(p1)
    # print()
    # print("p2:")
    # pprint(p2)

    logging.info("type(p1): {0}".format(type(p1)))
    logging.info("type(p2): {0}".format(type(p2)))
    logging.info("len(p1): {0}".format(len(p1)))
    logging.info("len(p2): {0}".format(len(p2)))
    # logging.info("cv2.contourArea(p1): {0}".format(cv2.contourArea(p1)))
    # logging.info("cv2.contourArea(p2): {0}".format(cv2.contourArea(p2)))
    logging.info("p1[0]: {0}".format(p1[0]))
    logging.info("p1[-1]: {0}".format(p1[-1]))
    logging.info("p2[0]: {0}".format(p2[0]))
    logging.info("p2[-1]: {0}".format(p2[-1]))

    p1_hull = copy.deepcopy(p1)  # cv2.convexHull(p1).reshape(-1, 2)
    p2_hull = copy.deepcopy(p2)  # cv2.convexHull(p2).reshape(-1, 2)

    # p1_hull = concave_hull(p1.reshape(-1, 2))
    # p2_hull = concave_hull(p2.reshape(-1, 2))

    pc = PolygonCalc()

    intersection_area = pc.poly_intersection_area(p1_hull.tolist(), p2_hull.tolist())

    # intersection_area = poly_intersection_area(p1_hull.tolist(), p2_hull.tolist())

    logging.info("intersection_area: {0}".format(intersection_area))

    # total_area = cv2.contourArea(p1_hull) + cv2.contourArea(p2_hull)

    logging.info("p1_hull area: {0}".format(pc.poly_area(p1_hull.tolist())))
    logging.info("p2_hull area: {0}".format(pc.poly_area(p2_hull.tolist())))

    total_area = pc.poly_area(p1_hull.tolist()) + pc.poly_area(p2_hull.tolist())

    logging.info("total_area: {0}".format(total_area))

    area_deviation_ratio = 2 * (total_area - 2 * intersection_area) / total_area

    logging.info("area_deviation_ratio: {0}".format(area_deviation_ratio))

    del pc

    return area_deviation_ratio


# check if an array of the shape (n, 2) is a circle (returning 2), an ellipse (returning 1) or neither (returning 0)
def check_ellipse_or_circle(arr):

    arr = arr.astype(np.int64)

    if any([arr.shape[1] != 2, arr.ndim != 2]):
        raise ValueError("Invalid input shape: {0}".format(arr.shape))

    # X = arr[:, 0:1]
    # Y = arr[:, 1:]
    #
    # A = np.hstack([X**2, X * Y, Y**2, X, Y])
    #
    # b = np.ones_like(X)
    #
    # x, residuals, rank, s = np.linalg.lstsq(A, b)
    #
    # x = x.squeeze()

    # print("arr:")
    # pprint(arr)

    try:

        reg = LsqEllipse().fit(arr)

        reg_params = reg.as_parameters()

        logging.info("Original ellipse parameters: {0}".format(reg_params))

        reg_params = np.hstack(reg_params).tolist()

        reg_params = [complex_to_real(el) for el in reg_params]

    except ValueError:
        logging.warning("ValueError when trying to fit ellipse")
        return 0, {}

    except IndexError:
        logging.warning("IndexError when trying to fit ellipse")
        return 0, {}

    logging.info("Ellipse parameters: {0}".format(reg_params))

    center_x, center_y, width, height, phi = reg_params

    logging.info("center_x: {0}".format(center_x))
    logging.info("center_y: {0}".format(center_y))
    logging.info("width: {0}".format(width))
    logging.info("height: {0}".format(height))
    logging.info("phi: {0}".format(phi))

    t_values = np.linspace(0.0, 2 * np.pi, 1000)

    # width = 2 * width
    # height = 2 * height

    x_values = center_x + (width * np.cos(t_values) * np.cos(phi) - height * np.sin(t_values) * np.sin(phi))

    y_values = center_y + (width * np.cos(t_values) * np.sin(phi) + height * np.sin(t_values) * np.cos(phi))

    res_arr = np.column_stack([x_values, y_values])

    # print("res_arr:")
    # pprint(res_arr)
    # print()
    # print("arr:")
    # pprint(arr)

    # arr = 1000 * arr
    # res_arr = 1000 * res_arr

    # res_arr = res_arr.astype(np.int64)
    # arr = arr.astype(np.int64)
    # x_values = 1000 * x_values
    # y_values = 1000 * y_values

    try:

        area_deviation_ratio = get_area_deviation_ratio(arr.reshape(-1, 2), res_arr.reshape(-1, 2))

        logging.info("Successful execution of get_area_deviation_ratio!")

    except ValueError:
        logging.warning("ValueError when trying to fit ellipse")
        return 0, {}

    except IndexError:
        logging.warning("IndexError when trying to fit ellipse")
        return 0, {}

    logging.info("area_deviation_ratio: {0}".format(area_deviation_ratio))

    # fig = plt.figure(figsize=(6, 6))
    # ax = plt.subplot()
    # ax.axis('equal')
    # ax.plot(arr[:, 0], arr[:, 1], 'bo')
    # ax.plot(x_values, y_values, 'r-')
    # plt.show()

    if area_deviation_ratio <= MAX_AREA_DEVIATION:

        circle_deviation = abs(width - height) * 2 / (width + height)

        if circle_deviation <= MAX_CIRCLE_DEVIATION:
            logging.info("Circle detected!")

            data = {
                "a": (width + height) / 2
            }

            return 2, data

        logging.info("Ellipse detected!")

        data = {
            "a": max([width, height]),
            "b": min([width, height])
        }

        return 1, data

    return 0, {}


# check if an array of four points and shape (4, 2) is a square (returning 2), a rectangle (returning 1) or neither
#   (returning 0)
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

    data = {}

    if mid_dist > md * MAX_DEVIATION:
        logging.info("Rectangle check negative!")
        return 0, data

    else:
        logging.info("Rectangle check positive!")

    dist_diff = [abs(md - el) for el in distances_lst]

    if max(dist_diff) < max_deviation * md:
        logging.info("Square check positive!")

        a = np.mean(distances_lst)

        data = {"a": a}

        return 2, data

    a = (distances_lst[0] + distances_lst[2]) / 2

    b = (distances_lst[1] + distances_lst[3]) / 2

    if a < b:
        a, b = b, a

    # also find delta_x and delta_y (difference between max and min of the coordinate)

    x_min = arr[:, 0].min()
    x_max = arr[:, 0].max()
    delta_x = x_max - x_min

    y_min = arr[:, 1].min()
    y_max = arr[:, 1].max()
    delta_y = y_max - y_min

    data = {"a": a, "b": b, "delta_x": delta_x, "delta_y": delta_y}

    return 1, data


# detect shapes in black-white RGB formatted cv2 image
def detect_shapes(img, interactive=True):

    res_dict = {
        "rectangles": [],
        "squares": [],
        "circles": [],
        "ellipses": []
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

    logging.info("len(contours): {0}".format(len(contours)))
    logging.info("len(hierarchy): {0}".format(len(hierarchy)))
    # logging.info("contours.shape: {0}".format(contours.shape))
    logging.info("hierarchy.shape: {0}".format(hierarchy.shape))
    # pprint(hierarchy)

    reshaped_hierarchy = hierarchy.reshape(-1, 4)
    logging.info("reshaped_hierarchy.shape: {0}".format(reshaped_hierarchy.shape))

    reshaped_hierarchy_list = reshaped_hierarchy.tolist()
    logging.info("len(reshaped_hierarchy_list): {0}".format(len(reshaped_hierarchy_list)))

    logging.info("Number of found contours for shape detection: {0}".format(len(contours)))

    # vis = img.copy()
    # cv2.drawContours(vis, contours, -1, (0, 255, 0), 2)
    # cv2.imshow('vis', vis)
    # cv2.waitKey(0)

    for i in range(len(contours)):

        blob_data = {}

        contour = contours[i]
        hierarchy_elem = reshaped_hierarchy_list[i]

        # contour = cv2.convexHull(contour)
        # contour = concave_hull(contour.reshape(-1, 2).astype(np.float64))

        parents_count = get_parents_count(i, reshaped_hierarchy_list)

        area = cv2.contourArea(contour)

        blob_data.update({
            "parents_count": parents_count,
            "has_child": hierarchy_elem[2] > 0,
            "area": area,
            "area_ratio": area / total_area
        })

        # if parents_count % 2 != 0:
        #     logging.info("Odd parents count. Skipping.")
        #     # cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
        #     continue
        #
        # if hierarchy_elem[2] > 0:
        #     logging.info("Contour {0} has a child! Skipping.")
        #     cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
        #     continue
        #
        # if area < MIN_SHAPE_AREA:
        #     logging.warning("Area too small: {0}. Skipping.".format(area))
        #     # cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
        #     continue
        #
        # if area > MAX_SHAPE_AREA_RATIO * total_area:
        #     logging.warning("Area ratio too big: {0}. Skipping.".format(area / total_area))
        #     # cv2.drawContours(vis, [contour], -1, (255, 0, 0), 2)
        #     continue

        # approx = cv2.approxPolyDP(contour, approx_poly_accuracy * cv2.arcLength(contour, True), True)

        x, y, w, h = cv2.boundingRect(contour)

        approx = np.array([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])

        cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)

        # logging.info("approx: {0}".format(approx))

        try:

            area_deviation_ratio = get_area_deviation_ratio(contour.reshape(-1, 2).astype(np.float64),
                                                            approx.astype(np.float64))

            if area_deviation_ratio > MAX_RECT_AREA_DEVIATION:
                logging.info("area_deviation_ratio too big: {0}".format(area_deviation_ratio))
                approx = contour.reshape(-1, 2)

            logging.info("Successful execution of get_area_deviation_ratio!")

        except ValueError as e:
            logging.exception(e)
            logging.warning("ValueError when executing get_area_deviation_ratio")
            approx = contour.reshape(-1, 2)

        # cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)

        la = len(approx)

        # find the center of the shape

        M = cv2.moments(contour)

        if M['m00'] == 0.0:
            logging.warning("Unable to compute shape center! Skipping.")
            continue

        x = int(M['m10'] / M['m00'])
        y = int(M['m01'] / M['m00'])

        blob_data.update({"position": (x, y), "approx": approx})

        if la < 3:
            logging.warning("Invalid shape detected! Skipping.")
            continue

        if la == 3:
            logging.info("Triangle detected at position {0}".format((x, y)))

        elif la == 4:
            logging.info("Quadrilateral detected at position {0}".format((x, y)))

            cv2.drawContours(vis, [approx], -1, (0, 0, 255), 2)

            # if approx.shape != (4, 1, 2):
            #     raise ValueError("Invalid shape before reshape to (4, 2): {0}".format(approx.shape))

            approx = approx.reshape(4, 2)

            r_check, data = check_rect_or_square(approx)

            blob_data.update(data)

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

            approx = approx.reshape(-1, 2)

            c_check, data = check_ellipse_or_circle(approx)

            blob_data.update(data)

            if c_check == 2:
                res_dict["circles"].append(blob_data)

            elif c_check == 1:
                res_dict["ellipses"].append(blob_data)

        # cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        # cv2.imshow('vis', vis)
        # cv2.resizeWindow('vis', 800, 800)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    if interactive:
        cv2.namedWindow('vis', cv2.WINDOW_NORMAL)
        # cv2.drawContours(vis, [contour], -1, (0, 255, 0), 2)
        cv2.imshow('vis', vis)
        cv2.resizeWindow('vis', 800, 800)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # logging.info("res_dict: {0}".format(res_dict))

    return res_dict


# add color info to a list of shapes
def add_color_info(input_shapes, img, colors_num, erosion_kernel_size=COLOR_DETECTION_EROSION_KERNEL_SIZE):

    output_shapes = {k: [] for k in input_shapes.keys()}

    for k, v in input_shapes.items():

        for i in range(len(v)):

            data = input_shapes[k][i]

            contour = data['approx']

            color_pixels = get_image_color_pixels(img, contour, erosion_kernel_size).astype(float)

            if color_pixels.size == 0:
                logging.warning("Empty color_pixels encountered. Skipping shape.")
                continue

            dominant_color = get_cv2_dominant_color_3(color_pixels, colors_num, reshape=False)

            data.update({'dominant_color': dominant_color})

            output_shapes[k].append(data)

    return output_shapes


# filter the main chart circle / ellipse from the res_dict of detected shapes
def filter_chart_ellipse(detected_shapes):

    filtered_shapes = copy.deepcopy(detected_shapes)

    # remove rectangles and squares
    filtered_shapes.pop("rectangles")
    filtered_shapes.pop("squares")

    logging.info("initial shapes: {0}".format(filtered_shapes))

    # Remove inner (hole) contours
    filtered_shapes["circles"] = [el for el in filtered_shapes["circles"] if el["parents_count"] % 2 == 0]
    filtered_shapes["ellipses"] = [el for el in filtered_shapes["ellipses"] if el["parents_count"] % 2 == 0]

    # Remove elements below the minimum area
    filtered_shapes["circles"] = [el for el in filtered_shapes["circles"] if el["area"] >= MIN_SHAPE_AREA]
    filtered_shapes["ellipses"] = [el for el in filtered_shapes["ellipses"] if el["area"] >= MIN_SHAPE_AREA]

    # Remove elements that are too big
    filtered_shapes["circles"] = [el for el in filtered_shapes["circles"] if el["area_ratio"] <= MAX_SHAPE_AREA_RATIO]
    filtered_shapes["ellipses"] = [el for el in filtered_shapes["ellipses"] if el["area_ratio"] <= MAX_SHAPE_AREA_RATIO]

    # Remove elements that are too small for the main circle / ellipse of the chart
    filtered_shapes["circles"] = [el for el in filtered_shapes["circles"]
                                  if el["area_ratio"] >= MIN_CHART_ELLIPSE_AREA_RATIO]
    filtered_shapes["ellipses"] = [el for el in filtered_shapes["ellipses"]
                                   if el["area_ratio"] >= MIN_CHART_ELLIPSE_AREA_RATIO]

    logging.info("filtered_shapes: {0}".format(filtered_shapes))

    filtered_shape_tuples = [(shape_type, shape_data) for shape_type, el in filtered_shapes.items() for shape_data in el]

    if not bool(filtered_shape_tuples):
        logging.warning("No chart ellipse found!")
        return None

    chart_ellipse = max(filtered_shape_tuples, key=lambda x: x[1]['area'])

    logging.info("chart_ellipse: {0}".format(chart_ellipse))

    return chart_ellipse


# filter the main chart circle / ellipse from the res_dict of detected shapes
def filter_legend_squares(detected_shapes, img, colors_num):

    filtered_shapes = copy.deepcopy(detected_shapes)

    # remove rectangles, ellipses and circles
    filtered_shapes.pop("rectangles")
    filtered_shapes.pop("circles")
    filtered_shapes.pop("ellipses")

    logging.info("initial shapes: {0}".format(filtered_shapes))

    # Remove inner (hole) contours
    filtered_shapes["squares"] = [el for el in filtered_shapes["squares"] if el["parents_count"] % 2 == 0]

    # Remove elements below the minimum area
    filtered_shapes["squares"] = [el for el in filtered_shapes["squares"] if el["area"] >= MIN_SHAPE_AREA]

    # Remove elements that are too big
    filtered_shapes["squares"] = [el for el in filtered_shapes["squares"] if el["area_ratio"] <= MAX_SHAPE_AREA_RATIO]

    # Remove elements that are too big for a legend square
    filtered_shapes["squares"] = [el for el in filtered_shapes["squares"]
                                  if el["area_ratio"] <= MAX_LEGEND_SHAPE_AREA_RATIO]

    logging.info("filtered_shapes: {0}".format(filtered_shapes))

    # add color info
    filtered_shapes = add_color_info(filtered_shapes, img, colors_num)

    filtered_square_a_values = [el["a"] for el in filtered_shapes['squares']]

    a_clusters = cluster_abs_1d(filtered_square_a_values, MAX_LEGEND_ATOL)

    shape_clusters = []

    for temp_cluster in a_clusters:

        shape_clusters.append([])

        for val in list(set(temp_cluster)):
            shape_clusters[-1] += [el for el in filtered_shapes['squares'] if el['a'] == val]

    shape_clusters = [shape_cluster for shape_cluster in shape_clusters
                      if len(shape_cluster) >= MIN_LEGEND_SHAPE_CLUSTER_LEN]

    logging.info("shape_clusters: {0}".format(shape_clusters))

    if not bool(shape_clusters):
        return None

    shape_cluster = shape_clusters[0]

    return shape_cluster


# filter the main chart circle / ellipse from the res_dict of detected shapes
def filter_legend_rectangles(detected_shapes, img, colors_num):

    filtered_shapes = copy.deepcopy(detected_shapes)

    # remove squares, ellipses and circles
    filtered_shapes.pop("squares")
    filtered_shapes.pop("circles")
    filtered_shapes.pop("ellipses")

    logging.info("initial shapes: {0}".format(filtered_shapes))

    # Remove inner (hole) contours
    filtered_shapes["rectangles"] = [el for el in filtered_shapes["rectangles"] if el["parents_count"] % 2 == 0]

    # Remove elements below the minimum area
    filtered_shapes["rectangles"] = [el for el in filtered_shapes["rectangles"] if el["area"] >= MIN_SHAPE_AREA]

    # Remove elements that are too big
    filtered_shapes["rectangles"] = [el for el in filtered_shapes["rectangles"]
                                     if el["area_ratio"] <= MAX_SHAPE_AREA_RATIO]

    # Remove elements that are too big for a legend square
    filtered_shapes["rectangles"] = [el for el in filtered_shapes["rectangles"]
                                     if el["area_ratio"] <= MAX_LEGEND_SHAPE_AREA_RATIO]

    # Remove rectangles with wrong aspect ratio
    filtered_shapes["rectangles"] = [el for el in filtered_shapes["rectangles"] if el["delta_x"] >= el["delta_y"]]

    logging.info("filtered_shapes: {0}".format(filtered_shapes))

    # add color info
    filtered_shapes = add_color_info(filtered_shapes, img, colors_num)

    filtered_square_ab_values = [[el["delta_x"], el["delta_y"]] for el in filtered_shapes['rectangles']]

    _, shape_clusters = cluster_dbscan(filtered_square_ab_values, 2 * MAX_LEGEND_ATOL,
                                       input_objects=filtered_shapes['rectangles'])

    # shape_clusters = []
    #
    # for temp_cluster in ab_clusters:
    #
    #     shape_clusters.append([])
    #
    #     for val in list(set(temp_cluster)):
    #         shape_clusters[-1] += [el for el in filtered_shapes['squares'] if el['a'] == val]

    shape_clusters = [shape_cluster for shape_cluster in shape_clusters
                      if len(shape_cluster) >= MIN_LEGEND_SHAPE_CLUSTER_LEN]

    logging.info("shape_clusters: {0}".format(shape_clusters))

    if not bool(shape_clusters):
        return None

    return shape_clusters[0]


# detect the sector positions of the chart ellipse. Takes a list of BGR colors as input, returns a list of sector
#   positions (center of mass).
def detect_ellipse_sectors(img, legend_colors, chart_ellipse, max_color_distance=CHART_ELLIPSE_SECTOR_COLOR_DISTANCE,
                           erosion_kernel_size=COLOR_DETECTION_EROSION_KERNEL_SIZE, erosion_iterations=1):

    cp = ColorProcesser()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    contour = chart_ellipse['approx']
    img_mask = np.full(img.shape, 0)
    img_mask = np.ascontiguousarray(img_mask, dtype=np.uint8)

    cv2.drawContours(img_mask, [contour], -1, color=(255, 255, 255), thickness=cv2.FILLED)

    # apply erode filter
    if erosion_iterations > 0:
        kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
        img_mask = cv2.erode(img_mask, kernel, iterations=erosion_iterations)

    logging.debug("img_mask.shape before: {0}".format(img_mask.shape))
    img_mask = img_mask[:, :, 0]
    logging.debug("img_Mask shape after: {0}".format(img_mask.shape))

    centers = []

    for legend_color in legend_colors:

        legend_color_rgb = (legend_color[2], legend_color[1], legend_color[0])

        color_distances = cp.array_color_distance(legend_color_rgb, img_rgb)

        sector_points = np.where((img_mask == 255) & (color_distances <= max_color_distance))

        logging.debug("sector_points: {0}".format(sector_points))
        logging.info("sector_points[0].shape: {0}".format(sector_points[0].shape))

        assert len(sector_points) == 2

        center = list(reversed([el.mean() for el in sector_points]))

        logging.info("center: {0}".format(center))

        centers.append(center)

    del cp

    logging.info("centers: {0}".format(centers))

    return centers


# optimize detected shapes by executing different erosion dilation operations
def optimize_detected_shapes(img, img_bin, colors_num, interactive=True):

    # separate operations for chart ellipse detection to deal with larger gaps
    chart_ellipse_operations_set = [[
        ("erosion", 7, i),
        ("dilation", 7, i + 1)
    ] for i in range(11)]

    operations_set = [[
        ("erosion", 5, i),
        ("dilation", 5, i + 1)
    ] for i in range(3)]

    chart_ellipse_results = []
    legend_squares_results = []
    legend_rectangles_results = []

    for i in range(len(chart_ellipse_operations_set)):

        chart_ellipse_operations = chart_ellipse_operations_set[i]

        img_bin_chart_ellipse = erosion_dilation_operations(img_bin, chart_ellipse_operations)

        chart_ellipse_detected_shapes = detect_shapes(img_bin_chart_ellipse, interactive=interactive)

        chart_ellipse = filter_chart_ellipse(chart_ellipse_detected_shapes)

        chart_ellipse_results.append(chart_ellipse)

        if (i == 0) and (chart_ellipse is not None):
            # do not continue to save computation time if the first search is already a hit
            break

    logging.debug("chart_ellipse_results: {0}".format(chart_ellipse_results))

    chart_ellipse_boolean_results = [el is not None for el in chart_ellipse_results]

    logging.debug("chart_ellipse_boolean_results: {0}".format(chart_ellipse_boolean_results))

    if not any(chart_ellipse_boolean_results):
        chart_ellipse = None

    else:
        if chart_ellipse_boolean_results[0]:
            chart_ellipse = chart_ellipse_results[0]

        else:
            middle_index = find_longest_sequence(chart_ellipse_boolean_results, lambda x: x)
            logging.debug("middle_index: {0}".format(middle_index))
            chart_ellipse = chart_ellipse_results[middle_index]

    logging.debug("chart_ellipse: {0}".format(chart_ellipse))

    for i in range(len(operations_set)):

        operations = operations_set[i]

        img_bin = erosion_dilation_operations(img_bin, operations)

        detected_shapes = detect_shapes(img_bin, interactive=interactive)

        legend_squares = filter_legend_squares(detected_shapes, img, colors_num)

        legend_rectangles = filter_legend_rectangles(detected_shapes, img, colors_num)

        legend_squares_results.append(legend_squares)
        legend_rectangles_results.append(legend_rectangles)

    legend_squares_int_results = []
    legend_rectangles_int_results = []

    for el in legend_squares_results:
        if el is None:
            legend_squares_int_results.append(0)

        else:
            legend_squares_int_results.append(len(el))

    for el in legend_rectangles_results:
        if el is None:
            legend_rectangles_int_results.append(0)

        else:
            legend_rectangles_int_results.append(len(el))

    logging.debug("legend_squares_int_results: {0}".format(legend_squares_int_results))
    logging.debug("legend_rectangles_int_results: {0}".format(legend_rectangles_int_results))

    legend_squares = None
    legend_rectangles = None

    if any(legend_squares_int_results):
        max_square_legends = max(legend_squares_int_results)
        middle_index = find_longest_sequence(legend_squares_int_results, lambda x: x == max_square_legends)
        logging.debug("middle_index: {0}".format(middle_index))
        legend_squares = legend_squares_results[middle_index]

    logging.debug("legend_squares: {0}".format(legend_squares))

    if any(legend_rectangles_int_results):
        max_rectangles_legends = max(legend_rectangles_int_results)
        middle_index = find_longest_sequence(legend_rectangles_int_results, lambda x: x == max_rectangles_legends)
        logging.debug("middle_index: {0}".format(middle_index))
        legend_rectangles = legend_rectangles_results[middle_index]

    logging.debug("legend_rectangles: {0}".format(legend_rectangles))

    return chart_ellipse, legend_squares, legend_rectangles
