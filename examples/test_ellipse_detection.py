# import matplotlib.pyplot as plt
#
# from skimage import data
# import ellipse_detection
#
# # Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
#
# ellipse_detection.detect_ellipses(image_rgb)

from piechartocr.ellipse_example import make_test_ellipse
import numpy as np
from piechartocr import shape_detection
from pprint import pprint
import matplotlib.pyplot as plt
# import alphashape
from piechartocr.hull_computation import order_edges, alpha_shape, edges_to_contour, concave_hull
import cv2
from shapely.geometry import Polygon


def main():

    X1, X2 = make_test_ellipse()

    points = np.array(list(zip(X1, X2)))

    # alpha = 0.95 * alphashape.optimizealpha(points)
    #
    # X = alphashape.alphashape(points, alpha)

    edges = alpha_shape(points, alpha=0.1, only_outer=True)

    X = points

    print("cv2.contourArea(X): {0}".format(cv2.contourArea(X.astype(np.int64))))

    # print("X.shape: {0}".format(X.shape))
    # print("X:")
    # # pprint(edges)
    #
    # edges_list = list(zip(*list(edges)))
    #
    # print("once: {0}".format([el for el in edges if el[1] not in edges_list[0]]))
    #
    # edges = order_edges(list(edges))
    #
    # # plt.plot(X[:, 0], X[:, 1], 'r-')
    # # plt.plot(points[:, 0], points[:, 1], '.')
    # # for i, j in edges:
    # #     plt.plot(X[:, 0], X[:, 1], 'r-')
    # #     plt.plot(points[[i, j], 0], points[[i, j], 1])
    # #     plt.show()
    #
    # contour = edges_to_contour(points, edges)

    contour = concave_hull(points)

    # contour = concave_hull(contour)
    #
    # contour = concave_hull(contour)
    #
    # contour = concave_hull(contour)
    #
    # contour = concave_hull(contour)
    #
    # contour = concave_hull(contour)
    #
    # contour = concave_hull(contour)

    plt.plot(contour[:, 0], contour[:, 1], 'r-')
    plt.show()

    # X = concave_hull(X)

    # X = X.astype(np.float64)

    # X = X * 1000

    # X = X.astype(np.int64)

    # concave_hull(X)

    poly1 = Polygon(X.tolist())
    print("poly1.is_valid: {0}".format(poly1.is_valid))

    # print("X:")
    # pprint(X.tolist())

    print("cv2.contourArea(X): {0}".format(cv2.contourArea(X.astype(np.int64))))

    shape_detection.get_area_deviation_ratio(X, X)

    # shape_detection.check_ellipse_or_circle(X)


if __name__ == "__main__":
    main()
