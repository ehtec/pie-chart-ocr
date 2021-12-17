# import matplotlib.pyplot as plt
#
# from skimage import data
# import ellipse_detection
#
# # Load picture, convert to grayscale and detect edges
# image_rgb = data.coffee()[0:220, 160:420]
#
# ellipse_detection.detect_ellipses(image_rgb)

from ellipse_example import make_test_ellipse
import numpy as np
import shape_detection
from pprint import pprint
import matplotlib.pyplot as plt
# import alphashape
from test_alphashape_2 import alpha_shape


def order_edges(unordered_edges):

    edges_l = list(zip(*list(unordered_edges)))

    if len(edges_l[0]) != len(list(set(edges_l[0]))):
        raise ValueError("Duplicate point A found!")

    if len(edges_l[1]) != len(list(set(edges_l[1]))):
        raise ValueError("Duplicate point B found!")

    ordered_edges = [unordered_edges[0]]

    while len(ordered_edges) < len(unordered_edges):

        # print("ordered_edges: {0}".format(ordered_edges))

        found_edge = False

        for el in unordered_edges:
            # print("el: {0}".format(el))
            if el[0] == ordered_edges[-1][1]:
                ordered_edges.append(el)
                found_edge = True
                break

        if not found_edge:
            print("len(ordered_edges): {0}".format(len(ordered_edges)))
            raise ValueError("This should not be reached.")

    return ordered_edges


X1, X2 = make_test_ellipse()

points = np.array(list(zip(X1, X2)))

# alpha = 0.95 * alphashape.optimizealpha(points)
#
# X = alphashape.alphashape(points, alpha)

edges = alpha_shape(points, alpha=0.1, only_outer=True)

X = points

print("X.shape: {0}".format(X.shape))
print("X:")
# pprint(edges)

edges_list = list(zip(*list(edges)))

print("once: {0}".format([el for el in edges if el[1] not in edges_list[0]]))

edges = order_edges(list(edges))

# plt.plot(X[:, 0], X[:, 1], 'r-')
# plt.plot(points[:, 0], points[:, 1], '.')
for i, j in edges:
    plt.plot(X[:, 0], X[:, 1], 'r-')
    plt.plot(points[[i, j], 0], points[[i, j], 1])
    plt.show()

X = X * 1000

X = X.astype(np.int64)

shape_detection.check_ellipse_or_circle(X)
