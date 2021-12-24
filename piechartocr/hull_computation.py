from scipy.spatial import Delaunay
import numpy as np
import logging
from shapely.geometry import Polygon
from shapely.validation import make_valid


# default alpha value for concave hull
DEFAULT_CONCAVE_HULL_ALPHA = 1.0


def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(the_edges, i, j):
        """
        Add a line between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in the_edges or (j, i) in the_edges:
            # already added
            assert (j, i) in the_edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it is not a boundary edge
                the_edges.remove((j, i))
            return
        the_edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


# order edges so that connected edges are next to each other, as per contour specification
def order_edges(unordered_edges):

    edges_l = list(zip(*list(unordered_edges)))
    print("edges_l: {0}".format(edges_l))

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


# return opencv style contour from ordered_edges
def edges_to_contour(points, ordered_edges):

    if not all([points is not None, ordered_edges is not None]):
        logging.info("Either edges or points are None!")
        return []

    if not all([len(points) > 0, len(ordered_edges) > 0]):
        logging.info("Either edges or points are empty!")
        return []

    i, j = ordered_edges[0]

    contour = [[points[i, 0], points[i, 1]]]

    for i, j in ordered_edges:
        contour.append([points[j, 0], points[j, 1]])

    contour = np.array(contour)

    # logging.info("contour: {0}".format(contour))

    return contour


# return concave hull of a contour
def concave_hull(points, alpha=DEFAULT_CONCAVE_HULL_ALPHA):

    the_poly = Polygon(points.tolist())

    if the_poly.is_valid:
        return points

    logging.info("points: {0}".format(points))

    edges = list(alpha_shape(points, alpha=alpha, only_outer=True))

    if not bool(edges):
        raise ValueError("No edges found")

    ordered_edges = order_edges(edges)

    contour = edges_to_contour(points, ordered_edges)

    the_poly = Polygon(contour.tolist())

    logging.info("the_poly.is_valid: {0}".format(the_poly.is_valid))

    if not the_poly.is_valid:
        logging.warning("The polygon is not valid. Trying to fix it...")

        multipoly = make_valid(the_poly)

        if isinstance(multipoly, Polygon):
            return np.column_stack(multipoly.exterior.coords.xy)

        # for el in multipoly:
        #     logging.info("el: {0}, el.area: {1}".format(el, el.area))

        raise NotImplementedError("Unknown object (maybe multiple polygons) returned from make_valid")

    return contour
