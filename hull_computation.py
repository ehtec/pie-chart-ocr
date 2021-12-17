from scipy.spatial import Delaunay
import numpy as np


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

    if not all([bool(points), bool(ordered_edges)]):
        return []

    i, j = ordered_edges[0]

    contour = [[points[i, 0], points[i, 1]]]

    for i, j in ordered_edges:
        contour.append([points[j, 0], points[j, 1]])

    contour = np.array(contour)

    return contour
