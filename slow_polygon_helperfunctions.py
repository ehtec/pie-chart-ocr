from sympy import Point, Polygon
import shapely.geometry


# calculate intersection area between polygons
# example polygon: ((0, 0), (1, 0), (1, 1), (0, 1))
def poly_intersection_area(p1, p2):

    poly1 = Polygon(*map(Point, list(p1)))
    poly2 = Polygon(*map(Point, list(p2)))

    polya = shapely.geometry.Polygon(map(Point, list(p1)))
    polyb = shapely.geometry.Polygon(map(Point, list(p2)))

    if not bool(poly1.intersection(poly2)):
        return 0.0

    intersection = polya.intersection(polyb)

    area = intersection.area

    return area
