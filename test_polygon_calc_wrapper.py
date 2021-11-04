import logging

logging.basicConfig(level=logging.DEBUG)

from polygon_calc_wrapper import PolygonCalc

pc = PolygonCalc

elements = [(1, 2, 3, 4),
            (5, 6, 7, 8),
            (9, 10, 11, 12)]

print(pc.group_elements(elements))
