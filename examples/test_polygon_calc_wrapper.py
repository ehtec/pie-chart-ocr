import logging

logging.basicConfig(level=logging.DEBUG)

from piechartocr.polygon_calc_wrapper import PolygonCalc


def main():

    pc = PolygonCalc()

    elements = [(1, 2, 3, 4),
                (5, 6, 7, 8),
                (9, 10, 11, 12)]

    print(pc.group_elements(elements=elements, threshold_dist=0.25, slov_ratio=0.92))


if __name__ == "__main__":
    main()
