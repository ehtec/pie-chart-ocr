import logging
logging.basicConfig(level=logging.DEBUG)
from piechartocr import helperfunctions
import numpy as np
from examples import test_polygon_calc_wrapper, test_superreshelper, test_pie_chart_ocr


# input_values = [0, 1, 2, 60, 3, 59, 62, 61, 79, -5, -100, -101, -103]
#
# abs_clusters = helperfunctions.cluster_abs_1d(input_values, 1.0)
# rel_clusters = helperfunctions.cluster_rel_1d(input_values, 0.05)
#
# logging.info("abs_clusters: {0}".format(abs_clusters))
# logging.info("rel_clusters: {0}".format(rel_clusters))
#
# data = np.random.rand(20, 2)
#
# dbscan_clusters = helperfunctions.cluster_dbscan(data, 0.12)
#
# logging.info("dbscan_clusters: {0}".format(dbscan_clusters))
#
# logging.info("root path: {0}".format(helperfunctions.get_root_path()))

# test_helperfunctions.main()
# test_ellipse_detection.main()
# test_mser_functions.main()
# test_polygon_calc_wrapper.main()
# test_superreshelper.main()
test_pie_chart_ocr.main()
