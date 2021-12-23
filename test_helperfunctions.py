import logging
logging.basicConfig(level=logging.DEBUG)
import helperfunctions


input_values = [0, 1, 2, 60, 3, 59, 62, 61, 79, -5]

abs_clusters = helperfunctions.cluster_abs_1d(input_values, 1.0)

logging.info("abs_clusters: {0}".format(abs_clusters))
