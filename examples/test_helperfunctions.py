import logging
logging.basicConfig(level=logging.DEBUG)
from piechartocr import helperfunctions
import numpy as np


def main():

    input_values = [0, 1, 2, 60, 3, 59, 62, 61, 79, -5, -100, -101, -103]

    abs_clusters = helperfunctions.cluster_abs_1d(input_values, 1.0)
    rel_clusters = helperfunctions.cluster_rel_1d(input_values, 0.05)

    logging.info("abs_clusters: {0}".format(abs_clusters))
    logging.info("rel_clusters: {0}".format(rel_clusters))

    data = np.random.rand(20, 2)

    dbscan_clusters = helperfunctions.cluster_dbscan(data, 0.12)

    logging.info("dbscan_clusters: {0}".format(dbscan_clusters))


if __name__ == "__main__":
    main()
