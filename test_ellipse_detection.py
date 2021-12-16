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

X1, X2 = make_test_ellipse()

X = np.array(list(zip(X1, X2)))

print("X.shape: {0}".format(X.shape))
print("X:")
pprint(X)

X = X * 1000

X = X.astype(np.int64)

shape_detection.check_ellipse_or_circle(X)
