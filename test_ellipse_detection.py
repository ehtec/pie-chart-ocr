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

X1, X2 = make_test_ellipse()

X = np.array(list(zip(X1, X2)))

shape_detection.check_ellipse_or_circle(X)
