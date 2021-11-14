import matplotlib.pyplot as plt

from skimage import data
import ellipse_detection

# Load picture, convert to grayscale and detect edges
image_rgb = data.coffee()[0:220, 160:420]

ellipse_detection.detect_ellipses(image_rgb)
