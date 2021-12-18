import cv2
from cv2 import dnn_superres
from data_helpers import get_steph_test_path
# import mser_functions
import os
import numpy as np
import math


# target pixel number after neural network upscale
TARGET_PIXEL_SIZE = 1000 * 1000


# upsample image using superres method
def upsample_image(image):

    pixel_size = np.prod(image.shape[:-1])

    upscale_factor = min(int(math.ceil(np.sqrt(TARGET_PIXEL_SIZE / pixel_size))), 4)

    if upscale_factor <= 1:

        print("No need to upsample image because it has already {0} pixels".format(pixel_size))

        result = image

    else:

        print("Upsampling image with upscaling factor {0} because it has {1} pixels".format(upscale_factor, pixel_size))

        sr = dnn_superres.DnnSuperResImpl_create()

        # models taken from https://github.com/Saafke/EDSR_Tensorflow
        path = "EDSR_x{0}.pb".format(upscale_factor)

        sr.readModel(path)

        sr.setModel("edsr", upscale_factor)

        result = sr.upsample(image)

    return result


# upscale image file from test dataset
def upscale_test_image_file(n):

    if not os.path.isfile('temp2/upscaled{0}.png'.format(n)):

        csvpath, imagepath = get_steph_test_path(n)

        image = cv2.imread(imagepath)

        result = upsample_image(image)

        cv2.imwrite('temp2/upscaled{0}.png'.format(n), result)
