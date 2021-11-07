import cv2
from cv2 import dnn_superres
from data_helpers import get_steph_test_path
import mser_functions
import os
import numpy as np
import math


# target pixel number after neural network upscale
TARGET_PIXEL_SIZE = 1000 * 1000


n = int(input("Image id: "))

if not os.path.isfile('temp2/upscaled{0}.png'.format(n)):

    sr = dnn_superres.DnnSuperResImpl_create()

    csvpath, imagepath = get_steph_test_path(n)

    image = cv2.imread(imagepath)

    pixel_size = np.prod(image.shape[:-1])

    upscale_factor = min(int(math.ceil(np.sqrt(TARGET_PIXEL_SIZE / pixel_size))), 4)

    if upscale_factor <= 1:

        print("No need to upsample image {0} because it has already {1} pixels".format(n, pixel_size))

        result = image

    else:

        print("Upsampling image {0} with upscaling factor {1} because it has {2} pixels".format(n, upscale_factor, pixel_size))

        # models taken from https://github.com/Saafke/EDSR_Tensorflow
        path = "EDSR_x{0}.pb".format(upscale_factor)

        sr.readModel(path)

        sr.setModel("edsr", upscale_factor)

        result = sr.upsample(image)

    # cv2.imshow(result)
    #
    # cv2.waitkey(0)

    cv2.imwrite('temp2/upscaled{0}.png'.format(n), result)

mser_functions.main('temp2/upscaled{0}.png'.format(n))
