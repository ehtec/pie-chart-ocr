import logging
import cv2
from cv2 import dnn_superres
from .data_helpers import get_steph_test_path, get_upscaled_steph_test_path, test_data_percentages
# import mser_functions
import os
import numpy as np
import math
from .helperfunctions import get_root_path
import shutil
from tqdm import tqdm


# target pixel number after neural network upscale
TARGET_PIXEL_SIZE = 1000 * 1000


# upsample image using superres method
def upsample_image(image, use_gpu=False):

    pixel_size = np.prod(image.shape[:-1])

    upscale_factor = min(int(math.ceil(np.sqrt(TARGET_PIXEL_SIZE / pixel_size))), 4)

    if upscale_factor <= 1:

        print("No need to upsample image because it has already {0} pixels".format(pixel_size))

        result = image

    else:

        print("Upsampling image with upscaling factor {0} because it has {1} pixels".format(upscale_factor, pixel_size))

        sr = dnn_superres.DnnSuperResImpl_create()

        # models taken from https://github.com/Saafke/EDSR_Tensorflow
        path = os.path.join(get_root_path(), "models", "EDSR_x{0}.pb".format(upscale_factor))

        sr.readModel(path)

        sr.setModel("edsr", upscale_factor)

        if use_gpu:
            # try to use GPU
            sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        result = sr.upsample(image)

    return result


# upscale image file from test dataset (old directory)
def old_upscale_test_image_file(n):

    if not os.path.isfile(os.path.join(get_root_path(), 'temp2', 'upscaled{0}.png'.format(n))):

        csvpath, imagepath = get_steph_test_path(n)

        image = cv2.imread(imagepath)

        result = upsample_image(image)

        output_path = os.path.join(get_root_path(), 'temp2', 'upscaled{0}.png'.format(n))

        cv2.imwrite(output_path, result)


# upscale image file from test dataset
def upscale_test_image_file(n):

    if not os.path.isfile(get_upscaled_steph_test_path(n)[1]):

        csvpath, imagepath = get_steph_test_path(n)

        image = cv2.imread(imagepath)

        result = upsample_image(image)

        csv_output_path, image_output_path = get_upscaled_steph_test_path(n)

        os.mkdir(os.path.dirname(image_output_path))

        shutil.copyfile(csvpath, csv_output_path)

        cv2.imwrite(image_output_path, result)


# upscale all valid charts from steph
def upscale_all_images():

    correct_numbers = test_data_percentages()

    logging.info("correct_numbers: {0}".format(correct_numbers))

    for n in tqdm(correct_numbers):
        upscale_test_image_file(n)
