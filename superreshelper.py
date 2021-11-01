import cv2
from cv2 import dnn_superres
from data_helpers import get_steph_test_path
import mser_functions
import os


n = int(input("Image id: "))

if not os.path.isfile('temp2/upscaled{0}.png'.format(n)):

    sr = dnn_superres.DnnSuperResImpl_create()

    csvpath, imagepath = get_steph_test_path(n)

    image = cv2.imread(imagepath)

    # models taken from https://github.com/Saafke/EDSR_Tensorflow
    path = "EDSR_x3.pb"

    sr.readModel(path)

    sr.setModel("edsr", 3)

    result = sr.upsample(image)

    # cv2.imshow(result)
    #
    # cv2.waitkey(0)

    cv2.imwrite('temp2/upscaled{0}.png'.format(n), result)

mser_functions.main('temp2/upscaled{0}.png'.format(n))
