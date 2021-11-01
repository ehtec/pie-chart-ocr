import cv2
from cv2 import dnn_superres
from data_helpers import get_steph_test_path

sr = dnn_superres.DnnSuperResImpl_create()

csvpath, imagepath = get_steph_test_path(2)

image = cv2.imread(imagepath)

# models taken from https://github.com/Saafke/EDSR_Tensorflow
path = "EDSR_x3.pb"

sr.readModel(path)

sr.setModel("edsr", 3)

result = sr.upsample(image)

# cv2.imshow(result)
#
# cv2.waitkey(0)

cv2.imwrite('temp2/upscaled.png', result)
