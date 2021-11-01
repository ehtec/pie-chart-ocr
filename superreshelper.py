import cv2
from cv2 import dnn_superres

sr = dnn_superres.DnnSuperResImpl_create()

image = cv2.imread('./pie_charts/image-024.jpg')

# models taken from https://github.com/Saafke/EDSR_Tensorflow
path = "EDSR_x3.pb"

sr.readModel(path)

sr.setModel("edsr", 3)

result = sr.upsample(image)

# cv2.imshow(result)
#
# cv2.waitkey(0)

cv2.imwrite('temp2/upscaled.png', result)
