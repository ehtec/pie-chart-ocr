#!/usr/bin/python

# taken from opencv_contrib. MIT license

import sys
import os
from tqdm import tqdm
import cv2 as cv
# import numpy as np

SCALING_FACTOR = 2

print('\ntextdetection.py')
print('       A demo script of the Extremal Region Filter algorithm described in:')
print('       Neumann L., Matas J.: Real-Time Scene Text Localization and Recognition, CVPR 2012\n')

if len(sys.argv) < 2:
    print(' (ERROR) You must call this script with an argument (path_to_image_to_be_processed)\n')
    quit()

pathname = os.path.dirname(sys.argv[0])

img = cv.imread(str(sys.argv[1]))

img = cv.resize(img, (img.shape[1] * SCALING_FACTOR, img.shape[0] * SCALING_FACTOR), interpolation=cv.INTER_AREA)

cv.imwrite('temp2/orig.png', img)

# for visualization
vis = img.copy()


# Extract channels to be processed individually
channels = list(cv.text.computeNMChannels(img))
# Append negative channels to detect ER- (bright regions over dark background)
cn = len(channels) - 1
for c in range(0, cn):
    channels.append(255 - channels[c])

# Apply the default cascade classifier to each independent channel (could be done in parallel)

erc1 = cv.text.loadClassifierNM1('trained_classifierNM1.xml')
# er1 = cv.text.createERFilterNM1(erc1,6,0.00005,0.08,0.2,True,0.1)
er1 = cv.text.createERFilterNM1(erc1, 16, 0.0000005, 0.08, 0.4, True, 0.2)
# er1 = cv.text.createERFilterNM1(erc1,1,0.00000015,0.13,0.005,False,0.1)
# er1 = cv.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

erc2 = cv.text.loadClassifierNM2('trained_classifierNM2.xml')
er2 = cv.text.createERFilterNM2(erc2, 0.5)

print("Extracting Class Specific Extremal Regions from " + str(len(channels)) + " channels ...")
print("    (...) this may take a while (...)")
for channel in tqdm(channels):

    regions = cv.text.detectRegions(channel, er1, er2)

    rects = cv.text.erGrouping(img, channel, [r.tolist() for r in regions])
    # rects = cv.text.erGrouping(img,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)

    # Visualization
    for rect in rects:
        cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 0), 2)
        cv.rectangle(vis, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)

# Visualization
# cv.imshow("Text detection result", vis)
# cv.waitKey(0)

cv.imwrite('temp2/img.png', vis)
