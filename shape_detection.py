import cv2


# detect shapes in black-white RGB formatted cv2 image
def detect_shapes(img):

    # Morphological closing: get rid of holes
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Morphological opening: get rid of extensions at the border of the objects
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (121, 121)))

    cv2.imshow('intermediate', img)
    cv2.waitKey(0)

    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vis = img.copy()
    vis = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('vis', vis)
    cv2.waitKey(0)
