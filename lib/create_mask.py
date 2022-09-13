import cv2
import numpy as np
import imutils

def mask_img(image_file, kf, leak, bright_thresh, vessel_area_thresh, hole_area_thresh, blur_thresh):
    image = cv2.imread(image_file, 0)
    image = cv2.filter2D(image, -1, np.ones((7, 7))/49)
    smooth_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1],
    ])/16
    image = cv2.filter2D(image, -1, smooth_kernel)
    big_kernel = (
        np.ones((int(image.shape[0]//kf), int(image.shape[1]//kf)))
        / (int(image.shape[0]//kf)*int(image.shape[1]//kf))
    )
    new_image = cv2.filter2D(image, -1, big_kernel)
    new_image = np.uint8(np.clip(np.subtract(np.int16(new_image), np.int16(image)) + np.ones_like(np.int16(image))*leak, 0, 255))
    image = new_image
    image = cv2.filter2D(image, -1, smooth_kernel)
    image[np.where(image >= bright_thresh)] = 255
    image[np.where(image < bright_thresh)] = 0
    image = cv2.erode(image, np.ones((7, 7)), iterations = 2)
    thresh = image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bad_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if(area < vessel_area_thresh):
            bad_cnts.append(c)
    thresh = cv2.drawContours(thresh, bad_cnts, -1, 0, -1)
    thresh = cv2.dilate(thresh, np.ones((7, 7)), iterations = 2)
    thresh = cv2.filter2D(thresh, -1, np.ones((7, 7))/49)
    thresh[np.where(thresh >= blur_thresh)] = 255
    thresh[np.where(thresh < blur_thresh)] = 0
    thresh = cv2.erode(thresh, np.ones((7, 7)), iterations = 1)
    thresh[:] = 255 - thresh[:]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    bad_cnts = []
    for c in cnts:
        area = cv2.contourArea(c)
        if(area < hole_area_thresh):
            bad_cnts.append(c)
    thresh = cv2.drawContours(thresh, bad_cnts, -1, 0, -1)
    thresh[:] = 255 - thresh[:]
    return thresh