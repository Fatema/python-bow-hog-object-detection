import os

import cv2
import numpy as np


# code for creating the mask was taken from the answer for this question in stackoverflow:
# this https://stackoverflow.com/questions/49093729/remove-background-of-any-image-using-opencv

# Parameters
BLUR = 21
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


def nothing(x):
    pass


def contour_edges(img, lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3, sobel_size=3):
    img_canny = canny(img, lower_threshold, upper_threshold,smoothing_neighbourhood,sobel_size)

    # convert the canny edges into contours
    # Find contours in edges, sort by area
    contour_info = []
    _, contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)

    # -- Create empty mask, draw filled polygon on it corresponding to largest contour ----
    # Mask is black, polygon is white
    mask = np.zeros(img_canny.shape)

    for c in contour_info:
        cv2.fillConvexPoly(mask, c[0], (255))

    # -- Smooth mask, then blur it
    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    # -- Blend masked img into MASK_COLOR background
    mask_stack = mask_stack.astype('float32') / 255.0
    img = img.astype('float32') / 255.0
    masked = (mask_stack * img) + ((1 - mask_stack) * MASK_COLOR)
    masked = (masked * 255).astype('uint8')

    return masked


def canny(img, lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3, sobel_size=3):

    # check neighbourhood is greater than 3 and odd

    smoothingNeighbourhood = max(3, smoothing_neighbourhood)
    if not (smoothingNeighbourhood % 2):
        smoothingNeighbourhood = smoothingNeighbourhood + 1

    sobelSize = max(3, sobel_size)
    if not (sobelSize % 2):
        sobelSize = sobelSize + 1

    # convert to grayscale

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # performing smoothing on the image using a 5x5 smoothing mark (see manual entry for GaussianBlur())

    smoothed = cv2.GaussianBlur(gray_img, (smoothingNeighbourhood, smoothingNeighbourhood), 0)

    # perform canny edge detection

    img_canny = cv2.Canny(smoothed, lower_threshold, upper_threshold, apertureSize=sobelSize)
    img_canny = cv2.dilate(img_canny, None)
    img_canny = cv2.erode(img_canny, None)

    return img_canny


if __name__ == '__main__':
    directory_to_cycle = os.environ['CV_HOME'] + "pedestrian/INRIAPerson/train_64x128_H96/pos/"
    for filename in sorted(os.listdir(directory_to_cycle)):
        if '.png' in filename:
            print(os.path.join(directory_to_cycle, filename))

            # read image data

            img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

            img_canny = canny(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)
            img_contour = contour_edges(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)

            cv2.imshow('img', img)
            cv2.imshow('canny', img_canny)
            cv2.imshow('contours', img_contour)

            key = cv2.waitKey()

            if (key == ord('x')):
                break