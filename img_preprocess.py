import os

import cv2
import numpy as np



# Parameters
BLUR = 21
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


def nothing(x):
    pass


# this code was taken from https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def adjust_gamma(img, gamma=1.0):
   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(img, table)


# code for creating the mask was taken from the answer for this question in stackoverflow:
# this https://stackoverflow.com/questions/49093729/remove-background-of-any-image-using-opencv
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


def mask_colour(img, light_colour, dark_colour, remove=True):
    # convert the img to HSV range
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    print(hsv_img)

    window_name = 'mask options' + str(light_colour)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # add some track bar controllers for settings
    hue, saturation, brightness = light_colour
    cv2.createTrackbar("hue",window_name , hue, 180, nothing)
    cv2.createTrackbar("saturation", window_name, saturation, 255, nothing)
    cv2.createTrackbar("brightness", window_name, brightness, 255, nothing)
    hue2, saturation2, brightness2 = dark_colour
    cv2.createTrackbar("hue2",window_name , hue2, 180, nothing)
    cv2.createTrackbar("saturation2", window_name, saturation2, 255, nothing)
    cv2.createTrackbar("brightness2", window_name, brightness2, 255, nothing)


    while(True):
        hue = cv2.getTrackbarPos("hue", window_name)
        saturation = cv2.getTrackbarPos("saturation", window_name)
        brightness = cv2.getTrackbarPos("brightness", window_name)
        hue2 = cv2.getTrackbarPos("hue2", window_name)
        saturation2 = cv2.getTrackbarPos("saturation2", window_name)
        brightness2 = cv2.getTrackbarPos("brightness2", window_name)

        print([hue, saturation, brightness],[hue2, saturation2, brightness2])

        # create a mask based on the colour range specified
        mask = cv2.inRange(hsv_img,
                           np.array([hue, saturation, brightness], dtype="uint8"),
                           np.array([hue2, saturation2, brightness2], dtype="uint8"))
        mask = cv2.dilate(mask, None)
        mask = cv2.erode(mask, None)

        if remove:
            mask = cv2.bitwise_not(mask)

        cv2.imshow(window_name, mask)

        masked_img = cv2.bitwise_and(hsv_img, img, mask=mask)

        cv2.imshow('img', masked_img)

        key = cv2.waitKey()

        if key == ord('x'):
            break

    # return the bitwise representation
    return cv2.cvtColor(masked_img, cv2.COLOR_HSV2BGR)


def remove_shadows(img):
    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = np.array([])
        norm_img = cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)

    cv2.imshow('shadows_out', result)
    cv2.imshow('shadows_out_norm', result_norm)

    key = cv2.waitKey()


# segment roads based on the colour grey
def remove_roads(img):
    light_grey = [75, 13, 13]
    dark_grey = [180, 255, 112]
    return mask_colour(img, light_grey, dark_grey)


# segment trees based on the colour green
# colours were picked by using the answer provided here
# https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966
def remove_trees(img):
    # for first image in the shadows ([9, 0, 0] [49, 255, 255])
    light_green = [10, 0, 0]
    dark_green = [50, 255, 255]
    return mask_colour(img, light_green, dark_green)


# this function creates a mask for the car from
def create_car_front_mask(img_height, img_width):
    mask = cv2.imread(os.path.join(os.environ['CV_HOME'], 'car_mask.jpg'), cv2.IMREAD_GRAYSCALE)
    # this kernel is based on the naswer found in
    # https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    mask = cv2.filter2D(mask, -1, kernel)

    # there was some noise introduced after the kernel around the mask edges, this is due to the image quality
    # so I apllied the dilate and erode functions to remove the noise
    mask = cv2.dilate(mask, None)
    mask = cv2.erode(mask, None)

    # this type conversion is to make sure that it is the same as the image
    mask = mask.astype('uint8')

    # the mask is bigger than the original an image, hence the added shift (the values are based on trial and error)
    mask = mask[22:img_height + 22, 42:img_width + 42]
    return cv2.bitwise_not(mask)


def remove_car_front(img, mask):
    return cv2.bitwise_and(img, img, mask=mask)


if __name__ == '__main__':
    directory_to_cycle = os.environ['CV_HOME'] + "TTBB-durham-02-10-17-sub10/left-images"
    for filename in sorted(os.listdir(directory_to_cycle)):
        if '.png' in filename:
            print(os.path.join(directory_to_cycle, filename))

            # read image data

            img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            mask = create_car_front_mask(h,w)
            img = remove_car_front(img, mask)
            img = remove_roads(img)
            img = remove_trees(img)
            cv2.imshow('gamma img', adjust_gamma(img, 1.5))
            img_canny = canny(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)
            img_contour = contour_edges(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)

            cv2.imshow('img', img)
            cv2.imshow('canny', img_canny)
            cv2.imshow('contours', img_contour)

            key = cv2.waitKey()

            if (key == ord('x')):
                break