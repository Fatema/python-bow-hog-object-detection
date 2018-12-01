import math
import os

import cv2
import numpy as np

# Parameters
from sliding_window import non_max_suppression_fast
from utils import *

################################################################################

BLUR = 21
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format

################################################################################

camera_focal_length_px = 399.9745178222656  # focal length in pixels
camera_focal_length_m = 4.8 / 1000          # focal length in metres (4.8 mm)
stereo_camera_baseline_m = 0.2090607502     # camera baseline in metres

################################################################################

show_scan_window_process = True

################################################################################

# load dictionary and SVM data

# try:
#     # dictionary = np.load(params.BOW_DICT_PATH)
#     # svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
#     svm = cv2.ml.SVM_load(params.HOG_SVM_PATH)
# except:
#     print("Missing files - dictionary and/or SVM!")
#     print("-- have you performed training to produce these files ?")
#     exit()

################################################################################


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
    ## blur the image to remove some of the noise
    smoothed = cv2.GaussianBlur(img, (3, 3), 0)

    # convert the img to HSV range
    hsv_img = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    # create a mask based on the colour range specified
    mask = cv2.inRange(hsv_img,
                       np.array(light_colour, dtype="uint8"),
                       np.array(dark_colour, dtype="uint8"))
    mask = cv2.dilate(mask, None)
    mask = cv2.erode(mask, None)

    if remove:
        mask = cv2.bitwise_not(mask)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # return the bitwise representation
    return masked_img


def mask_colour_interactive(img, light_colour, dark_colour, remove=True):
    # blur the image to remove some of the noise
    smoothed = cv2.GaussianBlur(img, (3, 3), 0)

    # convert the img to HSV range
    hsv_img = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)

    window_name = 'mask options'

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

    print([hue, saturation, brightness], [hue2, saturation2, brightness2])
    print(light_colour, dark_colour)

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

        masked_img = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('img1', masked_img)

        key = cv2.waitKey()

        if key == ord('w'):
            break

    # return the bitwise representation
    return masked_img


# segment trees based on the colour green
# colours were picked by using the answer provided here
# https://stackoverflow.com/questions/47483951/how-to-define-a-threshold-value-to-detect-only-green-colour-objects-in-an-image/47483966#47483966
def remove_trees(img, light_green, dark_green):
    # for first image in the shadows ([9, 0, 0] [49, 255, 255])
    return mask_colour(img, light_green, dark_green)


# this function creates a mask for the car from
def create_car_front_mask(img_height, img_width):
    mask = cv2.imread(os.path.join(os.environ['CV_HOME'], 'car_mask.jpg'), cv2.IMREAD_GRAYSCALE)
    # this kernel is used to sharpen the edges of the mask it is based on the answer found in
    # https://stackoverflow.com/questions/4993082/how-to-sharpen-an-image-in-opencv
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    mask = cv2.filter2D(mask, -1, kernel)

    # there was some noise introduced after the kernel around the mask edges, this is due to the image quality
    # so I applied the dilate and erode functions to remove the noise
    mask = cv2.dilate(mask, None)
    mask = cv2.erode(mask, None)

    # this type conversion is to make sure that it is the same as the image
    mask = mask.astype('uint8')

    # the mask is bigger than the original an image, hence the added shift (the values are based on trial and error)
    mask = mask[20:img_height + 20, 42:img_width + 42]
    return cv2.bitwise_not(mask)


def remove_car_front(img, mask):
    print('adding mask to front of car')
    return cv2.bitwise_and(img, img, mask=mask)


def selective_search(img, ss, disparity):
    img_copy = img.copy()

    img_h, img_w = imgL.shape[:2]

    ss.setBaseImage(img_copy)
    # ss.switchToSelectiveSearchQuality()
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    # number of region proposals to show
    # 600 seems to be a good number of rectangle to capture all important objects
    numShowRects = 600
    j = 0

    object_rects = []
    # iterate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            rect_area = w * h
            # if the area of the rectangle is less than 3000px, more than 100000px and the height is less
            # than the width then ignore this rect and move to the next one
            # if 2 >= h // 128 >= 0  and 6 >= w // 64 >= 0 and h > w :
            # if 370 < i < 390:

            # use disparity to check the validity of the rectangle
            # make sure the rect is within the frame of the disparity if not then shift x to 135
            min_x = x
            if min_x < 135:
                min_x = 135

            # make sure that the rect is within the boundary of the image
            max_x = min(min_x + w, img_w)
            max_y = min(y + h, img_h)

            print(max_x, max_y, '/')

            points = np.array([disparity[y, x] for y in range(y, max_y) for x in range(min_x, max_x)])
            stn_d = np.std(points)
            max_d = np.max(points)
            median_d = np.median(points)
            mean_d = np.mean(points)

            # print(i, x, y, h, w, stn_d, max_d, median_d, mean_d)

            # the rect is either very close or the height is at max 250 and min 20
            # the rect is not far away
            # the max disparity of the rect is not way way higher than the mean (80% more than the mean)
            # the width of the rect should not be more than half the image
            if (max_d > 1000 or 20 < h < 250) \
                    and max_d > 80 \
                    and np.divide(np.subtract(max_d, mean_d), max_d) < 0.57 \
                    and h / 4 < w < img_w / 2:
                print('/selected', max_d)
                j += 1
                object_rects.append(np.float32([x, y, max_x, max_y]))
                cv2.rectangle(img_copy, (min_x, y), (max_x, max_y), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(img_copy, (x, y), (max_x, max_y), (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(img_copy, str(i), (x - 1, y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255),
                        1, cv2.LINE_AA)
        else:
            break

    print(j)

    cv2.imshow('img copy', img_copy)

    # get a list of big rectangle
    big_recs = non_max_suppression_fast(np.int32(object_rects), 0.5)

    return big_recs, object_rects


def selective_search_interactive(img, ss):
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchQuality()
    rects = ss.process()

    window_name = 'selective search'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # number of region proposals to show
    # 400 seems to be a good number of rectangle to capture all important objects
    numShowRects = 400
    cv2.createTrackbar("numShowRects", window_name , numShowRects, 1000, nothing)

    while(True):
        img_copy = img.copy()
        numShowRects = cv2.getTrackbarPos("numShowRects", window_name)
        print(numShowRects)
        object_rects = []
        # iterate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                rect_area = w * h
                if rect_area > 3000 and rect_area < 100000 and h * 2 > w:
                    print('rect size', w * h)
                    object_rects.append(np.float32([x, y, x + w, y + h]))
                    # cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # print(rects[1:5])
        # object_rects = np.asarray(object_rects)
        # print(object_rects[1:5])

        big_recs = non_max_suppression_fast(np.int32(object_rects), 0.4)
        # big_recs = cv2.groupRectangles(object_rects, 100)[0]

        for r in big_recs:
            cv2.rectangle(img_copy, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 1)

        # display image
        cv2.imshow(window_name, img_copy)

        key = cv2.waitKey()

        if key == ord('q'):
            break


def segmentation_options(img):
    smoothed = cv2.GaussianBlur(img, (3, 3), 0)

    # calculate the average over each channel
    blue_avg, green_avg, red_avg, _ = cv2.mean(smoothed)

    print(blue_avg, green_avg, red_avg)

    # when avg = 96.63224702722886 103.317626953125 101.04474954044117 [20, 0, 0] [90, 255, 255]
    #          = 85.43271412568934 88.92509550206802 86.99965892118566 [20, 0, 0] [76, 255, 255]
    #          = 86.72567928538602 89.17838781020221 87.35499123965992 [20, 0, 0] [80, 255, 255]
    #          = 81.39078117819393 87.50255629595588 82.92397173713235 [10, 0, 0] [50, 255, 255]
    #          = 86.8916841394761 90.78615794462316 88.80653112074909 [34, 0, 0] [79, 255, 255]
    # when avg = 95.78433227539062 96.7064406451057 94.64330695657169  [24, 0, 0] [90, 255, 255]
    #          = 86.83111931295956 86.97573493508732 85.88225061753216 [28, 0, 0] [82, 255, 255]
    #          =

    green_lower = 10
    green_upper = 50

    if green_avg > red_avg > blue_avg:
        green_lower = 30 - int(round(2 * green_avg - red_avg - blue_avg))
        green_upper = int(round(green_avg / 2 + red_avg / 3 + blue_avg / 4))
    elif green_avg > blue_avg > red_avg:
        green_lower = 30 - int(round(2 * green_avg - red_avg - blue_avg))
        green_upper = int(round(green_avg / 2 + blue_avg / 3 + red_avg / 4))

    light_green = [green_lower, 0, 0]
    dark_green = [green_upper, 255, 255]

    return light_green, dark_green


def ignore_far_objects_interactive(img, disparity):
    h, w = img.shape[:2]

    window_name = 'masked disparity'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    max_dis = 30
    cv2.createTrackbar("max_dis", window_name , max_dis, 1000, nothing)

    while (True):
        img_copy = img.copy()

        mask = np.zeros_like(img)

        max_dis = cv2.getTrackbarPos("max_dis", window_name)
        print(max_dis)
        for y in range(0, h):
            for x in range(135, w):
                dis = disparity[y, x]
                if dis < max_dis:
                    mask[y, x] = 255

        mask = cv2.bitwise_not(mask)

        masked_image = cv2.bitwise_and(img_copy, mask)

        cv2.imshow(window_name, masked_image)

        key = cv2.waitKey()

        if key == ord('a'):
            break

    return masked_image


def ignore_far_objects(img, disparity):
    h, w = img.shape[:2]

    max_dis = 20

    mask = np.zeros_like(img)

    for y in range(0, h):
        for x in range(135, w):
            dis = disparity[y, x]
            if dis < max_dis:
                mask[y, x] = 255

    mask = cv2.bitwise_not(mask)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


# this is based on https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
def segment_road_tri(img, disparity):
    h, w = img.shape[:2]

    # a triangle that represents the view right in-front of the car across the distance
    region_of_interest_vertices = [
        (0, h * 0.71),
        (w / 2, 0),
        (w, h * 0.71),
    ]

    # Convert to grayscale here.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # enhance the grey image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # blur the image to remove some of the noise
    smoothed = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(smoothed, 50, 150)

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=25
    )

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
            if math.fabs(slope) < 0.2:  # <-- Only consider extreme slope
                continue
            if slope <= 0:  # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = 0 # <-- The top of the image
    max_y = int(img.shape[0])  # <-- The bottom of the image

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    # end of region of interest
    max_h = int(h * 0.7646)

    left_x_start = int(poly_left(max_y))
    right_x_start = int(poly_right(max_y))

    d_h, d_w = disparity.shape

    for y in range(0, max_h):
        min_x, max_x = int(poly_left(y)), int(poly_right(y))
        # the points resulted from the polygon are out of reach
        if max_x > d_w or min_x < 0 or min_x >= max_x: continue
        # find the disparity of the pixels in this row
        points = np.array([disparity[y, x] for x in range(min_x, max_x)])
        stn_d = np.std(points)
        max_d = np.max(points)
        # print(y, stn_d, max_d)
        # if the disparity greatly varies in this pixel row
        # and the maximum disparity is greater than 300 then
        # set the minimum y value used for the contour to the
        # current y value and break the loop
        if stn_d > 10 and max_d > 300:
            min_y = y
            break

    left_x_end = int(poly_left(min_y))
    right_x_end = int(poly_right(min_y))

    # use the points computed to contour the region of interest
    contours = np.array([[left_x_end, min_y], [left_x_start, max_y], [right_x_start, max_y], [right_x_end, min_y]])

    # mask the region of interest to the img
    line_image = cv2.fillPoly(img, pts=[contours], color=(0, 0, 0))

    return line_image


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def find_disparity(imgL, imgR, max_disparity=128):
    stereoProcessor = cv2.StereoSGBM_create(0, max_disparity, 21)

    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # perform preprocessing - raise to the power, as this subjectively appears
    # to improve subsequent disparity calculation

    grayL = np.power(grayL, 0.75).astype('uint8');
    grayR = np.power(grayR, 0.75).astype('uint8');

    # compute disparity image from undistorted and rectified stereo images
    # that we have loaded
    # (which for reasons best known to the OpenCV developers is returned scaled by 16)

    disparity = stereoProcessor.compute(grayL, grayR)

    # filter out noise and speckles (adjust parameters as needed)

    dispNoiseFilter = 5  # increase for more agressive filtering
    cv2.filterSpeckles(disparity, 0, 4000, max_disparity - dispNoiseFilter)

    # scale the disparity to 8-bit for viewing
    # divide by 16 and convert to 8-bit image (then range of values should
    # be 0 -> max_disparity) but in fact is (-1 -> max_disparity - 1)
    # so we fix this also using a initial threshold between 0 and max_disparity
    # as disparity=-1 means no disparity available

    _, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
    disparity_scaled = (disparity / 16.).astype(np.uint8)

    return disparity, disparity_scaled


def bow_detector(img, rects):
    img_copy = img.copy()

    ped_detections = []
    car_detections = []

    for rect in rects:
        img_copy2 = img.copy()

        rect = np.int32(rect)
        rect_img = img[rect[1]:rect[3], rect[0]:rect[2]]

        cv2.imshow('rect img', rect_img)

        cv2.waitKey(10)

        img_data = ImageData(rect_img)
        img_data.compute_bow_descriptors()

        # generate and classify each window by constructing a BoW
        # histogram and passing it through the SVM classifier

        if img_data.bow_descriptors is not None:
            img_data.generate_bow_hist(dictionary)

            print("detecting with SVM ...")

            retval, [result] = svm.predict(np.float32([img_data.bow_histogram]))

            # if we get a detection, then record it

            if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                # if we want to see progress show each detection, at each scale

                if (show_scan_window_process):
                    cv2.rectangle(img_copy2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                    cv2.imshow('detecting', img_copy2)
                    cv2.waitKey(40)

                ped_detections.append(rect)

            elif result[0] == params.DATA_CLASS_NAMES["cars"]:
                # store rect as (x1, y1) (x2,y2) pair

                # if we want to see progress show each detection, at each scale

                if (show_scan_window_process):
                    cv2.rectangle(img_copy2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.imshow('detecting', img_copy2)
                    cv2.waitKey(40)

                car_detections.append(rect)

    # For the overall set of detections (over all scales) perform
    # non maximal suppression (i.e. remove overlapping boxes etc).

    big_ped_detections = non_max_suppression_fast(np.int32(ped_detections), 0.4)
    big_car_detections = non_max_suppression_fast(np.int32(car_detections), 0.4)

    # finally draw all the detection on the original image

    for rect in big_ped_detections:
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)

    for rect in big_car_detections:
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (255, 255, 0), 2)

    cv2.imshow('detecting', img_copy)

    cv2.waitKey(10)

    return big_ped_detections, big_car_detections


def hog_detector(img, rects):
    img_copy = img.copy()

    ped_detections = []
    car_detections = []

    for rect in rects:
        img_copy2 = img.copy()

        rect = np.int32(rect)
        rect_img = img[rect[1]:rect[3], rect[0]:rect[2]]

        cv2.imshow('rect img', rect_img)

        cv2.waitKey(10)

        img_data = ImageData(rect_img)
        img_data.compute_hog_descriptor()

        # generate and classify each window by constructing a BoW
        # histogram and passing it through the SVM classifier

        if img_data.hog_descriptor is not None:

            print("detecting with SVM ...")

            retval, [result] = svm.predict(np.float32([img_data.hog_descriptor]))

            print(retval, result)

            # if we get a detection, then record it

            if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                # if we want to see progress show each detection, at each scale

                if (show_scan_window_process):
                    cv2.rectangle(img_copy2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                    cv2.imshow('detecting', img_copy2)
                    cv2.waitKey(40)

                ped_detections.append(rect)

            elif result[0] == params.DATA_CLASS_NAMES["cars"]:
                # store rect as (x1, y1) (x2,y2) pair

                # if we want to see progress show each detection, at each scale

                if (show_scan_window_process):
                    cv2.rectangle(img_copy2, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                    cv2.imshow('detecting', img_copy2)
                    cv2.waitKey(40)

                car_detections.append(rect)

    # For the overall set of detections (over all scales) perform
    # non maximal suppression (i.e. remove overlapping boxes etc).

    big_ped_detections = non_max_suppression_fast(np.int32(ped_detections), 0.4)
    big_car_detections = non_max_suppression_fast(np.int32(car_detections), 0.4)

    # finally draw all the detection on the original image

    for rect in big_ped_detections:
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)

    for rect in big_car_detections:
        cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)

    cv2.imshow('detected objects', img_copy)

    cv2.waitKey(10)

    return big_ped_detections, big_car_detections


# adjust the brightness and contrast of an image based on its histogram
def adjust_contrast_gray(img, clip_hist_percent=0):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hist_size = 256

    # calculate histogram
    hist_range = [0, 256]
    accumulate = False

    hist = cv2.calcHist(gray_img, [0], None, [hist_size], hist_range, accumulate=accumulate)

    norm_gray_image = cv2.normalize(gray_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    gray_equ = cv2.equalizeHist(gray_img)


    # calculate cumulative distribution from the histogram
    accumulator = [hist[0]]
    for i in range(1, hist_size):
        accumulator += [accumulator[i - 1] + hist[i]]

    # locate points that cuts at required value float
    max_ = accumulator[len(accumulator) - 1]
    clip_hist_percent *= (max_ / 100.0) # make percent as absolute
    clip_hist_percent /= 2.0 # left and right wings

    # locate left cut
    min_gray = 0
    while accumulator[min_gray] < clip_hist_percent:
        min_gray += 1

    # locate right cut
    max_gray = hist_size - 1
    while accumulator[max_gray] >= (max_ - clip_hist_percent):
        max_gray -= 1

    # current range
    input_range = max_gray - min_gray

    alpha = (hist_size - 1) / input_range # alpha expands current range to histsize range
    beta = -min_gray * alpha # beta shifts current range so that minGray will go to 0

    # Apply brightness and contrast normalization
    cons_gray_img = np.zeros_like(gray_img)
    cons_gray_img = cv2.convertScaleAbs(gray_img, cons_gray_img, alpha, beta)

    cv2.imshow('img', img)
    cv2.imshow('norm_image', norm_image)
    cv2.imshow('gray_img', gray_img)
    cv2.imshow('norm_gray_image', norm_gray_image)
    cv2.imshow('cons_gray_img', cons_gray_img)
    cv2.imshow('gray_equ', gray_equ)

    # cv2.waitKey()

    return gray_img


if __name__ == '__main__':
    left_imgs = os.environ['CV_HOME'] + "TTBB-durham-02-10-17-sub10/left-images"
    right_imgs = os.environ['CV_HOME'] + "TTBB-durham-02-10-17-sub10/right-images"
    right_imgs = os.environ['CV_HOME'] + "TTBB-durham-02-10-17-sub10/right-images"

    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    max_disparity = 128

    for filename_left in sorted(os.listdir(left_imgs)):

        filename_right = filename_left.replace("_L", "_R")
        full_path_filename_left = os.path.join(left_imgs, filename_left)
        full_path_filename_right = os.path.join(right_imgs, filename_right)

        if '.png' in filename_left and os.path.isfile(full_path_filename_right):
            print(full_path_filename_left)
            print(full_path_filename_right)

            # read image data

            imgL = cv2.imread(full_path_filename_left, cv2.IMREAD_COLOR)
            imgR = cv2.imread(full_path_filename_right, cv2.IMREAD_COLOR)

            disparity, disparity_scaled = find_disparity(imgL, imgR)
            cv2.imshow("disparity", (disparity_scaled * (255. / max_disparity)).astype(np.uint8))

            img_copy = imgL.copy()

            # cv2.imshow('gamma imgL', adjust_gamma(imgL, 1.5))

            # print('applying gamma correction')
            # start_t = cv2.getTickCount()
            imgL = adjust_gamma(imgL, 1.5)
            # adjust_contrast_gray(imgL, clip_hist_percent=1)

            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            h, w = imgL.shape[:2]

            # print('creating mask for front of the car')
            # start_t = cv2.getTickCount()
            mask = create_car_front_mask(h,w)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            # print('segmenting front of the car')
            # start_t = cv2.getTickCount()
            imgL = remove_car_front(imgL, mask)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            # apply gaussian filter
            # imgL = cv2.GaussianBlur(imgL, (5, 5), 0)

            # print('segmenting roads')
            # start_t = cv2.getTickCount()
            # imgL = remove_roads(imgL)
            imgL = segment_road_tri(imgL, disparity)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            imgL = ignore_far_objects(imgL, disparity)

            # light_green, dark_green = segmentation_options(imgL)
            # print(light_green, dark_green)
            # the order in which the colours are removed affects the general parameter for the following segmentation
            # print('segmenting trees')
            # start_t = cv2.getTickCount()
            # imgL = remove_trees(imgL,light_green, dark_green)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            cv2.imshow('no trees or roads', imgL)

            # img_canny = canny(imgL,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)
            # img_contour = contour_edges(imgL,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)

            # print('running selective search')
            # start_t = cv2.getTickCount()
            big_rects, rects = selective_search(imgL, ss, disparity)
            # for r in big_rects:
            #     cv2.rectangle(img_copy, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 1)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            # detections_rects1 = hog_detector(img_copy, rects)
            # detections_rects2 = hog_detector(img_copy, big_rects)

            cv2.imshow('img3', img_copy)
            # cv2.imshow('canny', img_canny)
            # cv2.imshow('contours', img_contour)

            key = cv2.waitKey()

            if (key == ord('x')):
                break
