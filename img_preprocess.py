import math
import os

import cv2
import numpy as np



# Parameters
from sliding_window import non_max_suppression_fast

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

        masked_img = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('img1', masked_img)

        key = cv2.waitKey()

        if key == ord('w'):
            break

    # return the bitwise representation
    return masked_img


# segment roads based on the colour grey
def remove_roads(img):
    # possible parameters [59, 0, 54] [119, 255, 139], [32, 39, 39] [111, 119, 97]
    light_grey = [59, 0, 54]
    dark_grey = [119, 255, 139]
    return mask_colour_interactive(img, light_grey, dark_grey)


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


def selective_search(img, ss):
    img_copy = img.copy()

    ss.setBaseImage(img_copy)
    # ss.switchToSelectiveSearchQuality()
    ss.switchToSelectiveSearchFast()
    rects = ss.process()

    # number of region proposals to show
    # 600 seems to be a good number of rectangle to capture all important objects
    numShowRects = 600

    object_rects = []
    # iterate over all the region proposals
    for i, rect in enumerate(rects):
        # draw rectangle for region proposal till numShowRects
        if (i < numShowRects):
            x, y, w, h = rect
            rect_area = w * h
            # if the area of the rectangle is less than 3000px, more than 100000px and the height is less
            # than the width then ignore this rect and move to the next one
            if rect_area > 3000 and rect_area < 100000:
                object_rects.append(np.float32([x, y, x + w, y + h]))
                cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)
        else:
            break

    cv2.imshow('img copy', img_copy)

    # get a list of big rectangle
    big_recs = non_max_suppression_fast(np.int32(object_rects), 0.5)

    return big_recs


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
    # calculate the average over each channel
    blue_avg, green_avg, red_avg, _ = cv2.mean(img)

    print(blue_avg, green_avg, red_avg)

    # when avg = 96.63224702722886 103.317626953125 101.04474954044117 [20, 0, 0] [90, 255, 255]
    #          = 85.43271412568934 88.92509550206802 86.99965892118566 [20, 0, 0] [76, 255, 255]
    #          = 86.72567928538602 89.17838781020221 87.35499123965992 [20, 0, 0] [80, 255, 255]
    #          = 81.39078117819393 87.50255629595588 82.92397173713235 [10, 0, 0] [50, 255, 255]
    #          = 86.8916841394761 90.78615794462316 88.80653112074909 [34, 0, 0] [79, 255, 255]
    # when avg = 95.78433227539062 96.7064406451057 94.64330695657169  [24, 0, 0] [90, 255, 255]
    #          = 86.83111931295956 86.97573493508732 85.88225061753216 [28, 0, 0] [82, 255, 255]
    #          =

    # road reaching sun 82.31891228170956 92.83636115579044 90.31332397460938
    #                   96.63224702722886 103.317626953125 101.04474954044117
    #                   99.60952758789062 103.52056705250459 100.57451674517463
    #                   95.78433227539062 96.7064406451057 94.64330695657169
    #                   89.61135146197151 87.94548842486213 86.43010397518383
    #                   86.83111931295956 86.97573493508732 85.88225061753216
    # for the sun-lit road [10, 0, 0] [40, 255, 255] 62.8368350758272 69.68562047621784 66.02278585994945
    #                      [4, 0, 0] [40, 255, 255]  82.31891228170956 92.83636115579044 90.31332397460938
    #                      [0, 0, 0] [15, 255, 255]  96.63224702722886 103.317626953125 101.04474954044117 will need to not inverse the mask here
    #                      [134, 0, 0] [180, 255, 255]  96.63224702722886 103.317626953125 101.04474954044117
    #                      [95, 0, 0] [180, 255, 255]  86.8916841394761 90.78615794462316 88.80653112074909
    #                      [7, 0, 0] [36, 255, 255]  81.86627915326287 86.75275555778953 83.80855605181526
    #                      [111, 0, 0] [180, 255, 255] [10, 0, 0] [40, 255, 255] 87.12228393554688 91.03091071633732 85.47778679342831

    green_lower = 10
    green_upper = 50

    if green_avg > red_avg > blue_avg:
        green_lower = 30 - int(round(2 * green_avg - red_avg - blue_avg))
        green_upper = int(round(green_avg / 2 + red_avg / 8 + blue_avg / 16))
    elif green_avg > red_avg and blue_avg > red_avg:
        green_lower = 30 - int(round(2 * green_avg - red_avg - blue_avg))
        green_upper = int(round(green_avg / 2 + blue_avg / 4 + red_avg / 8))

    print(green_lower, green_upper)

    light_green = [green_lower, 0, 0]
    dark_green = [green_upper, 255, 255]

    return light_green, dark_green


# this is based on https://medium.com/@mrhwick/simple-lane-detection-with-opencv-bfeb6ae54ec0
def segment_road_tri(img):
    h, w = img.shape[:2]

    # a triangle that represents the view right in-front of the car across the distance
    region_of_interest_vertices = [
        (0, h * 0.7646),
        (w / 2, 0),
        (w, h * 0.7646),
    ]

    # mask the image so only the region of interest remains (just realized that I didn't use it ops :p
    # cropped_image = region_of_interest(
    #     img,
    #     np.array([region_of_interest_vertices], np.int32),
    # )

    # Convert to grayscale here.
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # enhance the grey image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # blur the image to remove some of the noise
    smoothed = cv2.GaussianBlur(clahe_img, (5, 5), 0)

    # Call Canny Edge Detection here.
    cannyed_image = cv2.Canny(smoothed, 50, 150)

    # cv2.imshow('canny image', cannyed_image)
    #
    # cv2.waitKey()

    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32)
    )

    cv2.imshow('cropped image lines', cropped_image)

    cv2.waitKey()

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

    line_img = img.copy()

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
    #         cv2.line(line_img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
    #         cv2.putText(line_img, str(round(slope, 3)), (x1 + 1, y1 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 2, cv2.LINE_AA)
    #
    # cv2.imshow('image lines', line_img)
    #
    # cv2.waitKey()

    min_y = int(img.shape[0] * (3 / 5))  # <-- Just below the horizon << I can change this to manipulate the distance of the road segmenttaion
    max_y = int(img.shape[0])  # <-- The bottom of the image

    # print(left_line_y, left_line_x)

    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    # print(right_line_y, right_line_x)

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    # use the points computed to contour the region of interest
    contours = np.array([[left_x_end, min_y], [left_x_start, max_y], [right_x_start, max_y], [right_x_end, min_y]])

    # mask the region of interest to the img
    line_image = cv2.fillPoly(img, pts=[contours], color=(0, 0, 0))

    # line_image = draw_lines(
    #     img,
    #     [[
    #         [left_x_start, max_y, left_x_end, min_y],
    #         [right_x_start, max_y, right_x_end, min_y],
    #     ]],
    #     thickness=5,
    # )

    return line_image


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8
    )
    img = np.copy(img)
    if lines is None:
        return

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img


if __name__ == '__main__':
    directory_to_cycle = os.environ['CV_HOME'] + "TTBB-durham-02-10-17-sub10/left-images"
    cv2.setUseOptimized(True)
    cv2.setNumThreads(4)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    for filename in sorted(os.listdir(directory_to_cycle)):
        if '.png' in filename:
            print(os.path.join(directory_to_cycle, filename))

            # read image data

            img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)
            img_copy = img.copy()
            # cv2.imshow('gamma img', adjust_gamma(img, 1.5))

            print('applying gamma correction')
            start_t = cv2.getTickCount()
            img = adjust_gamma(img, 1.5)
            stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            print('Processing time (ms): {}'.format(stop_t))

            h, w = img.shape[:2]

            print('creating mask for front of the car')
            start_t = cv2.getTickCount()
            mask = create_car_front_mask(h,w)
            stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            print('Processing time (ms): {}'.format(stop_t))

            print('segmenting front of the car')
            start_t = cv2.getTickCount()
            img = remove_car_front(img, mask)
            stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            print('Processing time (ms): {}'.format(stop_t))

            # apply gaussian filter
            # img = cv2.GaussianBlur(img, (5, 5), 0)

            light_green, dark_green = segmentation_options(img)

            # the order in which the colours are removed affects the general parameter for the following segmentation
            # print('segmenting trees')
            # start_t = cv2.getTickCount()
            # img = remove_trees(img,light_green, dark_green)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            print('segmenting roads')
            start_t = cv2.getTickCount()
            # img = remove_roads(img)
            img = segment_road_tri(img)
            stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            print('Processing time (ms): {}'.format(stop_t))

            cv2.imshow('no trees or roads', img)


            # img_canny = canny(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)
            # img_contour = contour_edges(img,lower_threshold=10, upper_threshold=100, smoothing_neighbourhood=3)

            # print('running selective search')
            # start_t = cv2.getTickCount()
            # big_recs = selective_search(img, ss)
            # for r in big_recs:
            #     cv2.rectangle(img_copy, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 1)
            # stop_t = ((cv2.getTickCount() - start_t) / cv2.getTickFrequency()) * 1000
            # print('Processing time (ms): {}'.format(stop_t))

            cv2.imshow('img3', img_copy)
            # cv2.imshow('canny', img_canny)
            # cv2.imshow('contours', img_contour)

            key = cv2.waitKey(40)

            if (key == ord('x')):
                break
