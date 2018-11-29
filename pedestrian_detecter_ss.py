#####################################################################

# Example : performs selective search bounding box identification

# Author : Toby Breckon, toby.breckon@durham.ac.uk
# Copyright (c) 2018 Department of Computer Science, Durham University, UK

# License: MIT License

# ackowledgements: based on the code and examples presented at:
# https://www.learnopencv.com/selective-search-for-object-detection-cpp-python/

#####################################################################

import cv2
import os
import sys
import math
import numpy as np

#####################################################################

# press all the go-faster buttons - i.e. speed-up using multithreads
from utils import *

cv2.setUseOptimized(True);
cv2.setNumThreads(4);

#####################################################################

directory_to_cycle = os.environ['CV_HOME'] + "pedestrian/INRIAPerson/Test/pos/"

# load dictionary and SVM data

try:
    dictionary = np.load(params.BOW_DICT_PATH)
    svm = cv2.ml.SVM_load(params.BOW_SVM_PATH)
except:
    print("Missing files - dictionary and/or SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# print some checks

print("dictionary size : ", dictionary.shape)
print("svm size : ", len(svm.getSupportVectors()))
print("svm var count : ", svm.getVarCount())

#####################################################################

# create Selective Search Segmentation Object using default parameters

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

#####################################################################

# loop all images in directory (sorted by filename)

for filename in sorted(os.listdir(directory_to_cycle)):

    # if it is a PNG file

    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename));

        # read image from file

        frame = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

        # start a timer (to see how long processing and display takes)

        start_t = cv2.getTickCount();

        # set input image on which we will run segmentation

        ss.setBaseImage(frame)

        # Switch to fast but low recall Selective Search method
        # ss.switchToSelectiveSearchFast()

        # Switch to high recall but slow Selective Search method (slower)
        ss.switchToSelectiveSearchQuality()

        # run selective search segmentation on input image
        rects = ss.process()
        print('Total Number of Region Proposals: {}'.format(len(rects)))

        # number of region proposals to show
        numShowRects = 100

        detections = []

        # iterate over all the region proposals
        for i, rect in enumerate(rects):
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect

                window = frame[y:y + h, x:x + w]

                img_data = ImageData(window)
                img_data.compute_bow_descriptors()

                # generate and classify each window by constructing a BoW
                # histogram and passing it through the SVM classifier

                if img_data.bow_descriptors is not None:
                    img_data.generate_bow_hist(dictionary)

                    print("detecting with SVM ...")

                    retval, [result] = svm.predict(np.float32([img_data.bow_histogram]))

                    print(result)

                    # if we get a detection, then record it

                    if result[0] == params.DATA_CLASS_NAMES["pedestrian"]:

                        # store rect as (x1, y1) (x2,y2) pair

                        rect = np.float32([x, y, x + w, y + h])

                        detections.append(rect)

                    # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break

        # display image

        for rect in detections:
            cv2.rectangle(frame, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)

        cv2.imshow('Selective Search - Object Region Proposals', frame);

        # stop the timer and convert to ms. (to see how long processing and display takes)

        stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000;

        print('Processing time (ms): {}'.format(stop_t))
        print()

        # start the event loop - essential

        # cv2.waitKey() is a keyboard binding function (argument is the time in milliseconds).
        # It waits for specified milliseconds for any keyboard event.
        # If you press any key in that time, the program continues.
        # If 0 is passed, it waits indefinitely for a key stroke.
        # (bitwise and with 0xFF to extract least significant byte of multi-byte response)
        # here we use a wait time in ms. that takes account of processing time already used in the loop

        # wait 40ms or less depending on processing time taken (i.e. 1000ms / 25 fps = 40 ms)

        key = cv2.waitKey(max(40, 40 - int(math.ceil(stop_t)))) & 0xFF;

        # It can also be set to detect specific key strokes by recording which key is pressed

        # e.g. if user presses "x" then exit / press "f" for fullscreen

        if (key == ord('x')):
            break
        elif (key == ord('f')):
            cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

        ss.clearImages()

# close all windows

cv2.destroyAllWindows()

#####################################################################
