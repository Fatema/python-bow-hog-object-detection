################################################################################

# functionality: perform detection based on HOG feature descriptor / SVM classification
# using a very basic multi-scale, sliding window (exhaustive search) approach

# This version: (c) 2018 Toby Breckon, Dept. Computer Science, Durham University, UK
# License: MIT License

# Minor portions: based on fork from https://github.com/nextgensparx/PyBOW

################################################################################

from utils import *
from sliding_window import *

################################################################################

directory_to_cycle = os.environ['CV_HOME'] + "pedestrian/INRIAPerson/Test/pos/"

show_scan_window_process = True

################################################################################

# load dictionary and SVM data

try:
    dictionary = np.load(params.BOW_DICT_PATH)
    svm_bow = cv2.ml.SVM_load(params.BOW_SVM_PATH)
except:
    print("Missing files - dictionary and/or SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# print some checks

print("dictionary size : ", dictionary.shape)
print("svm size : ", len(svm_bow.getSupportVectors()))
print("svm_bow var count : ", svm_bow.getVarCount())


# load SVM from file

try:
    svm_hog = cv2.ml.SVM_load(params.HOG_SVM_PATH)
except:
    print("Missing files - SVM!")
    print("-- have you performed training to produce these files ?")
    exit()

# print some checks

print("svm_hog size : ", len(svm_hog.getSupportVectors()))
print("svm_hog var count : ", svm_hog.getVarCount())

pedestrian_class = params.DATA_CLASS_NAMES["pedestrian"]

################################################################################

# process all images in directory (sorted by filename)

# see the detection for a single image

for filename in sorted(os.listdir(directory_to_cycle)):

    # if it is a PNG file

    if '.png' in filename:
        print(os.path.join(directory_to_cycle, filename))

        # read image data

        img = cv2.imread(os.path.join(directory_to_cycle, filename), cv2.IMREAD_COLOR)

        # make a copy for drawing the output

        output_img = img.copy()
        bow_img = img.copy()
        hog_img = img.copy()
        just_bow_img = img.copy()
        just_hog_img = img.copy()

        # for a range of different image scales in an image pyramid

        current_scale = -1
        detections_hog_bow = []
        detections_hog = []
        detections_bow = []
        detections_just_hog = []
        detections_just_bow = []
        rescaling_factor = 1.25

        ################################ for each re-scale of the image

        for resized in pyramid(img, scale=rescaling_factor):

            # at the start our scale = 1, because we catch the flag value -1

            if current_scale == -1:
                current_scale = 1

            # after this rescale downwards each time (division by re-scale factor)

            else:
                current_scale /= rescaling_factor

            rect_img = resized.copy()

            # if we want to see progress show each scale

            if (show_scan_window_process):
                cv2.imshow('current scale',rect_img)
                cv2.waitKey(10)

            # loop over the sliding window for each layer of the pyramid (re-sized image)

            window_size = params.DATA_WINDOW_SIZE
            step = math.floor(resized.shape[0] / 16)

            if step > 0:

                ############################# for each scan window

                for (x, y, window) in sliding_window(resized, window_size, step_size=step):

                    # if we want to see progress show each scan window

                    if (show_scan_window_process):
                        cv2.imshow('current window',window)
                        key = cv2.waitKey(10) # wait 10ms

                    # for each window region get the BoW feature point descriptors

                    img_data = ImageData(window)
                    img_data.compute_hog_descriptor()
                    img_data.compute_bow_descriptors()

                    # generate and classify each window by constructing a BoW
                    # histogram and passing it through the SVM classifier

                    if img_data.hog_descriptor is not None and img_data.bow_descriptors is not None:

                        print("detecting with HOG SVM ...")
                        retval_hog, [result_hog] = svm_hog.predict(np.float32([img_data.hog_descriptor]))

                        img_data.generate_bow_hist(dictionary)

                        print("detecting with BOW SVM ...")
                        retval_bow, [result_bow] = svm_bow.predict(np.float32([img_data.bow_histogram]))

                        print('hog result', result_hog)
                        print('bow result', result_bow)

                        # if we get a detection, then record it

                        if result_hog[0] == pedestrian_class \
                                and result_bow[0] == pedestrian_class:

                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
                                cv2.imshow('current scale',rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections_hog_bow.append(rect)
                        if result_hog[0] == pedestrian_class:
                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
                                cv2.imshow('current scale', rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections_hog.append(rect)
                        if result_bow[0] == pedestrian_class:
                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
                                cv2.imshow('current scale', rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections_bow.append(rect)
                    if img_data.hog_descriptor is not None:
                        print("detecting with HOG SVM ...")

                        retval_hog, [result_hog] = svm_hog.predict(np.float32([img_data.hog_descriptor]))

                        print('hog result', result_hog)

                        # if we get a detection, then record it

                        if result_hog[0] == pedestrian_class:

                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                                cv2.imshow('current scale', rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections_just_hog.append(rect)
                    if img_data.bow_descriptors is not None:
                        img_data.generate_bow_hist(dictionary)

                        print("detecting with BOW SVM ...")
                        retval_bow, [result_bow] = svm_bow.predict(np.float32([img_data.bow_histogram]))

                        print('bow result', result_bow)

                        # if we get a detection, then record it

                        if result_bow[0] == pedestrian_class:

                            # store rect as (x1, y1) (x2,y2) pair

                            rect = np.float32([x, y, x + window_size[0], y + window_size[1]])

                            # if we want to see progress show each detection, at each scale

                            if (show_scan_window_process):
                                cv2.rectangle(rect_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
                                cv2.imshow('current scale', rect_img)
                                cv2.waitKey(40)

                            rect *= (1.0 / current_scale)
                            detections_just_bow.append(rect)
                ########################################################

        # For the overall set of detections_hog_bow (over all scales) perform
        # non maximal suppression (i.e. remove overlapping boxes etc).

        detections_hog_bow = non_max_suppression_fast(np.int32(detections_hog_bow), 0.4)
        detections_hog = non_max_suppression_fast(np.int32(detections_hog), 0.4)
        detections_bow = non_max_suppression_fast(np.int32(detections_bow), 0.4)
        detections_just_bow = non_max_suppression_fast(np.int32(detections_just_bow), 0.4)
        detections_just_hog = non_max_suppression_fast(np.int32(detections_just_hog), 0.4)

        # finally draw all the detection on the original image

        for rect in detections_hog_bow:
            cv2.rectangle(output_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 2)
        for rect in detections_hog:
            cv2.rectangle(hog_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 255), 2)
        for rect in detections_bow:
            cv2.rectangle(bow_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 255), 2)
        for rect in detections_just_bow:
            cv2.rectangle(just_bow_img, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        for rect in detections_just_hog:
            cv2.rectangle(just_hog_img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)

        cv2.imshow('detected objects',output_img)
        cv2.imshow('detected objects bow_img',bow_img)
        cv2.imshow('detected objects hog_img',hog_img)
        cv2.imshow('detected objects just_hog_img',just_hog_img)
        cv2.imshow('detected objects just_bow_img',just_bow_img)
        cv2.waitKey(0) # wait 200ms
        if (key == ord('x')):
            break

# close all windows

cv2.destroyAllWindows()

#####################################################################
