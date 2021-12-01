# Credit: adapted from https://gist.github.com/eLtronicsVilla/b089a0134193524f208e88a811c0f88d

import cv2
import numpy as np
import os
from multiprocessing import Pool
from imutils.object_detection import non_max_suppression
import time
from datetime import datetime

### Text blur function

def blurtext(filename):

    try:

        print('Processing',filename)

        basepath = os.path.dirname(os.path.realpath(__file__))
        fileloc = os.path.join(basepath, "source_images", filename)
        image = cv2.imread(fileloc)

        norm_img = np.zeros((300, 300))
        image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)

        eighth = image.size // 8

        orig = image.copy()
        (H, W) = image.shape[:2]

        # set the new width and height and then determine the ratio in change
        # for both the width and height
        (newW, newH) = (320*4, 320*4)
        rW = W / float(newW)
        rH = H / float(newH)

        # resize the image and grab the new image dimensions
        image = cv2.resize(image, (newW, newH))
        (H, W) = image.shape[:2]

        # define the two output layer names for the EAST detector model that
        # we are interested -- the first is the output probabilities and the
        # second can be used to derive the bounding box coordinates of text
        layerNames = [
                "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]

        basepath = os.path.dirname(os.path.realpath(__file__))
        textdetect = os.path.join(basepath, 'frozen_east_text_detection.pb')
        net = cv2.dnn.readNet(textdetect)

        # construct a blob from the image and then perform a forward pass of
        # the model to obtain the two output layer sets
        blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False)
        start = time.time()
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        end = time.time()

        # grab the number of rows and columns from the scores volume, then
        # initialize our set of bounding box rectangles and corresponding
        # confidence scores
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        # loop over the number of rows
        for y in range(0, numRows):
                # extract the scores (probabilities), followed by the geometrical
                # data used to derive potential bounding box coordinates that
                # surround text
                scoresData = scores[0, 0, y]
                xData0 = geometry[0, 0, y]
                xData1 = geometry[0, 1, y]
                xData2 = geometry[0, 2, y]
                xData3 = geometry[0, 3, y]
                anglesData = geometry[0, 4, y]

                # loop over the number of columns
                for x in range(0, numCols):
                        # if our score does not have sufficient probability, ignore it
                        if scoresData[x] < 0.25:
                                continue

                        # compute the offset factor as our resulting feature maps will
                        # be 4x smaller than the input image
                        (offsetX, offsetY) = (x * 4.0, y * 4.0)

                        # extract the rotation angle for the prediction and then
                        # compute the sin and cosine
                        angle = anglesData[x]
                        cos = np.cos(angle)
                        sin = np.sin(angle)

                        # use the geometry volume to derive the width and height of
                        # the bounding box
                        h = xData0[x] + xData2[x]
                        w = xData1[x] + xData3[x]

                        # compute both the starting and ending (x, y)-coordinates for
                        # the text prediction bounding box
                        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                        endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                        startX = int(endX - w)
                        startY = int(endY - h)

                        # add the bounding box coordinates and probability score to
                        # our respective lists
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scoresData[x])

        # apply non-maxima suppression to suppress weak, overlapping bounding
        boxes = non_max_suppression(np.array(rects), probs=confidences,overlapThresh=0.80)

        # Sharpen image

        kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])

        orig = cv2.filter2D(orig, -1, kernel)

        surfaces = []

        for (startX, startY, endX, endY) in boxes:
            box = [[[]]]
            # scale the bounding box coordinates based on the respective
            # ratios
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)

            surfaces.append((endX-startX)*(endY-endX))

            try:

                scope = 20

                if startX-int((round(1.5*scope,0))) > 0 and startY-int((round(1.5*scope,0))) > 0 and endX+(round(1.5*scope,)) < W and endY+int((round(1.5*scope,0))) < H:
                    ROI = orig[startY-int((round(1.5*scope,0))):endY+int((round(1.5*scope,0))),startX-int((round(1.5*scope,0))):endX+int((round(1.5*scope,0)))]
                    blur = cv2.GaussianBlur(ROI, (19,19),4)
                    orig[startY-int((round(1.5*scope,0))) :endY+int((round(1.5*scope,0))),startX-int((round(1.5*scope,0))) :endX+int((round(1.5*scope,0)))] = blur

                else:
                    ROI = orig[startY:endY,startX:endX]
                    blur = cv2.GaussianBlur(ROI, (19,19),4)
                    orig[startY:endY,startX:endX] = blur

            except:
                print('>>> Error on image blurring',filename)

        targetloc = os.path.join(basepath, 'source_images_clean',filename)
        cv2.imwrite(targetloc,orig)

    except:
        print('Error on',filename)

### Run script

if __name__ == '__main__':

    print('Image cleaning script initiated',datetime.now())

    ### Read files

    basepath = os.path.dirname(os.path.realpath(__file__))

    subdir = "source_images"
    filepath = os.path.join(basepath, subdir)

    try:
        os.mkdir(os.path.join(basepath, 'source_images_clean'))
    except:
        pass

    files = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.jpg' in file:
                files.append(file)

    start = time.time()

    p = Pool()

    executefunction = p.map(blurtext,files)

    p.close()
    p.join()

    runtime = time.time()-start

    print('Script runtime:',round(runtime,2),'seconds',round(runtime/3600,2),'hours')
