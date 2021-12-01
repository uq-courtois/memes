import cv2
import numpy as np
import os
from multiprocessing import Pool
import time
import imutils
from datetime import datetime

def processimgs(template_target):

    basepath = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(basepath, "source_images_clean")

    templates = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templates.append(file)

    orb = cv2.ORB_create()

    fileloc1 = os.path.join(filepath, template_target)
    #print(fileloc1)
    img1 = cv2.imread(fileloc1)
    img1 = imutils.resize(img1, width=800)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    kp1, des1 = orb.detectAndCompute(img1, None)

    if type(des1) is np.ndarray:

        matchdir = 'matches'

        donepath = os.path.join(basepath, matchdir)

        done = []

        for r, d, f in os.walk(donepath):
            for file in f:
                if '.DS_Store' not in file:
                    if '.txt' in file:
                        doneitem = file.replace('.txt','.jpg')
                        done.append(doneitem)

        #print(done)

        print('Processing',template_target,'-',len(done),'in process/done','-',datetime.now())

        for indexb,template_compare in enumerate(templates):
            if template_target != template_compare:
                if template_compare not in done:

                    processfileloc = os.path.join(basepath, matchdir, template_target.replace('.jpg','.txt'))
                    processfile = open(processfileloc,"w")
                    processfile.close()

                    fileloc2 = os.path.join(filepath, template_compare)
                    #print(fileloc2)
                    img2 = cv2.imread(fileloc2)
                    img2 = imutils.resize(img2, width=800)
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                    kp2, des2 = orb.detectAndCompute(img2, None)

                    if type(des2) is np.ndarray:

                        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                        matches = bf.match(des1, des2)

                        goodmatches = []

                        for match in matches:
                            if match.distance <= 25:
                                goodmatches.append(match)

                        if len(goodmatches) >= 10:
                            print('Matched:',template_target,template_compare)
                            matching_result = cv2.drawMatches(img1, kp1, img2, kp2, goodmatches, None, flags=2)
                            targetdir = matchdir
                            savename = template_target.replace('.jpg','') + '_' + template_compare.replace('.jpg','') + '_' + str(len(goodmatches)) + '.jpg'
                            fileloc3 = os.path.join(basepath, targetdir, savename)
                            cv2.imwrite(fileloc3, matching_result)

if __name__ == '__main__':

    start = time.time()

    basepath = os.path.dirname(os.path.realpath(__file__))
    subdir = "source_images_clean"
    filepath = os.path.join(basepath, subdir)

    templatesbase = []

    for r, d, f in os.walk(filepath):
        for file in f:
            if '.DS_Store' not in file:
                templatesbase.append(file)

    p = Pool()

    executefunction = p.map(processimgs,templatesbase)

    p.close()
    p.join()

    runtime = time.time()-start
    print('Script runtime:',round(runtime,2),'seconds',round(runtime/3600,2),'hours')
