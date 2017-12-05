# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 23:01:46 2017

@author: MONIK RAJ
"""

import faceDetection
import cv2
import os
import numpy as np
import sys


print("Executing "+str(sys.argv[0]))

#TEST_IMAGE_PATH = str(sys.argv[2])
DATASET_PATH = "D:/face-recognition-final-year/ht"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

#snippet for file testing
'''
detectedFace = faceDetection.faceDetectedMat(TEST_IMAGE_PATH, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
cv2.namedWindow("Face Landmarks Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Face Landmarks Detection",detectedFace)
'''


recognizer = cv2.face.createLBPHFaceRecognizer()

def getTrainData(path):
    image_files = []
    labels = []
    labelInt_key = []
    labelInt_val = []
    main_dir = path
    v = 0
    for person in os.listdir(main_dir):
        foldername = person
        labelInt_key.append(foldername)
        labelInt_val.append(v)
        v=v+1
        person_dir = os.path.join(main_dir,foldername)
        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir,file)
            file_path = file_path.replace("\\","/")
            print("analyzing "+ file_path)
            x = faceDetection.faceDetectedMat(file_path, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
            if x!=None:
                #print(x)
                image_files.append(x)
                labels.append(foldername)           
            
    return image_files, labels, labelInt_key, labelInt_val


images, labels, labelKey, labelVal = getTrainData(DATASET_PATH)
np.save("NUMPY_OBJECTS/HT_INPUT",np.array(images))
np.save("NUMPY_OBJECTS/HT_OUTPUT_R",np.array(labels))
print("Images length : "+str(len(images)))

