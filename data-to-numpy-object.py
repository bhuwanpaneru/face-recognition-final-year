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

DATASET_PATH = "D:/face-recognition-final-year/jaffeimages"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

#For database of Yale / LFW directory structure
'''

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
'''


#For database of jaffeimages 
def getTrainData(path):
    image_files = []
    labels = []
    main_dir = path
    for file in os.listdir(main_dir):
        file_path = os.path.join(main_dir,file)
        file_path = file_path.replace("\\","/")
        exp_str_full = file_path.split(".")
        exp_str = exp_str_full[1]
        exp_str = exp_str[:2]
        exp_str = exp_str.upper()
        expression = "NEUTRAL"
        if exp_str == "AN":
            expression = "ANGER"
        elif exp_str == "DI":
            expression = "DISGUSTING"
        elif exp_str =="FE":
            expression = "FEAR"
        elif exp_str == "HA":
            expression = "HAPPY"
        elif exp_str == "SA":
            expression = "SAD"
        elif exp_str == "SU":
            expression = "SURPRISED"
        else:
            expression = "NEUTRAL"
        print("analyzing " + file_path)
        x = faceDetection.faceDetectedMat(file_path, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
        if x!=None:
            image_files.append(x)
            labels.append(expression)
            
    return image_files, labels

images, labels = getTrainData(DATASET_PATH)
np.save("NUMPY_OBJECTS/JAFFE_INPUT",np.array(images))
np.save("NUMPY_OBJECTS/JAFFE_OUTPUT_E",np.array(labels))
print("Images length : "+str(len(images)))

