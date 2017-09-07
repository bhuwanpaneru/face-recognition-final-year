import faceDetection
import cv2
import os
import numpy as np

TEST_IMAGE_PATH = "2.pgm"
DATASET_PATH = "dbc"
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
    main_dir = path
    for person in os.listdir(main_dir):
        foldername = person
        person_dir = os.path.join(main_dir,foldername)
        for file in os.listdir(person_dir):
            file_path = os.path.join(person_dir,file)
            print(file_path)
            x = faceDetection.faceDetectedMat(file_path, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
            if x!=None:
                print(x)
                image_files.append(x)
                labels.append(foldername)           
            
    return image_files, labels


images, labels = getTrainData(DATASET_PATH)
recognizer.train(images, np.array(labels))

name_predicted, cnf = recognizer.predict(faceDetection.faceDetectedMat(TEST_IMAGE_PATH, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH))

print("The predicted Name : " + str(name_predicted)+" with confidence of "+str(cnf))

