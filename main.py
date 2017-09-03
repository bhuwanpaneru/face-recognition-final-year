import faceDetection
import cv2
from matplotlib import pyplot as plt


detectedFaces = faceDetection.faceDetectedMat("b.jpg")
for i in range(0, len(detectedFaces)):
    cv2.namedWindow("Face "+str(i+1), cv2.WINDOW_NORMAL)
    cv2.imshow("Face "+str(i+1),detectedFaces[i])

