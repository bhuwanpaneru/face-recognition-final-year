import faceDetection
import cv2

cv2.namedWindow("show", cv2.WINDOW_NORMAL)
cv2.imshow('show',faceDetection.faceDetectedMat("c.jpg"))
