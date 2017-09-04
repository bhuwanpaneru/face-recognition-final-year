import faceDetection
import cv2

TEST_IMAGE_PATH = "1.pgm"
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

detectedFace, landmarkPoints = faceDetection.faceDetectedMat(TEST_IMAGE_PATH, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
cv2.namedWindow("Face Landmarks Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Face Landmarks Detection",detectedFace)

