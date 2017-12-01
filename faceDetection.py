#This code function returns face with landmarks list and landmarks being pointed out with circles

'''
def faceDetectedMat(imagePath,HAAR_DETECTOR_PATH, PREDICTOR_PATH):
    import numpy as np
    import cv2
    import dlib
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    face_cascade = cv2.CascadeClassifier(HAAR_DETECTOR_PATH)
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if type(face)==np.ndarray:
        x = face[0][0]
        y = face[0][1]
        w = face[0][2]
        h = face[0][3]
        rect = img[y:y+h, x:x+w]
        rect = cv2.resize(rect, (150,150))
        return rect
    else:
        return None

 '''       

        

def faceDetectedMat(imagePath,HAAR_DETECTOR_PATH, PREDICTOR_PATH):
    from imutils.face_utils import FaceAligner
    from imutils.face_utils import rect_to_bb
    import argparse
    import imutils
    import dlib
    import cv2
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
    fa = FaceAligner(predictor, desiredFaceWidth=256)    
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #face = face_cascade.detectMultiScale(img, 1.3, 5)
    rects = detector(img, 2)
    
    for rect in rects:
        (x, y, w, h) = rect_to_bb(rect)
        faceAligned = fa.align(img, img, rect)
        #resImage = faceAligned[y:y+h, x:x+w]
        resImage = cv2.resize(faceAligned, (150,150))
        return resImage
    else:
        return None
