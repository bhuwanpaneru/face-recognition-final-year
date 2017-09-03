def faceDetectedMat(imagePath):
    import numpy as np
    import cv2
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    roi_img = []
    for (x,y,w,h) in face:
        roi_img.append(img[y:y+h, x:x+w])
    return roi_img

