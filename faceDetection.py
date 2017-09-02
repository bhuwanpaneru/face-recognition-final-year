def faceDetectedMat(imagePath):
    import numpy as np
    import cv2
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    face = face_cascade.detectMultiScale(img, 1.3, 5)
    for (x,y,w,h) in face:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (210,170,20), 5)
        roi_img = img[y:y+h, x:x+w]
    return img

