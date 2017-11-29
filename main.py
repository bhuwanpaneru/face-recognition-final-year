import faceDetection
import cv2
import os
import numpy as np
import sys
from skimage import io
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

print("Executing "+str(sys.argv[0]))

#TEST_IMAGE_PATH = str(sys.argv[2])
DATASET_PATH = "D:/face-recognition-final-year/Yale database"
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

c = np.array(images)
labelNum = []
for i in labels:
    labelNum.append(int(labelVal[labelKey.index(i)]))

    
print("Images length : "+str(len(images)))
'''
recognizer.train(images, np.array(labelNum))
tImg = faceDetection.faceDetectedMat(TEST_IMAGE_PATH, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
#print(tImg)
name_predicted = recognizer.predict(tImg)
print("The predicted Name : " + str(name_predicted) +" - "+ str(labelKey[int(name_predicted)]))
print("Execution completed")
'''


X_train, X_test, y_train, y_test = train_test_split(np.array(images),np.array(labels), train_size=0.9, random_state = 20)

X_train = np.array(X_train)
X_test = np.array(X_test)

'''
nb_classes = 43
y_train = np.array(y_train) 
y_test = np.array(y_test)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
'''

nb_classes = len(labelKey)



cy_train = []
for i in y_train:
    cy_train.append(int(labelVal[labelKey.index(i)]))

cy_test = []
for i in y_test:
    cy_test.append(int(labelVal[labelKey.index(i)]))
    
Y_train = cy_train
Y_test = cy_test
    
xtrainLength = X_train.shape[0]
xtestLength = X_test.shape[0]
X_train = X_train.reshape(xtrainLength, 150*150)
X_test = X_test.reshape(xtestLength, 150*150)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

'''
model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
'''

model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print(model.summary())

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=50, verbose=1, validation_data=(X_test, Y_test))
loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)

print("Loss : "+str(loss))

print("Accuracy :"+str(accuracy))