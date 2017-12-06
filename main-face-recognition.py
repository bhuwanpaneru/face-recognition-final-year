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
from keras.layers import Dense, Dropout, Activation, Conv2D,MaxPooling2D, Flatten
import keras

print("Executing "+str(sys.argv[0]))

#TEST_IMAGE_PATH = str(sys.argv[2])
#DATASET_PATH = "D:/face-recognition-final-year/lfw_15"
#SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
#HAAR_CASCADE_PATH = "haarcascade_frontalface_default.xml"

#snippet for file testing
'''
detectedFace = faceDetection.faceDetectedMat(TEST_IMAGE_PATH, HAAR_CASCADE_PATH, SHAPE_PREDICTOR_PATH)
cv2.namedWindow("Face Landmarks Detection", cv2.WINDOW_NORMAL)
cv2.imshow("Face Landmarks Detection",detectedFace)
'''

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

#Since No more calling in the main module for image information retrieval, direct reading of Numpy array
#images, labels, labelKey, labelVal = getTrainData(DATASET_PATH)    
#print("Images length : "+str(len(images)))


images = np.load("NUMPY_OBJECTS/JAFFE_INPUT.npy")
labels = np.load("NUMPY_OBJECTS/JAFFE_OUTPUT_E.npy")
X_train, X_test, y_train, y_test = train_test_split(np.array(images),np.array(labels), train_size=0.9, random_state = 20)

X_train = np.array(X_train)
X_test = np.array(X_test)

nb_classes = np.unique(labels).shape[0]

labelKey = np.unique(labels).tolist()
labelVal = []
for v in range(0,len(labelKey)):
    labelVal.append(v)

cy_train = []
for i in y_train:
    cy_train.append(int(labelVal[labelKey.index(i)]))

cy_test = []
for i in y_test:
    cy_test.append(int(labelVal[labelKey.index(i)]))
    
Y_train = np.array(cy_train)
Y_test = np.array(cy_test)



xtrainLength = X_train.shape[0]
xtestLength = X_test.shape[0]

#For ANN - (xtestLength,150*150)
#For CNN - (xtestLength,1,150,150)
X_train = X_train.reshape(xtrainLength,150*150)
X_test = X_test.reshape(xtestLength,150*150)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255



#Artificial Neural Network Model
model = Sequential()
model.add(Dense(512,input_shape=(X_train.shape[1],)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))



#Convolution Model 
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=(1,150,150),data_format="channels_first"))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
'''


print(model.summary())

#model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])


model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, epochs=50, verbose=1, validation_data=(X_test, Y_test))
loss, accuracy = model.evaluate(X_test,Y_test, verbose=0)
print("Loss : "+str(loss))

print("Accuracy :"+str(accuracy))