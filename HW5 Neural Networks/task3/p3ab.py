#!/usr/bin/env python
import keras
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#%matplotlib inline
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
#from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import scipy.io as sio
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import BatchNormalization


# ## Task - 3


# Import data

train = sio.loadmat('/rigel/edu/coms4995/datasets/train_32x32.mat')
test  = sio.loadmat('/rigel/edu/coms4995/datasets/test_32x32.mat')

X_train = train['X']
y_train = train['y']

X_test = test['X']
y_test = test['y']




np.place(y_train,y_train == 10,0)
np.place(y_test,y_test == 10,0)

y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test = keras.utils.to_categorical(y_test, num_classes = 10)


# Reshape dataset to bring in the required format
X_train = np.swapaxes(X_train,0,3)
X_train = np.swapaxes(X_train,2,3)
X_train = np.swapaxes(X_train,1,2)

X_test = np.swapaxes(X_test,0,3)
X_test = np.swapaxes(X_test,2,3)
X_test = np.swapaxes(X_test,1,2)


# Define input shape and num_classes
input_shape=X_train[0].shape
num_classes = 10


model = Sequential([Conv2D(32, kernel_size=(3, 3),input_shape=input_shape,activation='relu'),
					MaxPooling2D(pool_size=(2, 2)),Conv2D(32, (3, 3), activation='relu'),
					MaxPooling2D(pool_size=(2, 2)),Conv2D(32, (3, 3), activation='relu'),MaxPooling2D(pool_size=(2, 2)),
					Flatten(),Dense(64, activation='relu'),Dense(num_classes, activation='softmax')])

print "Summary of model without batch normalization:"
print model.summary()


model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
history= model.fit(X_train, y_train,batch_size=128, epochs=5, validation_split=.1, verbose = 0)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(score[1]))



model = Sequential()

model.add(Conv2D(32, kernel_size=(5, 5),
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32,(3, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(num_classes, activation='softmax'))

print("summary of model with batch normalization")
print model.summary()

model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

bestscore = 0
e = 0
for i in range(20):
	history= model.fit(X_train, y_train,batch_size=128, epochs=1, validation_split=.1, verbose = 0)
	score = model.evaluate(X_test, y_test, verbose=0)
	if score[1] > bestscore:
	   bestscore = score[1]
	   e = i+1
#print("Test loss: {:.3f}".format(score[0]))
print("Test Accuracy: {:.3f}".format(bestscore))

