import numpy as np
np.random.seed(3)
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import datasets
from sklearn.cross_validation import train_test_split


import pandas as pd
from keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train=X_train.reshape(60000,784)
X_train=X_train.astype('float32')
X_train/=255
X_test=X_test.reshape(10000,784)
X_test=X_test.astype('float32')
X_test/=255
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

model=Sequential([Dense(256, input_shape=(784,)),Activation('relu'),
                                  Dense(24),Activation('relu'),
                                  Dense(10), Activation('softmax')])
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X_train,y_train,epochs=10,verbose=0)
res = model.evaluate(X_test,y_test,verbose=0)
print("test score without dropout")
print(res[1])


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.figure(0)
plt.plot(pd.DataFrame(history.history))
plt.xlabel("epochs")
plt.ylabel("loss and accuracy")
plt.savefig("p2nodropout.png")



model2=Sequential([Dense(256, input_shape=(784,)),Activation('relu'),Dropout(0.5),
                                  Dense(24),Activation('relu'),Dropout(0.5),
                                  Dense(10), Activation('softmax')])
model2.compile('adam','categorical_crossentropy', metrics=['accuracy'])
history2=model2.fit(X_train,y_train,epochs=10,verbose=0)
res2 = model2.evaluate(X_test,y_test,verbose=0)
print("test score with dropout")
print(res2[1])

matplotlib.use('Agg')
plt.figure(1)
plt.plot(pd.DataFrame(history2.history))
plt.xlabel("epochs")
plt.ylabel("loss and accuracy")
plt.savefig("p2dropout.png")

