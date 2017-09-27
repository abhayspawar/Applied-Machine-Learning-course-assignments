import numpy as np
np.random.seed(3)
from keras.layers import Dropout, Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn import datasets
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()
X = iris.data  # we only take the first two features.
y = iris.target
y=to_categorical(y,3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

hidden_size1 = [32,64,256]
hidden_size2 = [6,12,24]
epochs = [20,50,75,100]

bestscore = 0
s1 = 0
s2 = 0
e = 0
for i in hidden_size1:
    for j in hidden_size2:
        for k in epochs :
            model=Sequential([Dense(i, input_shape=(4,)),Activation('relu'), 
                              Dense(j),Activation('relu'),
                              Dense(3), Activation('softmax')])
            model.compile('adam','categorical_crossentropy', metrics=['accuracy'])

            model.fit(X_train,y_train,epochs=k,verbose=0)

            res = model.evaluate(X_test,y_test,verbose=0)
            if res[1] > bestscore:
                s1 = i
                s2 = j
                e = k
                bestscore = res[1]

model=Sequential([Dense(s1, input_shape=(4,)),Activation('relu'), 
                              Dense(s2),Activation('relu'),
                              Dense(3), Activation('softmax')])
model.compile('adam','categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train,y_train,epochs=e,verbose=0)

res = model.evaluate(X_test,y_test,verbose=0)

print(res[1])
