import pandas as pd
import numpy as np

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

model = VGG16(weights='imagenet', include_top=False)

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('/rigel/edu/coms4995/datasets/pets') if isfile(join('/rigel/edu/coms4995/datasets/pets', f))]

onlyfiles.remove('Abyssinian_100.mat')
onlyfiles.remove('Abyssinian_101.mat')
onlyfiles.remove('Abyssinian_102.mat')
p=len(onlyfiles)

images=np.array([image.img_to_array(image.load_img('/rigel/edu/coms4995/datasets/pets/'+onlyfiles[i],target_size=(224,224))) for i in range(p)])

ids=[]
for files in onlyfiles:
    for i in range(len(files)):
        if files[len(files)-i-1]=='_':
            category=files[0:len(files)-i-1]
            ids.append(category)
            break
y=pd.factorize(ids)[0]

X_pre=preprocess_input(images)

features=model.predict(X_pre)
feat_train=features.reshape(p,-1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(feat_train, y[0:p], stratify=y[0:p])
print('Fitting model ')
rf=RandomForestClassifier(n_estimators=200, criterion='entropy', max_depth=10, min_samples_split=30, min_samples_leaf=30)

rf.fit(X_train,y_train)

print(rf.score(X_train,y_train))
print(rf.score(X_test,y_test))
#Accuracy 73%

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(X_train,y_train)
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))
#Accuracy 87%