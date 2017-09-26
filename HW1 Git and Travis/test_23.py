import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score 
def checkscore():
	K = KNeighborsClassifier(n_neighbors = 5)
	iris = load_iris()
	cvs = cross_val_score(K,iris.data,iris.target,cv=5)
	return cvs

def test_checkscore():
	assert np.mean(checkscore())>0.7
	