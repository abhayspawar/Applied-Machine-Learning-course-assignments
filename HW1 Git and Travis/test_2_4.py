# KNeighbors classifier accuracy

from sklearn import datasets, neighbors
from sklearn.model_selection import cross_val_score

def Kneighbors_score():

	iris=datasets.load_iris()
	X = iris.data
	y = iris.target
	knn=neighbors.KNeighborsClassifier(n_neighbors=5)
	sc=cross_val_score(knn,X,y,cv=5)
	return sum(sc)/len(sc)

def test_accuracy():
	assert Kneighbors_score()>=0.70

