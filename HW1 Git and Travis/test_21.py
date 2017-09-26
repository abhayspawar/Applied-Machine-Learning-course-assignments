from __future__ import division
import numpy as np
def divide(a,b):
	return a/b

def divide2(a,b):
	return np.true_divide(np.array([a]),np.array([b]))
def test_divide():
	assert divide(2,8) == 0.25
def test_divide2():
	assert divide2(2,8) == 0.25
