from __future__ import division 
import numpy as np

def divide_num():
	return 2/8
	
def divide_arr():
	return np.true_divide(np.array([2.0]),np.array([8]))

def test_1():
	assert divide_num()==0.25

def test_2():
	assert divide_arr()==0.25

