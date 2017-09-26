import io
def readfile(filename):
	f1 = io.open(filename,mode='r',encoding="utf-8")
	return len(f1.read())
def test_readfile():
	assert readfile("input.txt") == 7

