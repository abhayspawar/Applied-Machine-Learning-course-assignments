#function to read file and count characters
import io
def read_chars():
	txt=io.open('input.txt','r',encoding="utf-8")
	return len(txt.read())

def test_read():
	assert read_chars()==7