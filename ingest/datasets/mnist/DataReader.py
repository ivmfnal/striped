import numpy as np
from striped.ingestion import BaseDataReader
from keras.datasets import mnist
from keras.utils import to_categorical

class DataReader(BaseDataReader):

    def __init__(self, file_path, schema):
	assert file_path in ("train", "test")
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	if file_path is "train":
		self.x = np.asarray(x_train, dtype=np.float32)/256.
		self.y = to_categorical(y_train, 10)
	else:
		self.x = np.asarray(x_test, dtype=np.float32)/256.
		self.y = to_categorical(y_test, 10)
        self.Schema = schema
        
    def profile(self):
	return None
                    
    def reopen(self):
        pass
        
    def nevents(self):
        return len(self.x)
        
    def branchSizeArray(self, bname):
	pass

    def stripesAndSizes(self, groups, bname, attr_name, attr_desc):
        src = attr_desc["source"]
	src_array = {'x': self.x, 'y': self.y }[src]
	i = 0
	for g in groups:
		yield np.ascontiguousarray(src_array[i:i+g]), None
		i += g
