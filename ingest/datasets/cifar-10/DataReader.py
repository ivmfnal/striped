import numpy as np
from BaseDataReader import BaseDataReader
from keras.datasets import cifar10
from keras.utils import to_categorical

class DataReader(BaseDataReader):

    def __init__(self, file_path, schema):
	(x_train, y_train), _ = cifar10.load_data()
	self.x_train = np.asarray(x_train, dtype=np.float32)/256.
	self.y_train = to_categorical(y_train, 10)
        self.Schema = schema
        
    def profile(self):
	return None
                    
    def reopen(self):
        pass
        
    def nevents(self):
        return len(self.x_train)
        
    def branchSizeArray(self, bname):
	pass

    def stripesAndSizes(self, groups, bname, attr_name, attr_desc):
        src = attr_desc["source"]
	src_array = {'x': self.x_train, 'y': self.y_train }[src]
	i = 0
	for g in groups:
		yield np.ascontiguousarray(src_array[i:i+g]), None
		i += g
