class BaseDataReader(object):
    # Abstract base class for the data reader

    def __init__(self, file_path):
	self.FilePath = file_path

    
    def metadata(self):
	# metadata to apply to all frames produced from the file
        return None

    def fileID(self):
	return self.FilePath

    def stripesAndSizes(self, group_sizes, branch, attr_name, descriptor):
        raise NotImplementedError
        
    def branchSizeArray(self, branch_name, bdesc):
        raise NotImplementedError
        
    def nevents(self):
        raise NotImplementedError
                

