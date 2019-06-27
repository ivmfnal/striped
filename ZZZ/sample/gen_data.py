import numpy as np
import sys

array = np.random.random((10000,))*200.0
array = np.asarray(array, dtype="<f4")
print np.mean(array)
open(sys.argv[1], "wb").write(array.data)
