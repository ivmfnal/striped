from nnet import Model
import numpy as np
import time

class Worker:

        Columns = ["image", "labels"]

        def __init__(self):
                self.Model = None

        def unpack_model(self, params):
                self.Model = Model.from_config(params["config"])

        def run(self, data, job, db):
                t0 = time.time()
                        
                if self.Model is None:
                        with self.Trace["create_model"]:
                                print job["model"].keys()
                                self.unpack_model(job["model"])
                model = self.Model
                
                return {"images": np.sum(data.image, axis=0)}

class Reducer:

        def __init__(self):
                self.Sum = None
                self.N = 0

        def reduce(self, nevents, data):
            if "images" in data:
                if self.Sum is None:
                    self.Sum = data["images"].copy()
                else:
                    self.Sum += data["images"]
                self.N += nevents

        def flush(self, nevents):
                return { "sum": self.Sum, "n": self.N }



                
                

        

        
