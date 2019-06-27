from nnet import Model
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
                
                x = data.image
                y = data.labels

                job.message("end of frame, time=%f" % (time.time() - t0))

                
                

        

        
