import random, time

class Worker:

        Columns = ["image"]

        def __init__(self, params, bulk, job_interface, db_interface):
                self.Params = params
                self.Job = job_interface
                self.Bulk = bulk

        def frame(self, data):
                #self.Job.message("Shape: %s" % (self.Bulk["data"].shape,))
                time.sleep(0.2*random.random())
                return {"count": len(data.image)}
            
                

class Accumulator:

        def __init__(self, params, bulk, job_interface, db_interface):
                self.N = 0

        def add(self, data):
                if random.random() < 0.5:
                    self.N += data.get("count", 0)
                else:
                    return {"count": data.get("count", 0)}

        def values(self):
                return {"count": self.N}