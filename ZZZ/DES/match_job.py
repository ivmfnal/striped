from striped.job import SinglePointStripedSession as Session
import numpy as np
import fitsio

job_server_address = ("dbwebdev.fnal.gov", 8765) #development
#job_server_address = ("ifdb01.fnal.gov", 8765) #production

session = Session(job_server_address)

sample = fitsio.read("/data/ivm/DES/sample.fits")
np.sort(sample, order="HPIX")
sample_data = np.array(zip(sample["RA"],sample["DEC"],
        np.asarray(sample["HPIX"], dtype=np.float64)))

def callback(type, nevents, data):
    if type == "update_streams":
        if "matches" in data:
            matches = data["matches"].reshape((-1,2))
            nerrors = 0
            for idx, oid in matches:
                if oid != sample["COADD_OBJECT_ID"][idx]:
                    print "error"
                    nerrors += 1
            print nerrors, "errors total"

        
job = session.createJob("Y3A2", 
                            user_callback = callback,
                            worker_class_file="match_worker.py",
                            input_data = sample_data)
job.run()
runtime = job.TFinish - job.TStart
catalog_objects = job.EventsProcessed
print "Compared %d observations against %d catalog objects, elapsed time=%f" % (len(sample), catalog_objects, runtime)



