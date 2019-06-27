from striped.common import Tracer
T = Tracer()

with T["run"]:
        with T["imports"]:
                        from striped.job import SinglePointStripedSession as Session
                        import numpy as np
                        from numpy.lib.recfunctions import append_fields
                        import fitsio, healpy as hp
                        import sys, time


        #job_server_address = ("dbwebdev.fnal.gov", 8765) #development
        job_server_address = ("ifdb01.fnal.gov", 8765) #production

        session = Session(job_server_address)

        input_file = sys.argv[1]
        input_filename = input_file.rsplit("/",1)[-1].rsplit(".",1)[-1]

        with T["fits/read"]:
                input_data = fitsio.read(input_file, ext=2, columns=["ALPHAWIN_J2000","DELTAWIN_J2000"])
        with T["hpix"]:
                hpix = hp.ang2pix(nside=16384,theta=input_data['ALPHAWIN_J2000'],phi=input_data['DELTAWIN_J2000'],
                        lonlat=True, nest=True)

        hpix = np.asarray(hpix, np.float64)
        input_data = append_fields(input_data, "HPIX", hpix)
        np.sort(input_data, order="HPIX")

        input_data = np.array(zip(input_data['ALPHAWIN_J2000'], input_data['DELTAWIN_J2000'], input_data['HPIX']))
        matches = []

        class Callback:

            def on_streams_update(self, nevents, data):
                        if "matches" in data:
                            for m in data["matches"]:
                                    matches.append(m)
                                    for obs_i, cat_id, obs_ra, obs_dec, cat_ra, cat_dec in m:
                                        print "Match:      index: %10d    RA: %9.4f    Dec: %9.4f" % (int(obs_i), obs_ra, obs_dec)
                                        print "   COADD oject id: %10d        %9.4f         %9.4f" % (int(cat_id), cat_ra, cat_dec)
                        if "message" in data:
                            for msg in data["message"]:
                                print msg
            def on_exception(self, wid, info):
                print "Worker exception:\n--------------------"
                print info
                print "--------------------"                   

                
        job = session.createJob("Y3A2", 
                                    user_callback = Callback(),
                                    worker_class_file="bliss_match_worker.py",
                                    user_params = {"observations":input_data})
        with T["job"]:
                job.run()
        runtime = job.TFinish - job.TStart
        catalog_objects = job.EventsProcessed
        print "Compared %d observations against %d catalog objects, elapsed time=%f" % (len(input_data), catalog_objects, runtime)


        if matches:
            matches = np.concatenate(matches, axis=0)
            matches = np.array(matches, dtype=[("INDEX", int),("COADD_OBJECT_ID", int)])
            save_fn = input_filename + "_match.fits"
            with T["fits/write"]:
                fitsio.write(save_fn, matches, clobber=True)
            print "Saved %d matches in %s" % (len(matches), save_fn)
        else:
            print "No matches"

T.printStats()







