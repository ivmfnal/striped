class ProgressBarCallback(object):

    def __init__(self, force_text = False, desc="Events processed",
            unit_scale = True):
        self.NTotal = 0
        self.NDone = 0
        self.Bar = None
        self.ForceText = force_text
        self.Desc = desc
        self.UnitScale = unit_scale
        
    def on_job_start(self, job):
        self.NTotal = job.EventsToProcess
        if not self.ForceText and job.IPython:
            from tqdm import tqdm_notebook as tqdm
        else:
            from tqdm import tqdm
        self.Bar = tqdm(total=self.NTotal, desc=self.Desc,
            unit_scale = self.UnitScale)

    def on_update(self, nevents):
        self.update(nevents)    
    
    def on_job_finish(self, nevents, error):
        self.update(nevents)
        self.Bar.close()

    def update(self, nprocessed):
        delta = nprocessed - self.NDone
        self.NDone = nprocessed
        self.Bar.update(delta)
        
class PrintCallback(object):

    def on_job_start(self, job):
        self.Job = job
        self.EventsToProcess = job.EventsToProcess
        print("job started: jid=%s, %d total events, %d events to process, dataset: %s" % (
            job.JID, job.EventsInDataset, job.EventsToProcess, job.DatasetName))
        
    def on_update(self, nevents):
        print("%d events processed = %.2f%%" % (nevents, float(nevents)/float(self.EventsToProcess)*100.0))
        
    def on_job_finish(self, nevents, error):
        print("Job finished, %d events processed" % (nevents,))
        if error:
            print("Error: %s" % (error,))

class TraceAllCallback(object):

    def on_callback(self, name, *params, **args):
        print("%s %s %s" % (name, params, args))
        
class HistogramUpdateCallback(object):

    def __init__(self, display):
        self.Display = display
        display.init()
        
    def on_histograms_update(self, nevents):
        self.Display.update()    
    
