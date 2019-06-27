from ..client import StripedClient
from ..common import Lockable, synchronized
from ..common import JobTracer as JT
from striped.hist import HAggregator
import cPickle, base64, time, sys

class UserCallbackList(object):

    class _FuncAsCallback(object):

        def __init__(self, func):
            self.F = func

        def on_histogram_update(self, nevents):
            return self.F("histogram_update", nevents, None)

        def on_stream_update(self, nevents, data):
            return self.F("stream_update", nevents, data)

        def on_exception(self, wid, info):
            return self.F("exception", None, {"worker_id":wid, "info":info})
            
        def on_data(self, wid, nevents, data):
            return self.F("data", nevents, data)

        def on_data_load_failure(self, wid, info):
            return self.F("data_load_failure", None, {"worker_id":wid, "info":info})

    def __init__(self, callbacks):
        self.Callbacks = []
        for c in callbacks:
            self.addCallback(c)
                            
    def addCallback(self, callback):
        self.Callbacks.append(self._FuncAsCallback(callback) if hasattr(callback, "__code__") 
                    else callback)

    def callback(self, name, *params, **args):
        #print "UserCallbackList: %s(%s)" % (name, params)
        for c in self.Callbacks:
            if hasattr(c, name):
                stop = getattr(c, name)(*params, **args)
                if stop:    return True
            if hasattr(c, "on_callback"):
                stop = c.on_callback(name, *params, **args)
                if stop:    return True
        return False
        
    __call__ = callback
        
class StripedJob(Lockable):

    def __init__(self, url_head, dataset_name, callbacks, user_params, stdout=None, stderr=None):
        Lockable.__init__(self)
        self.URLHead = url_head
        self.Client = StripedClient(url_head)
        self.DatasetName = dataset_name
        self.DatasetEvents = 0
        self.Finished = False
        self.EventsProcessed = self.EventsInDataset = self.EventsToProcess = 0
        self.FramesToProcess = []
        self.TStart = None
        self.Stderr = stderr or sys.stderr
        self.Stdout = stdout or sys.stdout
        
        if callbacks is None:   callbacks = []
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
            
        self.CallbackList = UserCallbackList(callbacks)
        
        self.DataCallbackDelegate = DataCallbackObject(self.CallbackList, stdout, stderr)
            
        self.UserParams = user_params
        self.IPython = None
        try:    self.IPython = __IPYTHON__
        except NameError:   self.IPython = False
        
    def toJSON(self):
        # makes JSON'able dictionary with job parameters
        return dict(
            ClientURL = self.URLHead,
            DatasetName = self.DatasetName,
            Histograms = {hid:h.descriptor() for hid, h in self.histograms().items()},
            UserParams = self.UserParams
        )

    @property
    def runtime(self):
        assert self.TStart is not None, "The job has not started yet"
        return time.time() - self.TStart

    def addHistogram(self, h, inputs=None, constants={}):
        if inputs is None:
            inputs = h.fields
        self.DataCallbackDelegate.addHistogram(h, inputs, constants=constants)
        
    def histograms(self):
        return self.DataCallbackDelegate.histograms()
        
    def start(self, nworkers=1):
        raise NotImplementedError
            
    def wait(self):
        # called by end user
        raise NotImplementedError

    def initDisplay(self, figure, ipython):
        # called by concrete subclass method start()
        return self.DataCallbackDelegate.initDisplay(figure, ipython)

    def updateDisplay(self, iframe):
        # called by concrete subclass refresh() method
        return self.DataCallbackDelegate.updateDisplay(iframe)


    def destroy(self):
        # clean-up, remove potentially circular dependencies
        self.DataCallbackDelegate = None
        self.CallbackList = None

    #
    # callbacks ----------
    #

    @synchronized
    def addCallback(self, c):
        self.CallbackList.addCallback(c)

    @synchronized
    def jobStarted(self):
        self.CallbackList.callback("on_job_start", self)

    @synchronized
    def updateReceived(self, wid, data_dict, nevents):
        self.CallbackList.callback("on_update", nevents)
        self.EventsProcessed = max(nevents, self.EventsProcessed)
        self.DataCallbackDelegate.updateReceived(self, data_dict, self.EventsProcessed)
        self.DataReceivedCounter += 1
        
    @synchronized
    def dataReceived(self, wid, events_delta, data):
        self.CallbackList.callback("on_data", wid, events_delta, data)
        
    @synchronized
    def messageReceived(self, wid, nevents, message):
        handled = self.CallbackList.callback("on_message", wid, nevents, message)
        if not handled:
            self.Stdout.write(message)
            self.Stdout.write("\n")
            self.Stdout.flush()

    @synchronized
    def exceptionReceived(self, wid, info):
        handled = self.CallbackList.callback("on_exception", wid, info)
        if not handled:
            self.Stderr.write("Job %s: striped worker #%d excepton:\n%s\n" % (self.JID, wid, info))
            self.Stderr.flush()
        
    @synchronized
    def workerExited(self, wid, address, status, t, total_events, nrunning):
        self.CallbackList.callback("on_worker_exit", wid, address, status, t, total_events, nrunning)
        
    @synchronized
    def jobFinished(self):
        #print "jobFinished()"
        self.Finished = True
        self.TFinish = time.time()
        self.DataCallbackDelegate.jobEnded()
        
        self.CallbackList.callback("on_job_finish", self.EventsProcessed, None)

    @synchronized
    def jobFailed(self, error):
        #print "jobFailed()"
        self.Finished = True
        self.TFinish = time.time()

        self.DataCallbackDelegate.jobEnded()
        
        handled = self.CallbackList.callback("on_job_finish", self.EventsProcessed, error)
        if not handled:
            self.Stderr.write("Job failed after %d events:\n%s\n" % (nevents, error))
            self.Stderr.flush()

    @synchronized
    def dataLoadFailureReceived(self, wid, frameid):
        handled = self.CallbackList.callback("on_data_load_failure", wid, frameid)
        if not handled:
            self.Stderr.write("Job %s: worker #%d failed to load frame %d\n" % (self.JID, wid, frameid))
            self.Stderr.flush()
            
class DataCallbackObject(Lockable):

    def __init__(self, callback_list, stdout, stderr):
        Lockable.__init__(self)
        self.Histograms = {}            # {key: histogram aggregator}
        self.CallbackList = callback_list
        self.LastUpdateCallback = 0
        self.EventsProcessed = 0
        self.Fig = None                 # if this remains None, it means no plotting is needed
        self.Subplots = None
        self.Stdout = stdout or sys.stdout
        self.Stderr = stderr or sys.stderr
        
    def addHistogram(self, h, inputs, display = True, constants={}):
        if isinstance(inputs, (str, unicode)):
                inputs = [inputs]
        ha = HAggregator(h)
        self.Histograms["___h_%s" % (h.id,)] = ha
        
    def histograms(self):
        return self.Histograms
        
    def initDisplay(self, figure, ipython):
        self.Fig = figure
        histograms = sorted([hn for hn, (h, display) in self.Histograms.items() if display])
        nsubplots = len(histograms)
        self.Subplots = {}              # key -> subplot
        for i, hn in enumerate(histograms):
            self.Subplots[hn] = self.Fig.add_subplot(nsubplots, 1, i+1)
        return True

    def plotHistogram(self, name):
        with JT["GenericJobCallbackObject.plotHistogram"]:
                hist, display = self.Histograms[name]
                subplot = self.Subplots.get(name)
                if subplot is not None:
                    counts, edges = hist.bins
                    #open("/tmp/qq.out", "a").write("hist %s:\n%s\n%s\n" % (name, edges, counts))
                    subplot.clear()
                    subplot.bar(edges, counts[1:-1], align="edge", width=hist.Bin)
                    if hist.Title:  subplot.set_title(hist.Title)
                    if hist.LogScale:       subplot.set_yscale("log")

    def plotHistograms(self):
        if self.Fig is not None:
            for hn in self.Histograms.keys():
                self.plotHistogram(hn)
        
    def updateDisplay(self, iframe):
        if self.Fig is not None:
            self.plotHistograms()
            
    #
    # callbacks --------------
    #
        
    @synchronized
    def updateReceived(self, job, data_dict, nevents):
        self.EventsProcessed = max(nevents, self.EventsProcessed)
        stream_data = {}
        update_hist = False
        for key, data in data_dict.items():
            if key in self.Histograms:
                ha = self.Histograms[key]
                ha.add(data)
                update_hist = True
            else:
                stream_data[key] = data
        #print "updateRedeived:  stream_data.keys", stream_data.keys()
        
        if update_hist and time.time() > self.LastUpdateCallback + 1:
            self.CallbackList.callback("on_histograms_update", self.EventsProcessed)
            self.LastUpdateCallback = time.time()

        if stream_data:
            self.CallbackList.callback("on_streams_update", self.EventsProcessed, stream_data)
                
    @synchronized
    def jobEnded(self):
        if self.Fig is not None:
            self.plotHistograms()            
