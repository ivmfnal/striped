import json, cPickle
import numpy as np
from .DataExchange import DXMessage
from .Meta import MetaNode
from .bulk_data_transport import encodeData

class JobDescription:

    def __init__(self, dataset_name, fraction, worker_text, user_params, 
                frame_selector, worker_tags, use_data_cache, auth_token, username, identity, 
                data_mod_url, data_mod_token, bulk_data):
        self.DatasetName = dataset_name
        self.Fraction = fraction
        self.WorkerText = worker_text
        self.HDescriptors = {}
        self.UserParams = user_params
        self.BulkData = bulk_data or {}
        self.WorkerTags = worker_tags
        self.UseDataCache = use_data_cache
        self.FrameSelector = frame_selector       # Meta expression or None
        self.AuthToken = auth_token
        self.Username = username
        self.Identity = identity
        self.DataModificationToken = data_mod_token
        self.DataModificationURL = data_mod_url
        
    def addHistograms(self, hdescriptors):
        self.HDescriptors = hdescriptors
        
    def toDXMsg(self):
        msg = DXMessage("job_request", 
                auth_token = self.AuthToken,
                data_mod_token = self.DataModificationToken,
                data_mod_url = self.DataModificationURL,
                fraction=self.Fraction, 
                dataset_name=self.DatasetName, 
                format="pickle", 
                username=self.Username,
                use_data_cache = "yes" if self.UseDataCache else "no")
        encoded_params = encodeData(self.UserParams)
        encoded_bulk = encodeData(self.BulkData)
        #print "JobDescription: pickled_params=", len(pickled_params)
        msg.append(
            bulk_data = encoded_bulk,
            worker_text=self.WorkerText,
            worker_tags=cPickle.dumps(self.WorkerTags),
            user_params=encoded_params,
            histograms=json.dumps(self.HDescriptors),
            identity=self.Identity
        )
        if self.FrameSelector is not None:
            msg.append(frame_selector=json.dumps(self.FrameSelector.serialize()))        
        return msg
        
    @staticmethod
    def fromDXMsg(msg):
        format = msg["format"]
        assert format == "pickle", "Unknown job description encoding format %s" % (format,)
        dataset_name    = msg["dataset_name"]
        username        = msg["username"]
        fraction        = msg["fraction"]
        auth_token      = msg.get("auth_token")
        data_mod_token  = msg.get("data_mod_token")
        data_mod_url    = msg.get("data_mod_url")
        identity        = msg.get("identity")
        hdescriptors    = json.loads(msg["histograms"])
        user_params     = msg["user_params"] # do not unpickle, pass as is to the workers
        bulk_data       = msg["bulk_data"]  # do not unpickle, pass as is to the workers
        worker_tags     = cPickle.loads(msg["worker_tags"])
        worker_text     = msg["worker_text"]
        use_data_cache  = msg.get("use_data_cache", "yes") != "no"
        frame_selector  = msg.get("frame_selector")
        if frame_selector is not None:
            frame_selector = MetaNode.deserialize(json.loads(frame_selector))
        
        desc = JobDescription(dataset_name, fraction, worker_text, user_params, frame_selector, worker_tags, use_data_cache, 
            auth_token, username, identity, data_mod_url, data_mod_token,
            bulk_data)
        desc.addHistograms(hdescriptors)
        return desc

        
        
        
        
