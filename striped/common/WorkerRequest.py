import json, cPickle
from .DataExchange import DXMessage
from .signature import Signer

class WorkerRequest:

    #
    # Sent from job server (Contract) to worker master
    #

    def __init__(self, jid, wid, data_server_url, dataset_name, rgids, nworkers, worker_module_name, worker_text, hdesriptors, 
                user_params, use_data_cache, data_mod_url, data_mod_token, bulk_data_name):
        #
        # hdescriptors: {hist_name: hist_descriptor}
        #
        self.JID = jid
        self.WID = wid
        self.DataServerURL = data_server_url         # not used for now, but worth keeping
        self.DatasetName = dataset_name
        self.RGIDs = rgids
        self.NWorkers = nworkers
        self.WorkerModuleName = worker_module_name
        self.WorkerText = worker_text
        self.HDescriptors = hdesriptors
        self.UserParams = user_params
        self.UseDataCache = use_data_cache
        self.DataModURL = data_mod_url
        self.DataModToken = data_mod_token
        self.BulkDataName = bulk_data_name
        
    def generateSignature(self, key):
        signer = Signer(key)
        data = (self.WorkerText, self.DatasetName)
        signature, t, salt, alg = signer.sign(data)
        return signature, t, salt, alg
        
    def verifySignature(self, key, client_signature, t, salt, alg):
        signer = Signer(key)
        data = (self.WorkerText, self.DatasetName)
        verified, reason = signer.verify(client_signature, data, t, salt, alg, time_tolerance=3600)
        return verified, reason
        
    def toDXMsg(self):

        msg = DXMessage("request", wid=self.WID, nworkers=self.NWorkers, bulk_data_name=self.BulkDataName, 
                        use_data_cache = "yes" if self.UseDataCache else "no"
        )
        msg.append(
            job_id = self.JID,
            dataset_name = self.DatasetName,
            worker_module_name = self.WorkerModuleName,
            data_server_url = self.DataServerURL,
            rgids = cPickle.dumps(self.RGIDs),
            worker_text = self.WorkerText,
            histograms = json.dumps(self.HDescriptors),
            data_mod_url = self.DataModURL,
            data_mod_token = self.DataModToken,
            user_params = self.UserParams       # this comes picked straight from the client, the job server passes it without unpickling
        )
        return msg
        
    @staticmethod
    def fromDXMsg(msg):
        assert msg.Type == "request"
        jid = msg["job_id"]
        wid = msg["wid"]
        dataset_name = msg["dataset_name"]
        nworkers = msg["nworkers"]
        worker_module_name = msg["worker_module_name"]
        data_server_url = msg["data_server_url"]
        rgids = cPickle.loads(msg["rgids"])
        worker_text = msg["worker_text"]
        histograms = json.loads(msg["histograms"])
        user_params = msg.get("user_params")
        use_data_cache  = msg.get("use_data_cache", "yes") != "no"
        data_mod_token = msg.get("data_mod_token")
        data_mod_url = msg.get("data_mod_url")
        bulk_data_name = msg.get("bulk_data_name")
        return WorkerRequest(jid, wid, data_server_url, dataset_name, rgids, nworkers, worker_module_name, worker_text, histograms, 
            user_params, use_data_cache, data_mod_url, data_mod_token, bulk_data_name
            )
        
