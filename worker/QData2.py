from striped.common import Tracer
from QArrays2 import QAEventGroup
from Vault import Vault
import numpy as np, sys, traceback

T = Tracer()

class Frame:
    
    def __init__(self, rginfo, column_to_branch, raw_stroll, tagged_event_ids,):
        self.RGInfo = rginfo
        #self.RGID = rginfo.RGID
        self.NEvents = rginfo.NEvents
        #print "Frame: created for rgid=%d, nevents=%d" % (rginfo.RGID, rginfo.NEvents) 
        self.AttrVault = Vault()
        self.VarAttrVaults = {}
        self.BranchVaults = {}              # name -> branch vault
        
        for cn, (bn, sc) in column_to_branch.items():
            if bn:
                prefix = bn + '.'
                assert cn.startswith(prefix)
                aname = cn[len(prefix):]
                if sc != bn+".@size":
                    raise NotImplementedError("Variable length branch attributes not implemented yet")
                size = raw_stroll[bn+".@size"]
                #print "Frame: Size array for branch %s: %s" % (bn, size)
                v = self.BranchVaults.get(bn)
                if v is None:
                    v = Vault(size)
                    self.BranchVaults[bn] = v
                try:    
                    v.addStripe(aname, raw_stroll[cn])
                except:
                    info = traceback.format_exc()
                    message = "Exception in VaultaddStripe: \n%s\nRGInfo: %s\nAttribute name: %s, column name: %s" % (
                        info, rginfo, aname, cn)
                    raise RuntimeError(message)
            else:
                #print "Frame: size column for %s: %s" % (cn, sc)
                if sc:
                    #print "Frame: var attr %s size array:%s" % (cn, raw_stroll[sc])
                    v = Vault(raw_stroll[sc])       # attributes with variable size
                    self.VarAttrVaults[cn] = v
                else:
                    v = self.AttrVault
                v.addStripe(cn, raw_stroll[cn])

    def eventGroup(self, iframe, nframes):
        return QAEventGroup(self.RGInfo, self.RGInfo.NEvents, self.AttrVault, self.VarAttrVaults, self.BranchVaults,
            iframe, nframes)
        
    

class Dataset(object):
    
    def __init__(self, striped_client, data_buffer, dataset_name, columns, schema = None, trace = None):
        self.T = trace or Tracer()
        global T
        T = self.T
        self.Name = dataset_name

        self.BranchNames = set()
        self.AttrNames = set()
        self.Columns = set(columns)
        data_columns = set(columns)
        
        if not schema:
            self.ClientDataset = striped_client.dataset(dataset_name, columns)
            columns_dict = self.ClientDataset.columns(columns, include_size_columns=True)
            
            # check if any columns are missing in the dataset
            missing = [cn for cn in columns if not cn in columns_dict]
            if len(missing):
                raise KeyError("The following columns are not found in the dataset: %s" % (",".join(missing),))
            
            self.ColumnToBranch = { cn: (cc.descriptor.ParentArray, cc.descriptor.SizeColumn) for cn, cc in columns_dict.items() }
            for cn in columns:
                bn, sn = self.ColumnToBranch.get(cn)
                if bn:
                    self.BranchNames.add(bn)
                    data_columns.add(bn + ".@size")
                else:
                    self.AttrNames.add(cn)
                    if sn:
                        data_columns.add(sn)
            self.FetchColumns = self.ClientDataset.columnsAndSizes(columns)
        else:
            self.ClientDataset = None
            columns_to_branch = {}
            fetch_columns = set()
            missing = []
            for cn in columns:
                if '.' in cn:
                    bn, an = cn.split('.', 1)
                    sn = bn + ".@size"
                    columns_to_branch[cn] = (bn, sn)
                    fetch_columns.add(sn)
                    self.BranchNames.add(bn)
                else:
                    columns_to_branch[cn] = (None, None)
                    self.AttrNames.add(cn)
                fetch_columns.add(cn)
                
            self.ColumnToBranch = columns_to_branch
            self.FetchColumns = list(fetch_columns)
            
        self.TagConditions = []
        self.ProcessedEvents = 0
        #print self.EventTemplate.branchTemplate
        #print "Q Dataset: fetch columns:", self.FetchColumns
        self.Filter = None
        self.DataBuffer = data_buffer
        
    @property
    def event(self):
        return self.EventTemplate
    
    def frame(self, rginfo, data):
        return Frame(rginfo, self.ColumnToBranch, data, self.TagConditions)
            
    def filter(self, qexp):
        self.Filter = qexp
        
    def emit(self, name, q):
        self.EmitItems[name] = q

    def events(self, frame_ids=None):
        for f in self.frames(frame_ids):
            with self.T["frame/evaluate"]:
                if self.Filter is not None: self.Filter.reset()
                self.EventTemplate.evaluate(f.Storage)
                if self.Filter is not None:
                    event_mask = self.Filter.evaluate(f.Storage)
                    #print "Filter evaluated:", event_mask
                else:
                    event_mask = None

            for name, q in self.EmitItems.items():
                data = q.evaluate(f.Storage)
                if event_mask is not None:
                    size_array = q.sizeArray()
                    if size_array is not None:
                        i = 0
                        for ievent, filtered_in in enumerate(event_mask):
                            n = size_array[ievent]
                            if filtered_in:
                                self.DataBuffer.addArray(name, data[i:i+n])
                            i += n
                    else:
                        self.DataBuffer.addArray(name, data[event_mask])
                else:
                    self.DataBuffer.addArray(name, data)
                

            for ie, e in enumerate(f.events()):
                self.ProcessedEvents += 1
                if event_mask is None or event_mask[ie]:
                    yield e
                    
            self.DataBuffer.endOfFrame(self.ProcessedEvents)
    
