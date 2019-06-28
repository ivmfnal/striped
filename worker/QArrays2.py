#
# Frame.events - QAEventGroup                                   setattr     filter
# Frame.events.EventID - ndarray
# Frame.events.hits - [ndarray, ...] --- ???
# Frame.events.Muon - QABranch                                  setattr     filter
# Frame.events.Muon.pairs - QABranchCombo                       setattr     filter
# Frame.events.Muon.pt - ndarray
# Frame.events.Muon.pairs[sibling] - QABranchComboSibling
# Frame.events.Muon.pairs[sibling].pt - ndarray
#

import numpy as np
from QAFilters import QAEventFilter, QABranchFilter, Filterable
from Vault import Vault
from QAPointers import QAEventIterator
from attribute_setter import constructor, attr_setter

class QAEventGroup(Filterable):

    @constructor
    def __init__(self, rginfo, nevents, attr_vault, var_attr_vaults, branch_vaults, iframe, nframes):
        self._RGInfo = rginfo
        self._NEvents = nevents
        self._AttrVault = attr_vault
        self._VarAttrVaults = var_attr_vaults
        self._BranchVaults = branch_vaults
        self._IFrame = iframe
        self._NFrames = nframes
        
        #for bn, bv in branch_vaults.items():
        #    print "QAEventGroup: vault %s: size: %s" % (bn, bv.SizeArray)
        
        self._Branches = {}                    # QABranch objects
        self._ExpandedAsBranches = {}

    def __getattr__(self, name):
        if name in self._BranchVaults:
            return self.branch(name)
        elif name in self._VarAttrVaults:
            return self._VarAttrVaults[name].arrays[name]
        else:
            return self._AttrVault[name].values
            
    def meta(self, name):
        return self.metadata.get(name)
        
    __getitem__ = meta

    @property
    def iframe(self):
        return self._IFrame

    @property
    def nframes(self):
        return self._NFrames

    @property
    def metadata(self):
        return self._RGInfo.Metadata
        
    dot = __getattr__
    attr = __getattr__

    def branch(self, name):
        b = self._Branches.get(name)
        if b is None:
            b = QABranch(name, self._BranchVaults[name])
            self._Branches[name] = b
        return b
            
    def hasBranch(self, name):
        return name in self._BranchVaults
        
    def branchVault(self, name):
        return self.BranchVault
    
    @property
    def rgid(self):
        return self._RGInfo.RGID
        
    @property
    def rginfo(self):
        return self._RGInfo

    @property
    def nevents(self):
        return self._NEvents
        
    count = nevents
    
    def __len__(self):
        return self._NEvents
            
    @attr_setter
    def __setattr__(self, name, val):
        self._AttrVault.addStripe(name, val)

    def __iter__(self):
        return QAEventIterator(self, self.nevents)
        
    def filter(self, mask_or_filter = None, operation = "any"):
        # always returns an event filter, converts branch filter to event filter if needed
        if isinstance(mask_or_filter, QAEventFilter):
            assert mask_or_filter.Parent is self, "Can not convert filter made for another event group"   
            return mask_or_filter 
        elif isinstance(mask_or_filter, QABranchFilter):
            # convert branch filter to event filter
            return QAEventFilter(self, mask_or_filter.reduceMask(operation))
        else:
            mask = mask_or_filter
            if mask is None:
                # no-op filter
                mask = np.ones((self.nevents,), dtype=np.dtype("bool"))
            # assume mask
            return QAEventFilter(self, mask)
            
    def apply_event_filter(self, filter):
        new_attr_vault = filter(self._AttrVault)
        new_var_vaults = {n:filter(v) for n, v in self._VarAttrVaults.items()}
        new_branch_vaults = {n:filter(v) for n, v in self._BranchVaults.items()}
        _, nevents = filter.counts
        return QAEventGroup(self._RGInfo, nevents, new_attr_vault, new_var_vaults, new_branch_vaults)
            
class QABranch(Filterable):

    @constructor
    def __init__(self, branch_name, attr_vault):
        self._AttrVault = attr_vault
        #print "QABranch %s created with attr_vault.size: %s" % (branch_name, attr_vault.SizeArray)
        self._Name = branch_name
        self._Pairs = None             # Combo branch

    def expand(self, array):
        return self._AttrVault.expandArray(array)
        
    @property
    def pairs(self):
        if self._Pairs is None:
                siblings_vault = self._AttrVault.makePairsVault()
                #print "siblings_vault = ", siblings_vault
                self._Pairs = QABranchCombo(self._Name, 2, siblings_vault)
        return self._Pairs
        
    def __getattr__(self, name):
        return self._AttrVault[name].values
        
    @property
    def count(self):
        return self._AttrVault.SizeArray
        
    def length(self, ievent):
        return self._AttrVault.SizeArray[ievent]
        
    @attr_setter
    def __setattr__(self, name, val):
        self._AttrVault.addStripe(name, val)

    def filter(self, mask_or_filter = None):

        if isinstance(mask_or_filter, QABranchFilter):
            assert mask_or_filter.Parent is self, "Can not convert filter made for another branch"   
            return mask_or_filter 
        elif isinstance(mask_or_filter, QAEventFilter):
            mask = mask_or_filter.expandMask(self._AttrVault.SizeArray)
            return QABranchFilter(self, mask, self._AttrVault.SizeArray)
        else:
            mask = mask_or_filter
            if mask is None:
                # no-op filter
                mask = np.ones((self._AttrVault.Length,), dtype=np.dtype("bool"))
            # assume mask
            return QABranchFilter(self, mask, self._AttrVault.SizeArray)
            
    def apply_row_filter(self, filter):
        return QABranch(self._Name, filter(self._AttrVault))

    apply_event_filter = apply_row_filter
        
        
class QABranchCombo(Filterable):

    @constructor
    def __init__(self, branch_name, cardinality, siblings_vault, attr_vault = None):
        self._Name = branch_name + "/%d" % (cardinality,)
        self._Cardinality = cardinality
        self._DataVault = siblings_vault
        self._AttrVault = attr_vault or Vault(siblings_vault.SizeArray)
        #print "QABranchCombo: dict=", self.__dict__
        
    @property
    def count(self):
        return self._AttrVault.SizeArray

    def __getitem__(self, sibling):
        if sibling > 1 or sibling < -2: raise IndexError("index = %d" % (sibling,))
        return QABranchComboSibling(self._Name, self._Cardinality, sibling, self._DataVault)

    def __getattr__(self, name):
        return self._AttrVault[name].values
        
    def __len__(self):
        return self._Cardinality

    @attr_setter
    def __setattr__(self, name, val):
        self._AttrVault.addStripe(name, val)

    def filter(self, mask_or_filter = None):
        size_array = self._AttrVault.SizeArray
        if isinstance(mask_or_filter, QABranchFilter):
            assert mask_or_filter.Parent is self, "Can not convert filter made for another branch"   
            return mask_or_filter 
        elif isinstance(mask_or_filter, QAEventFilter):
            mask = mask_or_filter.expandMask(size_array)
            return QABranchFilter(self, mask, size_array)
        else:
            mask = mask_or_filter
            if mask is None:
                # no-op filter
                mask = np.ones((sum(size_array),), dtype=np.bool)
            # assume mask
            return QABranchFilter(self, mask, size_array)

    def apply_event_filter(self, filter):
        return QABranchCombo(self._Name, self._Cardinality, 
            filter(self._DataVault), filter(self._AttrVault))

    apply_row_filter = apply_event_filter

            
class QABranchComboSibling(object):

    def __init__(self, name, cardinality, sibling, vault):
        assert sibling >= 0 and sibling < cardinality
        self._Name = name
        self._Cardinality = cardinality
        self._Sibling = sibling
        self._Vault = vault
        
    def __getattr__(self, name):
        data = self._Vault[name][self._Sibling]
        return data

        
