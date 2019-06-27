class EventList(object):
    def __init__(self, frame):
        self.Frame = frame
        
    def __getitem__(self, ievent):
        return EventPointer(self.Frame, ievent)
        
class QAEventIterator(object):

    def __init__(self, event_group, nevents):
        self.EventGroup = event_group
        self.NEvents = nevents
        self.I = 0

    def __iter__(self):
        return self

    def next(self):
        if self.I >= self.NEvents:
            raise StopIteration
        p = QAEventPointer(self.EventGroup, self.I)
        self.I += 1
        return p

class QAEventPointer:
    
    # e.g. event_group[i]
    
    def __init__(self, event_group, ievent=0):
        self.EventGroup = event_group
        self.IEvent = ievent
        
    def __str__(self):
        return "[Event %d]" % (self.IEvent,)
        
    def dot(self, a):
        #
        # events[i].Muon or
        # events[i].ID
        #
        if self.EventGroup.hasBranch(a):
            return QABranchPointer(self.EventGroup.branch(a), self.IEvent)
        else:
            return self.EventGroup.dot(a)[self.IEvent]

    __getattr__ = dot
    attr = dot

class QABranchPointer:
    
    #
    # E.g.: Frame.events[i].Muon
    #
    
    def __init__(self, branch, ievent):
        self.IEvent = ievent
        self.Branch = branch        # QABranch object
        self.Index = 0
        self.Length = 0
    
    @property
    def pairs(self):
        return QAComboPointer(self.Branch.pairs, self.IEvent, 2)
    
    def __getitem__(self, i):           # e.g. Frame.event[ievent].Muon[i]
        if isinstance(i, int):
            return QABranchTerminal(self.Branch._Name, self.Branch._AttrVault, self.IEvent, i)
        elif isinstance(i, slice):
            return [QABranchTerminal(self.Branch._Name, self.Branch._AttrVault, self.IEvent, j) 
                        for j in xrange(*i.indices(len(self)))]
    
    def __len__(self):
        return self.Branch.length(self.IEvent)

    #
    # column access Event[i].Muon.pt - app pt's of all muons in the event
    #
    def dot(self, a):
        return self.Branch._AttrVault.event(a, self.IEvent)

    __getattr__ = dot    
            
    #
    # Iterator protocol
    #
    def iterate(self):
        for i in xrange(len(self)):
            yield self[i]

    def __iter__(self):
        self.Index = 0
        self.Length = len(self)
        return self
        
    def next(self):
        if self.Index >= self.Length:
            raise StopIteration
        x = self[self.Index]
        self.Index += 1
        return x
        
class QABranchTerminal:
    
    #
    # E.g.: events[i].Muon[j]
    # Also events[i].Muon.pairs[j][s]
    #
    def __init__(self, name, vault, ievent, iitem, sibling=None):
        self.IEvent = ievent
        self.Vault = vault
        self.IItem = iitem
        self.Sibling = sibling
        self.Name = name
        
    def __str__(self):
        return "[QABranchTerminal branch=%s ievent=%s iitem=%s sibling=%s]" % (self.Name, self.IEvent, self.IItem, self.Sibling)

    __repr__ = __str__

    def dot(self, a):
        event = self.Vault.event(a, self.IEvent)
        if self.Sibling is None:
            return event[self.IItem]
        else:
            return event[self.IItem, self.Sibling]

    __getattr__ = dot    

class QAComboIterator(object):
    def __init__(self, combo_branch, ievent, ncombos):
        self.IEvent = ievent
        self.I = 0
        self.N = ncombos
        self.ComboBranch = combo_branch
        self.PairPointer = QAComboItemPointer(combo_branch, self.IEvent, 0)
        
    def __iter__(self):
        return self
        
    def next(self):
        if self.I >= self.N:
            raise StopIteration
        self.PairPointer.IPair = self.I
        self.I += 1
        return self.PairPointer

class QAComboPointer(object):
    
    #
    # E.g.: events[i].Muon.pairs
    #
    
    def __init__(self, combo_branch, ievent, cardinality):
        if cardinality != 2:
            raise NotImplementedError("Only pairs are implemented")
        self.IEvent = ievent
        self.ComboBranch = combo_branch
        
    def __len__(self):
        #print self.ComboBranch
        return self.ComboBranch.count[self.IEvent]
        
    def __iter__(self):
        return QAComboIterator(self.ComboBranch, self.IEvent, len(self))
        
    def __getitem__(self, inx):
        return QAComboItemPointer(self.ComboBranch, self.IEvent, inx)
        
class QAComboItemPointer:
    
    #
    # Represents specific pair, e.g.:
    # event.Muon.pairs[i]
    # 
    # 
    
    def __init__(self, combo_branch, ievent, ipair):
        self.ComboBranch = combo_branch
        #print combo_branch, dir(combo_branch)
        self.AttrVault = combo_branch._AttrVault
        self.DataVault = combo_branch._DataVault
        self.IEvent = ievent
        self.IPair = ipair
        
    def __len__(self):
        return 2
        
    def __getitem__(self, sibling):
        if sibling > 1 or sibling < -2: raise IndexError("index = %d" % (sibling,))
        return QABranchTerminal(self.ComboBranch._Name, self.DataVault, self.IEvent, self.IPair, sibling)
    
    def dot(self, aname):                   # e.g. pair.max_p
        if aname == "__iter__":
            raise AttributeError
        #print "__getattr__(%s)" % (aname,)
        return self.AttrVault.event(aname, self.IEvent)[self.IPair]

    __getattr__ = dot
    
