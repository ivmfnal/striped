import os, time, glob, sys, getopt
import numpy as np
import uproot

class RowGroupMap:

	def __init__(self, rgsize = 10000, start_rgid = 0, from_file = None):
		self.RGSize = rgsize
		self.StartRGID = start_rgid
		self.Map = {}		# { filename -> (start_rgid, [rgsize,...]) }
		if from_file is not None:
			self.readFile(from_file)

	@staticmethod
	def fromDict(dct):
		m = RowGroupMap(dct["RGSize"], dct["StartRGID"])
		m.Map = {
			fn: (d["StartRGID"], d["Groups"])
			for fn, d in dct["Map"].items()
		}
		return m

	@staticmethod
	def partition(nevents, target_group_size):
		if nevents <= target_group_size:
			return [nevents]
		if nevents % target_group_size == 0:
			return [target_group_size]*(nevents/target_group_size)
		M = (nevents+target_group_size-1)//target_group_size     # number of groups
		n = nevents % M
		k = nevents // M
		return [k+1]*n + [k]*(M-n)

	def scanFiles(self, files):
		self.Map = {}
		rgid = self.StartRGID
		for f in sorted(files):
			fn = f.split("/")[-1]
			utree = uproot.open(f, memmap=False)["Events"]
			# get event number
			nevents = utree.numentries
			groups = self.partition(nevents, self.RGSize)
			self.Map[fn] = (rgid, groups)
			rgid += len(groups)

	def asDict(self):
		dct = {
			"RGSize": self.RGSize,
			"StartRGID": self.StartRGID,
			"Map": 	{fn:{"StartRGID":rgid, "Groups":groups} for fn, (rgid, groups) in self.Map.items()}
		}
		return dct
			

	def load(self, file_object):
		self.Map = {}
		for l in file_object.readlines():
			l = l.strip()
			if l:
				words = l.split()
				numbers = [int(x) for x in words[1:]]
				self.Map[words[0]] = (numbers[0], numbers[1:])

	def __len__(self):
		return len(self.Map)

	def __getitem__(self, fn):
		return self.Map[fn]

	def items(self):
		return self.Map.items()

	def dump(self, file_object):
		for fn, (rgid, groups) in sorted(self.items()):
			file_object.write("%s %d %s\n" % (fn, rgid, " ".join(["%d" % (n,) for n in groups])))

