#!/bin/env python

import numpy as np
import sys, getopt
from striped.client import StripedClient 

Usage = """
striped [<opts> ...] <command> [<args> ...]

Options:
  -s <Striped data server URL>

Commands/args:
  help
  stripe [-a] <dataset> <column> <rgid>
    -a = assembled
  columns <dataset>
  column <dataset> <column>
"""

ServerURL = None

global_opts, rest = getopt.getopt(sys.argv[1:], "s:")
global_opts = dict(global_opts)

if not rest or rest[0] == "help" or not "-s" in global_opts:
    print Usage
    sys.exit(1)

ServerURL = global_opts["-s"]

command, args = rest[0], rest[1:]

client = StripedClient(ServerURL)

if command == "stripe":
    print "stripe..."
    opts, args = getopt.getopt(args, "a")
    opts = dict(opts)
    assembled = "-a" in opts
    dataset, column, rgid = args[0], args[1], int(args[2])
    ds = client.dataset(dataset)
    col = ds.column(column)
    s = ds.stripe(column, rgid)
    if hasattr(s, "shape"):
        print s.shape, s
    else:
        for x in s[:10]:
            print x
        print "%d items" % (len(s),)

elif command == "column":
    dataset, column = args[0], args[1]
    ds = client.dataset(dataset)
    col = ds.column(column)
    desc = col.descriptor   
    print "Dataset:        ", dataset
    print "Column:         ", column
    print "Type:           ", desc.Type
    print "Numpy type:     ", desc.NPType
    print "Convert to:     ", desc.ConvertToNPType
    print "Depth:          ", desc.Depth
    print "Size column:    ", desc.SizeColumn
    print "Shape:          ", desc.Shape
    print "Parent array:   ", desc.ParentArray

elif command == "columns":
    dataset = args[0]
    columns = client.dataset(dataset).columnNames
    for c in sorted(columns):
        print c
           
