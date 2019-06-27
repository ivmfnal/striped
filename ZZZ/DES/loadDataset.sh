#!/bin/bash

if [ "$1" = "" ]; then
	echo Usage: `basename $0` dataset_dir bucket
	exit 1
fi

dataset_dir=$1
bucket=$2
schema="cms/zjets_corrected.json"
correct="cms/cms_correct.py"
treetop="TreeMaker2/PreSelection"

dataset=`basename $dataset_dir`

python createDataset2.py $schema $bucket $dataset
python loadDataset.py -O -C $correct -s 5 -m 15 $dataset_dir $treetop $bucket
python listDataset.py $bucket $dataset
