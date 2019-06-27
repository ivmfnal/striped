#!/bin/sh

for fn in $@; do 
	echo
	echo $fn 
	python match_bliss.py $fn
done
