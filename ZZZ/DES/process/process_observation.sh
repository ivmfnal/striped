#!/bin/sh 

dirpath=$1
oid=`basename $dirpath`
tmp=./tmp

rm -f ${tmp}/${oid}*.fits

time python match_exposure.py -s 0.1 -m 10 ${tmp}/$oid ${dirpath}/*.fits

if [ -f ${tmp}/${oid}_matches.fits ]; then
	time python add_observations.py DES Bliss ${tmp}/${oid}_matches.fits
fi

if [ -f ${tmp}/${oid}_unmatches.fits ]; then
	time python add_objects.py DES Bliss ${tmp}/${oid}_unmatches.fits
fi


