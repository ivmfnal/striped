#!/bin/bash

#source setup.sh

nworkers=${1:-15}

URL=http://dbweb7.fnal.gov:9091/striped/app

cd QWorker

python socket_worker_spawner2.py $URL /tmp 9000 $nworkers ifdb01.fnal.gov 7867
