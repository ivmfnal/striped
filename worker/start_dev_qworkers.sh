#!/bin/bash

#source setup.sh

nworkers=${1:-15}

URL=http://dbweb7.fnal.gov:9091/striped/app
#URL=http://dbwebdev.fnal.gov:9090/striped/app

cd QWorker

#python socket_worker_spawner.py <striped server URL> <module storage> <starting port> <nworkers> <registry host> <registry port>

python socket_worker_spawner2.py $URL /tmp 9100 $nworkers dbwebdev.fnal.gov 7666
