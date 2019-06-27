#!/bin/bash

nworkers=2
logfile=""
URL=${DATA_SERVER_URL}

echo URL=$URL

while [ "$1" != "" ]
do
	case "$1" in
		-n)	nworkers=$2
			shift
			;;
		-l)	logfile="-l $2"
			shift
			;;
		-u)	URL="$2"
			shift
			;;
	esac
	shift
done
			

exec python product/worker/socket_worker_spawner3.py $logfile $URL /tmp 9000 $nworkers localhost 7556
