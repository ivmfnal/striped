#!/bin/sh

here=`pwd`

cd ~/striped_home
source ./setup.sh
cd $here

port=""

if [ "$1" = "-p" ]; then
	port="--port=$2"
	shift
	shift
fi

jupyter notebook --config=demo_config.py $port >/dev/null </dev/null 2>&1 &

