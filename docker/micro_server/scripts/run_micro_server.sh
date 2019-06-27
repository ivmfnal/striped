#!/bin/sh

cd ${HOME}

export STRIPED_HOME=`pwd`
export PYTHONPATH=${STRIPED_HOME}:${STRIPED_HOME}/product
export DATA_SERVER_URL="http://dbdata0vm.fnal.gov:9091/striped/app"

./start_qworkers.sh &
./start_job_server.sh &

echo Striped server started at port 8765

wait
