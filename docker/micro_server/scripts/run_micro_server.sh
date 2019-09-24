#!/bin/sh

./run_job_server.sh > /dev/null 2>&1 </dev/null &
./run_worker_master.sh > /dev/null 2>&1 </dev/null &

wait
