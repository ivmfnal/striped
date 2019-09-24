#!/bin/bash

. setup.sh

exec python product/worker/worker_master.py -c config/worker_master.yaml 