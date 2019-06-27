#!/bin/bash

exec python product/job_server/striped_job_server.py -d $DATA_SERVER_URL -r 7556
