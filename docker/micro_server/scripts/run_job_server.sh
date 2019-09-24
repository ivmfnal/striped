#!/bin/bash

. setup.sh

STRIPED_JOB_SERVER_CFG=config/job_server.yaml \
JINJA_TEMPLATES_LOCATION=product/job_server \
	exec python product/job_server/striped_job_server.py -- job_server 

