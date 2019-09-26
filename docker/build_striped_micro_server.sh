#!/bin/sh

./build_striped_base.sh

docker build -t striped_micro_server:latest micro_server