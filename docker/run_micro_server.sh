#!/bin/sh

Usage="run_micro_server.sh [-c <config>] [-i]\n"

config=`pwd`/config
cmd=""
ti=""

while [ "$1" != "" ]
do
	case "$1" in
		-i)	
        	ti="-ti"
        	cmd=/bin/bash
        	;;
		-c)	
            if [ ! -d $2 ]; then
                echo $2 is invalid path
                exit 1
            fi
            config=`cd $2; pwd`
			shift
			;;
        -\?|-h|--help)
            echo "Usage: run_micro_server.sh [-c <config>] [-i]"
            echo "   -c <config> - absolute path to config directory. Default: $PWD/config"
            echo "   -i - do not start Striped server. Instead, run interactive shell"
            echo
            exit 2
            ;;
	esac
	shift
done

if [ ! -f ${config}/job_server.yaml ]; then
    echo Error: ${config}/job_server.yaml file not found
    exit 1
fi 

if [ ! -f ${config}/worker_master.yaml ]; then
    echo Error: ${config}/worker_master.yaml file not found
    exit 1
fi 

echo will use $config as the configuration directory

if [ "$ti" == "" ]; then
    echo starting the Striped server
else 
    echo starting interactive shell instead of the Striped server    
fi

docker run --rm \
	-p 8766:8766 -p 8765:8765 \
	--mount type=bind,source=${config},target=/home/striped/striped_home/config \
	$ti \
	striped_micro_server:latest $cmd
