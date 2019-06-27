#!/bin/sh



#export PATH=/opt/couchbase/bin:$PATH

cluster_node=$1
bucket=$2

# edit password !

curl -X PUT \
	-u admin:password \
	-H 'Content-Type: application/json' \
	-d "@views.json" \
	"http://${cluster_node}:8092/${bucket}/_design/views"
