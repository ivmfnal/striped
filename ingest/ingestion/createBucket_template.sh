#!/bin/sh



#export PATH=/opt/couchbase/bin:$PATH

cluster_node=$1
bucket=$2

/opt/couchbase/bin/couchbase-cli bucket-create -c ${cluster_node}:8091 -u admin -p admin501 \
	--bucket=$bucket --bucket-type=couchbase \
	--bucket-ramsize=400 --bucket-replica=0 \
	--enable-flush=1 \
	--wait

sleep 1

# edit password !

curl -X PUT \
	-u admin:admin_password \
	-H 'Content-Type: application/json' \
	-d "@views.json" \
	"http://${cluster_node}:8092/${bucket}/_design/views"
