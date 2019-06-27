FROM striped_base:latest

RUN /sbin/adduser -d /home/striped striped; chown -R striped /home/striped

USER striped
RUN cd /home/striped; \
    git clone http://cdcvs.fnal.gov/projects/nosql-ldrd; \
    cd nosql-ldrd ;\
    python setup.py install --user ;\
    cp sample/job.py ~ ;\
    cd ~; rm -rf nosql-ldrd
    
CMD cd /home/striped; STRIPED_JOB_SERVER_ADDRESS=ifdb01.fnal.gov:8765 exec /bin/bash