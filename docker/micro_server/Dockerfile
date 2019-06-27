FROM striped_base:latest

RUN /sbin/adduser -d /home/striped striped; chown -R striped /home/striped

USER striped

WORKDIR /home/striped

RUN cd ${HOME}; git clone http://cdcvs.fnal.gov/projects/nosql-ldrd striped
RUN cd ${HOME}/striped ; make 
RUN cd ${HOME} ; rm -rf striped

ADD ./scripts ${HOME}

RUN ln -s ${HOME}/build/striped ${HOME}/product

EXPOSE 8765

CMD cd ${HOME}; ./run_micro_server.sh