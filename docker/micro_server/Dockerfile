FROM striped_base:latest

USER striped

WORKDIR /home/striped

RUN pip install uproot uproot-methods awkward 
RUN pip install fnal-column-analysis-tools
RUN pip install posix_ipc pyyaml webpie

RUN python -c "import yaml; print ('ok')"

RUN cd ~ && \
    git clone https://github.com/ivmfnal/striped.git striped && \
    mkdir -p build && \
    cd striped && \
    make
    
RUN mkdir ~/striped_home && \
    ln -s ~/build/striped ~/striped_home/product

ADD --chown=striped scripts/*.sh striped_home/
ADD --chown=striped config striped_home/config/

EXPOSE 8765 8766

CMD cd ~/striped_home  #; ./run_micro_server.sh
