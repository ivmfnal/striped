FROM centos:latest
RUN yum -y update
RUN yum install -y git gcc sudo make 
RUN yum install -y wget

USER striped
WORKDIR /home/striped

RUN echo miniconda3 ...
RUN mkdir -p ~/.conda && \
    wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p ~/miniconda3 && \
    rm ~/miniconda.sh && \
    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN cat ~/.bashrc
ENV PATH /home/striped/miniconda3/bin:$PATH
RUN python -V
RUN wget "https://bootstrap.pypa.io/get-pip.py" -O /tmp/get-pip.py && \
    python /tmp/get-pip.py && \
    rm /tmp/get-pip.py
RUN pip install numpy
RUN pip install scipy
RUN pip install pandas tables
RUN pip install jupyter IPython vega histbook numba cloudpickle




