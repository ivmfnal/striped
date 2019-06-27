FROM centos:latest
RUN yum -y update
RUN yum install -y git gcc python-devel sudo make
RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "/tmp/get-pip.py"
RUN python /tmp/get-pip.py
RUN pip install numpy==1.13.1
RUN pip install scipy
RUN pip install pandas tables
RUN pip install jupyter IPython vega histbook numba cloudpickle
RUN pip install uproot uproot-methods awkward 
RUN pip install fnal-column-analysis-tools




