FROM ubuntu:16.04


RUN apt-get update && \
    apt-get install nano && \
    echo "alias python=python3.6" >> ~/.bashrc && \
    echo "alias pip=pip3" >> ~/.bashrc 

RUN apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev && \
cd /tmp && \
wget https://www.python.org/ftp/python/3.6.12/Python-3.6.12.tgz && \
tar xvf Python-3.6.12.tgz && \
cd Python-3.6.12 && \
./configure --enable-optimizations && \
make -j 8 && \
make altinstall
