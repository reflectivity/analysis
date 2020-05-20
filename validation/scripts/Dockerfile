#
# Dockerfile for testing
#
# docker build . -t andyfaff/orso:latest
# docker push andyfaff/orso:latest
#
# docker run -it --rm -v /bin/bash


FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gnupg2 \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.7 python3.7-dev \
        python3.7-venv python3.7-distutils python3-pip \
    && apt-get install -y vim curl build-essential python3-wxgtk4.0 \
        libgtk-3-dev git\
        gfortran cmake libgsl-dev libboost-all-dev \
        libfftw3-dev libtiff5-dev qt5-default \
        libqt5designercomponents5 qttools5-dev libqt5svg5-dev \
    && python3.7 -m pip install numpy scipy cython pytest setuptools wheel \
    && python3.7 -m pip install refnx refl1d pytest \
    && apt-get clean

# make and install BornAgain
# to use source /usr/local/bin/thisbornagain.sh \
RUN cd ~ \
    && git clone --recursive https://github.com/scgmlz/BornAgain.git \
    && cd BornAgain && mkdir build && cd build \
    && cmake -DCMAKE_INSTALL_PREFIX=/usr/local ../ \
    && make -j4 \
    && make install \
    && cd ~ \
    && rm -rf BornAgain

RUN cd ~ \
    && git clone https://github.com/reflectivity/analysis.git
