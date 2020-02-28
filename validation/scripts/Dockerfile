#
# Dockerfile for testing
#
# docker build . -t andyfaff/orso:latest
# docker push andyfaff/orso:latest
#
# docker run -it --rm -v /bin/bash


FROM debian:latest

RUN apt-get update

RUN apt-get install -y python3.7 python3.7-dev git build-essential vim  curl\
                       python3.7-distutils python3-wxgtk4.0 libgtk-3-dev\
                       gfortran  cmake libgsl-dev libboost-all-dev \
                       libfftw3-dev libtiff5-dev qt5-default \
                       libqt5designercomponents5 qttools5-dev libqt5svg5-dev

# setup pips, pip3.6 and pip3.7
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.7 get-pip.py

# install reflectometry packages for Python testing
RUN pip3.7 install numpy scipy cython pytest && pip3.7 install refnx refl1d pytest

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
