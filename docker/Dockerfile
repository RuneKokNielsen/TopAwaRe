FROM nvidia/cuda
LABEL maintainer="rune@runekoknielsen.dk"


# ============ INSTALL ADDITIONAL PACKAGES ============

# Avoid interaction in installations
ENV DEBIAN_FRONTEND=noninteractive

# Installs build utilities and in particular:
# - OpenCV for reading/writing and displaying images
# - boost libraries

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libopencv-dev \
    wget \
    csh \
    libboost-all-dev



# ============ CLONE REPOSITORY ============

ARG GIT_USERNAME
ARG GIT_PASSWORD

WORKDIR /root
ARG CACHEBUST=1
RUN git clone https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/RuneKokNielsen/TopAwaRe.git

WORKDIR /root/TopAwaRe


# ============ BUILD ============

ARG CACHEBUST=1
RUN cmake .
ARG CACHEBUST=1
RUN make install