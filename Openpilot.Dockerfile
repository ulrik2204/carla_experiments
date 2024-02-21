FROM ubuntu:20.04

RUN apt-get update
# Install dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

RUN apt-get install -y \
    software-properties-common \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    wget \
    curl \
    libbz2-dev

RUN apt install -y git
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install -y python3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
RUN apt install -y python3.11-distutils

# git lfs
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
RUN apt-get update
RUN apt-get install git-lfs

# Poetry
RUN python3.11 -m pip install poetry


# Clone openpilot  
RUN git clone --recurse-submodules https://github.com/commaai/openpilot.git

# Setup openpilot
WORKDIR /openpilot 
RUN git lfs pull
RUN tools/ubuntu_setup.sh
RUN poetry run scons -u -j$(nproc)
# Keep the container running
CMD ["tail", "-f", "/dev/null"]
