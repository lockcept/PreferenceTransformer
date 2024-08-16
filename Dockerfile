# CUDA 지원하는 베이스 이미지 사용 
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# 필수 패키지 설치
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3-pip \
    unzip \
    cmake \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# patchelf 설치
RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG=C.UTF-8

# CUDA 및 cuDNN 버전 확인
RUN nvcc --version
RUN cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Xdummy 설치 및 권한 설정
COPY vendor/Xdummy /usr/local/bin/Xdummy
RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
COPY vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip3 install --upgrade pip 
RUN pip3 install "jax[cuda11_cudnn805]>=0.2.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install protobuf==3.20.1 gym==0.23.1 distrax==0.1.2 wandb transformers

# 작업 디렉토리 설정
WORKDIR /workspace

# Mujoco 설치
COPY requirements-mujoco.txt /workspace/
RUN pip3 install -r requirements-mujoco.txt

RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O mjpro150.zip \
    && unzip mjpro150.zip -d /root/.mujoco \
    && rm mjpro150.zip

COPY mjkey.txt /root/.mujoco/mjkey.txt

ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mjpro150
ENV LD_LIBRARY_PATH=/root/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/lib/nvidia:/usr/lib/nvidia-000:${LD_LIBRARY_PATH}

RUN pip show Cython > /cython_version.log
RUN cat /cython_version.log

# RUN pip3 install lockfile
# COPY requirements.txt /workspace/
# RUN pip3 install -r requirements.txt

# COPY requirements-extra.txt /workspace/
# RUN pip3 install -r requirements-extra.txt

# COPY . /workspace

# RUN cd d4rl && pip3 install -e . && cd ..

# 전체 코드를 마지막에 복사하고 설치
# COPY . /mujoco_py
# RUN python setup.py install

# Docker 컨테이너 시작 시 기본 명령어
# ENTRYPOINT ["/mujoco_py/vendor/Xdummy-entrypoint"]
# CMD ["pytest"]