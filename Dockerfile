# https://github.com/microsoft/onnxruntime/blob/main/dockerfiles/Dockerfile.tensorrt
# --------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------
# Dockerfile to run ONNXRuntime with TensorRT integration

# nVidia TensorRT Base Image
FROM nvcr.io/nvidia/tensorrt:24.10-py3

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_BRANCH=rel-1.20.0
ARG CMAKE_CUDA_ARCHITECTURES=37;50;52;60;61;70;75;80;89

RUN apt-get update &&\
    apt-get install -y sudo git bash unattended-upgrades
RUN unattended-upgrade

WORKDIR /code
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

# cmake 3.28 (CMake 3.26 or higher is required)
RUN apt-get -y update && apt install wget && wget https://github.com/Kitware/CMake/releases/download/v3.28.0-rc1/cmake-3.28.0-rc1-linux-x86_64.sh  && \
    chmod +x cmake-3.28.0-rc1-linux-x86_64.sh  && sh cmake-3.28.0-rc1-linux-x86_64.sh  --skip-license && \
    cp /code/bin/cmake /usr/bin/cmake && cp /code/bin/cmake /usr/lib/cmake && \
    cp /code/bin/cmake /usr/local/bin/cmake && cp -r /code/share/cmake-3.28 /usr/local/share/ && \
    rm -rf cmake-3.28.0-rc1-linux-x86_64.sh 

# Prepare onnxruntime repository & build onnxruntime with TensorRT
# --parallel crashes for me due to out of ram, only use it if you have ram
RUN git clone --single-branch --branch ${ONNXRUNTIME_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
    trt_version=${TRT_VERSION:0:3} &&\
    /bin/sh onnxruntime/dockerfiles/scripts/checkout_submodules.sh ${trt_version} &&\
    cd onnxruntime &&\
    PYTHONPATH=/usr/bin/python3 /bin/sh build.sh --nvcc_threads 8 --parallel 8 --allow_running_as_root --build_shared_lib \
        --cuda_home /usr/local/cuda --cudnn_home /usr/lib/x86_64-linux-gnu/ --use_tensorrt --tensorrt_home /usr/lib/x86_64-linux-gnu/ \
        --config Release --build_wheel --skip_tests --skip_submodule_sync --cmake_extra_defines '"CMAKE_CUDA_ARCHITECTURES='${CMAKE_CUDA_ARCHITECTURES}'"' && \
    pip install /code/onnxruntime/build/Linux/Release/dist/*.whl &&\
    cd .. && rm -rf onnxruntime

RUN apt-get update && apt-get install libopencv-dev libturbojpeg-dev libsm6 libxext6 -y

RUN pip install torch torchvision torchaudio --extra-index-url=https://download.pytorch.org/whl/cu124
RUN pip install opencv-python tqdm onnxsim onnxslim albumentations scikit-learn kornia efficientnet_pytorch x_transformers vit-pytorch swin-transformer-pytorch adamp \
    tensorboardX torchvision timm madgrad pytorch_lightning git+https://github.com/styler00dollar/pytorch-randaugment adan-pytorch \
    git+https://github.com/lilohuang/PyTurboJPEG.git ffcv ffcv_pl
RUN MAX_JOBS=2 pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable