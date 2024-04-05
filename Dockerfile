FROM nvidia/cuda:11.6.2-base-ubuntu20.04 as base

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# For nvidia GPU
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

RUN apt-get update && \
    apt-get install -y \
    python3-pip python-is-python3 vim git \
    ffmpeg libsm6 libxext6 vim git \
    xauth tzdata libgl1-mesa-glx libgl1-mesa-dri \
    libeigen3-dev lsb-release curl coinor-libclp-dev cmake freeglut3-dev python3-tk

RUN groupadd --gid $USER_GID $USERNAME && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME
USER $USERNAME

RUN mkdir -p /home/user
WORKDIR /home/user

RUN git clone https://github.com/NVlabs/storm.git && \
    cd storm && \
    pip install -e .

RUN git clone https://github.com/pkicki/air_hockey_challenge.git && \
    cd air_hockey_challenge && \
    git checkout piotr_exp && \
    pip install -e .
RUN pip install mujoco osqp nlopt PyYAML

RUN git clone https://github.com/pkicki/mushroom-rl.git && \
    cd mushroom-rl && \
    git checkout ePPO && \
    pip install --no-use-pep517 -e .
     
RUN pip install hiyapyco imageio mujoco dm_control wandb hydra urdf_parser_py mp-pytorch \
                protobuf==4.23.0 torch opencv-python scikit-learn pygame matplotlib


