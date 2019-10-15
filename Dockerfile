FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install pip
RUN apt-get update
RUN apt-get -y install python3 python3-pip python3-dev python3-tk
RUN apt-get -y install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev

# Install basic libraries
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN pip3 install numpy tensorflow-gpu==1.9 matplotlib scipy scikit-learn future
RUN pip3 install torch==1.0.1 torchvision==0.2.1 -f https://download.pytorch.org/whl/torch_stable.html 

RUN pip3 install gym pybullet
RUN apt-get update
RUN apt-get -y install unzip unetbootin wget

ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Install additional requirements
RUN pip3 install --upgrade six
RUN pip3 install datetime gitpython h5py tqdm dotmap cython jupyter

# Environment setup
# RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

WORKDIR /root/stage
CMD /bin/bash
