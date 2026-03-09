# Use the base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

ENV TORCH_CUDA_ARCH_LIST="8.9"
ENV DEBIAN_FRONTEND=noninteractive

# Update and install tzdata separately
RUN apt update && apt install -y tzdata

# Install necessary packages
RUN apt install -y git && \
    apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev && \
    apt clean && apt install wget && rm -rf /var/lib/apt/lists/*

# Create a workspace directory 
WORKDIR /workspace/3dgs_dependency
# Clone the repository (dev branch): https://github.com/graphdeco-inria/gaussian-splatting/issues/965
RUN git clone -b master https://gitlab.inria.fr/sibr/sibr_core.git \
    && git clone https://gitlab.inria.fr/bkerbl/simple-knn.git 
COPY submodules/geer-rasterizer /workspace/3dgs_dependency/geer-rasterizer

WORKDIR /workspace/3dgs_dependency/sibr_core
RUN git reset --hard d8856f6
WORKDIR /workspace/3dgs_dependency/simple-knn
RUN git reset --hard 44f7642

# Create a Conda environment and activate it
WORKDIR /workspace

# RUN conda init bash && . ~/.bashrc && conda create -n geer python=3.8 && conda activate geer && \
#     conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 cudatoolkit=11.8 -c pytorch -c nvidia && \
#     conda install tqdm && conda install plyfile -c defaults -c conda-forge && \
#     pip install 3dgs_dependency/simple-knn && \
#     pip install opencv-python psutil matplotlib

RUN pip install --upgrade pip && \
    pip install tqdm plyfile==1.1 opencv-python==4.10.0.84 psutil matplotlib numpy==1.24.3 && \
    pip install --no-build-isolation ./3dgs_dependency/simple-knn && \
    pip install --no-build-isolation ./3dgs_dependency/geer-rasterizer

# Tweak the CMake file for matching the existing OpenCV version. Fix the naming of FindEmbree.cmake
WORKDIR /workspace/3dgs_dependency/sibr_core/cmake/linux
RUN sed -i 's/find_package(OpenCV 4\.5 REQUIRED)/find_package(OpenCV 4.2 REQUIRED)/g' dependencies.cmake
RUN sed -i 's/find_package(embree 3\.0 )/find_package(EMBREE)/g' dependencies.cmake
RUN mv /workspace/3dgs_dependency/sibr_core/cmake/linux/Modules/FindEmbree.cmake /workspace/3dgs_dependency/sibr_core/cmake/linux/Modules/FindEMBREE.cmake

# Fix the naming of the embree library in the rayscaster's cmake
RUN sed -i 's/\bembree\b/embree3/g' /workspace/3dgs_dependency/sibr_core/src/core/raycaster/CMakeLists.txt

# Ready to build the viewer now.
WORKDIR /workspace/3dgs_dependency/sibr_core 
RUN cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j24 --target install
