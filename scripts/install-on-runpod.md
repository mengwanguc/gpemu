# Install custom pytorch on Runpod instances

## Uninstall existing pytorch

pip uninstall torch torchvision torchaudio -y

## Install pytorch from source
```
curl -o Miniconda3-py39_23.5.2-0-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh && bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -b -p $HOME/miniconda && export PATH="$HOME/miniconda/bin:$PATH"

conda init


cd /workspace

git clone git@github.com:mengwanguc/pytorch-gpemu.git


cd pytorch-gpemu
git checkout v2.2.0
git submodule sync
git submodule update --init --recursive

conda install -y cmake ninja

pip install -r requirements.txt

# pytorch 2.2 requires numpy 1.22
pip uninstall numpy -y
pip install numpy==1.22.4

conda install -y mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda121  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda-12.1
export CUDNN_INCLUDE_DIR=/usr/local/cuda-12.1/include
export CUDNN_LIB_DIR=/usr/local/cuda-12.1/lib64

python setup.py install
```