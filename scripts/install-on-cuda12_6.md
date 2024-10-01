# Install custom pytorch on Runpod instances

## Uninstall existing pytorch

pip uninstall torch torchvision torchaudio -y

## Install pytorch from source
```
<!-- curl -o Miniconda3-py39_23.5.2-0-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh && bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh -b -p $HOME/miniconda && export PATH="$HOME/miniconda/bin:$PATH" -->

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p $HOME/anaconda3
$HOME/anaconda3/bin/conda init
source ~/.bashrc
rm Anaconda3-2024.10-1-Linux-x86_64.sh


cd /workspace

git clone git@github.com:mengwanguc/pytorch-gpemu.git


cd pytorch-gpemu
git checkout v2.4.1
git submodule sync
git submodule update --init --recursive

conda install -y cmake ninja

pip install -r requirements.txt

# pytorch 2.2 requires numpy 1.22
pip uninstall numpy -y
pip install numpy==1.22.4

conda install -y mkl mkl-include
# CUDA only: Add LAPACK support for the GPU if needed
conda install -c pytorch magma-cuda126  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo

```


```
cd ~
wget https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.5.1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2204-9.5.1/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12

wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.5.1.17_cuda12-archive.tar.xz
tar -xvJf cudnn-linux-x86_64-9.5.1.17_cuda12-archive.tar.xz
mv cudnn-linux-x86_64-9.5.1.17_cuda12-archive cuda
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

sudo apt-get update
sudo apt-get -y install cudnn
sudo apt-get -y install cudnn-cuda-12
```



```
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=1
# export CUDA_HOME=/usr/local/cuda-12.1
# export CUDNN_INCLUDE_DIR=/usr/local/cuda-12.1/include
# export CUDNN_LIB_DIR=/usr/local/cuda-12.1/lib64



python setup.py install
```