## Installation on GPU nodes

Profiling should be done on GPU nodes. Below are the instructions for installing the necessary softwares.


1. System configuration

All our scripts have been tested on Ubuntu 20 OS with CUDA 11.

2. Set up ssh
```
ssh-keygen -t rsa -b 4096
ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
cat ~/.ssh/id_rsa.pub
```

Copy and paste into: https://github.com/settings/keys

Configure your username and email.
```
git config --global user.name "FIRST_NAME LAST_NAME"
git config --global user.email "MY_NAME@example.com"
```

for example:

```
git config --global user.name "Meng Wang"
git config --global user.email "mengwanguc@gmail.com"
```



3. clone this repo to local

```
git clone git@github.com:mengwanguc/gpemu.git
```

4. Install conda

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```
After installation, log out and log in bash again.

5. Install packages required for builing pytorch

**Note: the commands below assumes you have cuda 11.2 installed on your machine. If you have other cuda versions, please use the magma-cuda\* that matches your CUDA version from** https://anaconda.org/pytorch/repo.

For example, here our cuda version is 11.2 (check it by running nvidia-smi), that's why the command is `conda install -y -c pytorch magma-cuda112`

```
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

# CUDA only: Add LAPACK support for the GPU if needed
conda install -y -c pytorch magma-cuda112  # or the magma-cuda* that matches your CUDA version from https://anaconda.org/pytorch/repo
```

6. Install CuDNN:

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudnn8=8.1.1.33-1+cuda11.2
```

```
cd ~
mkdir cudnn
cd cudnn
pip install gdown
gdown https://drive.google.com/uc?id=1IJGIH7Axqd8E5Czox_xRDCLOyvP--ej-
tar -xvf cudnn-11.2-linux-x64-v8.1.1.33.tar
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

sudo apt-get update
sudo apt-get install libcudnn8=8.1.1.33-1+cuda11.2
```



7. Download our custom pytorch and build it

```
cd ~
git clone git@github.com:mengwanguc/pytorch-meng.git
cd pytorch-meng
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py install
```

8. Download our custom torchvision and build it

```
conda install -y aiofiles

cd ~
git clone git@github.com:mengwanguc/torchvision-meng.git
cd torchvision-meng/
python setup.py install
```

9. Install transformer for speeach

```
pip install transformers
```