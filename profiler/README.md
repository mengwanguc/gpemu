# GPEmu Profiler

This directory contains the code for profiling the time and memory metrics of the DL models. The data are organized to be easily used by GPEmu to emulate DL runs.

## Installation on GPU nodes

Profiling should be done on GPU nodes to collect the GPU-related metrics. Below are the instructions for installing the necessary softwares.

1. System configuration

All our scripts have been tested on Ubuntu 20 OS with CUDA 11.

2. Set up ssh
```
ssh-keygen -t rsa -b 4096
```

``` 
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
cd ~
git clone git@github.com:mengwanguc/gpemu.git
```

4. Install conda

```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh
```
After installation, log out and log in bash again.

Or install conda in the non-interactive mode:
```
wget https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh
bash Anaconda3-2021.11-Linux-x86_64.sh -b -p $HOME/anaconda3
$HOME/anaconda3/bin/conda init
source ~/.bashrc
```

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
bash ~/gpemu/profiler/install-cudnn.sh
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


## Profiling

- To profile the compute time of models, see [time-compute](time-compute/README.md)
- To profile the CPU-to-GPU data transfer time, see [time-transfer](time-transfer/README.md)
- To profile the GPU-based preprocessing time, see [time-preprocess](time-preprocess/README.md)
- To profile the memory usage of models, see [memory](memory/README.md)