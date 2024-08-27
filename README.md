# GPEmu

This repo contains the code, data, corresponding software/packages, and experiment guidelines for GPEmu (GPU Emulator).

All code/experiments have been tested on Chameleon Cloud (https://www.chameleoncloud.org) using Ubuntu20 machines and with CUDA 11.

Most of our experiments are conducted with PyTorch, using [our custom PyTorch Implementation](https://github.com/mengwanguc/pytorch-meng) and [custom TorchVision](https://github.com/mengwanguc/torchvision-meng), with different 
branches corresponding to different experiments. We will specify the branch name in the experiment guidelines.

GPEmu also supports other deep learning frameworks such as TensorFlow and NVIDIA DALI. For example, our reproduction of the FastFlow was 
based on the integration of GPEmu with TensorFlow.

## GPEmu Installation

1. Platform and Image

Our experiments have been tested on [Chameleon Cloud](https://www.chameleoncloud.org) using Ubuntu 20. Therefore, we suggest Please using "ubuntu20-xxx" images.

GPEmu is an emulator with the purpose of prototyping deep learning system research *without real GPUs*. Therefore, no real GPUs are needed for running GPEmu.

2. Set up ssh key for Github
```
bash setup-ssh-key.sh
```

Copy and paste into: https://github.com/settings/keys

3. clone this repo to local

```
cd ~
git clone https://github.com/mengwanguc/gpemu.git
```

4. Install conda

```
bash install-conda.sh
source ~/.bashrc
```

5. Download and build our mlock package (which is used to emulate page-locked (pinned) memory)

```
cd ~
git clone git@github.com:gustrain/mlock.git
cd mlock
python setup.py install
```


6. Install PyTorch

Install packages required for builing pytorch

```
conda install -y astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses
```
Download our custom pytorch and build it (Note that we use "export USE_CUDA=0" to not install any cuda/GPU-related things.)

```
cd ~
git clone https://github.com/mengwanguc/pytorch-meng.git
cd pytorch-meng
git submodule update --init --recursive

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=0
git checkout gus-emulator-minio
python setup.py install
```

9. Download our custom torchvision and build it

```
conda install -y aiofiles

cd ~
git clone https://github.com/mengwanguc/torchvision-meng.git
cd torchvision-meng/
git checkout gus-min-io
python setup.py install
```

10. Update `/etc/security/limits.conf`

```
sudo nano /etc/security/limits.conf
```

Add the following text to the end of the file:

```
*   soft    memlock     unlimited
*   hard    memlock     unlimited
```

11. Reboot the machine, which will take a while, and may require you to try to reopen/reconnect to your machine. 

```
sudo reboot
```


## Our other repos

- Our python library for supporting page-locked (pinned) memory using mlock: https://github.com/gustrain/mlock
- Our Kubernetes plugin for emulated GPU: https://github.com/mengwanguc/gpemu-k8s
- Our own implementation of MinIO cache (from DataStall, VLDB '21), as well as our new micro-optimization SSF (Small File First) cache: https://github.com/gustrain/minio
- Our own implementation of CoorDL (distributed MinIO) as well as Locality-Aware Distributed Cache (HiPC): https://github.com/gustrain/ladcache
- Our new micro-optimization Asycn Batch data loader: https://github.com/gustrain/async-loader
- Our dirty repository with unorganized code (we are working on organizing and moving code to this repo): https://github.com/mengwanguc/gpufs

## Annoucements/Notes
*2024/8/27*: We are always trying to polish our repos. However, since the students are all busy with internships this summer, our time is limited. We are expected to be back in early September. Please bear with us, and don't hesitate to reach out to me (wangm12@uchicago.edu) if you have any questions.
