cd ~
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install libcudnn8=8.1.1.33-1+cuda11.2
rm cuda-keyring_1.1-1_all.deb

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