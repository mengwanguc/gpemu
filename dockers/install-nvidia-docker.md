```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```


```
sudo nano /etc/docker/daemon.json
```

Add or update the following:

```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```

Test it:

```
sudo docker run --rm --gpus all nvidia/cuda:11.2.2-base-ubuntu20.04 nvidia-smi
```

```
docker run --rm --gpus all pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime python -c "import torch; print(torch.cuda.is_available())"
```

```
sudo docker pull wangm12/pytorch-1.7-cuda11.2-cudnn8-gpu:latest
```


```
tmux

sudo docker run -it --gpus all --name pytorch_gpu_container -v ~/data:/data wangm12/pytorch-1.7-cuda11.2-cudnn8-gpu:latest bash

sudo docker rm pytorch_gpu_container

```