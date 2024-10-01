sudo docker build -t wangm12/pytorch-1.7-cuda11.2-cudnn8-gpu .

sudo docker run --rm --gpus all -itd --name pytorch-container pytorch-1.7-cuda11.2-cudnn8-gpu bash

sudo docker exec -it pytorch-container bash


python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())

sudo docker push wangm12/pytorch-1.7-cuda11.2-cudnn8-gpu