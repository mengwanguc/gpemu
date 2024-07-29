## Implementing FFCV on imagenette

1. Install all depedencies

   To install gpufs contained ffcv folder. and others depedencies.
    
   Need doing all step on https://github.com/mengwanguc/gpufs#readme until you done clone gpufs repository.

   And move to naufal branch:
   ```
   git checkout naufal
   ```

   Install ffcv with this, it will created ffcv env:
   ```
    conda create -n ffcv python=3.9 cupy pkg-config libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.6 numba -c conda-forge -c pytorch && conda activate ffcv && conda update ffmpeg && pip install ffcv
                    
   ```

   After that install requirement package:
   ```
   cd ~/gpufs/ffcv
   pip install -r requirements.txt
   ```

2. Download the [imagenettee](https://github.com/fastai/imagenette) dataset:

   ```
   cd ~
   mkdir data
   cd data
   mkdir test-accuracy
   cd test-accuracy
   wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
   tar -zxvf imagenette2.tgz
   ```
   
3. Convert dataset into ffcv format

   FFCV required to have same data properties such as resolution, for that you can use write "write_imagenet.sh" for that .it will reproduce train and val with ffcv format.

   ```
   # Required environmental variables for the script:
   export IMAGENET_DIR=~/data/test-accuracy/imagenette2
   export WRITE_DIR=~/data/

   # Starting in the root of the Git repo:
   cd ~/gpufs/ffcv

   # Serialize images with:
   # - 500px side length maximum
   # - 50% JPEG encoded
   # - quality=90 JPEGs
   ./write_imagenet.sh 500 0.50 90
   ```
## Measuring Preprocessing Time FFCV
1. Modified ops.py from ffcv folder
   
   [FFCV ops.py file that has been modified for measuring time](https://github.com/NaufalRezkyA/ffcv-emulator/blob/original-measuring-time/ffcv/transforms/ops.py)
   
   Copy this script and replaced them into FFCV folder
   ```
   /home/cc/anaconda3/envs/ffcv/lib/python3.9/site-packages/ffcv/transforms/ops.py
   ```
2. Modified transform.py from torchvision folder
   
   [FFCV transform.py file that has been modified for emulate the GPU for measuring time](https://github.com/mengwanguc/gpufs/blob/naufal/custom-file/transform(shape-time).py)
   
   Copy this script and replaced them into FFCV folder
   ```
   /home/cc/torchvision-meng/torchvision/transforms/transforms.py
   ```

3. Training the model

   This training will produce preprocessing time log file in ffcv folder. You can see it on "~/gpufs/ffcv/{preprocessing-stage-name}.txt" precossing stage used in this schema is : toDevice(), ToTorchImage(), Convert(), and Normalize()

   ```
   cd ~/gpufs/ffcv/figure2
   python main-original-ffcv-v2-cuda-measuring.py -a resnet18 --lr 0.1 ~/data/test-accuracy/imagenette2/ --epochs 1
   ```

## Emulating CPU
1. To emulate FFCV on CPU.
   
   [FFCV ops.py file that has been modified for emulate the GPU](https://github.com/NaufalRezkyA/ffcv-emulator/blob/main/ffcv/transforms/ops.py)
   
   Copy this script and replaced them into FFCV folder
   ```
   /home/cc/anaconda3/envs/ffcv/lib/python3.9/site-packages/ffcv/transforms/ops.py
   ```
2. Training the model

   The main thing to implement FFCV into our code is modify the Dataloader, because FFCV speeds up model training by eliminating (often subtle) data bottlenecks from the training process. It have similar format with the pytorch and we doesnt have to modify others things. You can use this command to train the models. Prepocessing time and total training time will write in "test-ffcv.txt"

   ```
   cd ~/gpufs/ffcv/
   python main-original-ffcv-v2-emulatorv0-cpu-todevice.py -a resnet18 --lr 0.1 ~/data/test-accuracy/imagenette2/ --epochs 1
   ```

   
## Usage

```
usage: main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N]
               [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N]
               [--resume PATH] [-e] [--pretrained] [--world-size WORLD_SIZE]
               [--rank RANK] [--dist-url DIST_URL]
               [--dist-backend DIST_BACKEND] [--seed SEED] [--gpu GPU]
               [--multiprocessing-distributed]
               DIR

PyTorch ImageNet Training

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  model architecture: alexnet | densenet121 |
                        densenet161 | densenet169 | densenet201 |
                        resnet101 | resnet152 | resnet18 | resnet34 |
                        resnet50 | squeezenet1_0 | squeezenet1_1 | vgg11 |
                        vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19
                        | vgg19_bn (default: resnet18)
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  --start-epoch N       manual epoch number (useful on restarts)
  -b N, --batch-size N  mini-batch size (default: 256), this is the total
                        batch size of all GPUs on the current node when using
                        Data Parallel or Distributed Data Parallel
  --lr LR, --learning-rate LR
                        initial learning rate
  --momentum M          momentum
  --weight-decay W, --wd W
                        weight decay (default: 1e-4)
  --print-freq N, -p N  print frequency (default: 10)
  --resume PATH         path to latest checkpoint (default: none)
  -e, --evaluate        evaluate model on validation set
  --pretrained          use pre-trained model
  --world-size WORLD_SIZE
                        number of nodes for distributed training
  --rank RANK           node rank for distributed training
  --dist-url DIST_URL   url used to set up distributed training
  --dist-backend DIST_BACKEND
                        distributed backend
  --seed SEED           seed for initializing training.
  --gpu GPU             GPU id to use.
  --multiprocessing-distributed
                        Use multi-processing distributed training to launch N
                        processes per node, which has N GPUs. This is the
                        fastest way to use PyTorch for either single node or
                        multi node data parallel training
```
