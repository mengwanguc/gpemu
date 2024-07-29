# File Grouping

This repo contains our micro-optimization that uses file grouping to improve the data loading.

1. Install all depedencies

   To install gpufs(detection files), pytorch, and torchvision
    
   Need doing all step on [../profiler/README.md](../profiler/README.md) until torchvision installed


2. Download imagenet large dataset.

   You need to make new folder ".kaggle" in your root and then copied your api key file in it. Next you need to give permission to that file using "chmod 600 'api key file'".
```
mkdir data
cd data
pip install kaggle
kaggle competitions download -c imagenet-object-localization-challenge
pip install unzip
unzip imagenet-object-localization-challenge.zip

```

3. Training without grouping
```
python main-original.py -a resnet50 --lr 0.1 ~/data/ILSVRC/Data/CLS-LOC --epochs 50
```

4. Training with grouping
```
cd ~/gpufs/grouping/diffclass-grouping
python group-needle.py ~/ILSVRC/Data/CLS-LOC/train 64 ~/data/grouping-data
python main-group-accuracy.py ~/data/grouping-data/64/ ~/data/ILSVRC/Data/CLS-LOC/val/ --img_per_tar 64 --epoch 50 -a resnet18 --lr 0.1 
```

5. Training with no grouping method.
```
cd ~/gpufs/grouping/sequential-grouping
python group-needle-sequential.py ~/ILSVRC/Data/CLS-LOC/train 64 ~/data/sequential-data
python main-group-accuracy.py ~/data/sequential-data/64/ ~/data/ILSVRC/Data/CLS-LOC/val/ --img_per_tar 64 --epoch 50 -a resnet18 --lr 0.1 
```


`group-needle.py` assumes that the train folder has such a folder structure:

- train
  - class_1
    - img_1
    - img_2
    - ...
  - class_2
    - img_1
    - img_2
    - ...


example:
`python group-needle.py ~/data/test-accuracy/imagenette2/train 4 ~/data/test-accuracy/mytar/train`
