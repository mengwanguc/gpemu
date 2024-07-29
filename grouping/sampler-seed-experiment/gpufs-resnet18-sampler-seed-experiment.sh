#!/bin/bash

python ~/gpufs/gpufs/grouping/sampler-seed-experiment/main-original.py ~/data/test-accuracy/imagenette2/ --epoch 100 -a resnet18 -b 128 --sampler-name sampler-seed-4 
python ~/gpufs/gpufs/grouping/sampler-seed-experiment/main-group-accuracy-rand-perm.py ~/data/test-accuracy/mytar/train-perm-rand/64/ ~/data/test-accuracy/imagenette2/val/ --img_per_tar 64 --epoch 100 -a resnet18 -b 128 --sampler-name sampler-seed-4
python ~/gpufs/gpufs/grouping/sampler-seed-experiment/main-group-accuracy-sequential.py ~/data/test-accuracy/mytar/train-sequential/64/ ~/data/test-accuracy/imagenette2/val/ --img_per_tar 64 --epoch 100 -a resnet18 -b 128 --sampler-name sampler-seed-4
python ~/gpufs/gpufs/grouping/sampler-seed-experiment/main-group-accuracy-diffclass.py ~/data/test-accuracy/mytar/train-diffclass/64/ ~/data/test-accuracy/imagenette2/val/ --img_per_tar 64 --epoch 100 -a resnet18 -b 128 --sampler-name sampler-seed-4