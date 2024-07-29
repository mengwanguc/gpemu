#!/bin/bash
set -e

gpu_type="p100"
# "shufflenet_v2_x0_5" "resnet18" "vgg11"
models=("alexnet" "resnet18" "mobilenet_v2" "resnet50")

batch_size="256"
# "1" "2" "3" "6" "12" 
n_workers=("24")
data_path="~/data/imagenette2"
gpu_count="8"

# flush memory & caches
echo "Flushing memory/cache"
sudo ./clear-cache.sh
#run one epoch before the nested loop to make sure all data are in there. 
python main-measure-time-emulator.py  --epoch 1 --profile-batches -1 --workers 24 --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=resnet18 --emulator-version=1 ~/data/imagenette2 &> outputsfig6/output_fill_cache.txt
vmtouch ~/data/imagenette2/train/
for model in ${models[@]}; do
    for worker in ${n_workers[@]}; do
        echo "running training"
        python main-measure-time-emulator.py  --epoch 1 --profile-batches -1 --workers $worker --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=$model --emulator-version=1 ~/data/imagenette2 &> outputsfig6/output_$model.txt
        echo "finished model $model"
    done
    echo "Finished model: $model"
done
