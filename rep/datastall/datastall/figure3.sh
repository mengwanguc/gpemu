#!/bin/bash
set -e

gpu_type="v100"

# For SSD node, use ("8g" "8g" "7g" "9g" "8g" "8g" "8g" "8g")
# For HDD node, use ("13g" "13g" "14g" "13g" "14g" "13g" "13g" "13g")
cache_sizes=("21000m" "21500m" "21500m" "21500m" "21500m" "21500m" "21500m")
models=("shufflenet_v2_x0_5" "alexnet" "resnet18" "squeezenet1_0" "vgg11" "mobilenet_v2" "resnet50")
# cache_sizes=("14500m")
# models=("shufflenet_v2_x0_5")

# Saved cache size: "2g" "4g" "6g" "8g" "10g" "12g" "14g"
# to reach 35% of figrue 3, I calculated that it should be about 3.2GB = 3200MB
batch_size="256"
n_workers="40"
data_path="~/data/imagenette2"
gpu_count="8"

# set up control group
group_name="myGroup"
sudo cgcreate -g memory:$group_name
sudo chown -R ${USER} /sys/fs/cgroup/memory/$group_name
length=${#models[@]}

for ((i=0; i<length; i++))
do
    # flush memory & caches
    #echo "Flushing memory/cache"
    #sudo ./clear-cache.sh
    model=${models[i]}
    cache_size=${cache_sizes[i]}
    echo "Flushing memory/cache"
    sudo ./clear-cache.sh
    echo "set the cache size"
    sudo bash -c "echo $cache_size > /sys/fs/cgroup/memory/$group_name/memory.limit_in_bytes"
    # run training with limited memory (https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process)
    echo "running training"
    #cgexec -g memory:$group_name python main-measure-time-emulator.py --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=$model --epochs 2 --emulator-version=1 -j $n_workers $data_path &> outputsfig4/output$limit.txt
    cgexec -g memory:$group_name python main-measure-time-emulator.py  --epoch 2 --profile-batches -1 --workers $n_workers --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=$model --emulator-version=1 ~/data/imagenette2 &> outputsfig3/output$model.txt
    # check how much memory the dataset was actually using
    # NOTE this isn't entirely accurate since the amount can vary throughout, and the amount at the end may not be representative/precise/etc. 
    echo "checking memory usage and write into file"
    vmtouch ~/data/imagenette2/train/
    vmtouch ~/data/imagenette2/train/ &> outputsfig3/cache_result$model.txt
done

# tear down the control group
sudo cgdelete memory:$group_name