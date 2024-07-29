#!/bin/bash
set -e

gpu_type="v100"
model="resnet18"
# Saved cache size: "2g" "4g" "6g" "8g" "10g" "12g" "14g"
limits=("6g" "8g" "10g" "12g" "14g" "16g")
batch_size="256"
n_workers="8"
data_path="~/data/imagenette2"
gpu_count="8"

# set up control group
group_name="myGroup"
sudo cgcreate -g memory:$group_name
sudo chown -R ${USER} /sys/fs/cgroup/memory/$group_name

for limit in ${limits[@]}; do
    echo "Profiling model $model with $limit memory limit"

    # place the limit
    sudo bash -c "echo $limit > /sys/fs/cgroup/memory/$group_name/memory.limit_in_bytes"

    # flush memory & caches
    echo "Flushing memory/cache"
    sudo ./clear-cache.sh

    # run training with limited memory (https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process)
    echo "running training"
    
    #cgexec -g memory:$group_name python main-measure-time-emulator.py --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=$model --epochs 2 --emulator-version=1 -j $n_workers $data_path &> outputsfig4/output$limit.txt
    cgexec -g memory:$group_name python main-measure-time-emulator.py  --epoch 2 --profile-batches -1 --workers $n_workers --gpu-type=$gpu_type --gpu-count=$gpu_count --arch=$model --emulator-version=1 ~/data/imagenette2 &> outputsfig4/output$limit.txt
    # check how much memory the dataset was actually using
    # NOTE this isn't entirely accurate since the amount can vary throughout, and the amount at the end may not be representative/precise/etc. 
    echo "checking memory usage..."
    vmtouch ~/data/imagenette2/train/
    vmtouch ~/data/imagenette2/train/ &> outputsfig4/cache_result$limit.txt
    echo
done

# tear down the control group
sudo cgdelete memory:$group_name