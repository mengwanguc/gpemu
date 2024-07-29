#!/bin/bash
set -e

gpu_type="p100"
models=("vgg11")
limit="12G"
batch_size="256"
n_workers="12"
data_path="/home/cc/data/test-utilization/imagenette2"

# set up control group
group_name="gpufs"
sudo cgcreate -g memory:$group_name
sudo chown -R ${USER} /sys/fs/cgroup/memory/$group_name

for model in ${models[@]}; do
    echo "Profiling model $model with $limit GB memory limit"

    # place the limit
    sudo bash -c "echo $limit > /sys/fs/cgroup/memory/$group_name/memory.limit_in_bytes"

    # flush memory & caches
    echo "Flushing memory/cache"
    sudo ~/gpufs/exp/clear-cache.sh

    # run training with limited memory (https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process)
    echo "running training"
    cgexec -g memory:$group_name python main-measure-time.py --epoch 2 --workers $n_workers --gpu 0 --gpu-type $gpu_type -a $model --batch-size $batch_size $data_path --profile-batches -1
	
    # check how much memory the dataset was actually using
    # NOTE this isn't entirely accurate since the amount can vary throughout, and the amount at the end may not be representative/precise/etc. 
    echo "checking memory usage..."
    usage=$(vmtouch $data_path | grep "Resident Pages" | awk 'NF>1{print $NF}')
    echo "... $usage cached\n"

    # save our output to a meaningful filename
    mv ./$gpu_type/$model-batch$batch_size.csv ./$gpu_type/$model-$batch_size-batch_size-$n_workers-workers-$limit-limit-$usage-usage.csv
    echo
done

# tear down the control group
sudo cgdelete memory:$group_name

