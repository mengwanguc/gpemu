#!/bin/bash
set -e

gpu_type="p100"
model="alexnet"

workers=(1    1    1    1     1     1     1     1     1     1     1
	 2    2    2    2     2     2     2     2     2     2     2
	 4    4    4    4     4     4     4     4     4     4     4
	 6    6    6    6     6     6     6     6     6     6     6
	 8    8    8    8     8     8     8     8     8     8
	 10   10   10   10    10    10    10    10    10    10
	 12   12   12   12    12    12    12    12    12    12)
limits=("4G" "6G" "8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"4G" "6G" "8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"4G" "6G" "8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"4G" "6G" "8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"6G" "8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G"
	"8G" "10G" "12G" "14G" "16G" "18G" "20G" "22G" "24G")

batch_size="256"
data_path="/home/cc/data/test-utilization/imagenette2"

# set up control group
group_name="gpufs"
sudo cgcreate -g memory:$group_name
sudo chown -R ${USER} /sys/fs/cgroup/memory/$group_name

cd ..

for i in ${!workers[@]}; do
    n_workers=${workers[i]}
    limit=${limits[i]}
    echo "Profiling model $model with $n_workers workers and $limit GB memory limit"

    # place the limit
    sudo bash -c "echo $limit > /sys/fs/cgroup/memory/$group_name/memory.limit_in_bytes"

    # flush memory & caches
    echo "Flushing memory/cache"
    sudo ~/gpufs/exp/clear-cache.sh

    # run training with limited memory (https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process)
    echo "running training"
    cgexec -g memory:$group_name python main-measure-time-emulator.py --emulator-version=1 -j 4 --epoch 2 --workers $n_workers --gpu-count 1 --gpu-type $gpu_type -a $model --batch-size $batch_size --profile-batches -1 $data_path

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

