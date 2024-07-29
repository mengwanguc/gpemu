#!/bin/bash
set -e

gpu_type="p100"
models=("shufflenet_v2_x0_5" "squeezenet1_0" "resnet18" "mobilenet_v2" "alexnet" "vgg11")
batch_size="256"
n_workers="12"
data_path="/home/cc/data/test-utilization/imagenette2"

for model in ${models[@]}; do
    echo "Profiling model $model with $limit GB memory limit"

    # flush memory & caches
    echo "Flushing memory/cache"
    sudo ~/gpufs/exp/clear-cache.sh

    # run training with limited memory (https://unix.stackexchange.com/questions/44985/limit-memory-usage-for-a-single-linux-process)
    echo "running training"
    python main-measure-time.py --epoch 2 --workers $n_workers --gpu 0 --gpu-type $gpu_type -a $model --batch-size $batch_size $data_path --profile-batches -1
	
    # check how much memory the dataset was actually using
    # NOTE this isn't entirely accurate since the amount can vary throughout, and the amount at the end may not be representative/precise/etc. 
    echo "checking memory usage..."
    usage=$(vmtouch $data_path | grep "Resident Pages" | awk 'NF>1{print $NF}')
    echo "... $usage cached\n"

    # save our output to a meaningful filename
    mv ./$gpu_type/$model-batch$batch_size.csv ./$gpu_type/$model-$batch_size-batch_size-$n_workers-workers-$limit-limit-$usage-usage.csv
    echo
done

