gpu_type='rtx6000'

models=(alexnet)
profile_batches=55

for model in ${models[@]}; do
    for batch_size in {1..2}; do
        echo "Profiling for model $model with batch size $batch_size"
        # python main-original-ffcv-v2-cuda-measuring.py ~/data/test-accuracy/imagenette2 -a $model --batch-size $batch_size --epoch 1
        python main-original-ffcv-v2-cuda-measuring.py -a $model --lr 0.01 ~/data/imagenette2/ --epochs 1 -b $batch_size
        echo
    done
done