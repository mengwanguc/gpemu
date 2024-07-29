gpu_type="rtx6000"

models=("alexnet" "resnet18" "mobilenet_v2" "resnet50")

workers=(1 2 3 6 12 24)

batch_size=256

for model in ${models[@]}; do
	for worker_n in ${workers[@]}; do
		echo "Profiling model $model with $worker_n workers (cores)"
		python main-measure-time.py --epoch 1 --workers $worker_n ~/data/test-accuracy/imagenette2 --gpu 0 --gpu-type $gpu_type -a $model --batch-size $batch_size
		mv ./$gpu_type/$model-batch$batch_size.csv ./$gpu_type/$model-batch$batch_size-$worker_n-worker.csv
		echo
	done
done

