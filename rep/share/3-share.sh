python main-emulator-share.py  --epoch 4 --profile-batches -1 --workers 8 --batch-size 128 \
    --gpu-type=p100 --gpu-count=1 --arch=resnet50 --emulator-version=1 ~/data/test-accuracy/imagenette2/ &> app1.csv &
sleep 30
python main-emulator-share.py  --epoch 4 --profile-batches -1 --workers 8 --batch-size 128 \
    --gpu-type=p100 --gpu-count=1 --arch=resnet50 --emulator-version=1 ~/data/test-accuracy/imagenette2 &> app2.csv &
sleep 30
python main-emulator-share.py  --epoch 4 --profile-batches -1 --workers 8 --batch-size 128 \
    --gpu-type=p100 --gpu-count=1 --arch=resnet50 --emulator-version=1 ~/data/test-accuracy/imagenette2 &> app3.csv &