### LADCache

To run LADCache (locality aware distributed cache), you'll need to install all
of the following repos. Make sure to follow all of their install directions
(e.g., updating the kernel version as outlined in the `async-loader` readme).
You will also need to update the `nofiles` limit in `/etc/security/limits.conf`
to a very large number. I believe 2^18 worked for the 10GB imagenette dataset.
```
axboe/liburing              | master
gustrain/async-loader       | master
gustrain/mlock              | master
gustrain/minio              | master
gustrain/ladcache           | master
mengwanguc/pytorch-meng     | gus-ladcache-v2
mengwanguc/torchvision-meng | gus-ladcache-v2
mengwanguc/gpufs            | gus-ladcache-v2
```

To generate the data for our figures, use the scripts in the `gus-ladcache-v2`
branch of `gpufs`. They're all located in `gpufs/emulator/datastall/figures`.

If the instructions to generate results with just MinIO in the earlier section
do not work for you, it is possible to achieve equivalent results by using
ladcache on a single node with a bottleneck parameter of 1. For example:
```
python main-measure-time-emulator.py --use-ladcache=true --cache-size=6979321856 --loader-bottleneck 1 --gpu-type=p100 --gpu-count=8 --epoch 2 --skip-epochs=1 --workers 24 --arch=alexnet --batch-size 256 --profile-batches -1 --dist-url localhost --dist-backend gloo --world-size 1 --rank 1 /home/cc/data/test-accuracy/imagenette2
```