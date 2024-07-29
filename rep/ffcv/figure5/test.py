import os.path

test_data_root = os.environ['DALI_EXTRA_PATH']

# Caffe LMDB
lmdb_folder = os.path.join(test_data_root, 'db', 'lmdb')

N = 8             # number of GPUs
BATCH_SIZE = 128  # batch size per GPU
ITERATIONS = 32
IMAGE_SIZE = 3

from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types


@pipeline_def
def caffe_pipeline(num_gpus):
    device_id = Pipeline.current().device_id
    jpegs, labels = fn.readers.caffe(
        name='Reader', path=lmdb_folder, random_shuffle=True, shard_id=device_id, num_shards=num_gpus)
    images = fn.decoders.image(jpegs, device='mixed')
    images = fn.resize(
        images,
        resize_shorter=fn.random.uniform(range=(256, 480)),
        interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
            images,
            crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
            crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
            dtype=types.FLOAT,
            crop=(227, 227),
            mean=[128., 128., 128.],
            std=[1., 1., 1.])

    return images, labels

import numpy as np
from nvidia.dali.plugin.pytorch import DALIGenericIterator


label_range = (0, 999)
pipes = [caffe_pipeline(
    batch_size=BATCH_SIZE, num_threads=2, device_id=device_id, num_gpus=N) for device_id in range(N)]

for pipe in pipes:
    pipe.build()

dali_iter = DALIGenericIterator(pipes, ['data', 'label'], reader_name='Reader')

for i, data in enumerate(dali_iter):
    # Testing correctness of labels
    for d in data:
        label = d["label"]
        image = d["data"]
        ## labels need to be integers
        assert(np.equal(np.mod(label, 1), 0).all())
        ## labels need to be in range pipe_name[2]
        assert((label >= label_range[0]).all())
        assert((label <= label_range[1]).all())

print("OK")