from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_datasets.core.utils import read_config as read_config_lib

tfds.disable_progress_bar()

import fastflow as ff
import os

from eval_app_runner import App

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

"""
Reference: https://github.com/keras-team/keras-io/blob/master/examples/vision/learnable_resizer.py
We modified the source code. """

"""
## Define hyperparameters
"""

"""
In order to facilitate mini-batch learning, we need to have a fixed shape for the images
inside a given batch. This is why an initial resizing is required. We first resize all
the images to (300 x 300) shape and then learn their optimal representation for the
(150 x 150) resolution.
"""

INP_SIZE = (300, 300)
TARGET_SIZE = (150, 150)
INTERPOLATION = "bilinear"

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 64
# EPOCHS = 5

"""
In this example, we will use the bilinear interpolation but the learnable image resizer
module is not dependent on any specific interpolation method. We can also use others,
such as bicubic.
"""

"""
## Load and prepare the dataset
For this example, we will only use 40% of the total training dataset.
"""


def preprocess_dataset(image, label):
    image = tf.image.resize(image, (INP_SIZE[0], INP_SIZE[1]))
    label = tf.one_hot(label, depth=2)
    return (image, label)


def embed_label_in_image(image, label):
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    label_padded = tf.zeros_like(image)
    delta = tf.SparseTensor([[0, 0, 0]], [tf.cast(label, tf.uint8)], [h, w, 3])
    label_padded = label_padded + tf.sparse.to_dense(delta)
    image_stacked = tf.stack([image, label_padded])

    return image_stacked


@pipeline_def(device_id=0, num_threads=4, batch_size=64)
def preprocess_dataset_dali(device):
    ds = fn.external_source(name='data', dtype=types.UINT8, device=device)
    image = ds[0]
    label = fn.reductions.sum(ds[1], device=device)

    image = fn.resize(image, resize_x=INP_SIZE[0], resize_y=INP_SIZE[1],
                      dtype=types.FLOAT, device=device)
    label = fn.one_hot(label, num_classes=2, device=device)
    return image, label


"""
## Define the learnable resizer utilities
The figure below (courtesy: [Learning to Resize Images for Computer Vision Tasks](https://arxiv.org/abs/2103.09950v1))
presents the structure of the learnable resizing module:
![](https://i.ibb.co/gJYtSs0/image.png)
"""


def conv_block(x, filters, kernel_size, strides, activation=layers.LeakyReLU(0.2)):
    x = layers.Conv2D(filters, kernel_size, strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if activation:
        x = activation(x)
    return x


def res_block(x):
    inputs = x
    x = conv_block(x, 16, 3, 1)
    x = conv_block(x, 16, 3, 1, activation=None)
    return layers.Add()([inputs, x])


def get_learnable_resizer(filters=16, num_res_blocks=1, interpolation=INTERPOLATION):
    inputs = layers.Input(shape=[None, None, 3])

    # First, perform naive resizing.
    naive_resize = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(inputs)

    # First convolution block without batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=7, strides=1, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Second convolution block with batch normalization.
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.BatchNormalization()(x)

    # Intermediate resizing as a bottleneck.
    bottleneck = layers.Resizing(*TARGET_SIZE, interpolation=interpolation)(x)

    # Residual passes.
    for _ in range(num_res_blocks):
        x = res_block(bottleneck)

    # Projection.
    x = layers.Conv2D(
        filters=filters, kernel_size=3, strides=1, padding="same", use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)

    # Skip connection.
    x = layers.Add()([bottleneck, x])

    # Final resized image.
    x = layers.Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    final_resize = layers.Add()([naive_resize, x])

    return tf.keras.Model(inputs, final_resize, name="learnable_resizer")


# """
# ## Visualize the outputs of the learnable resizing module
# Here, we visualize how the resized images would look like after being passed through the
# random weights of the resizer.
# """

# sample_images, _ = next(iter(train_ds))


# plt.figure(figsize=(16, 10))
# for i, image in enumerate(sample_images[:6]):
#     image = image / 255

#     ax = plt.subplot(3, 4, 2 * i + 1)
#     plt.title("Input Image")
#     plt.imshow(image.numpy().squeeze())
#     plt.axis("off")

#     ax = plt.subplot(3, 4, 2 * i + 2)
#     resized_image = learnable_resizer(image[None, ...])
#     plt.title("Resized Image")
#     plt.imshow(resized_image.numpy().squeeze())
#     plt.axis("off")

"""
## Model building utility
"""


class MyModel(ff.FastFlowModel):
    def __init__(self):
        backbone = tf.keras.applications.DenseNet121(
            weights=None,
            include_top=True,
            classes=2,
            input_shape=((TARGET_SIZE[0], TARGET_SIZE[1], 3)),
        )
        backbone.trainable = True

        learnable_resizer = get_learnable_resizer()

        inputs = layers.Input((INP_SIZE[0], INP_SIZE[1], 3))
        x = layers.Rescaling(scale=1.0 / 255)(inputs)
        x = learnable_resizer(x)
        outputs = backbone(x)

        super().__init__(inputs, outputs)

    def __deepcopy__(self):
        return MyModel()


"""
The structure of the learnable image resizer module allows for flexible integrations with
different vision models.
"""

"""
## Compile and train our model with learnable resizer
"""

import os

os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"

EPOCHS = 3

import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))


class LearneableResizerApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)

        read_config = read_config_lib.ReadConfig(assert_cardinality=False)
        read_config.assert_cardinality = False

        self.train_ds, self.validation_ds = tfds.load(
            "cats_vs_dogs",
            data_dir=os.path.join(args.data_prefix, 'tensorflow_datasets'),
            download=False,
            # Reserve 10% for validation
            split=["train[:40%]", "train[40%:50%]"],
            as_supervised=True,
            read_config=read_config
        )

    def create_model(self):
        # Train the model
        model = MyModel()
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            optimizer="sgd",
            metrics=["accuracy"],
        )

        return model

    def create_dataset(self, num_parallel):
        return (self.train_ds.shuffle(BATCH_SIZE * 100)
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTO))

    def create_valid_dataset(self, num_parallel):
        return (self.validation_ds
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTO))

    def create_manual_offloaded_dataset(self, num_parallel):
        return (self.train_ds.shuffle(BATCH_SIZE * 100)
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTO))

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        return (self.validation_ds
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .batch(BATCH_SIZE, drop_remainder=True)
                .prefetch(AUTO))

    def create_all_offload_dataset(self, num_parallel):
        return (self.train_ds.shuffle(BATCH_SIZE * 100)
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .batch(BATCH_SIZE, drop_remainder=True)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .prefetch(AUTO))

    def create_all_offload_valid_dataset(self, num_parallel):
        return (self.validation_ds
                .repeat(10)
                .map(preprocess_dataset, num_parallel_calls=num_parallel)
                .batch(BATCH_SIZE, drop_remainder=True)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .prefetch(AUTO))

    def create_dali_dataset(self, num_parallel):
        data = self.train_ds.map(embed_label_in_image)
        data = data.shuffle(BATCH_SIZE * 100).repeat()

        data = data.apply(tf.data.experimental.copy_to_device('//gpu:0'))
        input_spec_dict = {'data': data}
        pipe = preprocess_dataset_dali('gpu')
        shapes = ((None, 300, 300, 3), (None, 2))
        dtypes = (tf.float32, tf.float32)
        ds = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=pipe,
            input_datasets=input_spec_dict,
            batch_size=BATCH_SIZE,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0)
        return ds

    def create_dali_valid_dataset(self, num_parallel):
        data = self.validation_ds.map(embed_label_in_image)
        data = data.shuffle(BATCH_SIZE * 100).repeat()

        data = data.apply(tf.data.experimental.copy_to_device('//gpu:0'))
        input_spec_dict = {'data': data}
        pipe = preprocess_dataset_dali('gpu')
        shapes = ((None, 300, 300, 3), (None, 2))
        dtypes = (tf.float32, tf.float32)
        ds = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=pipe,
            input_datasets=input_spec_dict,
            batch_size=BATCH_SIZE,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0)
        return ds

    def steps_per_epoch_for_dali(self):
        return 93050 // BATCH_SIZE
