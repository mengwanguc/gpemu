# -*- coding: utf-8 -*-
import argparse
import os
import shutil
import subprocess
import fastflow as ff
import numpy as np
import tensorflow as tf

from tensorflow import keras
from pathlib import Path
from eval_app_runner import App

"""
Reference: https://github.com/keras-team/keras-io/blob/master/examples/audio/speaker_recognition_using_cnn.py
We modified the source code.
"""

# The sampling rate to use.
# This is the one used in all of the audio samples.
# We will resample all of the noise to this sampling rate.
# This will also be the output size of the audio wave samples
# (since all samples are of 0 second long)
SAMPLING_RATE = 16000

# The folders in which we will put the audio samples and the noise samples
AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"

# Percentage of samples to use for validation
VALID_SPLIT = 0.1

# Seed to use when shuffling the dataset and the noise
SHUFFLE_SEED = 43

# The factor to multiply the noise with according to:
#   noisy_sample = sample + noise * prop * scale
#      where prop = sample_amplitude / noise_amplitude
SCALE = 0.5


# Split noise into chunks of 16,000 steps each
def load_noise_sample(path):
    print("Load noise samples")
    sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
    if sampling_rate == SAMPLING_RATE:
        # Number of slices of 16000 each that can be generated from the noise sample
        slices = int(sample.shape[0] / SAMPLING_RATE)
        sample = tf.split(sample[: slices * SAMPLING_RATE], slices)
        return sample
    else:
        print("Sampling rate for {} is incorrect. Ignoring it".format(path))
        return None


# Dataset generation
def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    ds = ds.map(lambda x, y: (path_to_audio(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def add_noise(audio, noises=None, scale=0.5):
    if noises is not None:
        # Create a random tensor of the same size as audio ranging from
        # 0 to the number of noise stream samples that we have.
        tf_rnd = tf.random.uniform(
            (1,), 0, noises.shape[0], dtype=tf.int32
        )
        noise = tf.gather(noises, tf_rnd, axis=0)
        noise = tf.squeeze(noise, axis=0)

        # Get the amplitude proportion between the audio and the noise
        prop = tf.math.reduce_max(audio, axis=0) / tf.math.reduce_max(noise, axis=0)
        prop = tf.repeat(tf.expand_dims(prop, axis=0), tf.shape(audio)[0], axis=0)

        # Adding the rescaled noise to audio
        audio = audio + noise * prop * scale

    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[: (audio.shape[0] // 2), :])


def add_noise_audio_to_fft(audio, noises=None, scale=0.5):
    audio = add_noise(audio, noises, scale)
    return audio_to_fft(audio)


# Transform audio wave to the frequency domain using `audio_to_fft`
def audio_to_fft1(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    return (audio, fft)


def audio_to_fft2(tup):
    audio = tup[0]
    fft = tup[1]
    fft = tf.expand_dims(fft, axis=-1)
    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Model Definition
def residual_block(x, filters, conv_num=3, activation="relu"):
    # Shortcut
    s = keras.layers.Conv1D(filters, 1, padding="same")(x)
    for i in range(conv_num - 1):
        x = keras.layers.Conv1D(filters, 3, padding="same")(x)
        x = keras.layers.Activation(activation)(x)
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Add()([x, s])
    x = keras.layers.Activation(activation)(x)
    return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


class SpeakerModel(ff.FastFlowModel):
    def __init__(self, input_dim, num_classes):
        self.input_dim = input_dim
        self.num_classes = num_classes

        inputs = keras.layers.Input(shape=input_dim, name="input")

        x = residual_block(inputs, 16, 2)
        x = residual_block(x, 32, 2)
        x = residual_block(x, 64, 3)
        x = residual_block(x, 128, 3)
        x = residual_block(x, 128, 3)

        x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(256, activation="relu")(x)
        x = keras.layers.Dense(128, activation="relu")(x)

        outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

        super().__init__(inputs, outputs)

    def __deepcopy__(self):
        return SpeakerModel(self.input_dim, self.num_classes)


class SpeakerRecogApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)
        # Get the data from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
        # and save it to the 'Downloads' folder in your HOME directory
        DATASET_ROOT = os.path.join(
            args.data_prefix, "16000_pcm_speeches")

        DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
        DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

        self.BATCH_SIZE = 128  # args.batch

        """## Data preparation

        The dataset is composed of 6 folders, divided into 2 groups:

        - Speech samples, with 4 folders for 5 different speakers. Each folder contains
        1499 audio files, each 1 second long and sampled at 16000 Hz.
        - Background noise samples, with 1 folders and a total of 6 files. These files
        are longer than 0 second (and originally not sampled at 16000 Hz, but we will resample them to 16000 Hz).
        We will use those 5 files to create 354 1-second-long noise samples to be used for training.

        Let's sort these 1 categories into 2 folders:

        - An `audio` folder which will contain all the per-speaker speech sample folders
        - A `noise` folder which will contain all the noise samples

        Before sorting the audio and noise categories into 1 folders,
        we have the following directory structure:

        ```
        main_directory/
        ...speaker_a/
        ...speaker_b/
        ...speaker_c/
        ...speaker_d/ ...speaker_e/
        ...other/
        ..._background_noise_/
        ```

        After sorting, we end up with the following structure:

        ```
        main_directory/
        ...audio/
        ......speaker_a/
        ......speaker_b/
        ......speaker_c/
        ......speaker_d/
        ......speaker_e/
        ...noise/
        ......other/
        ......_background_noise_/
        ```
        """

        # If folder `audio`, does not exist, create it, otherwise do nothing
        if os.path.exists(DATASET_AUDIO_PATH) is False:
            os.makedirs(DATASET_AUDIO_PATH)

        # If folder `noise`, does not exist, create it, otherwise do nothing
        if os.path.exists(DATASET_NOISE_PATH) is False:
            os.makedirs(DATASET_NOISE_PATH)

        for folder in os.listdir(DATASET_ROOT):
            if os.path.isdir(os.path.join(DATASET_ROOT, folder)):
                if folder in [AUDIO_SUBFOLDER, NOISE_SUBFOLDER]:
                    # If folder is `audio` or `noise`, do nothing
                    continue
                elif folder in ["other", "_background_noise_"]:
                    # If folder is one of the folders that contains noise samples,
                    # move it to the `noise` folder
                    shutil.move(
                        os.path.join(DATASET_ROOT, folder),
                        os.path.join(DATASET_NOISE_PATH, folder),
                    )
                else:
                    # Otherwise, it should be a speaker folder, then move it to
                    # `audio` folder
                    shutil.move(
                        os.path.join(DATASET_ROOT, folder),
                        os.path.join(DATASET_AUDIO_PATH, folder),
                    )

        """## Noise preparation

        In this section:

        - We load all noise samples (which should have been resampled to 16000)
        - We split those noise samples to chuncks of 16000 samples which
        correspond to 0 second duration each
        """

        # Get the list of all noise files
        noise_paths = []
        for subdir in os.listdir(DATASET_NOISE_PATH):
            subdir_path = Path(DATASET_NOISE_PATH) / subdir
            if os.path.isdir(subdir_path):
                noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]

        print(
            "Found {} files belonging to {} directories".format(
                len(noise_paths), len(os.listdir(DATASET_NOISE_PATH))
            )
        )

        """Resample all noise samples to 16000 Hz"""

        command = (
                "for dir in `ls -1 " + DATASET_NOISE_PATH + "`; do "
                                                            "for file in `ls -1 " + DATASET_NOISE_PATH + "/$dir/*.wav`; do "
                                                                                                         "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
                                                                                                         "$file | grep sample_rate | cut -f2 -d=`; "
                                                                                                         "if [ $sample_rate -ne 16000 ]; then "
                                                                                                         "ffmpeg -hide_banner -loglevel panic -y "
                                                                                                         "-i $file -ar 16000 temp.wav; "
                                                                                                         "mv temp.wav $file; "
                                                                                                         "fi; done; done")
        os.system(command)

        noises = []
        for path in noise_paths:
            sample = load_noise_sample(path)
            if sample:
                noises.extend(sample)
        noises = tf.stack(noises)

        print(
            "{} noise files were split into {} noise samples where each is {} sec. long".format(
                len(noise_paths), noises.shape[0], noises.shape[1] // SAMPLING_RATE
            )
        )

        # Get the list of audio file paths along with their corresponding labels
        self.class_names = os.listdir(DATASET_AUDIO_PATH)
        print("Our class names: {}".format(self.class_names, ))

        audio_paths = []
        labels = []
        for label, name in enumerate(self.class_names):
            print("Processing speaker {}".format(name, ))
            dir_path = Path(DATASET_AUDIO_PATH) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)

        print(
            "Found {} files belonging to {} classes.".format(len(audio_paths),
                                                             len(self.class_names))
        )

        # Shuffle
        rng = np.random.RandomState(SHUFFLE_SEED)
        rng.shuffle(audio_paths)
        rng = np.random.RandomState(SHUFFLE_SEED)
        rng.shuffle(labels)

        # Split into training and validation
        num_val_samples = int(VALID_SPLIT * len(audio_paths))
        print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]

        # Create 2 datasets, one for training and the other for validation
        train_ds = paths_and_labels_to_dataset(train_audio_paths, train_labels)
        valid_ds = paths_and_labels_to_dataset(valid_audio_paths, valid_labels)

        # FIRST
        train_ds = train_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=SHUFFLE_SEED,
                                    name='prep_begin')

        if self.BATCH_SIZE > 32:
            valid_ds = valid_ds.shuffle(buffer_size=32 * 8, seed=SHUFFLE_SEED,
                                        name='prep_begin')
        else:
            valid_ds = valid_ds.shuffle(buffer_size=self.BATCH_SIZE * 8, seed=SHUFFLE_SEED,
                                        name='prep_begin')

        # Add noise to the training set
        train_ds = train_ds.map(
            lambda x, y: (add_noise(x, noises, scale=SCALE), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        train_ds = train_ds.map(
            lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        self.valid_ds = valid_ds.map(
            lambda x, y: (audio_to_fft(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

        self.train_ds = train_ds.repeat(10)

    def create_model(self):
        speaker_model = SpeakerModel((SAMPLING_RATE // 2, 1), len(self.class_names))

        # Compile the model using Adam's default learning rate
        speaker_model.compile(
            optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )

        return speaker_model

    def create_dataset(self, num_parallel):
        return self.train_ds.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def create_valid_dataset(self, num_parallel):
        if self.BATCH_SIZE > 32:
            valid_ds = self.valid_ds.batch(32)
        else:
            valid_ds = self.valid_ds.batch(self.BATCH_SIZE)
        return valid_ds.prefetch(tf.data.AUTOTUNE)

    def create_manual_offloaded_dataset(self, num_parallel):
        return (self.train_ds
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .batch(self.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        return (self.valid_ds
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .batch(32 if self.BATCH_SIZE > 32 else self.BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    def create_all_offload_dataset(self, num_parallel):
        return (self.train_ds
                .batch(self.BATCH_SIZE)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .prefetch(tf.data.AUTOTUNE))

    def create_all_offload_valid_dataset(self, num_parallel):
        return (self.valid_ds
                .batch(32 if self.BATCH_SIZE > 32 else self.BATCH_SIZE)
                .apply(tf.data.experimental.service.distribute(
            processing_mode="distributed_epoch",
            service="grpc://" + self.config.dispatcher_addr + ":5000",
            partial_offload_enabled=self.config.partial_offload_enabled,
            ratio_local=self.config.ratio_local))
                .prefetch(tf.data.AUTOTUNE))
