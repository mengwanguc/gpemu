"""
Title: Data-efficient GANs with Adaptive Discriminator Augmentation
Author: [András Béres](https://www.linkedin.com/in/andras-beres-789190210)
Date created: 2021/10/28
Last modified: 2021/10/28
Description: Generating images from limited data using the Caltech Birds dataset.

"""

"""
Reference: https://keras.io/examples/audio/ctc_asr/
We modified the source code.
"""
import tensorflow as tf
import fastflow as ff
import pandas as pd
import os

from eval_app_runner import App
from tensorflow import keras
from tensorflow.keras import layers

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

batch_size = 32
# batch_size = 64


# The set of characters accepted in the transcription.
CHARACTERS = [x for x in "abcdefghijklmnopqrstuvwxyz'?! "]

# An integer scalar Tensor. The window length in samples.
frame_length = 256
# An integer scalar Tensor. The number of samples to step.
frame_step = 160
# An integer scalar Tensor. The size of the FFT to apply.
# If not provided, uses the smallest power of 2 enclosing frame_length.
fft_length = 384


def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss


class MyModel(ff.FastFlowModel):
    def __init__(self, input_dim, output_dim, rnn_layers=5, rnn_units=128,
                 model_name="DeepSpeech-2"):
        """Model similar to DeepSpeech2."""
        # Model's input

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rnn_layers = rnn_layers
        self.rnn_units = rnn_units
        self.model_name = model_name

        input_spectrogram = layers.Input((None, input_dim), name=model_name + "input")
        print(f"[build_model] input_spectrogram: {input_spectrogram}")
        # Expand the dimension to use 2D CNN.
        x = layers.Reshape((-1, input_dim, 1), name=model_name + "expand_dim")(input_spectrogram)
        # Convolution layer 1
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 41],
            strides=[2, 2],
            padding="same",
            use_bias=False,
            name=model_name + "conv_1",
        )(x)
        x = layers.BatchNormalization(name=model_name + "conv_1_bn")(x)
        x = layers.ReLU(name=model_name + "conv_1_relu")(x)
        # Convolution layer 2
        x = layers.Conv2D(
            filters=32,
            kernel_size=[11, 21],
            strides=[1, 2],
            padding="same",
            use_bias=False,
            name=model_name + "conv_2",
        )(x)
        x = layers.BatchNormalization(name=model_name + "conv_2_bn")(x)
        x = layers.ReLU(name=model_name + "conv_2_relu")(x)
        # Reshape the resulted volume to feed the RNNs layers
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        # RNN layers
        for i in range(1, rnn_layers + 1):
            recurrent = layers.GRU(
                units=rnn_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                use_bias=True,
                return_sequences=True,
                reset_after=True,
                name=model_name + f"gru_{i}",
            )
            x = layers.Bidirectional(
                recurrent, name=model_name + f"bidirectional_{i}", merge_mode="concat"
            )(x)
            if i < rnn_layers:
                x = layers.Dropout(rate=0.5)(x)
        # Dense layer
        x = layers.Dense(units=rnn_units * 2, name=model_name + "dense_1")(x)
        x = layers.ReLU(name=model_name + "dense_1_relu")(x)
        x = layers.Dropout(rate=0.5)(x)
        # Classification layer
        output = layers.Dense(units=output_dim + 1, activation="softmax")(x)
        # Model
        super().__init__(input_spectrogram, output, name=model_name)

    def __deepcopy__(self):
        return MyModel(self.input_dim, self.output_dim, self.rnn_layers, self.rnn_units,
                       self.model_name + "-copy")


class CtcAsrApp(App):
    def __init__(self, args, config):
        super().__init__(args, config)

        self.data_path = os.path.join(args.data_prefix, "LJSpeech-1.1")
        self.wavs_path = self.data_path + "/wavs/"
        self.metadata_path = self.data_path + "/metadata.csv"

        # Mapping characters to integers
        self.char_to_num = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=CHARACTERS,
            oov_token="")
        # Mapping integers back to original characters
        self.num_to_char = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), oov_token="", invert=True
        )

        print(
            f"The vocabulary is: {self.char_to_num.get_vocabulary()} "
            f"(size ={self.char_to_num.vocabulary_size()})"
        )

        # Read metadata file and parse it
        metadata_df = pd.read_csv(self.metadata_path, sep="|", header=None, quoting=3)
        metadata_df.columns = ["file_name", "transcription", "normalized_transcription"]
        metadata_df = metadata_df[["file_name", "normalized_transcription"]]
        metadata_df = metadata_df.sample(frac=1).reset_index(drop=True)
        metadata_df.head(3)

        split = int(len(metadata_df) * 0.90)
        self.df_train = metadata_df[:split]
        self.df_val = metadata_df[split:]

        print(f"Size of the training set: {len(self.df_train)}")
        print(f"Size of the validation set: {len(self.df_val)}")

        self.validation_steps = len(self.df_val) // batch_size

        # Get the model
        self.input_dim = fft_length // 2 + 1

        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (list(self.df_train["file_name"]), list(self.df_train["normalized_transcription"]))
        )

        self.validation_dataset = tf.data.Dataset.from_tensor_slices(
            (list(self.df_val["file_name"]), list(self.df_val["normalized_transcription"]))
        )

    def _read_file(self, wav_file, label):
        # Read wav file
        file = tf.io.read_file(self.wavs_path + wav_file + ".wav")
        return file, label

    def _encode_single_sample(self, file, label):
        ###########################################
        ##  Process the Audio
        ##########################################
        # 1. Decode the wav file
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        # 2. Change type to float
        audio = tf.cast(audio, tf.float32)
        # 3. Get the spectrogram
        spectrogram = tf.signal.stft(
            audio, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length
        )
        # 4. We only need the magnitude, which can be derived by applying tf.abs
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.math.pow(spectrogram, 0.5)
        # 5. normalisation
        means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
        stddevs = tf.math.reduce_std(spectrogram, 1, keepdims=True)
        spectrogram = (spectrogram - means) / (stddevs + 1e-10)
        ###########################################
        ##  Process the label
        ##########################################
        # 6. Convert label to Lower case
        label = tf.strings.lower(label)
        # 7. Split the label
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        # 8. Map the characters in label to numbers
        label = self.char_to_num(label)
        # 9. Return a dict as our model is expecting two inputs
        return spectrogram, label

    def _read_decode_audio(self, wav_file):
        file = tf.io.read_file(self.wavs_path + wav_file + ".wav")
        audio, _ = tf.audio.decode_wav(file)
        audio = tf.squeeze(audio, axis=-1)
        audio = tf.cast(audio, tf.float32)
        return audio

    def _process_label(self, label):
        label = tf.strings.lower(label)
        label = tf.strings.unicode_split(label, input_encoding="UTF-8")
        label = self.char_to_num(label)
        return label

    @pipeline_def(device_id=0, num_threads=4, batch_size=64)
    def _preprocess_dali(self, device):
        audio = fn.external_source(name='audio', dtype=types.FLOAT, device=device)
        label = fn.external_source(name='label', dtype=types.INT64, device=device)

        spectrogram = fn.spectrogram(audio, layout='tf', window_length=frame_length,
                                     window_step=frame_step, nfft=fft_length, power=1,
                                     center_windows=False, device=device)
        spectrogram = dali.math.pow(spectrogram, 0.5)
        spectrogram = fn.normalize(spectrogram, axes=(1,), epsilon=1e-10, device=device)

        # Add padding (original TF code uses padded_batch)
        spectrogram = fn.pad(spectrogram, fill_value=0, axes=(), device=device)
        label = fn.pad(label, fill_value=0, axes=(), device=device)

        return spectrogram, label

    def create_model(self):
        model = MyModel(
            input_dim=self.input_dim,
            output_dim=self.char_to_num.vocabulary_size(),
            rnn_units=512,
            model_name="DeepSpeech-2")

        # Optimizer
        opt = keras.optimizers.Adam(learning_rate=1e-4)
        # Compile the model and return
        model.compile(optimizer=opt, loss=CTCLoss)

        return model

    def create_dataset(self, num_parallel):
        # Define the trainig dataset
        return (self.train_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel, name='prep_begin')
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_valid_dataset(self, num_parallel):
        # Define the validation dataset
        return (self.validation_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel, name='prep_begin')
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_manual_offloaded_dataset(self, num_parallel):
        return (self.train_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel)
                .apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                               service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                               partial_offload_enabled=self.config.partial_offload_enabled,
                                                               ratio_local=self.config.ratio_local))
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        # define the validation dataset
        return (self.validation_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel)
                .apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                               service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                               partial_offload_enabled=self.config.partial_offload_enabled,
                                                               ratio_local=self.config.ratio_local))
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_all_offload_dataset(self, num_parallel):
        return (self.train_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel)
                .padded_batch(batch_size)
                .apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                               service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                               partial_offload_enabled=self.config.partial_offload_enabled,
                                                               ratio_local=self.config.ratio_local))
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_all_offload_valid_dataset(self, num_parallel):
        # define the validation dataset
        return (self.validation_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel)
                .padded_batch(batch_size)
                .apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                               service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                               partial_offload_enabled=self.config.partial_offload_enabled,
                                                               ratio_local=self.config.ratio_local))
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def create_dali_dataset(self, num_parallel):
        audio_ds = tf.data.Dataset.from_tensor_slices(list(self.df_train["file_name"]))
        audio = audio_ds.map(self._read_decode_audio, num_parallel_calls=num_parallel).repeat()

        label_ds = tf.data.Dataset.from_tensor_slices(
            list(self.df_train["normalized_transcription"]))
        label = label_ds.map(self._process_label, num_parallel_calls=num_parallel).repeat()

        audio = audio.apply(tf.data.experimental.copy_to_device('/gpu:0'))
        label = label.apply(tf.data.experimental.copy_to_device('/gpu:0'))

        input_spec_dict = {'audio': audio, 'label': label}
        pipe = self._preprocess_dali('gpu')

        shapes = ((None, None, 193), (None, None))
        dtypes = (tf.float32, tf.int64)

        return (dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=pipe,
            input_datasets=input_spec_dict,
            batch_size=batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0))

    def create_dali_valid_dataset(self, num_parallel):
        return (self.validation_dataset
                .map(self._read_file, num_parallel_calls=num_parallel)
                .map(self._encode_single_sample, num_parallel_calls=num_parallel)
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def steps_per_epoch_for_dali(self):
        return len(self.df_train) // batch_size
