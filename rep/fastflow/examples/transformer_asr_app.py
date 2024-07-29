import tensorflow as tf
import os
import fastflow as ff
from glob import glob

from eval_app_runner import App
from tensorflow import keras
from tensorflow.keras import layers

from nvidia.dali import pipeline_def
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.plugin.tf as dali_tf

"""
Reference: https://keras.io/examples/audio/transformer_asr/
We modified the source code.
"""


class TokenEmbedding(layers.Layer):
    def __init__(self, num_vocab=1000, maxlen=100, num_hid=64):
        super().__init__()
        self.emb = tf.keras.layers.Embedding(num_vocab, num_hid)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        x = self.emb(x)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class SpeechFeatureEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)


"""## Transformer Encoder Layer"""


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


"""## Transformer Decoder Layer"""


class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, num_heads, feed_forward_dim, dropout_rate=0.1):
        super().__init__()
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.self_att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.enc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.self_dropout = layers.Dropout(0.5)
        self.enc_dropout = layers.Dropout(0.1)
        self.ffn_dropout = layers.Dropout(0.1)
        self.ffn = keras.Sequential(
            [
                layers.Dense(feed_forward_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

    def causal_attention_mask(self, batch_size, n_dest, n_src, dtype):
        """Masks the upper half of the dot product matrix in self attention.
        This prevents flow of information from future tokens to current token.
        1's in the lower triangle, counting from the lower right corner.
        """
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        m = i >= j - n_src + n_dest
        mask = tf.cast(m, dtype)
        mask = tf.reshape(mask, [1, n_dest, n_src])
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
        )
        return tf.tile(mask, mult)

    def call(self, enc_out, target):
        input_shape = tf.shape(target)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)
        target_att = self.self_att(target, target, attention_mask=causal_mask)
        target_norm = self.layernorm1(target + self.self_dropout(target_att))
        enc_out = self.enc_att(target_norm, enc_out)
        enc_out_norm = self.layernorm2(self.enc_dropout(enc_out) + target_norm)
        ffn_out = self.ffn(enc_out_norm)
        ffn_out_norm = self.layernorm3(enc_out_norm + self.ffn_dropout(ffn_out))
        return ffn_out_norm


"""## Complete the Transformer model
Our model takes audio spectrograms as inputs and predicts a sequence of characters.
During training, we give the decoder the target character sequence shifted to the left
as input. During inference, the decoder uses its own past predictions to predict the
next token.
"""


class Transformer(ff.FastFlowModel):
    def __init__(
            self,
            num_hid=64,
            num_head=2,
            num_feed_forward=128,
            source_maxlen=100,
            target_maxlen=100,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=10,
    ):
        super().__init__()

        self.num_hid = num_hid
        self.num_head = num_head
        self.num_feed_forward = num_feed_forward
        self.source_maxlen = source_maxlen
        self.target_maxlen = target_maxlen

        self.loss_metric = keras.metrics.Mean(name="loss")
        self.num_layers_enc = num_layers_enc
        self.num_layers_dec = num_layers_dec
        self.target_maxlen = target_maxlen
        self.num_classes = num_classes

        self.enc_input = SpeechFeatureEmbedding(num_hid=num_hid, maxlen=source_maxlen)
        self.dec_input = TokenEmbedding(
            num_vocab=num_classes, maxlen=target_maxlen, num_hid=num_hid
        )

        self.encoder = keras.Sequential(
            [self.enc_input]
            + [
                TransformerEncoder(num_hid, num_head, num_feed_forward)
                for _ in range(num_layers_enc)
            ]
        )

        for i in range(num_layers_dec):
            setattr(
                self,
                f"dec_layer_{i}",
                TransformerDecoder(num_hid, num_head, num_feed_forward),
            )

        self.classifier = layers.Dense(num_classes)

    def __deepcopy__(self):
        return Transformer(self.num_hid,
                           self.num_head,
                           self.num_feed_forward,
                           self.source_maxlen,
                           self.target_maxlen,
                           self.num_layers_enc,
                           self.num_layers_dec,
                           self.num_classes)

    def decode(self, enc_out, target):
        y = self.dec_input(target)
        for i in range(self.num_layers_dec):
            y = getattr(self, f"dec_layer_{i}")(enc_out, y)
        return y

    def call(self, inputs):
        source = inputs[0]
        target = inputs[1]
        x = self.encoder(source)
        y = self.decode(x, target)
        return self.classifier(y)

    @property
    def metrics(self):
        return [self.loss_metric]

    def train_step(self, batch):
        """Processes one batch inside model.fit()."""
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        with tf.GradientTape() as tape:
            preds = self([source, dec_input])
            one_hot = tf.one_hot(dec_target, depth=self.num_classes)
            mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
            loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def test_step(self, batch):
        source = batch["source"]
        target = batch["target"]
        dec_input = target[:, :-1]
        dec_target = target[:, 1:]
        preds = self([source, dec_input])
        one_hot = tf.one_hot(dec_target, depth=self.num_classes)
        mask = tf.math.logical_not(tf.math.equal(dec_target, 0))
        loss = self.compiled_loss(one_hot, preds, sample_weight=mask)
        self.loss_metric.update_state(loss)
        return {"loss": self.loss_metric.result()}

    def generate(self, source, target_start_token_idx):
        """Performs inference over one batch of inputs using greedy decoding."""
        bs = tf.shape(source)[0]
        enc = self.encoder(source)
        dec_input = tf.ones((bs, 1), dtype=tf.int32) * target_start_token_idx
        dec_logits = []
        for i in range(self.target_maxlen - 1):
            dec_out = self.decode(enc, dec_input)
            logits = self.classifier(dec_out)
            logits = tf.argmax(logits, axis=-1, output_type=tf.int32)
            last_logit = tf.expand_dims(logits[:, -1], axis=-1)
            dec_logits.append(last_logit)
            dec_input = tf.concat([dec_input, last_logit], axis=-1)
        return dec_input


"""## Download the dataset
Note: This requires ~3.6 GB of disk space and
takes ~5 minutes for the extraction of files.
"""


class VectorizeChar:
    def __init__(self, max_len=50):
        self.vocab = (
                ["-", "#", "<", ">"]
                + [chr(i + 96) for i in range(1, 27)]
                + [" ", ".", ",", "?"]
        )
        self.max_len = max_len
        self.char_to_idx = {}
        for i, ch in enumerate(self.vocab):
            self.char_to_idx[ch] = i

    def __call__(self, text):
        text = text.lower()
        text = text[: self.max_len - 2]
        text = "<" + text + ">"
        pad_len = self.max_len - len(text)
        return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

    def get_vocabulary(self):
        return self.vocab


def read_file(path):
    audio = tf.io.read_file(path)
    return audio


def preprocess(audio):
    # spectrogram using stft
    audio, _ = tf.audio.decode_wav(audio, 1)
    audio = tf.squeeze(audio, axis=-1)
    stfts = tf.signal.stft(audio, frame_length=200, frame_step=80, fft_length=256)
    x = tf.math.pow(tf.abs(stfts), 0.5)
    # normalisation
    means = tf.math.reduce_mean(x, 1, keepdims=True)
    stddevs = tf.math.reduce_std(x, 1, keepdims=True)
    x = (x - means) / stddevs
    audio_len = tf.shape(x)[0]
    # padding to 10 seconds
    pad_len = 2754
    paddings = tf.constant([[0, pad_len], [0, 0]])
    x = tf.pad(x, paddings, "CONSTANT")[:pad_len, :]
    return x


def read_decode_audio(path):
    """ Read and decode w/ TF (CPU). This dataset is explicitly copied to GPU 
    and continues to be preprocessed w/ DALI in `preprocess_dali` (GPU).
    """
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1)
    return audio


@pipeline_def(device_id=0, num_threads=4, batch_size=64)
def preprocess_dali(device):
    audio = fn.external_source(name='audio_ds', dtype=types.FLOAT, device=device)
    label = fn.external_source(name='text_ds', dtype=types.INT32, device=device)

    audio = fn.squeeze(audio, axes=-1, device=device)
    x = fn.spectrogram(audio, layout='tf', window_length=200, window_step=80,
                       nfft=256, power=1, center_windows=False, device=device)
    x = dali.math.pow(x, 0.5)
    x = fn.normalize(x, axes=(1,))

    pad_len = 2754
    x = fn.pad(x, fill_value=0, axes=(0, 1), align=(pad_len, 129), device=device)[:pad_len, :]

    return x, label


"""## Callbacks to display predictions"""


class DisplayOutputs(keras.callbacks.Callback):
    def __init__(
            self, batch, idx_to_token, target_start_token_idx=27, target_end_token_idx=28
    ):
        """Displays a batch of outputs after every epoch
        Args:
            batch: A test batch containing the keys "source" and "target"
            idx_to_token: A List containing the vocabulary tokens corresponding to their indices
            target_start_token_idx: A start token index in the target vocabulary
            target_end_token_idx: An end token index in the target vocabulary
        """
        self.batch = batch
        self.target_start_token_idx = target_start_token_idx
        self.target_end_token_idx = target_end_token_idx
        self.idx_to_char = idx_to_token

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 != 0:
            return
        source = self.batch["source"]
        target = self.batch["target"].numpy()
        bs = tf.shape(source)[0]
        preds = self.model.generate(source, self.target_start_token_idx)
        preds = preds.numpy()
        for i in range(bs):
            target_text = "".join([self.idx_to_char[_] for _ in target[i, :]])
            prediction = ""
            for idx in preds[i, :]:
                prediction += self.idx_to_char[idx]
                if idx == self.target_end_token_idx:
                    break
            print(f"target:     {target_text.replace('-', '')}")
            print(f"prediction: {prediction}\n")


"""## Learning rate schedule"""


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
            self,
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=85,
            steps_per_epoch=203,
    ):
        super().__init__()
        self.init_lr = init_lr
        self.lr_after_warmup = lr_after_warmup
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.decay_epochs = decay_epochs
        self.steps_per_epoch = steps_per_epoch

    def calculate_lr(self, epoch):
        """ linear warm up - linear decay """
        a = ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1))

        epoch = tf.cast(epoch, dtype=tf.float32)
        decay_epochs = tf.cast(self.decay_epochs, dtype=tf.float32)
        warmup_epochs = tf.cast(self.warmup_epochs, dtype=tf.float32)

        b = a * epoch
        warmup_lr = (self.init_lr + b)
        #         warmup_lr = (
        #             self.init_lr
        #             + ((self.lr_after_warmup - self.init_lr) / (self.warmup_epochs - 1)) * epoch
        #         )
        decay_lr = tf.math.maximum(
            self.final_lr,
            self.lr_after_warmup
            - (epoch - warmup_epochs)
            * (self.lr_after_warmup - self.final_lr)
            / decay_epochs,
        )
        return tf.math.minimum(warmup_lr, decay_lr)

    def __call__(self, step):
        epoch = step // self.steps_per_epoch
        return self.calculate_lr(epoch)


class TransformerAsrApp(App):
    def __init__(self, args, config):
        def _create_ds(d):
            # Create audio_ds
            flist = [_["audio"] for _ in d]
            audio_ds = tf.data.Dataset.from_tensor_slices(flist)

            # Create text_ds
            texts = [_["text"] for _ in d]
            text_ds = [self.vectorizer(t) for t in texts]
            text_ds = tf.data.Dataset.from_tensor_slices(text_ds)

            return (audio_ds, text_ds)

        super().__init__(args, config)

        saveto = os.path.join(args.data_prefix, "LJSpeech-1.1")
        wavs = glob("{}/**/*.wav".format(saveto), recursive=True)

        id_to_text = {}
        with open(os.path.join(saveto, "metadata.csv"), encoding="utf-8") as f:
            for line in f:
                id = line.strip().split("|")[0]
                text = line.strip().split("|")[2]
                id_to_text[id] = text

        def get_data(wavs, id_to_text, maxlen=50):
            """ returns mapping of audio paths and transcription texts """
            data = []
            for w in wavs:
                id = w.split("/")[-1].split(".")[0]
                if len(id_to_text[id]) < maxlen:
                    data.append({"audio": w, "text": id_to_text[id]})
            return data

        """## Preprocess the dataset"""

        self.max_target_len = 200  # all transcripts in out data are < 200 characters
        self.data = get_data(wavs, id_to_text, self.max_target_len)
        self.vectorizer = VectorizeChar(self.max_target_len)
        print("vocab size", len(self.vectorizer.get_vocabulary()))

        split = int(len(self.data) * 0.99)
        self.train_audio_ds, self.train_text_ds = _create_ds(self.data[:split])
        self.test_audio_ds, self.test_text_ds = _create_ds(self.data[split:])

        self.train_batch_size = 64
        self.valid_batch_size = 4

        self.validation_steps = len(self.test_audio_ds) // self.valid_batch_size

        """## Create & train the end-to-end model"""

        # batch = next(iter(val_ds))
        os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

        # The vocabulary to convert predicted indices into characters
        idx_to_char = self.vectorizer.get_vocabulary()
        # display_cb = DisplayOutputs(
        #    batch, idx_to_char, target_start_token_idx=2, target_end_token_idx=3
        # )  # set the arguments as per vocabulary index for '<' and '>'

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=0.1,
        )

        learning_rate = CustomSchedule(
            init_lr=0.00001,
            lr_after_warmup=0.001,
            final_lr=0.00001,
            warmup_epochs=15,
            decay_epochs=85,
            steps_per_epoch=203,
        )

        import datetime

        # log_dir = "/tensorboard/" + "tasr-prd"
        # tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,
        #                                                     profile_batch=[204, 203*2 - 1])
        self.optimizer = keras.optimizers.Adam(learning_rate)

        """In practice, you should train for around 100 epochs or more.
        Some of the predicted text at or around epoch 35 may look as follows:
        ```
        target:     <as they sat in the car, frazier asked oswald where his lunch was>
        prediction: <as they sat in the car frazier his lunch ware mis lunch was>
        target:     <under the entry for may one, nineteen sixty,>
        prediction: <under the introus for may monee, nin the sixty,>
        ```
        """

    def create_model(self):
        model = Transformer(
            num_hid=200,
            num_head=2,
            num_feed_forward=400,
            target_maxlen=self.max_target_len,
            num_layers_enc=4,
            num_layers_dec=1,
            num_classes=34,
        )
        model.compile(optimizer=self.optimizer, loss=self.loss_fn)
        return model

    def _create_dataset(self, dataset, num_parallel, batch_size):
        # Zip and map
        ds = dataset.map(lambda x, y: (read_file(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=num_parallel,
                    name='prep_begin')
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_dataset(self, num_parallel):
        # Zip and map
        return self._create_dataset(
            tf.data.Dataset.zip((self.train_audio_ds, self.train_text_ds)),
            num_parallel, self.train_batch_size)

    def create_valid_dataset(self, num_parallel):
        return self._create_dataset(
            tf.data.Dataset.zip((self.test_audio_ds, self.test_text_ds)),
            num_parallel, self.valid_batch_size)

    def create_manual_offloaded_dataset(self, num_parallel):
        # Zip and map
        ds = tf.data.Dataset.zip((self.train_audio_ds, self.train_text_ds))
        ds = ds.map(lambda x, y: (read_file(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        ds = ds.apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                              service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                              partial_offload_enabled=self.config.partial_offload_enabled,
                                                              ratio_local=self.config.ratio_local))
        ds = ds.batch(self.train_batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_manual_offloaded_valid_dataset(self, num_parallel):
        ds = tf.data.Dataset.zip((self.test_audio_ds, self.test_text_ds))
        ds = ds.map(lambda x, y: (read_file(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        ds = ds.apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                              service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                              partial_offload_enabled=self.config.partial_offload_enabled,
                                                              ratio_local=self.config.ratio_local))
        ds = ds.batch(self.valid_batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_all_offload_dataset(self, num_parallel):
        # Zip and map
        ds = tf.data.Dataset.zip((self.train_audio_ds, self.train_text_ds))
        ds = ds.map(lambda x, y: (read_file(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        ds = ds.batch(self.train_batch_size)
        ds = ds.apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                              service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                              partial_offload_enabled=self.config.partial_offload_enabled,
                                                              ratio_local=self.config.ratio_local))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_all_offload_valid_dataset(self, num_parallel):
        ds = tf.data.Dataset.zip((self.test_audio_ds, self.test_text_ds))
        ds = ds.map(lambda x, y: (read_file(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: (preprocess(x), y), num_parallel_calls=num_parallel)
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        ds = ds.batch(self.valid_batch_size)
        ds = ds.apply(tf.data.experimental.service.distribute(processing_mode="distributed_epoch",
                                                              service="grpc://" + self.config.dispatcher_addr + ":5000",
                                                              partial_offload_enabled=self.config.partial_offload_enabled,
                                                              ratio_local=self.config.ratio_local))
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def create_dali_dataset(self, num_parallel):
        audio_ds = self.train_audio_ds.map(read_decode_audio, num_parallel_calls=num_parallel)
        audio_ds = audio_ds.repeat()
        text_ds = self.train_text_ds.repeat()

        # Copy to GPU for preprocessing the remaining pipeline w/ DALI
        audio_ds = audio_ds.apply(tf.data.experimental.copy_to_device('/gpu:0'))
        text_ds = text_ds.apply(tf.data.experimental.copy_to_device('/gpu:0'))

        # Define dictionary to ingest the TF dataset to DALI pipeline
        input_spec_dict = {
            'audio_ds': audio_ds,
            'text_ds': text_ds
        }

        # Create DALI pipeline with the TF dataset ingested
        pipe = preprocess_dali('gpu')

        shapes = ((None, None, 129), (None, 200))
        dtypes = (tf.float32, tf.int32)

        ds = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=pipe,
            input_datasets=input_spec_dict,
            batch_size=self.train_batch_size,
            output_shapes=shapes,
            output_dtypes=dtypes,
            device_id=0)
        ds = ds.map(lambda x, y: {"source": x, "target": y}, num_parallel_calls=num_parallel)
        return ds

    def create_dali_valid_dataset(self, num_parallel):
        return self._create_dataset(
            tf.data.Dataset.zip((self.test_audio_ds, self.test_text_ds)),
            num_parallel, self.valid_batch_size)

    def steps_per_epoch_for_dali(self):
        return len(self.train_audio_ds) // self.train_batch_size
