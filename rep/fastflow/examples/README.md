This directory contains example source codes to compare FastFlow's performance with other systems (TensorFlow and DALI).
The original example codes come from https://keras.io/examples, and we modified the codes to apply FastFlow in training.

## How to Run Examples

### Install dependencies
Check CUDA version using `nvcc --version` or `nvidia-smi` command.
Depending on the installed CUDA version, execute:
```bash
$ pip install -r <requirements file>
```
where `<requirements file>` is either `requirements_cuda10.txt` or `requirements_cuda11.txt`.

### Execute the eval_app_runner.py
```bash
$ python eval_app_runner.py <example.py> <data_path_prefix> <offloading_type> <yaml_path_for_ff_config> <gpu_type(optional)>
```

#### Parameters of eval_app_runner.py
##### 1) app_file_path: the example app python file path 
##### 2) data_path_prefix: the prefix of data path
##### 3) offloading_type:
  - 'tf': TensorFlow (no offloading) 
  - 'tf-dsr-all': TF+Remote Worker by offloading all operations
  - 'tf-dslr-all':TF+Local and Remote Worker by offloading all operations
  - 'dali': DALI
  - 'ff': FastFlow
##### 4) yaml_path: the fastflow config yaml path
##### 5) gpu_type (optional):
  - 'single' (default): single-gpu training
  - 'multi': multi-gpu training

Please see how to configure yaml config in fastflow/README and default_config.yaml in examples/


#### Example Apps
1) ctc_asr_app.py
	- dataset_path: <data_path_prefix>/LJSpeech-1.1
	- Download dataset from https://keithito.com/LJ-Speech-Dataset/  
2) gan_ada_app.py
	- dataset_path: <data_path_prefix>/tensorflow_datasets (it will be automatically downloaded from tfds.load)
3) learnable_resizer_app.py
	- dataset_path: <data_path_prefix>/tensorflow_datasets (it will be automatically downloaded from tfds.load)
4) melgan_app.py
	- dataset_path: <data_path_prefix>/LJSpeech-1.1
	- Download dataset from https://keithito.com/LJ-Speech-Dataset/  
5) randaug_imagenet_app.py
	- dataset_path: <data_path_prefix>/imagenet-tfrecords/data 
	- Download dataset from https://www.tensorflow.org/datasets/catalog/imagenet2012 (you must manually download the dataset and convert into tfrecords)
6) speaker_recog_app.py
	- dataset_path: <data_path_prefix>/16000_pcm_speeches
	- Download dataset from https://www.kaggle.com/kongaevans/speaker-recognition-dataset/download
7) transformer_asr_app.py 
	- dataset_path: <data_path_prefix>/LJSpeech-1.1 
	- Download dataset from https://keithito.com/LJ-Speech-Dataset/  




## How to Write an Example for Evaluation

### Inherit App class

You can find App class in the eval_app_runner.py.
You should inherit the App class and implement the methods. 


```python
class App:
    def create_model(self)
        pass
    def create_dataset(self, parallel)
        pass
    def create_valid_dataset(self, parallel)
        pass
    def create_manual_offloaded_dataset(self, parallel)
        pass
        ....
```

```python
class CtcAsrApp(App):
    def create_dataset(self, num_parallel):
        # Define the trainig dataset
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (list(df_train["file_name"]), list(df_train["normalized_transcription"]))
        )

        train_dataset = (
            train_dataset.map(encode_single_sample, num_parallel_calls=num_parallel)
                .padded_batch(batch_size)
                .prefetch(buffer_size=tf.data.AUTOTUNE)
        )

        return train_dataset
       
    ....
```

==============

