## Profile CPU-to-GPU data transfer time
This is the profiler for profiling the CPU-to-GPU data transfer time.

### File structure:

- `cpu2gpu_transfer_time_profiler.py`: The profiler script that profiles the cpu-to-gpu data transfer time. The profiling is done for a selected (gpu, DL model, [start_batch_size, end_batch_size]). The raw data is written to raw_data/{gpu_name}/transfer/{model_name}/{batch_size}/data_transfer_time.txt.
  - Note that we also measure model transfer time in the profiler for completeness. However, model transfer time is different from data transfer time in two aspects:
    - Model transfer is done only once during the training process, so it's not that critical to emulate. In the future, we might consider the GPU-to-CPU transfer time to transfer the model back to CPU for checkpoints.
    - Model transfer time is not batch-size dependent, while data transfer time is.
 - Also note that the data transfer time can be actually same for different models, if they have the same input size. However, we still profile the data transfer time for each model to provide a more complete profiling.

- `profile_all.py`: A script that profiles multiple models for a given gpu and batch size range.

- `organize_data.py`: 
    - Organizes the raw data transfer time and write them into csv files. 
    - Generates a linear regression model which fits the data and predicts the data transfer time for a given batch size.
    - Also copies the model transfer time to the target directory. No regression model is generated for model transfer time as it's not batch-size dependent.


### Example usage:

1. Profile a single model for a given gpu and batch size range:
```
python cpu2gpu_transfer_time_profiler.py --gpu 0 --arch resnet50 --start_batch_size 1 --end_batch_size 1
```

2. Profile multiple models for a given gpu and batch size range:
```
python profile_all.py
```

3. Organize the raw data and generate linear regression models for data transfer time vs batch size:
```
python organize_data.py
```
