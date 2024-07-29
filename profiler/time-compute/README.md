## Profile compute data
This is the profiler for profiling the compute time.

Our current profiling's granularity is model level and profiles forward and backward time respectively.

### File structure:

- `gpu_compute_time_profiler.py`: The profiler script that profiles the compute forward and backward time. The profiling is done for a selected (gpu, DL model, [start_batch_size, end_batch_size]). The raw data is written to raw_data/{gpu_name}/{time_type}/{model_name}/{batch_size}/time.txt.

- `profile_all.py`: A script that profiles multiple models for a given gpu and batch size range.

- `organize_data.py`: 
    - Organizes the raw data into csv files. 
    - And generates a linear regression model which fits the data and predicts the compute time for a given batch size.

### Example usage:

1. Profile a single model for a given gpu and batch size range:
```
python gpu_compute_time_profiler.py --gpu 0 --arch resnet50 --start_batch_size 1 --end_batch_size 1
```

2. Profile multiple models for a given gpu and batch size range:
```
python profile_all.py
```

3. Organize the raw data into csv files and generate linear regression models for each (gpu, model, time_type):
```
python organize_data.py
```
