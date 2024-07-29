## Profile Model GPU Memory Usage

This folder contains the code to profile the GPU memory usage during model training.

### Prerequisite

1. Install the required packages:
```
pip install pandas matplotlib pycuda
```

### Folder Structure

- `memory_profiler.py`: 
  - Profiles the peak and persistent memory usage during the training for a certain model and batch size.
  - With `--plot-time-series` option, it will plot the time series memory usage change during the training.

- `utils`
  - `memory.py`: Helper functions get the GPU memory usage.
  - `plot.py`: Helper functions to plot the change of memory usage during the training.

- `profile_all.py`: Profiles the memory usage for all the models and batch sizes.



### Usage

1. Profile the memory usage for a certain model and batch size:

```
python memory_profiler.py -a resnet50 --batch-size 64
```

2. Profile the memory usage for all the models and batch sizes:

```
python profile_all.py
```

3. Plot the time series memory usage change during the training:

```
python memory_profiler.py -a resnet50 --batch-size 64 --plot-time-series
```