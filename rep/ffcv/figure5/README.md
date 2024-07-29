### Measuring Preprocessing Time Dali


1. Install all depedencies

    To install gpufs(detection files), pytorch, and torchvision

    Need doing all step on https://github.com/mengwanguc/gpufs#readme until torchvision installed

    And move to naufal branch:
    ```
    git checkout naufal
    ```
    Installing Dali in Ubuntu:
    ```
    pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
    ```

2. Copying pipeline.py into dali folder
    this pipeline.py is script to measuring the time. 
    ```
    cp ~/gpufs/ffcv/figure5/cpu-gpu/pipeline.py /home/cc/anaconda3/lib/python3.9/site-packages/nvidia/dali/pipeline.py
    ```

3. Training a model.

    ```
    cd ~/gpufs/ffcv/figure5/
    python main-dali-original.py -a alexnet --lr 0.01 ~/data/imagenette2 --epochs 1 
    ```

3. Preprocessing Time.
    After the training process was finished it will create 2 txt file namely "preprocess-cpu-time.txt" and "preprocess-gpu-time.txt" that contained time measurement for CPU and GPU. 
    ```
    cd ~/gpufs/ffcv/figure5/
    ```

## 