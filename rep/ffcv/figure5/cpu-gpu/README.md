### Measuring Preprocessing Time Dali


1. Install all depedencies

    To install gpufs(detection files), pytorch, and torchvision

    Need doing all step on https://github.com/mengwanguc/gpufs#readme until torchvision and data installed

    And move to naufal branch:
    ```
    git checkout naufal
    ```
    Installing Dali in Ubuntu:
    ```
    pip install --extra-index-url https://pypi.nvidia.com --upgrade nvidia-dali-cuda110
    ```

2. Copying pipeline.py into dali folder

    copy and paste pipeline.py and pytorch.py that had been modified to measuring the time. for emulate the GPU you can use script that has pipeline_emulate.py and pytorch_emulate.py.
    ```
    cp ~/gpufs/ffcv/figure5/cpu-gpu/pipeline.py /home/cc/anaconda3/lib/python3.9/site-packages/nvidia/dali/pipeline.py

    cp ~/gpufs/ffcv/figure5/cpu-gpu/pytorch.py /home/cc/anaconda3/lib/python3.9/site-packages/nvidia/dali/plugin/pytorch.py
    ```

3. Training a model.

    ```
    cd ~/gpufs/ffcv/figure5/cpu-gpu
    python main-dali-original.py -a alexnet --lr 0.01 ~/data/imagenette2 --epochs 1 
    ```

3. Preprocessing Time.

    After the training process was finished it will create txt file namely "preprocess-gpu-time.txt" and "preprocess-gpu-time-emulate.txt" for the emulate that contained time measurement.
    ```
    cd ~/gpufs/ffcv/figure5/cpu-gpu/
    ```

4. Emulate the gpu
    After copy and paste pipeline_emulate.py and pytorch_emulate.py. We have to using CPU to training model. you can using --dali_cpu parameter to used CPU

    ```
    cd ~/gpufs/ffcv/figure5/cpu-gpu
    python main-dali-original.py -a alexnet --lr 0.01 ~/data/imagenette2 --epochs 1 --dali_cpu
    ```



### Measuring Preprocessing Time Dali for each stages 
    
Measuring time for each of preprocessing stages. theres 4 stages in preprocessing using dali. You can comment them to focus measurement on a certain stages. this preprocessing stages can be seen in main-dali-original.py in create_dali_pipeline(). 

NOTE: you can't measuring one by one the stages. so it should be like this, if you want to measured the resize stage we cant comment all code below and leave uncomment on resize. but you have to measuring it one by one from readers, image crop, and then resize. This is because we cant found way to modify the stages script in DALI. 

    ```
    # readers
    images, labels = fn.readers.file(file_root=data_dir,
                                     shard_id=shard_id,
                                     num_shards=num_shards,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader")

    # image crop
    images = fn.decoders.image_random_crop(images,
                                        device=decoder_device, output_type=types.RGB,
                                        device_memory_padding=device_memory_padding,
                                        host_memory_padding=host_memory_padding,
                                        preallocate_width_hint=preallocate_width_hint,
                                        preallocate_height_hint=preallocate_height_hint,
                                        random_aspect_ratio=[0.8, 1.25],
                                        random_area=[0.1, 1.0],
                                        num_attempts=100)

    # resize
    images = fn.resize(images,
                    device=dali_device,
                    resize_x=crop,
                    resize_y=crop,
                    interp_type=types.INTERP_TRIANGULAR)

    # normalize
    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    ```