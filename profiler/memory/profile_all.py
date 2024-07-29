import subprocess

models = [
    'alexnet',
    'densenet121',
    'densenet161',
    'densenet169',
    'densenet201',
    'googlenet',
    'mnasnet0_5',
    'mnasnet0_75',
    'mnasnet1_0',
    'mnasnet1_3',
    'mobilenet_v2',
    'mobilenet_v3_large',
    'mobilenet_v3_small',
    'resnet101',
    'resnet152',
    'resnet18',
    'resnet34',
    'resnet50',
    'resnext101_32x8d',
    'resnext50_32x4d',
    'shufflenet_v2_x0_5',
    'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5',
    'shufflenet_v2_x2_0',
    'squeezenet1_0',
    'squeezenet1_1',
    'vgg11',
    'vgg11_bn',
    'vgg13',
    'vgg13_bn',
    'vgg16',
    'vgg16_bn',
    'vgg19',
    'vgg19_bn',
    'wide_resnet101_2',
    'wide_resnet50_2',
]

input_dim = [3, 224, 224]
num_classes = 100
gpu = 0

for model in models:
    for batch_size in range(1,2):
        cmd = [
            'python', 'memory_profiler.py',
            '--arch', model,
            '--input_dim', *map(str, input_dim),
            '--num_classes', str(num_classes),
            '--batch-size', str(batch_size),
            '--gpu', str(gpu)
        ]

        try:
            # Run the command
            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Command failed with error: {e}")
            print(f"Continuing with the next configuration...")
