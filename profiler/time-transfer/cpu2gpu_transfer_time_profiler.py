import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from time_measure import GPUTimeMeasurement

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
from torch.optim.lr_scheduler import *
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from boto.provider import get_default
import os
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor, HubertForSequenceClassification, Data2VecAudioForSequenceClassification
import gc



class UserDefinedModel(nn.Module):

    def __init__(self, input_shape):
        super(UserDefinedModel, self).__init__()
        self.input_shape = input_shape
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self._get_flattened_size(), 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, 10)

    def _get_flattened_size(self):
        dummy_input = torch.zeros(1, *self.input_shape)
        return self.flatten(dummy_input).numel()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def get_optimizer(self):
        return optim.SGD(model.parameters(), lr=0.01)
    
    def get_criterion(self):
        return nn.MSELoss()


# Function to detect the GPU type
def get_default_gpu_name():
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_name = gpu_name.replace(" ", "_")
        return gpu_name
    else:
        return 'CPU'


# Function to save average time to a file
def save_average_time(gpu_name, time_type, model_name, batch_size, average_time, filename='time.txt'):
    directory = f"raw_data/{gpu_name}/{time_type}/{model_name}/{batch_size}/"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, "w") as file:
        file.write(f"{average_time}\n")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append("user_defined")
model_names.append("facebook/wav2vec2-base-960h")
model_names.append("facebook/hubert-base-ls960")
model_names.append("facebook/data2vec-audio-base-960h")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='GPEmu Profiler')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + 
                    ' | '.join(model_names) + 
                    ' (default: resnet18)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-sbs', '--single-batch-size', action='store_true',
                    help='use single batch size')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--gpu-type', default='unknown', type=str,
                    help='gpu type that you are using, e.g. p100/v100/rtx6000/...')
parser.add_argument('--profile_batches', default=10, type=int,
                    help='How many batches to run in order to profile the performance data')
parser.add_argument('--category', default='vision', type=str,
                    help='Task category, e.g. vision/speech/language')
parser.add_argument('--input_dim', type=int, nargs='+', default=[3, 224, 224],
                    help='Input dimensions (e.g., 3 224 224 for an image)')
parser.add_argument('--target_dim', type=int, nargs='+', default=[10],
                    help='Target dimensions (e.g., 10 for classification)')
parser.add_argument('--start_batch_size', type=int, default=64,
                    help='Start batch size for profiling')
parser.add_argument('--end_batch_size', type=int, default=64,
                    help='End batch size for profiling')
parser.add_argument('--num_classes', type=int, default=100,
                    help='Number of classes for classification')
parser.add_argument('--max_length', type=int, default=16000,
                    help='Maximum length of the audio sequences (in samples)')
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='Sample rate of the input audio')

args = parser.parse_args()

if not torch.cuda.is_available():
    print('GPU not found.')
    exit(0)

# Instantiate the model with user-defined dimensions and move it to the GPU
print(args)
if args.arch == "user_defined":
    model = UserDefinedModel(args.input_dim)
elif args.category == 'vision':
    model = models.__dict__[args.arch]()
elif args.category == 'speech':
    if 'wav2vec2' in args.arch:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    elif 'hubert' in args.arch:
        model = HubertForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    elif 'data2vec' in args.arch:
        model = Data2VecAudioForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    else:
        raise ValueError("Unsupported model type")
    scaler = GradScaler()


# Function to clear GPU memory and reset CUDA context
def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
model_measurement = GPUTimeMeasurement()
for i in range(args.profile_batches):
    clear_gpu_memory()  
    model_measurement.start()
    model = model.cuda(args.gpu)
    model_measurement.end()
    model = model.to('cpu')
    clear_gpu_memory()  

model = model.cuda(args.gpu)

model_transfer_time = model_measurement.get_adjusted_average_time()

gpu_name = get_default_gpu_name()
# model's transfer time is not related to batch size
save_average_time(gpu_name, "transfer", args.arch, '', model_transfer_time, filename='model_transfer_time.txt')

print(f'gpuname: {gpu_name}, ' + 
            f'model: {args.arch}, ' + 
            f'model transfer time: {model_transfer_time}')

    


# switch to train mode
model.train()

if args.single_batch_size:
    batch_sizes = [args.batch_size]
else:
    batch_sizes = range(args.start_batch_size, args.end_batch_size + 1)
for batch_size in batch_sizes:

    data_transfer_time_measurement = GPUTimeMeasurement()
    
    num_batches = int(args.profile_batches)
    for _ in range(num_batches):
        # Create dummy data input dimensions
        if args.category == 'vision':
            inputs = torch.randn(
                batch_size,
                *args.input_dim
            )

            if args.arch == 'user_defined':
                targets = torch.randint(
                    batch_size,
                    *args.target_dim
                )
            else:
                targets = torch.randint(
                    0,
                    args.num_classes,
                    (batch_size,)
                    )
        elif args.category == 'speech':
            dummy_audio = torch.randn(batch_size, args.max_length)
            inputs = processor(dummy_audio, sampling_rate=args.sample_rate, return_tensors='pt',
                    padding='max_length', truncation=True, max_length=args.max_length)
            targets = torch.randint(0, args.num_classes, (batch_size,))
        
        # Start timing
        data_transfer_time_measurement.start()

        if args.category == 'vision':
            inputs = inputs.cuda(args.gpu, non_blocking=False)
            targets = targets.cuda(args.gpu, non_blocking=False) 
        elif args.category == 'speech':
            inputs = {k: v.cuda(args.gpu) for k, v in inputs.items()} 
            targets = targets.cuda(args.gpu)
        
        # End timing
        data_transfer_time_measurement.end()
        
        del inputs, targets
        clear_gpu_memory()

    print(data_transfer_time_measurement.get_records())
    data_transfer_time = data_transfer_time_measurement.get_adjusted_average_time()
    print(f'gpuname: {gpu_name}, ' + 
            f'model: {args.arch}, ' + 
            f'batch size: {batch_size}, ' + 
            f'data transfer time: {data_transfer_time}')
    
    save_average_time(gpu_name, "transfer", args.arch, batch_size, data_transfer_time, filename='data_transfer_time.txt')
