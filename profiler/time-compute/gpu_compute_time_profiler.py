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
def save_average_time(gpu_name, time_type, model_name, batch_size, average_time):
    directory = f"raw_data/{gpu_name}/{time_type}/{model_name}/{batch_size}/"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, "time.txt")
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
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--use-minio', default=False, type=bool,
                    metavar='USE_MINIO', help='use MinIO cache')
parser.add_argument('-c', '--cache-size', default=16 * 1024 * 1024 * 1024,
                    type=int, metavar='CACHESIZE',
                    help='MinIO cache size, training gets 10/11, validation gets 1/11 (default=16GB)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
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
    criterion = model.get_criterion()
    optimizer = model.get_optimizer()
elif args.category == 'vision':
    model = models.__dict__[args.arch]()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
elif args.category == 'speech':
    if 'wav2vec2' in args.arch:
        model = Wav2Vec2ForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes).cuda(args.gpu)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    elif 'hubert' in args.arch:
        model = HubertForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes).cuda(args.gpu)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    elif 'data2vec' in args.arch:
        model = Data2VecAudioForSequenceClassification.from_pretrained(args.arch, num_labels=args.num_classes).cuda(args.gpu)
        processor = Wav2Vec2Processor.from_pretrained(args.arch)
    else:
        raise ValueError("Unsupported model type")
    scaler = GradScaler()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

model = model.cuda(args.gpu)
criterion = criterion.cuda(args.gpu)


# switch to train mode
model.train()

gpu_name = get_default_gpu_name()

if args.single_batch_size:
    batch_sizes = [args.batch_size]
else:
    batch_sizes = range(args.start_batch_size, args.end_batch_size + 1)
for batch_size in batch_sizes:
    # Create dummy data input dimensions and move it to the GPU
    if args.category == 'vision':
        inputs = torch.randn(
            batch_size,
            *args.input_dim
        ).cuda(
            args.gpu,
            non_blocking=False
        )

        if args.arch == 'user_defined':
            targets = torch.randint(
                batch_size,
                *args.target_dim
            ).cuda(
                args.gpu,
                non_blocking=False
            )
        else:
            targets = torch.randint(
                0,
                args.num_classes,
                (batch_size,)
                ).cuda(
                    args.gpu,
                    non_blocking=False) 
    elif args.category == 'speech':
        dummy_audio = torch.randn(batch_size, args.max_length)
        inputs = processor(dummy_audio, sampling_rate=args.sample_rate, return_tensors='pt',
                padding='max_length', truncation=True, max_length=args.max_length)
        inputs = {k: v.cuda(args.gpu) for k, v in inputs.items()}  # Move to GPU
        targets = torch.randint(0, args.num_classes, (batch_size,)).cuda(args.gpu)

    forward_time_measurement = GPUTimeMeasurement()
    backward_time_measurement = GPUTimeMeasurement()

    num_batches = int(args.profile_batches)
    for _ in range(num_batches):
        # Start timing
        forward_time_measurement.start()

        # Forward pass
        if args.category == 'vision':
            outputs = model(inputs)
        elif args.category == 'speech':
            with autocast():
                outputs = model(**inputs)

        forward_time_measurement.end()

        backward_time_measurement.start()

        # Calculate loss
        if args.category == 'vision':
            if args.arch in ['googlenet', 'inception_v3']:
                loss = criterion(outputs.logits, targets)
            else:
                loss = criterion(outputs, targets)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elif args.category == 'speech':
            if 'loss' in outputs:
                loss = outputs.loss
            else:
                loss = criterion(outputs.logits, targets)
            model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # End timing
        backward_time_measurement.end()

    # Calculate and print the average time
    forward_time = forward_time_measurement.get_adjusted_average_time()
    backward_time = backward_time_measurement.get_adjusted_average_time()
    print(f'gpuname: {gpu_name}, ' + 
            f'model: {args.arch}, ' + 
            f'batch size: {batch_size}, ' + 
            f'forward time: {forward_time}, ' + 
            f'backward time: {backward_time}')
    
    # Save the average time to a file
    save_average_time(gpu_name, "forward", args.arch, batch_size, forward_time)
    save_average_time(gpu_name, "backward", args.arch, batch_size, backward_time)

    del inputs, targets, outputs, loss
    torch.cuda.empty_cache()
