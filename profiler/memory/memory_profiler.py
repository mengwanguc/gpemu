# %% Imports
import pandas as pd
import torch
from torch import nn

from utils.memory import _add_memory_hooks
from utils.plot import plot_mem_by_time

import time

import torch.backends.cudnn as cudnn
import sys

import argparse
import torchvision.models as models
import os

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


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
model_names.append("user_defined")
model_names.append("facebook/wav2vec2-base-960h")
model_names.append("facebook/hubert-base-ls960")
model_names.append("facebook/data2vec-audio-base-960h")


parser = argparse.ArgumentParser(description='GPEmu Profiler')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' + 
                    ' | '.join(model_names) + 
                    ' (default: resnet18)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--category', default='vision', type=str,
                    help='Task category, e.g. vision/speech/language')
parser.add_argument('--input_dim', type=int, nargs='+', default=[3, 224, 224],
                    help='Input dimensions (e.g., 3 224 224 for an image)')
parser.add_argument('--target_dim', type=int, nargs='+', default=[10],
                    help='Target dimensions (e.g., 10 for classification)')
parser.add_argument('--num_classes', type=int, default=100,
                    help='Number of classes for classification')
parser.add_argument('--max_length', type=int, default=16000,
                    help='Maximum length of the audio sequences (in samples)')
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='Sample rate of the input audio')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('-pl', '--plot-time-series', action='store_true',
                    help='plot the time series chart of memory usage during the training')
parser.add_argument('--profile_batches', default=3, type=int,
                    help='How many batches to run in order to profile the performance data')


args = parser.parse_args()
cudnn.benchmark = True

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

model.train()

gpu_name = get_default_gpu_name()

batch_size = args.batch_size

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




mem_log = []
torch.cuda.reset_peak_memory_stats(0)

exp=f'batch size {batch_size}'
hr = []

for idx, module in enumerate(model.modules()):
    _add_memory_hooks(idx, module, mem_log, exp, hr)

mem_log.append({
        'batch': 0,
        'layer_idx': -1,
        'call_idx': -1,
        'layer_type': 'Start',
        'exp': exp,
        'hook_type': 'start',
        'mem_all': torch.cuda.memory_allocated(),
        'mem_cached': torch.cuda.memory_reserved(),
        'timestamp': time.time(),
    })

for i in range(args.profile_batches):
    # Forward pass
    if args.category == 'vision':
        outputs = model(inputs)
    elif args.category == 'speech':
        with autocast():
            outputs = model(**inputs)

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
        
    torch.cuda.synchronize()
    mem_log.append({
        'batch': i,
        'layer_idx': -1,
        'call_idx': mem_log[-1]["call_idx"] + 1,
        'layer_type': 'Final',
        'exp': exp,
        'hook_type': 'final',
        'mem_all': torch.cuda.memory_allocated(),
        'mem_cached': torch.cuda.memory_reserved(),
        'timestamp': time.time(),
    })

[h.remove() for h in hr]


torch.cuda.synchronize()
torch.cuda.empty_cache()

df = pd.DataFrame(mem_log)

# with pd.option_context('display.max_rows', None):
#     print(df)

peak_memory = df.mem_all.max()/1024**2
persistent_df = df[(df['hook_type'] == 'final') & (df['batch'] > 0)]
persistent_memory = persistent_df['mem_all'].max() / 1024**2

directory = f"raw_data/{gpu_name}/{args.arch}/{batch_size}/"
os.makedirs(directory, exist_ok=True)
peak_file_path = os.path.join(directory, "peak.txt")
with open(peak_file_path, "w") as file:
    file.write(f"{peak_memory}\n")
persistent_file_path = os.path.join(directory, "persistent.txt")
with open(persistent_file_path, "w") as file:
    file.write(f"{persistent_memory}\n")
          
if args.plot_time_series:
    print('plotting')
    base_dir = '.'
    plot_mem_by_time(df, output_file=f'{base_dir}/{args.arch}_{args.batch_size}_memory_over_time.png',
            title = f'GPU Memory Usage of {args.arch} with batch size {args.batch_size} over time')
    print('done plotting')



