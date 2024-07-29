# from apex import amp
from torch.utils.checkpoint import checkpoint_sequential
import torch
import torch.nn as nn
import pycuda
import time

def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), time.time()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached, timestamp = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached,
            'timestamp': timestamp,
        })

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)


def log_mem(model, inp, target, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []

    criterion = nn.CrossEntropyLoss().cuda(0)

    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    print(1)
    for i in range(5):
        out = model(inp)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        mem_log.append({
            'layer_idx': -1,
            'call_idx': mem_log[-1]["call_idx"] + 1,
            'layer_type': 'Final',
            'exp': exp,
            'hook_type': 'final',
            'mem_all': torch.cuda.memory_allocated(),
            'mem_cached': torch.cuda.memory_reserved(),
            'timestamp': time.time(),
        })
        time.sleep(0.001)
        print(i)
        
    print('hi')
    
    [h.remove() for h in hr]

    return mem_log


def log_mem_cp(model, inp, mem_log=None, exp=None, cp_chunks=3):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = checkpoint_sequential(model, cp_chunks, inp)
        loss = out.sum()
        loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def log_mem_amp(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_model, optimizer = amp.initialize(model, optimizer)
    for idx, module in enumerate(amp_model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = amp_model(inp)
        loss = out.sum()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log


def log_mem_amp_cp(model, inp, mem_log=None, exp=None, cp_chunks=3):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_model, optimizer = amp.initialize(model, optimizer)
    for idx, module in enumerate(amp_model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        out = checkpoint_sequential(amp_model, cp_chunks, inp)
        loss = out.sum()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    finally:
        [h.remove() for h in hr]

        return mem_log



