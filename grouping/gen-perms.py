import sys
import os
import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from os.path import basename
import tarfile

def gen_seed():
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    txt = "perms/seed10.txt"
    with open(txt, 'w') as f:
        f.write(str(seed))
    f.close()

#gen_seed()

def read_seed():
    with open('perms/seed1.txt') as f:
        seed = int(f.read())
    f.close()

    print(seed)
    print(type(seed))

read_seed()