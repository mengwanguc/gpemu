"""
   MIT License

   Copyright (c) 2023 Gus Waldspurger

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
"""

import os
import sys
import time
import math
import numpy as np
import multiprocessing as mp
import AsyncLoader as al
from typing import List
from glob import glob

# Get all filepaths descending from the provided ROOT directory.
def get_all_filepaths(root: str, extension: str = "*"):
    # Taken from https://stackoverflow.com/a/18394205
    filepaths = [y for x in os.walk(root) for y in glob(os.path.join(x[0], "*.{}".format(extension)))]
    total_size = sum([os.path.getsize(filepath) for filepath in filepaths])

    return filepaths, total_size

# Load all files in FILEPATHS using regular synchronous IO.
def load_normal(filepaths: List[str]):
    begin = time.time()
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            foo = file.read(-1) # Read the entire file
    end = time.time()

    return end - begin

# Load all files in FILEPATHS using an AsyncLoader worker context.
def load_async_worker_loop(filepaths: List[str], batch_size: int, worker: al.Worker):
    # Read everything, one batch at a time.
    while filepaths:
        # Submit requests
        n_this_batch = min(batch_size, len(filepaths))
        partial_batch = n_this_batch < batch_size
        for _ in range(n_this_batch):
            if worker.request(filepath = filepaths.pop()) != True:
                print("Worker request failed")

        # Retrieve results
        for _ in range(n_this_batch):
            entry = worker.wait_get()
            entry.release()

# Load all files in FILEPATHS using AsyncLoader with N_WORKERS worker threads.
def load_async(filepaths: List[str], batch_size: int, max_idle_iters: int, n_workers: int):
    n_files = len(filepaths)
    files_per_loader = int(math.ceil(n_files / n_workers))
    loader = al.Loader(queue_depth=batch_size,
                       n_workers=n_workers,
                       dispatch_n=batch_size,
                       max_idle_iters=max_idle_iters,
                       direct=False)
    
    # Spawn the loader
    loader_process = mp.Process(target=loader.become_loader)
    
    # Spawn N_WORKERS workers
    processes = []
    for i in range(n_workers):
        process = mp.Process(target=load_async_worker_loop, args=(
            filepaths[i * files_per_loader : (i + 1) * files_per_loader].copy(),
            batch_size,
            loader.get_worker_context(id=i)
        ))
        processes.append(process)

    # Don't count setup for timing.
    begin = time.time()

    # Start all the processes
    loader_process.start()
    for process in processes:
        process.start()

    # Wait on the workers
    for process in processes:
        process.join()
    end = time.time()

    # Kill the loader once the workers are done
    loader_process.kill()

    return end - begin

def verify_worker_loop(filepaths: List[str], batch_size: int, worker: al.Worker, reference_data):
    match_count = 0
    mismatch_count = 0
    batch_id = 0

    # Read everything, one batch at a time.
    while filepaths:
        batch_id += 1

        # Submit requests
        n_this_batch = min(batch_size, len(filepaths))
        partial_batch = n_this_batch < batch_size
        for _ in range(n_this_batch):
            if worker.request(filepath = filepaths.pop()) != True:
                print("Worker request failed")

        # Retrieve results
        for i in range(n_this_batch):
            entry = worker.wait_get()
            filepath = entry.get_filepath().decode('utf-8')
            data = entry.get_data()

            matched = True
            for i, (ref, ours) in enumerate(zip(reference_data[filepath], data)):
                if ref != ours:
                    matched = False

            if matched:
                match_count += 1
            else:
                mismatch_count += 1

            entry.release()
    
    print("Worker end. {} matches, {} mismatches".format(match_count, mismatch_count))

def verify_integrity(filepaths: List[str], batch_size: int, max_idle_iters: int, n_workers: int):
    # Read everything, and store the data
    data = {}
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            data[filepath] = file.read(-1)

    # Spin up an AsyncLoader and make sure it reads the same data
    loader = al.Loader(queue_depth=batch_size,
                       n_workers=n_workers,
                       dispatch_n=batch_size,
                       max_idle_iters=max_idle_iters,
                       direct=False)
    loader_process = mp.Process(target=loader.become_loader)
    worker_process =  mp.Process(target=verify_worker_loop, args=(filepaths, batch_size, loader.get_worker_context(id=0), data))

    loader_process.start()
    worker_process.start()
    worker_process.join()
    loader_process.kill()

def main():
    np.random.seed(42)

    if len(sys.argv) < 2:
        print("Please provide the filepath of a directory to load from.")
        return
    if len(sys.argv) < 3:
        print("Please provide the desired file extension to be loaded.")
        return
    
    max_idle_iters = 1024

    filepath = sys.argv[1]
    extension = sys.argv[2]
    filepaths, size = get_all_filepaths(filepath, extension)
    np.random.shuffle(filepaths)
    print("Filepaths: {}".format( len(filepaths)))

    # Get normal loading time
    os.system("sudo ./clear_cache.sh")
    time_normal = load_normal(filepaths.copy())
    print("Normal: {:.04}s ({:.04} MB/s).".format(time_normal, size / (1024 * 1024 * time_normal)))

    # Get async loading time(s)
    worker_configs = [1]
    batch_configs = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    for n_workers in worker_configs:
        for batch_size in batch_configs:
            os.system("sudo ./clear_cache.sh")
            time_async = load_async(filepaths.copy(), batch_size, max_idle_iters, n_workers)
            print("AsyncLoader ({} workers, {} batch size): {:.04}s ({:.04} MB/s)".format(n_workers, batch_size, time_async, size / (1024 * 1024 * time_async)))
    
    # Check integrity...
    print("\nChecking integrity with 1 worker/32 batch size...")
    verify_integrity(filepaths.copy(), 32, max_idle_iters, 1)


if __name__ == "__main__":
    main()