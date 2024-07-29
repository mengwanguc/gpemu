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
import minio
import numpy as np
from typing import List, Dict, Tuple
from glob import glob


# Get all filepaths descending from the provided ROOT directory.
def get_all_filepaths(root: str, extension: str = "*"):
    # Taken from https://stackoverflow.com/a/18394205
    filepaths = [y for x in os.walk(root) for y in glob(os.path.join(x[0], "*.{}".format(extension)))]
    total_size = sum([os.path.getsize(filepath) for filepath in filepaths])

    return filepaths, total_size

# Load everything in FILEPATHS and inspect for mismatches
def load_inspect(cache: minio.PyCache,
                 filepaths: List[str],
                 data: Dict[str, Tuple[bytearray, int]]):
    matches = 0
    mismatches = 0
    for filepath in filepaths:
        if hash(cache.read(filepath)[0]) == data[filepath][1]:
            matches += 1
        else:
            mismatches += 1
    
    return matches, mismatches


# Test that we're loading files correctly, and no corruption is occuring.
def test_integrity(size: int,
                   max_usable: int,
                   average_size: int,
                   filepaths: List[str],
                   data: Dict[str, int]):
    cache = minio.PyCache(size=size,
                          max_usable_file_size=max_usable,
                          average_file_size=average_size)

    success = True

    # Load the files for the first time (Mixed cold/capacity misses)
    matches, mismatches = load_inspect(cache, filepaths, data)
    if mismatches > 0:
        success = False

    
    # Load the files for the second time (Mixed hits/capacity misses)
    matches, mismatches = load_inspect(cache, filepaths, data)
    if mismatches > 0:
        success = False

    return success

def manual_read(cache: minio.PyCache, filepath: str, data: bytearray):
    if (cache.contains(filepath)):
        return cache.load(filepath)[0]
    else:
        if (cache.store(filepath, len(data), data)):
            return cache.load(filepath)[0]
        else:
            return data

def load_inspect_manual(cache: minio.PyCache,
                        filepaths: List[str],
                        data: Dict[str, Tuple[bytearray, int]]):
    matches = 0
    mismatches = 0
    for filepath in filepaths:
        if hash(manual_read(cache, filepath, data[filepath][0])) == data[filepath][1]:
            matches += 1
        else:
            mismatches += 1
    
    return matches, mismatches

def test_manual_methods(size: int,
                        max_usable: int,
                        average_size: int,
                        filepaths: List[str],
                        data: Dict[str, Tuple[bytearray, int]]):
    cache = minio.PyCache(size=size,
                          max_usable_file_size=max_usable,
                          average_file_size=average_size)

    success = True

    # Load the files for the first time (Mixed cold/capacity misses)
    matches, mismatches = load_inspect_manual(cache, filepaths, data)
    if mismatches > 0:
        print("{} matches, {} mismatches".format(matches, mismatches))
        success = False

    
    # Load the files for the second time (Mixed hits/capacity misses)
    matches, mismatches = load_inspect_manual(cache, filepaths, data)
    if mismatches > 0:
        print("{} matches, {} mismatches".format(matches, mismatches))
        success = False

    return success

def main():
    np.random.seed(42)
    MB = 1024 * 1024

    if len(sys.argv) < 2:
        print("Please provide the filepath of a directory to load from.")
        return
    if len(sys.argv) < 3:
        print("Please provide the desired file extension to be loaded.")
        return

    filepaths, size = get_all_filepaths(sys.argv[1], sys.argv[2])
    average_file_size = size // len(filepaths)
    print("{} filepaths, {} MB, {} KB average".format(len(filepaths),
                                                      size // MB,
                                                      average_file_size // 1024))

    # Read everything normally to have a ground truth.
    data = {}
    for filepath in filepaths:
        with open(filepath, 'rb') as file:
            bytes = file.read(-1)
            data[filepath] = (bytes, hash(bytes))

    # Read everything with various cache sizes and ensure everything matches.
    configs = [
        (64  * MB, 16 * MB, average_file_size),
        (128 * MB, 16 * MB, average_file_size),
        (256 * MB, 16 * MB, average_file_size),
        (512 * MB, 16 * MB, average_file_size),
    ]

    print("-- testing integrity --")
    for config in configs:
        print("testing {} MB cache...".format(config[0] // MB), end="")
        if (test_integrity(*config, filepaths, data)):
            print("OK.")
        else:
            print("FAIL.")

    print("-- testing manual methods integrity --")
    for config in configs:
        print("testing {} MB cache...".format(config[0] // MB), end="")
        if (test_manual_methods(*config, filepaths, data)):
            print("OK.")
        else:
            print("FAIL.")

if __name__ == "__main__":
    main()