# MinIO Cache

A Python MinIO file cache module implemented in C/CPython.

## Installation

### Package installation

#### Pip
```bash
pip install MinIO-Cache
```

#### Manual
```bash
python setup.py install
```

### System configuration

Unless you're willing to run with sudo privileges, you'll need to update the `memlock` ulimit. To do so, add the following lines to `/etc/security/limits.conf` (you'll need sudo privileges to edit this file).
```
*   soft    memlock     unlimited
*   hard    memlock     unlimited
```
Alternatively, instead of `unlimited` you can specify the maximum (total, across all caches) size you plan to use.

## Documentation

This module adds a new class, `minio.PyCache`, initialized with parameters `size`, `max_usable_file_size`, and `max_cacheable_file_size`, all specifying size in bytes. An area of `max_usable_file_size` bytes will be allocated as a temporary area for copying files in addition to the `size` bytes allocated for the cache. `max_cacheable_file_size` is optional, and any file exceeding this value (unless value is zero) will bypass the cache when read. 

### `PyCache.contains(filepath: str)`

Returns `True` if `filepath` has an entry in the cache, otherwise returns `False`.

### `PyCache.store(filepath: str, bytes: int, data: bytearray)`

Stores `bytes` bytes of data from `data` in the cache, indexed by `filepath`.

### `PyCache.load(filepath: str)`

Load the data at `filepath` from the cache, returning a tuple `(data, size)`, where `data` is the bytes read, and `size` is the number of bytes. On miss, does not issue any IO, instead returning `None`.

### `PyCache.read_file(filepath: str)`

Reads the file at `filepath` through the cache, returning a tuple `(data, size)`, where `data` is the bytes read, and `size` is the number of bytes.

### `PyCache.flush()`

Flushes the cache.

### `PyCache.get_size()`

Returns the size of the cache's data region in bytes.

### `PyCache.get_used()`

Returns the number of bytes currently used in the cache's data region.