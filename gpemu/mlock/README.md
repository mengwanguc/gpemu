# Python `mlock` Module

A Python interface to allocate page-locked memory, in order to accurately simulate memory pressure.

## Installation

### Package installation

#### Pip
```bash
pip install mlock
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

This module adds a new class, `mlock.PyBalloon`, initialized with a single `size` parameter, specifying the size of the page-locked region to be allocated in bytes.

### `PyBalloon.get_size()`

Returns the size of the balloon's page-locked region in bytes.

### `PyBalloon.set_used(used: Bool)`

Set's the balloon's allocation status to `used`. Intended to allow balloons to be reused and/or shared.

### `PyBalloon.get_used()`

Returns the balloon's allocation status.
