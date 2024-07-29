import pycuda.driver as cuda
from pycuda.tools import DeviceMemoryPool, OccupancyRecord
from pycuda.compiler import SourceModule

# Initialize the CUDA device
cuda.init()

# Select the first CUDA device
device = cuda.Device(0)

# Create a CUDA context on the device
context = device.make_context()

# Get memory information
free_memory, total_memory = cuda.mem_get_info()

print(f"Total GPU Memory: {total_memory / 1024**2} MB")
print(f"Free GPU Memory: {free_memory / 1024**2} MB")

# Clean up and free all memory allocated by PyCUDA
context.pop()
