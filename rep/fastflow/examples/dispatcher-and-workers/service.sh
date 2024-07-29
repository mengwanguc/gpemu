#!/bin/bash

# Check if the script is provided with an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <NUM_WORKERS>"
    exit 1
fi

# Get the value of the argument
n=$1

# Define the number of CPUs per worker
cpus_per_worker=4

python dispatcher.py &

# Calculate the CPU ranges and run the commands
for ((i = 0; i < n; i++)); do
    start_cpu=$((i * cpus_per_worker))
    end_cpu=$((start_cpu + cpus_per_worker - 1))
    port=$((5001 + i))
    taskset -c $start_cpu-$end_cpu python worker.py $port &
done

# Wait for all background tasks to finish
wait
