#!/bin/bash

# Check if the script is provided with an argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <NUM_CPUs>"
    exit 1
fi

python dispatcher.py &

taskset -c 1-$1 python worker.py 50001 &

# Wait for all background tasks to finish
wait
