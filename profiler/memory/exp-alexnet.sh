#!/bin/bash

# Loop from 1 to 256
for i in {1..256}
do
   # Call the python script with the current value of i
   python profile-memory.py -a alexnet $i
done
