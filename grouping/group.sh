#!/bin/bash
for i in 1 2 4 8 16 32 64
do
  echo "doing grouping mytar and tar for group size $i"
  python group-needle.py ~/data/test-accuracy/imagenette2/train $i ~/data/test-accuracy/mytar/train
done