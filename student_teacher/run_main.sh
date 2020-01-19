#!/usr/bin/env bash

#pick according to which one you want to train, dev, test on

python main.py --add_student True --which_gpu_to_use 2  --use_ema True
