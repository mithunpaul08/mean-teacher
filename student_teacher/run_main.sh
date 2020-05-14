#!/usr/bin/env bash

#pick according to which one you want to train, dev, test on

python main.py --add_student True --which_gpu_to_use 0  --use_ema False \
--load_model_from_disk_and_test False \
--lex_train_full_path fever/train/fever_train_lex.jsonl