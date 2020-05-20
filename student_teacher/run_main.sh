#!/usr/bin/env bash


python main.py --add_student True  --use_ema False \
--load_model_from_disk_and_test False \
--lex_train_full_path fever/train/fever_train_lex.jsonl --save_dir '/xdisk/msurdeanu/mithunpaul/model_storage/'
