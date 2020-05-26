#!/usr/bin/env bash


python main.py --add_student True  --use_ema False --load_model_from_disk_and_test False  --lex_train_full_path fever/train/fever_train_lex.jsonl --random_seed 45674 --create_new_comet_graph True --save_dir '/xdisk/msurdeanu/mithunpaul/model_storage/'