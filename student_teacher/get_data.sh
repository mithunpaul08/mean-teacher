#!/usr/bin/env bash

#pick according to which one you want to train, dev, test on


mkdir -p data/rte/fever/train/
mkdir -p data/rte/fever/dev/
mkdir -p data/rte/fnc/train/
mkdir -p data/rte/fnc/dev/
mkdir -p data/rte/fnc/test/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O data/rte/fever/train/fever_train_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl -O data/rte/fever/train/fever_train_delex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl -O data/rte/fever/dev/fever_dev_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl -O data/rte/fever/dev/fever_dev_delex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_train_delex_4labels.jsonl -O data/rte/fnc/train/fnc_train_delex.jsonl