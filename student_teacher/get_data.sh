#!/usr/bin/env bash

#pick according to which one you want to train, dev, test on

#train
mkdir -p data/rte/fever/train/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O data/rte/fever/train/fever_train_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl -O data/rte/fever/train/fever_train_delex.jsonl

#dev
mkdir -p data/rte/fever/dev/
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl -O data/rte/fever/dev/fever_dev_lex.jsonl
wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl -O data/rte/fever/dev/fever_dev_delex.jsonl


#mkdir -p data/rte/fnc/train/
#wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_train_delex_4labels.jsonl -O data/rte/fnc/train/fnc_train_delex.jsonl
#mkdir -p data/rte/fnc/dev/
#test
mkdir -p data/rte/fnc/test/
wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_test_delex.jsonl -O data/rte/fnc/test/fnc_test_delex.jsonl
