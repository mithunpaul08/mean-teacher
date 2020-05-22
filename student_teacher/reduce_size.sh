#!/usr/bin/env bash
head -1000 data/rte/fever/train/fever_train_delex.jsonl > temp
mv temp data/rte/fever/train/fever_train_delex.jsonl

head -1000 data/rte/fever/train/fever_train_lex.jsonl > temp
mv temp data/rte/fever/train/fever_train_lex.jsonl


head -200 data/rte/fever/dev/fever_dev_delex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_delex.jsonl

head -200 data/rte/fever/dev/fever_dev_lex.jsonl > temp
mv temp data/rte/fever/dev/fever_dev_lex.jsonl

#all fnc related
head -200 data/rte/fnc/test/fnc_test_delex.jsonl > temp
mv temp data/rte/fnc/test/fnc_test_delex.jsonl

head -200 data/rte/fnc/test/fnc_test_lex.jsonl > temp
mv temp data/rte/fnc/test/fnc_test_lex.jsonl

head -200 data/rte/fnc/dev/fnc_dev_lex.jsonl > temp
mv temp data/rte/fnc/dev/fnc_dev_lex.jsonl


head -200 data/rte/fnc/dev/fnc_dev_delex.jsonl > temp
mv temp data/rte/fnc/dev/fnc_dev_delex.jsonl

