#!/usr/bin/env bash

#pick according to which kind of dataset you want to use for  train, dev, test on. Eg: train on fever, test on fnc

#######train
mkdir -p data/rte/fever/train/

FILE=data/rte/fever/train/fever_train_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O $FILE
fi

FILE=data/rte/fever/train/fever_train_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_delex_oaner_4labels.jsonl -O $FILE
fi


##########dev
mkdir -p data/rte/fever/dev/
FILE=data/rte/fever/dev/fever_dev_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_lex_4labels.jsonl -O $FILE
fi

FILE=data/rte/fever/dev/fever_dev_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_dev_delex_oaner_split_4labels.jsonl -O $FILE
fi


#######test
mkdir -p data/rte/fnc/dev/
FILE=data/rte/fnc/dev/fnc_dev_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_delexicalized_person-c1_dev_mithun_carved_from_train.jsonl -O $FILE
fi

FILE=data/rte/fnc/dev/fnc_dev_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fake_news_challenge_lexicalized_data/fnc_dev_lexicalized_mithun_carved.jsonl -O $FILE
fi

#below are the corresponding code/paths for fnc test and  train partitions


#mkdir -p data/rte/fnc/train/
#wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_train_delex_4labels.jsonl -O data/rte/fnc/train/fnc_train_delex.jsonl


mkdir -p data/rte/fnc/test/
FILE=data/rte/fnc/test/fnc_test_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_delexicalized_person-c1_actual_fnc_test.jsonl -O data/rte/fnc/test/fnc_test_delex.jsonl -O $FILE
fi

mkdir -p data/rte/fnc/test/
FILE=data/rte/fnc/test/fnc_test_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fn_test_split_fourlabels.jsonl -O data/rte/fnc/test/fnc_test_lex.jsonl -O $FILE
fi




mkdir -p log_dir/





python main.py --add_student True --which_gpu_to_use 2  --use_ema False \
--load_model_from_disk_and_test True \
--lex_train_full_path fever/train/fever_train_lex.jsonl
