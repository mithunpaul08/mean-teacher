#!/bin/bash


mkdir -p log_dir
mkdir -p model_storage

#pick according to which kind of dataset you want to use for  train, dev, test on. Eg: train on fever, test on fnc

#######train
mkdir -p datasets/rte/fever/train/

FILE=datasets/rte/fever/train/fever_train_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_lexicalized/used_in_emnlp2019_bert/fever_train_lex.jsonl -O $FILE

fi


FILE=datasets/rte/fever/train/fever_train_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    #for oaner :
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_delexicalized/used_in_emnlp2019_bert/fever_train_smart_ner.jsonl -O $FILE
    #for simpler ner:
    #wget https://storage.googleapis.com/fact_verification_mithun_files/fever_delexicalized/used_in_emnlp2019_bert/fever_train_simple_ner.jsonl -O $FILE


fi


##########dev
mkdir -p datasets/rte/fever/dev/
FILE=datasets/rte/fever/dev/fever_dev_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_lexicalized/used_in_emnlp2019_bert/fever_dev_lex.jsonl -O $FILE

fi


FILE=datasets/rte/fever/dev/fever_dev_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    #for oaner
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_delexicalized/used_in_emnlp2019_bert/fever_dev_smart_ner.jsonl -O $FILE
    #for simpler ner:
    #wget https://storage.googleapis.com/fact_verification_mithun_files/fever_delexicalized/used_in_emnlp2019_bert/fever_dev_simple_ner.jsonl -O $FILE
fi


#######test partition#########which usually is the fnc-dev loaded so that we can test on it while training itself.
mkdir -p datasets/rte/fnc/dev/

FILE=datasets/rte/fnc/dev/fnc_dev_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fake_news_challenge_lexicalized_data/used_in_emnlp2019_bert/FNC_dev_lex.jsonl -O $FILE
fi


FILE=datasets/rte/fnc/dev/fnc_dev_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/used_in_emnlp2019_bert/FNC_dev_smart_ner.jsonl -O $FILE
fi


#below are the corresponding code/paths for fnc test and  train partitions- uncomment if and when needed


#mkdir -p datasets/rte/fnc/train/
#wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_train_delex_4labels.jsonl -O datasets/rte/fnc/train/fnc_train_delex.jsonl


mkdir -p datasets/rte/fnc/test/
FILE=datasets/rte/fnc/test/fnc_test_delex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/person-c1/fnc_delexicalized_person-c1_actual_fnc_test.jsonl  -O $FILE
fi

mkdir -p datasets/rte/fnc/test/
FILE=datasets/rte/fnc/test/fnc_test_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fake_news_challenge_lexicalized_data/fn_test_split_fourlabels.jsonl -O $FILE
fi








