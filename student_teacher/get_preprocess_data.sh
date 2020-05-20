#!/bin/bash

./get_glove.sh
./get_gigaword.sh

mkdir -p log_dir
mkdir -p model_storage

#pick according to which kind of dataset you want to use for  train, dev, test on. Eg: train on fever, test on fnc

#######train
mkdir -p data/rte/fever/train/

FILE=data/rte/fever/train/fever_train_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fever_train_lex_4labels.jsonl -O $FILE
fi

#this is for the experiment of trying train a teacher model first, and then load a trained teacher model inside student
# teacher architecture
#delete or comment this after training teacher independently is done @ march 25th 20202
#head -59599 $FILE > temp
#mv temp $FILE


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

#this is for the experiment of trying train a teacher model first, and then load a trained teacher model inside student teacher architecture
#delete or comment this after training teacher independently is done @ march 25th 20202
#head -13126 $FILE > temp
#mv temp $FILE


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
    wget https://storage.googleapis.com/fact_verification_mithun_files/fnc_delexicalized/personc1/fn_dev_9k_mihaicarved_oaner_neutered.jsonl -O $FILE
fi

FILE=data/rte/fnc/dev/fnc_dev_lex.jsonl
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://storage.googleapis.com/fact_verification_mithun_files/fake_news_challenge_lexicalized_data/fn_dev_split_fourlabels_the_one_mihai_carved_from_train.jsonl -O $FILE
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




mkdir -p data/rte/fever/allnli/
mkdir -p data/rte/fnc/allnli/

python mean_teacher/utils/convert_to_allnli_format.py
rm -rf data/rte/fever/allnli/*.gz
rm -rf data/rte/fnc/allnli/*.gz
for each in data/rte/fever/allnli/*;
do
gzip  $each
done
for each in data/rte/fnc/allnli/*;
do
gzip  $each
done


