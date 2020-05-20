#!/usr/bin/env bash

#when in student-teacher code base, this will have to be called inside get_data_run.sh- after the files are downloaded

mkdir -p data/rte/fever/allnli/
mkdir -p data/rte/fnc/allnli/

python training_transformers/fact_verification/utils.py
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
rm sentence-transformers/examples/training_transformers/bert.log