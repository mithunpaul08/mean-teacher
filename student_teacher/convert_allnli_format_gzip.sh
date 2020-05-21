#!/usr/bin/env bash

#when in student-teacher code base, this will have to be called after get_data_run.sh- after the files are downloaded

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