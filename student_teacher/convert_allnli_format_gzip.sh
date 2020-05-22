#!/usr/bin/env bash

#when in student-teacher code base, this will have to be called after get_data_run.sh- after the files are downloaded

mkdir -p data/rte/fever/allnli/lex/
mkdir -p data/rte/fever/allnli/delex/
mkdir -p data/rte/fnc/allnli/lex/
mkdir -p data/rte/fnc/allnli/delex/

python mean_teacher/utils/convert_to_allnli_format.py
rm -rf data/rte/fever/allnli/lex/*.gz
rm -rf data/rte/fever/allnli/delex/*.gz
rm -rf data/rte/fnc/allnli/lex/*.gz
rm -rf data/rte/fnc/allnli/delex/*.gz


for each in data/rte/fever/allnli/lex/*;
do
gzip  $each
done
for each in data/rte/fever/allnli/delex/*;
do
gzip  $each
done


for each in data/rte/fnc/allnli/lex/*;
do
gzip  $each
done

for each in data/rte/fnc/allnli/delex/*;
do
gzip  $each
done