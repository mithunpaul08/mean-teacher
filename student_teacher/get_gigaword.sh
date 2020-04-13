#!/usr/bin/env bash
mkdir -p data
mkdir -p data/gigaword

FILE=data/gigaword/gigawordDocFreq.sorted.freq.txt
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget https://osf.io/nhqkb/download -O $FILE
fi


