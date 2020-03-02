#!/usr/bin/env bash
mkdir -p data
mkdir -p data/glove

FILE=data/glove/glove.840B.300d.txt
if test -f "$FILE";then
    echo "$FILE exists"
else
    wget http://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip glove.840B.300d.zip -d data/glove
fi


