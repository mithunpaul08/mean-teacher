#!/usr/bin/env bash

./get_glove_small.sh
get_preprocess_data.sh
./reduce_size.sh
./run_main.sh
