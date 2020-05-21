#!/usr/bin/env bash

./get_glove_small.sh
get_data.sh
./reduce_size.sh
./run_main.sh
