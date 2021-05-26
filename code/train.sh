#! /bin/bash

##F3net
python ./F3net_dup/train.py

##SOD
python ./SOD100k/main.py --train_root ../s2_data/expand_data/train --train_list ../s2_data/expand_data/train.txt

