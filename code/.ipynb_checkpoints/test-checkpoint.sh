#! /bin/bash

##F3net
python ./F3net_dup/predict.py

##SOD
python ./SOD100k/main.py --mode='test' --model='../user_data/model_data/SOD_epoch_27.pth' --test_fold='../prediction_result/images' --sal_mode='n'

