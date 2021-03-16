#!/bin/bash
#Author：changshu
#Date：2021-02-01
#Version：1.0
#Description：The first script		
echo "run test_encryption_project GPU version!"
cd /home/project/zhrtvc/zhrtvc && source activate && conda activate zhrtvc_env && export CUDA_VISIBLE_DEVICES=0,1 && ps -aux | grep mellotron_train.py | awk '{print $2}' && python mellotron_train.py;

