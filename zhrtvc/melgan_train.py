#!usr/bin/env python
# -*- coding: utf-8 -*-
# author: kuangdd
# date: 2020/2/23
"""
"""
import os

# from utils.argutils import print_args
from melgan.train import train_melgan, parse_args

args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

if __name__ == "__main__":
    try:
        from setproctitle import setproctitle

        setproctitle('zhrtvc-melgan-train')
    except ImportError:
        pass

    # print_args(args, parser)
    train_melgan(args)
