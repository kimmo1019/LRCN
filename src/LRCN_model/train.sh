#!/usr/bin/env bash

GPU_ID=5

#./build/tools/caffe train \
/home/zhangboheng/caffe-ssd/build/tools/caffe train \
    -solver ./examples/zh_caption/lstm_lm_solver.prototxt \
    -gpu $GPU_ID \
2>&1 | tee ./examples/zh_caption/log/log_init_train.log
