#!/usr/bin/env sh

/home/shh/tmp/caffe-1.0/build/tools/caffe train \
    --solver=models/alexnet_finetune_style/solver.prototxt \
    --weights=models/bvlc_alexnet.caffemodel
