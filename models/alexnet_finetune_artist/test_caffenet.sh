#!/usr/bin/env sh

/home/shh/tmp/caffe-1.0/build/tools/caffe test \
    --model=models/alexnet_finetune_artist/deploy.prototxt \
    --weights=models/alexnet_finetune_artist/caffe_alexnet_artist_train.caffemodel \
    --iterations=50 --gpu=0
