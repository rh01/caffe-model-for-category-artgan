#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/opt/shh/wikiart
DATA=data/style
TOOLS=/home/shh/tmp/caffe-1.0/build/tools

$TOOLS/compute_image_mean $EXAMPLE/style_train_lmdb \
  $DATA/style_mean.binaryproto

echo "Done."
