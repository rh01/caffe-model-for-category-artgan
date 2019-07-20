#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/opt/shh/wikiart
DATA=data/artist
TOOLS=/home/shh/tmp/caffe-1.0/build/tools

$TOOLS/compute_image_mean $EXAMPLE/artist_train_lmdb \
  $DATA/artist_mean.binaryproto

echo "Done."
