#!/bin/bash
# convert images to lmdb

DATA=/opt/shh/wikiart/
IMGLIST=/home/shh/ArtGAN/WikiArtDataset/Style/style_train.csv
LMDBNAME=style_train_lmdb

rm -rf $DATA/$LMDBNAME
echo 'converting images...'
/home/shh/tmp/caffe-1.0/build/tools/convert_imageset --shuffle=true \
--resize_height=227 --resize_width=227 $DATA $IMGLIST $DATA/$LMDBNAME
