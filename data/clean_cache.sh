#!/bin/bash

# ./data/clean_cache.sh 1 1 1
# Working dir: /home/wanghao/tf-faster-rcnn
# 1. clean data/cache/*
# 2. clean data/VOCdevkit2007/annotations_cache/*
# 3. clean soft link VOC2007
# Done.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR
echo 'Working dir:' $DIR

clean_cache=$1
clean_voc_annos_cache=$2
link_voc2007=$3


if [ $clean_cache -eq '1' ]; then
   echo '1. clean data/cache/*'
   rm -rf data/cache   
fi
if [ $clean_voc_annos_cache -eq '1' ]; then
   echo '2. clean data/VOCdevkit2007/annotations_cache/*'
   rm -rf data/VOCdevkit2007/annotations_cache/*
fi

# soft link need abs path 
if [ $link_voc2007 -eq '1' ]; then
   rm -rf data/VOCdevkit2007/VOC2007
   echo '3. clean soft link VOC2007'
   ln -s $DIR/data/VOCdevkit2007/VOC2007_1* $DIR/data/VOCdevkit2007/VOC2007
elif [ $link_voc2007 -eq '2' ]; then
   rm -rf data/VOCdevkit2007/VOC2007
   echo '3. clean soft link VOC2007'
   ln -s $DIR/data/VOCdevkit2007/VOC2007_2* $DIR/data/VOCdevkit2007/VOC2007
else
   rm -rf data/VOCdevkit2007/VOC2007
   echo '3. clean soft link VOC2007 ***'
   ln -s $DIR/data/VOCdevkit2007/VOC2007_$link_voc2007* $DIR/data/VOCdevkit2007/VOC2007
fi

echo "Done."
