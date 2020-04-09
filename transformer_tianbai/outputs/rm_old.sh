#!/bin/bash

search_dir=`pwd`
kept_dir='roberta_normal_2400'

for outer_dir in "$search_dir"/*/ ; do
  if [[ $outer_dir != *"$kept_dir"* ]];
  then
    for file in "${outer_dir}nbest_"* ; do
      rm $file
      done
  fi

done
