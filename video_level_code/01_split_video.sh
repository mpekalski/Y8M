#!/bin/bash

INPUT_DIR=${HOME}/YouTube.Kaggle/input
OUTPUT_DIR=${HOME}/YouTube.Kaggle/input

/usr/bin/python3.5 split_video.py \
  --input_data_pattern=${INPUT_DIR}/frame_level/validate*.tfrecord \
  --file_from=0 --file_to=4096 \
  --output_path=${OUTPUT_DIR}/frame_stats/stg/ 
