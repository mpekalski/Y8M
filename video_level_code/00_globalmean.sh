#!/bin/bash

INPUT_DIR=${HOME}/YouTube.Kaggle/input/video_level
OUTPUT_DIR=${HOME}/YouTube.Kaggle/input/model_params

python get_global_moments.py \
    --input_data_pattern=${INPUT_DIR}/train/train*.tfrecord \
    --output_=${OUTPUT_DIR}/video_level_global_moments.csv \
    --feature_names="mean_rgb, mean_audio" \
    --feature_sizes="1024,128" 
