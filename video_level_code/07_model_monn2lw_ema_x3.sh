#!/bin/bash

MODEL=MoNN2Lw
FOLD=0

if [ $# -gt 0 ]; then
    MODEL=$1
fi
if [ $# -gt 1 ]; then
    FOLD=$2
fi
if [ $# -gt 2 ]; then
    CKPT=$3
fi

INPUT_DIR=${HOME}/youtube-8m/

for FOLD in 1 2 3 4 5
do

  MODEL_DECOR=EMA_${MODEL}_f${FOLD}
  echo "running $MODEL_DECOR"

  /usr/bin/python3.5 train.py \
    --model=$MODEL \
    --MoNN_num_experts=2 \
    --train_data_pattern="${INPUT_DIR}/frame_stats/split/train/*.tfrecord" \
    --train_data_pattern2="${INPUT_DIR}/frame_stats/split/validation/*.tfrecord" \
    --fold=${FOLD} \
    --feature_names="x3_rgb,x3_audio,std_rgb,std_audio,num_frames,std_all_rgb,std_all_audio" \
    --feature_size="1024,128,1024,128,1,1,1" \
    --batch_size=1024 \
    --decode_zlib=True \
    --truncated_num_classes=4716 \
    --apply_global_normalization=False \
    --apply_batch_l2_normalization=False \
    --base_learning_rate=0.00008 \
    --restart_learning_rate=-0.00002 \
    --learning_rate_decay_examples=12000000 \
    --bst_model=NN2Lc \
    --bst_learnin6_rate=0.0001 \
    --num_epochs=5 \
    --save_model_minutes=240 \
    --export_model_steps=8000 \
    --use_ema=True \
    --ema_halflife=600 \
    --train_dir=${MODEL_DIR}/${MODEL_DECOR}  #\ --start_new_model
  sleep 60s
done


