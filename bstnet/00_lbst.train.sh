#!/bin/bash

ROOT=${HOME}/YouTube.Kaggle
VIDEO_LEVEL_DIR=$ROOT/input/video_level
F2V_DATA_DIR=$ROOT/input/f2v_splits
MODEL_DIR=$ROOT/models

MODEL_GRP=gbm
#MODEL=LogisticModel
MODEL=NN3L
MODEL_DECOR=${MODEL}
TRAIN_DIR=$MODEL_DIR/${MODEL_GRP}/${MODEL_DECOR}

python train.py --model $MODEL \
  --feature_names="mean_rgb,mean_audio" \
  --feature_size="1024,128" \
  --decode_zlib=False \
  --truncated_num_classes=4716 \
  --apply_global_normalization=False \
  --apply_batch_l2_normalization=False \
  --base_learning_rate=0.0001 \
  --restart_learning_rate=-0.002 \
  --learning_rate_decay_examples=5000000 \
  --bst_model=NN2Lc \
  --bst_learning_rate=0.0001 \
  --num_epochs=20 \
  --save_model_minutes=60 \
  --export_model_steps=5000 \
  --train_data_pattern=${VIDEO_LEVEL_DIR}/train/train*.tfrecord \
  --eval_data_pattern=${VIDEO_LEVEL_DIR}/validate/validate*.tfrecord \
  --train_dir=$TRAIN_DIR \
  --start_new_model
