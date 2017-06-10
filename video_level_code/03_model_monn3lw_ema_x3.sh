#!/bin/bash

MODEL=MoNN3Lw
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

INPUT_DIR=${HOME}/YouTube.Kaggle/input

for FOLD in 1 2 
do
   MODEL_DECOR=OFE_EMA_x3_${MODEL}_f${FOLD}
   echo "running $MODEL_DECOR"

   /usr/bin/python3.5 train.py \
     --model=$MODEL \
     --MoNN_num_experts=3 \
     --train_data_pattern="${INPUT_DIR}/frame_stats/split/train/*.tfrecord" \
     --train_data_pattern2="${INPUT_DIR}/frame_stats/split/validation/*.tfrecord" \
     --fold=${FOLD} \
     --feature_names="x3_rgb,x3_audio,std_rgb,std_audio,num_frames" \
     --feature_size="1024,128,1024,128,1" \
     --decode_zlib=True \
     --truncated_num_classes=4716 \
     --apply_global_normalization=False \
     --apply_batch_l2_normalization=False \
     --base_learning_rate=0.00008 \
     --restart_learning_rate=-0.00002 \
     --learning_rate_decay_examples=12000000 \
     --bst_model=NN2Lc \
     --bst_learning_rate=0.0001 \
     --num_epochs=12 \
     --save_model_minutes=240 \
     --export_model_steps=6000 \
     --ema_halflife=500 \
     --max_steps=70050 \
     --use_ema=True \
     --train_dir=${MODEL_DIR}/${MODEL_DECOR}  #\ --start_new_model
   sleep 60s
   for CKPT in 36050 42050 48050 54050 60050 66050
   do 
   /usr/bin/python3.5 inference.py \
     --model=$MODEL \
     --MoNN_num_experts=3 \
     --train_data_pattern='${INPUT_DIR}/frame_stats/split/train/*.tfrecord' \
     --train_data_pattern2='${INPUT_DIR}/frame_stats/split/validation/*.tfrecord' \
     --check_point=${CKPT} \
     --output_file="${MODEL_DIR}/${MODEL_DECOR}/${MODEL_DECOR}_predict_ckpt${CKPT}.csv" \
     --input_data_pattern="${INPUT_DIR}/frame_stats/split/test/O*.tfrecord" \
     --use_ema_var=True \
     --fold=${FOLD} \
     --feature_names="x3_rgb,x3_audio,std_rgb,std_audio,num_frames" \
     --feature_size="1024,128,1024,128,1" \
     --decode_zlib=True \
     --truncated_num_classes=4716 \
     --apply_global_normalization=False \
     --apply_batch_l2_normalization=False \
     --base_learning_rate=0.00008 \
     --restart_learning_rate=-0.00002 \
     --learning_rate_decay_examples=12000000 \
     --bst_model=NN2Lc \
     --bst_learning_rate=0.0001 \
     --num_epochs=12 \
     --save_model_minutes=240 \
     --export_model_steps=6000 \
     --ema_halflife=500 \
     --max_steps=70050 \
     --train_dir=${MODEL_DIR}/${MODEL_DECOR}  #\ --start_new_model

    sleep 60s
   done
done

