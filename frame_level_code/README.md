# Reproduction Readme 

This file contains details on how to generate and predict on frame level models. 
These models were used by You8M team as part of Kaggle's "Google Cloud & YouTube-8M Video Understanding Challenge".
 
Duo to random initializations the results may not be completely reproducible.
 
## LSTM Models

We trained 3 models with same parameter settings. You can start one training by running following commands:

```sh
SAVEPATH="/some/path/lstm_model_run1"
DATAPATH="/some/other/path/train/train*.tfrecord"
python train.py \
  --train_data_pattern=$DATAPATH \
  --model=Lstmbidirect \
  --frame_features \
  --feature_names="rgb, audio" \ 
  --feature_sizes="1024, 128" \
  --batch_size=256  \
  --train_dir=$SAVEPATH \
  --base_learning_rate=0.00025 \
  --lstm_cells=1200 \
  --num_epochs=6
```

The scripts used both training nad validation sets. Make sure you have both in the same folder e.g.
`/some/other/path/train/` and `/some/other/path/validate/`. Training was done for >135 batches.

After that select checkpoints that will be used as starting point for EMA training. 
We used first checkpoint that was >115000, >125000 and >135000. So three checkpoints in total. 

Using the following command you can make a copy of model and store EMA of the weights:
```
WEIGHTSSOURCE=${SAVEPATH}/model.ckpt-115035
MODELSOURCE=${SAVEPATH}/model.ckpt-115035
SAVEPATH2=${SAVEPATH}/ema_cp115035

/workspace5/miha/yt8m/results/fold2/bidirect_lstm_all_run3_EMA/cp115035/model.ckpt-115035
generate_EMAmodel.py  "$WEIGHTSSOURCE" "$MODELSOURCE" "$SAVEPATH2/model.ckpt-115035" 
```

Continue training your model for 3000 batches:
```sh
python train.py \
  --train_data_pattern=$DATAPATH \
  --model=Lstmbidirect \
  --frame_features \
  --feature_names="rgb, audio" \ 
  --feature_sizes="1024, 128" \
  --batch_size=256  \
  --train_dir="$SAVEPATH2" \
  --base_learning_rate=0.00025 \
  --lstm_cells=1200 \
  --num_epochs=1
```

finally use the inference.py script to generate the predictions:

```sh
CHECKPOINT=118203
TESTDATAPATH="/some/other/path/test/test*.tfrecord"
python inference.py \
 --checkpoint_file="${SAVEPATH2}/model.ckpt-${CHECKPOINT}" \
 --train_dir="${SAVEPATH2}" \
 --frame_features --feature_names="rgb, audio" --feature_sizes="1024, 128" \
 --top_k=30 \
 --use_ema_var=True \
 --input_data_pattern="${TESTDATAPATH}" \
 --output_file="${SAVEPATH2}/predictions_lstm_ema_ckpt-${CHECKPOINT}.csv" \
 --batch_size=300 \
 --num_readers=10
```

With training 3 models and taking 3 predictions from each run 
we ended up having 9 prediction files. All the predictions were assembled with equal weights. 
Merging code (main branch) can be used to aggregate the predictions.

## GRU Models

Unlike LSTM models we trained GRU models in 5 fold. For training use the following code:

```sh
DATAPATH="/some/other/path/train/train*.tfrecord"
for FOLD in {0..4}; do 
  SAVEPATH="/some/path/GRU_model_${FOLD}"

  python train.py \
    --train_data_pattern="${DATAPATH}"
    --model=GRUbidirect \
    --video_level_classifier_model="MoeModel" \
    --frame_features \
    --feature_names="rgb, audio" \
    --feature_sizes="1024, 128" \
    --batch_size=256 \
    --use_cv=True \
    --fold=${FOLD} \
    --split_seed=11 \
    --train_dir="${SAVEPATH} \
    --base_learning_rate=0.00025 \
    --lstm_cells=1250
    --num_epochs=6
done
```

To reproduce our results you will need to infer from checkpoints after 82, 94, 96, 98 and 100 steps.
Do the inference for all the folds and checkpoints:
```sh
FOLD=0
CHECKPOINT=94398
SAVEPATH="/some/path/GRU_model_${FOLD}"
TESTDATAPATH="/some/other/path/test/test*.tfrecord"
python inference.py \
 --checkpoint_file="${SAVEPATH2}/model.ckpt-${CHECKPOINT}" \
 --train_dir="${SAVEPATH2}" \
 --frame_features \
 --feature_names="rgb, audio" \
 --feature_sizes="1024, 128" \
 --top_k=30 \
 --input_data_pattern="${TESTDATAPATH}" \
 --output_file="${SAVEPATH}/predictions_GRU_ckpt-${CHECKPOINT}.csv" \
 --batch_size=300 \
 --num_readers=10
```

The 5 checkpoints from 5 folds were weighted equally and used for final prediction.
