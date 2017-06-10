## Introduction
This folder contains code for video level models.

During those almost three months we have worked on the competition we have added a lot of different pieces to the code. This repository contains merged version of most of the code used to create video level models we have tried. It also contains a short description of the canges and usage, please see below. In this repository there is also a set of bash scripts that can be used to run video level models that we have used for final submission. Keep in mind that there is some randomness in the code so the results might differ a bit. We run everything on `python >= 3.5`. 

For reference, one of bigger video level models, `MoNN3Lw` with `EMA` on `mean_rgb`, `mean_audio`, `std_rgb`, `std_audio`, `num_frames` takes about 9 hours to train one fold. More details can be found in our paper describing the solution (if the link is not available here at the time of writing, we will post it later). 

If you want to run the models we encurage you to use Tensorboard to monitor the progress. If provided with evaluation dataset our code also produces evalGAP that can be compared with trainGAP as the training progress.

Below is a list of files and some key features we have added, or a short description what they do. 

## train.py

`train_data_pattern`, `train_data_pattern2`, `train_data_pattern3`  some of us had scattered data across multiple hard drives, that is why we introduced flags to read train data from 3 different locations.

`eval_data_pattern` - during training you can track evaluated GAP score on a validations set, it is visible as evalGAP in the tensorboard.

`layers_keep_probs` - if your network includes dropout you can pass up to 10 values of keep_probs to the network from the command line. They will default to 1.0 during inference or evaluation scoring. Within the model you should be reading keep_probs from a list `layers_keep_probs`.

`fold` - this flags assigns number to a fold that is currently running. CV is deterministic here in a sence it takes four of every five files, which four are those depends on the value of fold flag.

`apply_global_normalization` - boolean flag indicating whether the dataset should be L2 normalized with precalculated mean/std. deviation
`apply_batch_l2_normalization` - boolean flag indicating whether the batch should be L2 normalized, the raw input.

`c_vars` - this variable takes a string being a list separated by commas. It allows for definition of new variables that will be created during training. Lets assume we want to create new variables from mean_rgb:
```
c_sq_mean_rgb = square of mean_rgb
c_log_mean_rgb = log(abs(1+x))
c_inv_mean_rgb = 1/mean_rgb
c_abs_mean_rgb = abs(mean_rgb)
c_sin_mean_rgb = sin(mean_rgb)
c_cos_mean_rgb = cos(mean_rgb)
c_sqrt_mean_rgb = sqrt(mean_rgb)
c_rsqrt_mean_rgb = 1/sqrt(mean_rgb)
c_diff_mean_rgb_10:20_mean_audio_10:20 = mean_rgb[10:20] - mean_audio[10:20]
c_over_mean_rgb_mean_audio = 
c_interaction_mean_rgb_10:20_mean_audio_10:20 = [mean_rgb[i]*mean_audio[i] for i in range(10m,0)]
```
Basically if you want to create a new calculated variable you have to include the base variable you do calculations on, if you want to exclude it from the model you may use another flag `r_vars`.

`r_vars` - takes a string being a list of variables that you want to be removed from the training (you may do calculations on them with c_vars and then they will get removed with `r_vars`).

The code that makes calulations and exclusion can be found in readers.py.

`restart_learning_rate` - if you stopped learning and you want to restart from where you stopped but with differnt learning rate, just set this variable to some positive value that will be a new learning_rate.

`ema_halflife` - halflife of exponential moving average.

`use_ema` - boolean indicating if exponential moving average of variables should be calculated or not.

`gpu_only` - 1 if gpu should be used, 0 if cpu use should be forced.

`model_export_gpu_only` - simliar to gpu_only, but that variable affects only export_model not training. Really big models may take whole GPU memory, so there would be nothing left for export process. In that case export can be run on cpu and the traning will not fail. It will take much longer but you can train slightly bigger models.

`truncated_num_classes` - if you want to train model only on top X classes set this flag to X.

`decoded_zlib` - if your dataset is compressed with zlib set it to True, False if not.

## xp_video_level_models.py 
That's the file where we defined our models used in final submission. The models we used are in the beginning of the file, in the latter part we kept some models we treid out but did not use.


## inspect_ckpt.py
This file will print all variables and their values used in a specified checkpoint. The paths and checkpoint name are hardcoded.

## split_video.py
Our training dataset was augmented with data coming from first and second part of each file. This scrip does the splitting and additionally calculates some additional variables. It takes frame level files as input. For each input file it outputs three files with prefixes `E`, `O`, `F`, denoting End (second half), whOle file, and First half respectively. The script will check which files are already transformed and will not split them again. You may split only a subset of files by specifying flags `file_from` and `file_to`, which are file numbers in a folder starting from 0. A sample use can be found in script `01_split_video.sh`. 

## get_global_moments.py
This script is used to calculate mean and std. deviation of selected variables. They can later be used in `train.py` when applying global_normalization. Sample use is in file `00_globalmean.sh`

`input_data_pattern`, `input_data_pattern2`, `input_data_pattern3` - up to three different input paths.
`feature_names`, `feature_sizes` - a list of features to calculate mean and std. deviation for, and their sizes.
`output_file` - there to save the calculated values.


## scripts 
All *.sh scripts with numbers from 02 to 10 are scripts used for training our video level models we have used in the final submission. Some of them also include inference loops. So predictions are made straight after each fold has finished. This way we could submit partial results and see how the model was performing on the public LB.
