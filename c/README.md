# README #

C++ implementation of two utility commands for project yt8m. 
### How to build?
Type 

``` $ make ```

2017.05.23

Added ```ensemble``` and ```predCorr``` commands.

### ```ensemble``` 
```ensemble``` calculates the weighted sum of predictions listed in an input file. The input file is a csv of *prediction file* and *weight* pairs. An example is included ```file_wgt_example.csv```. An example session:

```
$ ensemble file_wgt_example.csv 
2017-05-23.22:19:36: Total 4 files.
../byModelPred/ensemble/GRU_fold01234_cp82cp94k96kcp98k100k_meanvalue_k20.csv : wgt = 0.453125
../byModelPred/ensemble/lstmpairs_fold01234_cp90k95k100k105110k_meanvalue_k20.csv : wgt = 0.09375
../byModelPred/ensemble/dbof11k_5folds_cp90k93k95k99k104k85k.csv : wgt = 0.140625
../byModelPred/ensemble/MoNN3Lw_combo_5folds_CPS40k_60k_20files.csv : wgt = 0.3125
2017-05-23.22:19:55: Pred vecs sorted...
2017-05-23.22:19:56: Pred vecs sorted...
2017-05-23.22:20:13: Added 2
2017-05-23.22:20:13: Added 2
2017-05-23.22:20:17: Pred vecs sorted...
2017-05-23.22:20:40: Added 4
2017-05-23.22:20:53: Saved data to /tmp/scrib_combo.csv
```

### ```predCorr```
```predCorr``` takes the same (*prediction*, *weight*) pair csv file, and calculates the covariance and correlations between the predictions. *weight* is ignored.

2017.05.01

### calcGAP: calculate the average precision of a prediction csv.

   > calcGAP true_label.csv prediction.csv 

    Notice the GAP calculated by this command is about 0.012 lower than Kaggle GAP. The difference is because we're averaging over the examples GAP.

### addPred: add two predictions by weights

   > addPred prediction1.csv prediction2.csv wgt1 wgt2 output.csv

    add3Pred: A variant to add 3 predictions files, useful for checkpoints averaging.

### labPerf: calculate the predictions performance by label

   > labPerf true_label.csv prediction.csv output.csv

* Version 0.1

## An example session

The session below has 4 steps:

1. calculate GAP of the first model prediction. 
2. calculate GAP of the second prediction. 
3. add the two predictions with weights 0.4, 0.6. 
4. calculate GAP of the combined prediction.
5. calculate the by-label performance of a prediction on validate data set.

```
$ calcGAP ../input/validate_labels.csv ../predictions/validate.MoNN2L_d12K_nl2_70020.csv
2017-04-26.09:57:26: Starting...
2017-04-26.09:58:06: Read file #1: ../predictions/validate.MoNN2L_d12K_nl2_70020.csv
2017-04-26.09:58:10: Read file #2: ../input/validate_labels.csv
2017-04-26.09:58:47: Done.
GAP = 0.796602

$ calcGAP ../input/validate_labels.csv ../predictions/validate.MoNN3L_TAB_gl2_35020.csv  
2017-04-26.10:01:15: Starting...
2017-04-26.10:01:57: Read file #1: ../predictions/validate.MoNN3L_TAB_gl2_35020.csv
2017-04-26.10:02:01: Read file #2: ../input/validate_labels.csv
2017-04-26.10:02:40: Done.
GAP = 0.798249

$ addPred ../predictions/validate.MoNN2L_d12K_nl2_70020.csv ../predictions/validate.MoNN3L_TAB_gl2_35020.csv 0.4 0.6 /tmp/comb_v5.csv
2017-04-26.10:15:59: Starting...
2017-04-26.10:16:40: Read file #1: ../predictions/validate.MoNN2L_d12K_nl2_70020.csv
2017-04-26.10:17:21: Read file #2: ../predictions/validate.MoNN3L_TAB_gl2_35020.csv
2017-04-26.10:17:29: Pred vecs sorted...
2017-04-26.10:18:07: Added two predictions with (w1, w2) = (0.4, 0.6)
2017-04-26.10:18:43: Saved data to /tmp/comb_v5.csv

$ calcGAP ../input/validate_labels.csv /tmp/comb_v5.csv  
2017-04-26.10:19:39: Starting...
2017-04-26.10:20:20: Read file #1: /tmp/comb_v5.csv
2017-04-26.10:20:25: Read file #2: ../input/validate_labels.csv
2017-04-26.10:21:02: Done.
GAP = 0.803114

$ labPerf ../input/validate_labels.csv ../predictions/validate.MoNN3L_TAB_gl2_35020.csv  
2017-05-02.23:49:08: Starting...
2017-05-02.23:49:47: Read prediction file: ../predictions/validate.MoNN3L_TAB_gl2_35020.csv
2017-05-02.23:49:52: Read true label file: ../input/validate_labels.csv
2017-05-02.23:50:23: Done. Saved to /tmp/label_perf_output.csv
```