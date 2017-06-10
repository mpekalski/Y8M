This is a bit older version of the code, but it includes a boosting part. Basically with additional NN we tried to correct for errors in predictions we have made so far in the main network.

#Add a boosting stage

Add a second layer model, bst_model, to fit the misses of the first model prediction:

 bst_target =  target - first_model_predictions

## train.py 
1. bst_model
2. bst_optimizer

bst_model works on the misses of the first layer model prediction:
   target - prediction_by_mdoel

## losses.py
Added an L2Loss to take into account nature of the bst_model 

## xp_video_level_model.py 
Sigmoid does not fit as the final activation for bst_model, whose target is in [-1.0, 1.0].

## thoughts for testing:
1. Using different features for the models. Rationale: If the first layer model already digested the input data well, there won't be much for the bst_model to add. If different data is used, there's a better chance for the bst_model to add.
