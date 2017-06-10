#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
model_dir = '/media/m/3TB/kaggle/google/model/MoNN2Lw_f1/'
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
checkpoint_path = os.path.join(model_dir, "model.ckpt-50")

print(checkpoint_path)
# List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=True)

# List contents of v0 tensor.
# Example output: tensor_name:  v0 [[[[  9.27958265e-02   7.40226209e-02   4.52989563e-02   3.15700471e-02
print_tensors_in_checkpoint_file(file_name=checkpoint_path, 
    tensor_name='tower/Logi/weights',
    all_tensors=False)

print_tensors_in_checkpoint_file(file_name=checkpoint_path, 
    tensor_name='tower/tower/Logi/weights/ExponentialMovingAverage',
    all_tensors=False)


#%%

from tensorflow.python import pywrap_tensorflow
checkpoint_path = os.path.join(model_dir, "model.ckpt-50")
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in var_to_shape_map:
    print("tensor_name: ", key)
    #print(reader.get_tensor(key)) # Remove this is you want to print only variable names
    
xm2 = reader.get_tensor('tower/tower/Logi/weights/ExponentialMovingAverage')
x02 = reader.get_tensor('tower/Logi/weights')
