import datetime
import numpy as np
import tensorflow as tf

def mylog(s): 
  print('{:%Y-%m-%d %H:%M:%S}: {}'.format(datetime.datetime.now(), s))

#%% tensorflow features

def _byteslist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#%% top_k matrix way

def top_k_along_row(arr, k, ordered=True):
    """ top k of a 2d np.array, along the rows    
    http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array/18691983
    """    
    assert k>0, "top_k_along_row/column() requires k>0."
    if ordered:
        indices = np.argsort(arr, axis=1)[:,-k:]
    else:
        indices = np.argpartition(arr, -k, axis=1)[:, -k:]        
    x = arr.shape[0]
    return arr[np.repeat(np.arange(x), k), indices.ravel()].reshape(x, k)

def top_k_along_column(arr, k, ordered=True):
    return top_k_along_row(arr.T, k, ordered).T

def bottom_top_k_along_row(arr, k, ordered=True):
    """ bottom and top k of a 2d np.array, along the rows    
    http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array/18691983
    """    
    assert k>0, "bottom_top_k_along_row/column() requires k>0."
    rows = arr.shape[0]
    if ordered:
        tmp = np.argsort(arr, axis=1)      
        idx_bot = tmp[:, :k]
        idx_top = tmp[:,-k:]
    else:
        idx_bot = np.argpartition(arr, k, axis=1)[:,:k]
        idx_top = np.argpartition(arr, -k, axis=1)[:,-k:]
    
    indices = np.concatenate((idx_bot, idx_top), axis=1)
    vals = arr[np.repeat(np.arange(rows), 2*k), indices.ravel()].reshape(rows,2*k)
    return vals, indices

def bottom_top_k_along_column(arr, k, ordered=True):
    val, idx =  bottom_top_k_along_row(arr.T, k, ordered)
    return val.T, idx.T