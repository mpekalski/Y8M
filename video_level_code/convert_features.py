#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import app

from my_utils import mylog
import my_utils

#%%
FLAGS = flags.FLAGS
if __name__ == "__main__":
  flags.DEFINE_string("input_data_pattern",
        "input/GENERATED_DATA/f2train/*.tfrecord",
        "files to process")
  flags.DEFINE_string("output_path","/tmp/", "Path for generated data.")
  flags.DEFINE_integer("file_from", 11, "start from, eg., the 11th file")
  flags.DEFINE_integer("file_to",   15, "process 15 - 11 files")
  flags.DEFINE_bool("parallel", True, "parallel processing")
  flags.DEFINE_string("feature_names",
        "video_id,labels,mean_rgb,mean_audio,num_frames,std_rgb,std_audio",
        "features to pick")
  
#%%
# basic operation example
if False:
  filename = 'input/GENERATED_DATA/f2train/Atrain__.tfrecord'
  opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
      
  ex_iter = tf.python_io.tf_record_iterator(filename, options=opts)
  example = next(ex_iter)
  in_ex = tf.train.Example.FromString(example)

  mean_rgb = in_ex.features.feature['mean_rgb'].float_list.value
  median_rgb = in_ex.features.feature['median_rgb'].float_list.value  
  skew_rgb = np.array(mean_rgb) - np.array(median_rgb)
  
  mean_audio = in_ex.features.feature['mean_audio'].float_list.value
  median_audio = in_ex.features.feature['median_audio'].float_list.value  
  skew_audio = np.array(mean_audio) - np.array(median_audio)
  
  out_ex = tf.train.Example(features=tf.train.Features(feature={
             'video_id':     in_ex.features.feature['video_id'],
             'labels':       in_ex.features.feature['labels'],
             'mean_rgb':     in_ex.features.feature['mean_rgb'],
             'mean_audio':   in_ex.features.feature['mean_audio'],
             'skew_rgb':     my_utils._floatlist_feature(skew_rgb),
             'skew_audio':   my_utils._floatlist_feature(skew_audio),
             'std_rgb':      in_ex.features.feature['std_rgb'],
             'std_audio':    in_ex.features.feature['std_audio'],
             'top_1_rgb':    in_ex.features.feature['top_1_rgb'],
             'top_2_rgb':    in_ex.features.feature['top_2_rgb'],
             'top_3_rgb':    in_ex.features.feature['top_3_rgb'],
             'top_4_rgb':    in_ex.features.feature['top_4_rgb'],
             'top_5_rgb':    in_ex.features.feature['top_5_rgb'],                                                   
             'num_frames':   in_ex.features.feature['num_frames'],
             'std_all_rgb':      in_ex.features.feature['std_all_rgb'],
             'std_all_audio':    in_ex.features.feature['std_all_audio']
             } ) )
                                                                
#%%
def select_features_from_tfexample(input_tfexample, feats=None):
    if feats==None:
        feats = ['video_id','labels','mean_rgb','mean_audio','num_frames',
                 'std_rgb','std_audio',
                 'top_1_rgb','top_2_rgb','top_3_rgb','top_4_rgb','top_5_rgb',
                 'top_1_audio','top_2_audio','top_3_audio',
                 'top_4_audio','top_5_aduio']
    feats = set(['video_id','labels'] + feats) # make sure uniqueness and vid/labs

    fdict = { x : input_tfexample.features.feature[x] for x in feats 
             if x not in ['mmdf_rgb', 'mmdf_audio']}
    
    if 'mmdf_rgb' in feats:
        mean_rgb = input_tfexample.features.feature['mean_rgb'].float_list.value
        median_rgb = input_tfexample.features.feature['median_rgb'].float_list.value  
        mmdf_rgb = np.array(mean_rgb) - np.array(median_rgb)
        fdict['mmdf_rgb'] = my_utils._floatlist_feature(mmdf_rgb)
    if 'mmdf_audio' in feats:
        mean_audio = input_tfexample.features.feature['mean_audio'].float_list.value
        median_audio = input_tfexample.features.feature['median_audio'].float_list.value  
        mmdf_audio = np.array(mean_audio) - np.array(median_audio)
        fdict['mmdf_audio'] = my_utils._floatlist_feature(mmdf_audio)
        
    output_tfexample = tf.train.Example(features=tf.train.Features(feature=fdict))
    return output_tfexample

def pick_features_from_file(input_fn, out_fn, feats=None):
    start_time = time.time()
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    ex_iter = tf.python_io.tf_record_iterator(input_fn, options=opts)
    
    num_examples = 0
    with tf.python_io.TFRecordWriter(out_fn, options=opts) as tfwriter:
        out_examples = []
        for input_bytes in ex_iter: #two loops to split read/write operations
            input_example = tf.train.Example.FromString(input_bytes)
            out_examples.append(select_features_from_tfexample(input_example, feats))
        for example in out_examples:
            tfwriter.write(example.SerializeToString())
            num_examples += 1
    
    seconds_per_file = time.time() - start_time
    num_examples_per_sec = num_examples / seconds_per_file
    mylog("Processed in {:.0f} sec: {}, Examples: {}, Examples/second: {:.0f}.".format(
            seconds_per_file,input_fn, num_examples, num_examples_per_sec))

def process_one_file(filenames):
    input_file, output_file = filenames
    feats=None or FLAGS.feature_names.split(',')
    pick_features_from_file(input_file, output_file, feats)
    
def main(unused_argv):

  print("tensorflow version: %s" % tf.__version__)

  all_frame_files = gfile.Glob(FLAGS.input_data_pattern)
  f_fullpath = all_frame_files[FLAGS.file_from : FLAGS.file_to]
  f_fns = [x.split('/')[-1] for x in f_fullpath]

  exist_files = gfile.Glob(FLAGS.output_path + "C*tfrecord")
  exist_fn = [x.split('/')[-1].replace('CAtr', 'Atr')  for x in exist_files]

  yet_2_split = [x for x,y in zip(f_fullpath, f_fns) if y not in exist_fn]

  vf = [FLAGS.output_path + 'C' + x.split('/')[-1] for x in yet_2_split]
  
  mylog('number of files suggested: %d'%len(f_fullpath))
  mylog('number of files yet to process: %d'%len(yet_2_split))
      
  if FLAGS.parallel:
    from concurrent import futures
    executor = futures.ProcessPoolExecutor(max_workers=2)
    executor.map(process_one_file, zip(yet_2_split, vf))
  else: 
    for filenames in zip(yet_2_split, vf):
        #mylog('processing: {}'.format(filenames))
        process_one_file(filenames)

  mylog("done")

if __name__ == "__main__":

    app.run()
