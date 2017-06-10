#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import numpy as np
import tensorflow as tf
from tensorflow import flags
from tensorflow import gfile
from tensorflow import app
from tensorflow import logging

import utils
from my_utils import mylog
import my_utils

#%%
FLAGS = flags.FLAGS
if __name__ == "__main__":
  flags.DEFINE_string("input_data_pattern",
        "YouTube.Kaggle/input/GENERATED_DATA/f2train/*.tfrecord",
        "files to process")
  flags.DEFINE_string("output_path","/tmp/",
        "Path for generated data.")
  flags.DEFINE_integer("file_from", 11, "start from, eg., the 11th file")
  flags.DEFINE_integer("file_to",   15, "process 15 - 11 files")
  flags.DEFINE_bool("skip_shorts", False, "video with 1 frame")

#%% scribble for splitting a frame level example
def frame_example_2_np(seq_example_bytes, 
                       max_quantized_value=2,
                       min_quantized_value=-2):
  feature_names=['rgb','audio']
  feature_sizes = [1024, 128]
  with tf.Graph().as_default():
    contexts, features = tf.parse_single_sequence_example(
        seq_example_bytes,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in feature_names
        })

    decoded_features = { name: tf.reshape(
        tf.cast(tf.decode_raw(features[name], tf.uint8), tf.float32),
        [-1, size]) for name, size in zip(feature_names, feature_sizes)
        }
    feature_matrices = {
        name: utils.Dequantize(decoded_features[name],
          max_quantized_value, min_quantized_value) for name in feature_names}
        
    with tf.Session() as sess:
      vid = sess.run(contexts['video_id'])
      labs = sess.run(contexts['labels'].values)
      rgb = sess.run(feature_matrices['rgb'])
      audio = sess.run(feature_matrices['audio'])
      
  return vid, labs, rgb, audio

       
#%% Split frame level file into three video level files: all, 1st half, 2nd half.
def fExample_2_vExamples(fex):
  vid,labs,rgb,audio = frame_example_2_np(fex)
  nframes = audio.shape[0]
  
  if nframes < 10: #ignore short videos, let the called handle None's
      return None, None
  
  half = np.int(audio.shape[0]/2)
  
  return np_2_vExample(vid, labs, rgb[:half], audio[:half]), \
         np_2_vExample(vid, labs, rgb[half:], audio[half:])
  
def np_2_vExample(vid, labs, rgb, audio):
  nframes = audio.shape[0]

  if False:
      # top 5
      k = 5
      if nframes > 10:
        tk_rgb = my_utils.top_k_along_column(rgb, k)
        tk_audio = my_utils.top_k_along_column(audio, k)
      else:
        tk_rgb = np.repeat(rgb[0].reshape([1, rgb.shape[1]]), k, axis=0)
        tk_audio = np.repeat(audio[0].reshape([1, audio.shape[1]]), k, axis=0)      
              
  # std of all rgb or audio entries
  s_rgb = np.std(rgb)
  s_aud = np.std(audio)

  rgb_sq = rgb * rgb
  aud_sq = audio * audio

  vExample  = tf.train.Example(features=tf.train.Features(feature={
     'video_id':     my_utils._byteslist_feature([vid]),
     'labels':       my_utils._int64list_feature(labs),
     'mean_rgb':     my_utils._floatlist_feature(np.mean(rgb, axis=0)),
     'mean_audio':   my_utils._floatlist_feature(np.mean(audio, axis=0)),
     'std_rgb':      my_utils._floatlist_feature(np.std(rgb, axis=0)),
     'std_audio':    my_utils._floatlist_feature(np.std(audio, axis=0)),
     'x3_rgb':       my_utils._floatlist_feature(np.cbrt(np.mean(rgb_sq * rgb, axis=0))),
     'x3_audio':     my_utils._floatlist_feature(np.cbrt(np.mean(aud_sq * audio, axis=0))),
     'num_frames':   my_utils._floatlist_feature([(nframes-151.)/300.]),
     'std_all_rgb':    my_utils._floatlist_feature([s_rgb]),
     'std_all_audio':  my_utils._floatlist_feature([s_aud])
     }))      
      
     #'top_1_rgb':    my_utils._floatlist_feature(tk_rgb[-1]),
     #'top_3_rgb':    my_utils._floatlist_feature(tk_rgb[-3]),
     #'top_5_rgb':    my_utils._floatlist_feature(tk_rgb[-5]),
     #'top_1_audio':  my_utils._floatlist_feature(tk_audio[-1]),
     #'top_3_audio':  my_utils._floatlist_feature(tk_audio[-3]),
     #'top_5_audio':  my_utils._floatlist_feature(tk_audio[-5]),

  return vExample

#%%
def build_graph():
    feature_names=['rgb','audio']
    feature_sizes = [1024, 128] 
    max_quantized_value=2
    min_quantized_value=-2

    seq_example_bytes = tf.placeholder(tf.string)
    contexts, features = tf.parse_single_sequence_example(
        seq_example_bytes,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in feature_names
        })

    decoded_features = { name: tf.reshape(
        tf.cast(tf.decode_raw(features[name], tf.uint8), tf.float32),
        [-1, size]) for name, size in zip(feature_names, feature_sizes)
        }
    feature_matrices = {
        name: utils.Dequantize(decoded_features[name],
          max_quantized_value, min_quantized_value) for name in feature_names}
        
    tf.add_to_collection("vid_tsr", contexts['video_id'])
    tf.add_to_collection("labs_tsr", contexts['labels'].values)
    tf.add_to_collection("rgb_tsr", feature_matrices['rgb'])
    tf.add_to_collection("audio_tsr", feature_matrices['audio'])
    tf.add_to_collection("seq_example_bytes", seq_example_bytes)
 
#   with tf.Session() as sess:
#       writer = tf.summary.FileWriter('./graphs', sess.graph)

def split_files(filenames):
    
    t0 = time.time()
    with tf.Session() as sess:
      vid_tsr = tf.get_collection("vid_tsr")[0]
      labs_tsr = tf.get_collection("labs_tsr")[0]
      rgb_tsr = tf.get_collection("rgb_tsr")[0]
      audio_tsr = tf.get_collection("audio_tsr")[0]
      seq_example_bytes = tf.get_collection("seq_example_bytes")[0]

      for k, file_grp in enumerate(filenames):
          start_time = time.time()
          infn, outfn0, outfn1, outfn2 = file_grp

          ex_iter = tf.python_io.tf_record_iterator(infn)
          v0Examples = []
          v1Examples = []
          v2Examples = []
          for in_ex in ex_iter:
              vid,labs,rgb,audio = sess.run([vid_tsr, labs_tsr, rgb_tsr, audio_tsr],
                feed_dict = {seq_example_bytes: in_ex} ) 

              nframes = audio.shape[0]
              half = np.int(nframes/2)

              if nframes > 10: 
                  rgb_1 = rgb[:half]
                  rgb_2 = rgb[half:]
                  audio_1 = audio[:half]
                  audio_2 = audio[half:]
              else: #ignore short videos, let the called handle None's
                  mylog("One frame video encountered: {}, num_frames: {}, labels: {}".format(
                           vid, nframes, labs) )
                  rgb_1 = rgb
                  rgb_2 = rgb
                  audio_1 = audio
                  audio_2 = audio
                  if FLAGS.skip_shorts:
                      continue
  
              #try:
              v0Examples.append(np_2_vExample(vid, labs, rgb, audio))
              v1Examples.append(np_2_vExample(vid, labs, rgb_1, audio_1))
              v2Examples.append(np_2_vExample(vid, labs, rgb_2, audio_2))
              #except:
              #  mylog("failed. nframes: {}, rgb shape: {}".format(nframes, rgb.shape))
          opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
          with tf.python_io.TFRecordWriter(outfn0, options=opts) as tfwriter0: 
              for v0_ex in v0Examples:
                  tfwriter0.write(v0_ex.SerializeToString())
          with tf.python_io.TFRecordWriter(outfn1, options=opts) as tfwriter1: 
              for v1_ex in v1Examples:
                  tfwriter1.write(v1_ex.SerializeToString())
          with tf.python_io.TFRecordWriter(outfn2, options=opts) as tfwriter2: 
              for v2_ex in v2Examples:
                  tfwriter2.write(v2_ex.SerializeToString())
            
          seconds_per_file = time.time() - start_time
          num_examples_per_sec = len(v1Examples) / seconds_per_file
    
          mylog("Processed file number {} in {:.2f} sec: {}, Examples: {}, Examples/second: {:.0f}.".format(
            k, seconds_per_file, infn.split('/')[-1], len(v1Examples), num_examples_per_sec))

    ttl_time = time.time() - t0
    mylog("Processed total {} files in {:.2f}.".format(k, ttl_time))
    return k

if False: #testing
  filename = 'YouTube.Kaggle/input/frame_level/train-1.tfrecord'
  ex_iter = tf.python_io.tf_record_iterator(filename)
  in_ex_bytes = next(ex_iter)
  vid, labs, rgb, audio = frame_example_2_np(in_ex_bytes)
  
#%%
def main(unused_argv):

  logging.set_verbosity(tf.logging.ERROR)
  print("tensorflow version: %s" % tf.__version__)

  all_frame_files = gfile.Glob(FLAGS.input_data_pattern)
  f_fullpath = all_frame_files[FLAGS.file_from : FLAGS.file_to]
  f_fns = [x.split('/')[-1] for x in f_fullpath]

  exist_files = gfile.Glob(FLAGS.output_path + "E*tfrecord")
  exist_fn = [x.split('/')[-1].replace('Etr', 'tr')  for x in exist_files]
  exist_fn = [x.split('/')[-1].replace('Eval', 'val')  for x in exist_fn]
  exist_fn = [x.split('/')[-1].replace('Etes', 'tes')  for x in exist_fn]

  yet_2_split = [x for x,y in zip(f_fullpath, f_fns) if y not in exist_fn]

  vf0 = [FLAGS.output_path + 'O' + x.split('/')[-1] for x in yet_2_split]
  vf1 = [FLAGS.output_path + 'E' + x.split('/')[-1] for x in yet_2_split]
  vf2 = [FLAGS.output_path + 'F' + x.split('/')[-1] for x in yet_2_split]
  
  mylog('number of files suggested: %d'%len(f_fullpath))
  mylog('number of files yet to process: %d'%len(yet_2_split))
      
  #with tf.device("/gpu:0"):
  with tf.Graph().as_default():
      build_graph()
      split_files(zip(yet_2_split, vf0, vf1, vf2))
  mylog("done")

if __name__ == "__main__":
    app.run()
