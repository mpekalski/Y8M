# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for generating mean and stdev of all video level examples."""

import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging

import readers
import utils


#%%
FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("input_data_pattern2", "", "Additional data files.")
  flags.DEFINE_string("input_data_pattern3", "", "More data files.")
 
  flags.DEFINE_string("output_file", "",
                      "The file to save the l2 params to.")
  
  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 8192,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "mean_rgb,mean_audio", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024,128", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")

def get_input_data_tensors(
        reader, 
        data_pattern1,
        data_pattern2,
        data_pattern3,
        batch_size, 
        num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files1 = gfile.Glob(data_pattern1)
    files2 = gfile.Glob(data_pattern2)
    files3 = gfile.Glob(data_pattern3)
    files = files1 + files2 + files3
    
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern1 + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch = True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch

def calculate_moments(
        reader,
        feature_names,
        feature_sizes,
        data_pattern1,
        data_pattern2,
        data_pattern3,
        out_file_location, 
        batch_size):
    
  with tf.Session() as sess:
      
    video_id_batch, video_batch, num_frames_batch = get_input_data_tensors(
            reader, data_pattern1, data_pattern2, data_pattern3, batch_size)
    
    feat_sum = tf.Variable(tf.zeros([sum(feature_sizes)]), name="feat_sum", )
    feat_sq_sum = tf.Variable(tf.zeros([sum(feature_sizes)]), name="feat_sq_sum")
    num_examples = tf.Variable(0, dtype=tf.int32, name = "num_examples")

    feat_sum += tf.reduce_sum(video_batch, axis=0)
    feat_sq_sum += tf.reduce_sum( tf.square(video_batch), axis=0)
    num_examples += tf.shape(video_batch)[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        if "train_input" in variable.name:
          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))
    
    sess.run(tf.global_variables_initializer() )

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()

    try:
      fetches = [num_examples, feat_sum, feat_sq_sum, video_batch]
      while not coord.should_stop():
          num_examples_val, feat_sum_val, feat_sq_sum_val, video_batch_val = sess.run(fetches)
          now = time.time()
          num_examples_processed += len(video_batch_val)
          logging.info("num examples processed: " + str(num_examples_processed)
               + " elapsed seconds: " + "{0:.2f}".format(now-start_time))

    except tf.errors.OutOfRangeError:
        logging.info('Done with summation. num_examples = {}.'.format(num_examples_processed))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    
    global_mean = feat_sum_val / num_examples_val
    global_std  = np.sqrt(feat_sq_sum_val / num_examples_val - global_mean * global_mean)
    
    res = pd.DataFrame({'global_mean':global_mean, 'global_std':global_std})
    res.to_csv(out_file_location)


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  calculate_moments(reader, 
                    feature_names,
                    feature_sizes,
                    FLAGS.input_data_pattern, 
                    FLAGS.input_data_pattern2, 
                    FLAGS.input_data_pattern3,
                    FLAGS.output_file,
                    FLAGS.batch_size)


if __name__ == "__main__":
  app.run()
