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

"""Provides readers configured for different datasets."""

import tensorflow as tf
import utils
import re
from tensorflow import logging
def resize_axis(tensor, axis, new_size, fill_value=0):
  """Truncates or pads a tensor to new_size on on a given axis.

  Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
  size increases, the padding will be performed at the end, using fill_value.

  Args:
    tensor: The tensor to be resized.
    axis: An integer representing the dimension to be sliced.
    new_size: An integer or 0d tensor representing the new value for
      tensor.shape[axis].
    fill_value: Value to use to fill any new entries in the tensor. Will be
      cast to the type of tensor.

  Returns:
    The resized tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  shape = tf.unstack(tf.shape(tensor))

  pad_shape = shape[:]
  pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

  shape[axis] = tf.minimum(shape[axis], new_size)
  shape = tf.stack(shape)

  resized = tf.concat([
      tf.slice(tensor, tf.zeros_like(shape), shape),
      tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
  ], axis)

  # Update shape.
  new_shape = tensor.get_shape().as_list()  # A copy is being made.
  new_shape[axis] = new_size
  resized.set_shape(new_shape)
  return resized

class BaseReader(object):
  """Inherit from this class when implementing new readers."""

  def prepare_reader(self, unused_filename_queue):
    """Create a thread for generating prediction and label tensors."""
    raise NotImplementedError()


class YT8MAggregatedFeatureReader(BaseReader):
  """Reads TFRecords of pre-aggregated Examples.

  The TFRecords must contain Examples with a sparse int64 'labels' feature and
  a fixed length float32 feature, obtained from the features in 'feature_name'.
  The float features are assumed to be an average of dequantized values.
  """
  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["mean_inc3"],
               feature_calcs="",
               feature_remove="",
               decode_zlib=True):
    """Construct a YT8MAggregatedFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
    """
    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))
    new_feature_names = None
    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.decode_zlib = decode_zlib
    self.feature_remove = feature_remove.replace(' ','').split(',')
    #
    # New features' names
    #
    new_feature_names = []
    if feature_calcs != "":
       new_feature_names = [x.replace(' ','') for x in feature_calcs.split(',')]
    #
    # Determine new features' sizes
    #
    new_feature_sizes = [] 
    if feature_calcs != "":
       for feat in new_feature_names:
           if re.findall('audio$', feat) != []:
              new_feature_sizes = new_feature_sizes + [128]
           elif re.findall('rgb$', feat) != []:
              new_feature_sizes = new_feature_sizes + [1024]
           elif feat[:14] == 'c_interaction_' or feat[:7] == 'c_diff_':
              x = -1
              for g in re.findall('(\d+):(\d+)',feat):
                  if x < int(g[1])-int(g[0]):
                     x = int(g[1])-int(g[0])
              new_feature_sizes = new_feature_sizes + [x]
    #
    # Update old with new
    #
    if new_feature_sizes != []:
       self.feature_sizes = self.feature_sizes + new_feature_sizes
    if new_feature_names != []:
       self.feature_names = self.feature_names + new_feature_names
    #
    # Remove features
    #
    #if feature_remove != '':
    #  for feat in feature_remove.replace(' ','').split(','):
    #    i = self.feature_names.index(feat)
    #    print(' removing: ' + str(self.feature_names[i]))
    #    del self.feature_names[i]
    #    del self.feature_sizes[i] 

    print('Identified features: ' + str(len(self.feature_names)) + " | " + str(self.feature_names))
    print('            lengths: ' + str(len(self.feature_sizes)) + " | " + str(self.feature_sizes))
    print('            removed: ' + str(len(self.feature_remove))+ " | " + str(self.feature_remove))

  def prepare_reader(self, filename_queue, batch_size=1024):
    """Creates a single reader thread for pre-aggregated YouTube 8M Examples.

    Args:
      filename_queue: A tensorflow queue of filename locations.

    Returns:
      A tuple of video indexes, features, labels, and padding data.
    """
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    if self.decode_zlib:
      reader = tf.TFRecordReader(options=opts)
    else:
      reader = tf.TFRecordReader()
    _, serialized_examples = reader.read_up_to(filename_queue, batch_size)

    tf.add_to_collection("serialized_examples", serialized_examples)
    return self.prepare_serialized_examples(serialized_examples)

  def prepare_serialized_examples(self, serialized_examples):
    # set the mapping from the fields to data types in the proto
    num_features = len(self.feature_names)
    assert num_features > 0, "self.feature_names is empty!"
    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    # 2017-04-28 - now num_frames is float so I had to change it from int64
    feature_map = {"video_id": tf.FixedLenFeature([], tf.string),
                   "labels": tf.VarLenFeature(tf.int64), 
                   "num_frames": tf.VarLenFeature(tf.float32)}
    
    for feature_index in range(num_features):
      if self.feature_names[feature_index][:2] != 'c_':
          feature_map[self.feature_names[feature_index]] = tf.FixedLenFeature(
              [self.feature_sizes[feature_index]], tf.float32)

    #ssert False, feature_map
    features = tf.parse_example(serialized_examples, features=feature_map)
    labels = tf.sparse_to_indicator(features["labels"], 4716)
    labels.set_shape([None, 4716])
    labels = resize_axis(labels, 1, self.num_classes)
    
    #assert False, features['mean_rgb']
    #assert False, tf.slice(features['mean_rgb'],[?,0],[?,self.num_interactions])
    #assert False, self.feature_names
    for feature_name in self.feature_names:
         if   feature_name[:5] == 'c_sq_':
             features[feature_name] = features[feature_name[5:]] * features[feature_name[5:]]
         elif feature_name[:6] == 'c_log_':
             features[feature_name] = tf.log1p(tf.abs(features[feature_name[6:]]))
         elif feature_name[:6] == 'c_inv_':
             features[feature_name] = 1/features[feature_name[6:]]
         elif feature_name[:6] == 'c_abs_':
             features[feature_name] = tf.abs(features[feature_name[6:]])
         elif feature_name[:6] == 'c_sin_':
             features[feature_name] = tf.sin(features[feature_name[6:]])
         elif feature_name[:6] == 'c_cos_':
             features[feature_name] = tf.cos(features[feature_name[6:]])
         elif feature_name[:7] == 'c_sqrt_':
             features[feature_name] = tf.sqrt(features[feature_name[7:]])
         elif feature_name[:8] == 'c_rsqrt_':
             features[feature_name] = tf.rsqrt(features[feature_name[8:]])
         elif feature_name[:7] == 'c_diff_':
             x = re.findall('c_diff_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)', feature_name)[0]
             feat_1 = x[0]+'_'+x[1]
             feat_2 = x[3]+'_'+x[4]
             y_1 = [int(y) for y in x[2].split(':')]
             y_2 = [int(y) for y in x[5].split(':')]
             features[feature_name] = tf.subtract(features[feat_1][:,y_1[0]:y_1[1]], features[feat_2][:,y_2[0]:y_2[1]])
         elif feature_name[:8] == 'c_over_':
             x = re.findall('c_over_([^_]+)_([^_]+)_(.+)', feature_name)[0]
             feat_1 = x[0]+'_'+x[2]
             feat_2 = x[1]+'_'+x[2]
             features[feature_name] = tf.divide(features[feat_1], features[feat_2])
         elif feature_name[:14] == 'c_interaction_':
             # example: c_interaction_mean_rgb_0:128_mean_audio_0:128
             #          that is mean_rgb*mean_audio for the first 128 coordinates of both of them
             x = re.findall('c_interaction_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)_([^_]+)', feature_name)[0]
             feat_1 = x[0]+'_'+x[1]
             feat_2 = x[3]+'_'+x[4]             
             y_1 = [int(y) for y in x[2].split(':')]
             y_2 = [int(y) for y in x[5].split(':')]
             features[feature_name] = tf.multiply(features[feat_1][:,y_1[0]:y_1[1]], features[feat_2][:,y_2[0]:y_2[1]])
         #elif feature_name == 'num_frames':
         #    features[feature_name] = tf.cast(features[feature_name], tf.float32)

    #assert False, features
    #assert False, [self.feature_sizes, self.feature_names, features]
    #if self.feature_remove != '':
    #  for feat in self.feature_remove.replace(' ','').split(','):
    #    i = self.feature_names.index(feat)
    #    print(' removing: ' + str(self.feature_names[i]))
    #    del self.feature_names[i]
    #    del self.feature_sizes[i]
    #    del features[self.feature_names[i]]

    concatenated_features = tf.concat([
        features[feature_name] for feature_name in self.feature_names if feature_name not in self.feature_remove]
        , 1)
    return features["video_id"], concatenated_features, labels, tf.ones([tf.shape(serialized_examples)[0]])

class YT8MFrameFeatureReader(BaseReader):
  """Reads TFRecords of SequenceExamples.

  The TFRecords must contain SequenceExamples with the sparse in64 'labels'
  context feature and a fixed length byte-quantized feature vector, obtained
  from the features in 'feature_names'. The quantized features will be mapped
  back into a range between min_quantized_value and max_quantized_value.
  """

  def __init__(self,
               num_classes=4716,
               feature_sizes=[1024],
               feature_names=["inc3"],
               max_frames=300):
    """Construct a YT8MFrameFeatureReader.

    Args:
      num_classes: a positive integer for the number of classes.
      feature_sizes: positive integer(s) for the feature dimensions as a list.
      feature_names: the feature name(s) in the tensorflow record as a list.
      max_frames: the maximum number of frames to process.
    """

    assert len(feature_names) == len(feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(feature_names), len(feature_sizes))

    self.num_classes = num_classes
    self.feature_sizes = feature_sizes
    self.feature_names = feature_names
    self.max_frames = max_frames

  def get_video_matrix(self,
                       features,
                       feature_size,
                       max_frames,
                       max_quantized_value,
                       min_quantized_value):
    """Decodes features from an input string and quantizes it.

    Args:
      features: raw feature values
      feature_size: length of each frame feature vector
      max_frames: number of frames (rows) in the output feature_matrix
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      feature_matrix: matrix of all frame-features
      num_frames: number of frames in the sequence
    """
    decoded_features = tf.reshape(
        tf.cast(tf.decode_raw(features, tf.uint8), tf.float32),
        [-1, feature_size])

    num_frames = tf.minimum(tf.shape(decoded_features)[0], max_frames)
    feature_matrix = utils.Dequantize(decoded_features,
                                      max_quantized_value,
                                      min_quantized_value)
    feature_matrix = resize_axis(feature_matrix, 0, max_frames)
    return feature_matrix, num_frames

  def prepare_reader(self,
                     filename_queue,
                     max_quantized_value=2,
                     min_quantized_value=-2):
    """Creates a single reader thread for YouTube8M SequenceExamples.

    Args:
      filename_queue: A tensorflow queue of filename locations.
      max_quantized_value: the maximum of the quantized value.
      min_quantized_value: the minimum of the quantized value.

    Returns:
      A tuple of video indexes, video features, labels, and padding data.
    """
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    return self.prepare_serialized_examples(serialized_example,
        max_quantized_value, min_quantized_value)

  def prepare_serialized_examples(self, serialized_example,
      max_quantized_value=2, min_quantized_value=-2):

    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"video_id": tf.FixedLenFeature(
            [], tf.string),
                          "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name : tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })

    # read ground truth labels
    labels = (tf.cast(
        tf.sparse_to_dense(contexts["labels"].values, (4716,), 1,
            validate_indices=False),
        tf.bool))

    # loads (potentially) different types of features and concatenates them
    num_features = len(self.feature_names)
    assert num_features > 0, "No feature selected: feature_names is empty!"

    assert len(self.feature_names) == len(self.feature_sizes), \
    "length of feature_names (={}) != length of feature_sizes (={})".format( \
    len(self.feature_names), len(self.feature_sizes))

    num_frames = -1  # the number of frames in the video
    feature_matrices = [None] * num_features  # an array of different features
    for feature_index in range(num_features):
      feature_matrix, num_frames_in_this_feature = self.get_video_matrix(
          features[self.feature_names[feature_index]],
          self.feature_sizes[feature_index],
          self.max_frames,
          max_quantized_value,
          min_quantized_value)
      if num_frames == -1:
        num_frames = num_frames_in_this_feature
      else:
        tf.assert_equal(num_frames, num_frames_in_this_feature)

      feature_matrices[feature_index] = feature_matrix

    # cap the number of frames at self.max_frames
    num_frames = tf.minimum(num_frames, self.max_frames)

    # concatenate different features
    video_matrix = tf.concat(feature_matrices, 1)

    # convert to batch format.
    # TODO: Do proper batch reads to remove the IO bottleneck.
    batch_video_ids = tf.expand_dims(contexts["video_id"], 0)
    batch_video_matrix = tf.expand_dims(video_matrix, 0)
    batch_labels = tf.expand_dims(labels, 0)
    batch_frames = tf.expand_dims(num_frames, 0)
    #return batch_video_ids, batch_video_matrix, batch_labels, batch_frames
    return batch_video_ids, batch_video_matrix,  batch_labels, batch_frames

