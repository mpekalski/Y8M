#opyright 2016 Google Inc. All Rights Reserved.
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
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time
import numpy as np
import eval_util
import export_model
import losses
import frame_level_models
import video_level_models
import xp_frame_level_models
import xp_video_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import utils
import model_utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  # There are three data pattern variables in case data were scattered across 
  # multiple hard-drives. On single machine it helps with IO.
  flags.DEFINE_string("train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("train_data_pattern2", "",
      "additional training dataset.")
  flags.DEFINE_string("train_data_pattern3", "",
      "additional training dataset.")
  flags.DEFINE_string("eval_data_pattern", "",
      "File glob for the evaluation dataset.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                                                   "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")
  # Vector defining the probabilities of keeping a neuron in dropout. Set max to 10.
  flags.DEFINE_string("layers_keep_probs", "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0", "List of parameters for dropout")
  # Fold works on file level, not observation/example level.
  flags.DEFINE_integer("fold", -1, "fold, take four of every five files,"
                                   "starting from FLAGS.fold.")
  # Model flags.
  # Global normalization is read froma file, see  model_utils.load_global_moments
  flags.DEFINE_bool("apply_global_normalization", True,
      "By default, apply l2_normalization with calculated global mean/std.")
  flags.DEFINE_bool("apply_batch_l2_normalization", True, "By default, apply l2 batch normalization")
  flags.DEFINE_bool("frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string("model", "LogisticModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool("start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")
  flags.DEFINE_float("l2_penalty", 1e-8,"what should be l2 penalty")
  flags.DEFINE_string("c_vars", "", "A list of variables to compute."
                                  " Available transformations sq_a log_a inv_a abs_a sqrt_a diff_a_b interaction__a_b") 
  flags.DEFINE_string("r_vars", "", "A list of variables to be removed and not passed to the model")
  flags.DEFINE_float("loss_epsilon", 1e-6, "log(x+epsilon)")
  # Training flags.
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float("regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
  # In case you want to restart learning but forcing different learning rate.
  flags.DEFINE_float("restart_learning_rate", -100.0,
                     "learning rate when restart, ignored if less than zero.")
  flags.DEFINE_float("learning_rate_decay", 0.95,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
  flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
  flags.DEFINE_integer("num_epochs", 5,
                       "How many passes to make over the dataset before "
                       "halting training.")
  # ema = Exponential Moving Average
  flags.DEFINE_float("ema_halflife", 3000.0, "halflife time/steps")
  flags.DEFINE_bool("use_ema",False,"Set to True is you want to use EMA variables")
  flags.DEFINE_integer("max_steps", None,
                       "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer("export_model_steps", 5000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")
  flags.DEFINE_integer("save_model_minutes", 30,
                       "How many passes to make over the dataset before "
                       "halting training.")
  # You may want to force running on cpu istead of gpu
  flags.DEFINE_integer("gpu_only", 1, "0 if CPU only, 1 if GPU.")
  # When trainig really big models you may run out of memory on GPU to export model, 
  # you may force export to run on cpu. 
  flags.DEFINE_integer("model_export_gpu_only",1,"0 if CPU only, 1 if GPU, for saving model")
  # Other flags.
  flags.DEFINE_integer("truncated_num_classes", 4716,
                       "Number of classes to train for.")
  # If you want to pass compressed files. Helps with IO.
  flags.DEFINE_bool("decode_zlib", True,
      		    "Whether or not the data files are zlib compressed.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool("log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_files,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    logging.info("Fold number {}, use files {}.".format( FLAGS.fold,
        len(data_files)))
    filename_queue = tf.train.string_input_producer(
        data_files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def build_graph(reader,
                model,
                train_data_pattern,
                train_data_pattern2,
                train_data_pattern3,
                eval_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None, 
                l2_penalty=1e-8,
		gpu_only=1
                ):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """
  # data files
  files1 = gfile.Glob(train_data_pattern)
  files2 = gfile.Glob(train_data_pattern2)
  files3 = gfile.Glob(train_data_pattern3)
  files = files1 + files2 + files3
  if not files:
    raise IOError("Unable to find training files. data_pattern='" +
                  data_pattern + "'.")
  logging.info("Total number of training files: %s + %s + %s =  %s.",
                 str(len(files1)), str(len(files2)), str(len(files3)), str(len(files)))

  files4 = gfile.Glob(eval_data_pattern)
  logging.info("Total number of eval files: %s.", str(len(files4)))

  if FLAGS.fold == -1:
    validate_files = files4
    train_files = files
  else:
    validate_files = files[FLAGS.fold::5]
    train_files = [x for x in files if x not in validate_files]

  logging.info("train files: {}, first is: {}.".format(len(train_files),
      train_files[0].split('/')[-1]))
  logging.info("eval files: {}, first is: {}.".format(len(validate_files),
      validate_files[0].split('/')[-1]))

  # label weights for loss function. ugly hard coded for now.
  wgts_np = np.ones(FLAGS.truncated_num_classes)
  over_weight_labels = False
  if over_weight_labels:
      labels_to_overwgt = [38,47,49,55,72,76,86,89,93,94,95,98,99,101,102,110,111,113,114,115,120,121]
      wgts_np[labels_to_overwgt] = 2.0
  wgts_4_lossfn= tf.constant(wgts_np, dtype=tf.float32)

  global_step = tf.Variable(0, trainable=False, name="global_step")
  restart_learning_rate = tf.Variable(
    base_learning_rate, trainable=False, name="restart_learning_rate")
  
  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  num_gpus = len(gpus)
  
  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'

  learning_rate = tf.train.exponential_decay(
      restart_learning_rate,
      global_step * batch_size * num_towers,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)
  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_files,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)

  # model params
  # probabilities for keeping a neuron in a layer, assuming max 10 layers, below default value
  with tf.variable_scope("tower", reuse=True) as scope:
       layers_keep_probs = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=False, name="layers_keep_probs")
  model_input = model_input_raw
  if FLAGS.apply_global_normalization:
     g_mean, g_std = model_utils.load_global_moments()
     g_inv_std = 1.0/g_std
     global_mean = tf.constant(g_mean, dtype=tf.float32)
     # expand global mean to match new dimension and fill rest with zeros
     new_dim = tf.cast(model_input_raw.shape[1],tf.int32)
     zero_padding = tf.zeros(new_dim-tf.shape(global_mean), tf.float32)
     global_mean_padded = tf.concat([global_mean, zero_padding],0)
     # expand global inv std to match new dimension and fill rest with ones
     global_inv_std = tf.constant(g_inv_std, dtype=tf.float32)
     one_padding = tf.ones(new_dim - tf.shape(global_inv_std), tf.float32)
     global_inv_std_padded = tf.concat([global_inv_std, one_padding], 0)
     # apply normalizations (can do both) if requested
     # global L2 normalization
     model_input = tf.multiply(tf.subtract(model_input, global_mean_padded), global_inv_std_padded)
  # regular L2 normalization
  if FLAGS.apply_batch_l2_normalization:
     feature_dim = len(model_input.get_shape()) - 1
     model_input = tf.nn.l2_normalize(model_input, feature_dim)
  
  tower_inputs = tf.split(model_input, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_num_frames = tf.split(num_frames, num_towers)
  tower_gradients = []
  tower_predictions = []
  tower_label_losses = []
  tower_reg_losses = []
  
  # eval graph - to monitor performance out of sample during training
  e_video_id, e_input_raw, e_labels_batch, e_num_frames = (
      get_input_data_tensors(
          reader,
          validate_files,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=2 * num_epochs))
  e_input = e_input_raw
  if FLAGS.apply_global_normalization:
    e_input = tf.multiply(tf.subtract(e_input, global_mean_padded),  global_inv_std_padded)
  if FLAGS.apply_batch_l2_normalization:
    feature_dim = len(model_input.get_shape()) - 1
    e_input = tf.nn.l2_normalize(e_input, feature_dim)
  
  e_tower_inputs = tf.split(e_input, num_towers)
  e_tower_labels = tf.split(e_labels_batch, num_towers)
  e_tower_num_frames = tf.split(e_num_frames, num_towers)
  e_tower_predictions = []
  e_tower_layers_keep_probs = tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], trainable=False, name="layers_keep_probs")
  logging.info(e_tower_inputs)
  # end eval
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    logging.info('For tower: ' + str(i))
    with tf.device(device_string % i):
      with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")): 
          logging.info(layers_keep_probs)
          result = model.create_model(
            tower_inputs[i],
            num_frames=tower_num_frames[i],
            vocab_size=reader.num_classes,
            labels=tower_labels[i], 
            layers_keep_probs=layers_keep_probs,
            l2_penalty=l2_penalty,
            is_training=True)
          for variable in slim.get_model_variables():
            logging.info(variable)
            tf.summary.histogram(variable.op.name, variable)

          # create shadow moving average model variables
          if FLAGS.use_ema==True:
            model_vars = [x for x in slim.get_model_variables()]
            ema = tf.train.ExponentialMovingAverage(decay=1.0 - 1.0/FLAGS.ema_halflife)
            ema_op = ema.apply(model_vars)
            logging.info("model_vars:")
            logging.info(" || ".join([str(x) for x in model_vars]))
            ema_vars = [ema.average(x) for x in model_vars]
            ema_vars_pair_dict = { ema.average_name(x) : x.op.name for x in model_vars}
            logging.info("ema_vars_pair_dict:")
            for x, y in ema_vars_pair_dict.items():
              logging.info(x + ': ' + y)
            for v in ema_vars:
              tf.summary.histogram(v.op.name, v)        
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, ema_op)
            tf.add_to_collection("ema_op", ema_op)


          predictions = result["predictions"]
          tower_predictions.append(predictions)

          if "loss" in result.keys():
            label_loss = result["loss"]
          else:
            label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i], FLAGS.loss_epsilon)

          if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
          else:
            reg_loss = tf.constant(0.0)

          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss += tf.add_n(reg_losses)

          tower_reg_losses.append(reg_loss)

          # Adds update_ops (e.g., moving average updates in batch normalization) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if "update_ops" in result.keys():
            update_ops += result["update_ops"]
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

          tower_label_losses.append(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = optimizer.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
          tower_gradients.append(gradients)
          
          # eval ops
          logging.info("eval ops")
          e_result = model.create_model(
            e_tower_inputs[i],
            num_frames=e_tower_num_frames[i],
            vocab_size=reader.num_classes,
            labels=e_tower_labels[i],
            layers_keep_probs=e_tower_layers_keep_probs, #tf.Variable([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], tf.float32, name="layers_keep_probs")
            l2_penalty=l2_penalty,
            is_training=False
           )
          
          e_predictions = e_result["predictions"]
          e_tower_predictions.append(e_predictions)
          # end eval ops
          
  label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
  tf.summary.scalar("label_loss", label_loss)
  if regularization_penalty != 0:
    reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
    tf.summary.scalar("reg_loss", reg_loss)
  merged_gradients = utils.combine_gradients(tower_gradients)

  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

  train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("restart_learning_rate", restart_learning_rate)
  tf.add_to_collection("layers_keep_probs", layers_keep_probs)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("train_op", train_op)
  #tf.add_to_collection("ema_op", ema_op)

  # add eval graph
  e_label_loss = label_loss_fn.calculate_loss(
          tf.concat(e_tower_predictions,0), e_labels_batch, FLAGS.loss_epsilon)
  tf.summary.scalar("e_label_loss", e_label_loss)
 
  tf.add_to_collection("e_predictions", tf.concat(e_tower_predictions, 0))
  tf.add_to_collection("e_labels", tf.cast(e_labels_batch, tf.float32))
  tf.add_to_collection("e_loss", e_label_loss)
  
class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
               log_device_placement=True, max_steps=None,
               export_model_steps=5000, gpu_only=1):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement,  device_count = {'GPU': gpu_only})
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0

#     if self.is_master and self.task.index > 0:
#       raise StandardError("%s: Only one replica of master expected",
#                           task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
      self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

    with tf.Graph().as_default() as graph:

      if meta_filename:
        saver = self.recover_model(meta_filename)

      with tf.device(device_fn):
        if not meta_filename:
          saver = self.build_model(self.model, self.reader)

        global_step = tf.get_collection("global_step")[0]
        restart_learning_rate = tf.get_collection("restart_learning_rate")[0]
        layers_keep_probs = tf.get_collection("layers_keep_probs")[0]
        loss = tf.get_collection("loss")[0]
        predictions = tf.get_collection("predictions")[0]
        labels = tf.get_collection("labels")[0]
        train_op = tf.get_collection("train_op")[0]
        if FLAGS.use_ema == True:
           ema_op = tf.get_collection("ema_op")[0]      
        
        e_loss = tf.get_collection("e_loss")[0]
        e_labels = tf.get_collection("e_labels")[0]
        e_predictions = tf.get_collection("e_predictions")[0]
        
        init_op = tf.global_variables_initializer()
        restart_op = tf.assign(restart_learning_rate, FLAGS.restart_learning_rate)
        # getting a proper number of keep_prob parameters for dropout
        # max is 10 and we have to pad the vector with 1s 
        # not the nicest solution, but works
        tmp_layers = []
        if FLAGS.layers_keep_probs is not None:
           tmp_layers = [float(x) for x in FLAGS.layers_keep_probs.replace(' ','').split(',')]
    
        tmp_layers_padded = tmp_layers + [1.0 for x in range(10-len(tmp_layers))]
        with tf.variable_scope("tower", reuse=True) as scope:
            keep_op = tf.assign(layers_keep_probs, tmp_layers_padded)

    sv = tf.train.Supervisor(
        graph,
        logdir=self.train_dir,
        init_op=init_op,
        is_chief=self.is_master,
        global_step=global_step,
        save_model_secs=FLAGS.save_model_minutes * 60,
        save_summaries_secs=120,
        saver=saver)

    logging.info("%s: Starting managed session.", task_as_string(self.task))
    with sv.managed_session(target, config=self.config) as sess:
      try:
        if FLAGS.restart_learning_rate > 0.0:
          sess.run(restart_op)
          logging.info("restart learning rate: %f\n" % FLAGS.restart_learning_rate)
        if FLAGS.layers_keep_probs != "1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0":
          logging.info("============")
          sess.run(keep_op)
          logging.info("layers keep probabilites: %s" % FLAGS.layers_keep_probs)
        logging.info("%s: Entering training loop.", task_as_string(self.task))
        while (not sv.should_stop()) and (not self.max_steps_reached):
          batch_start_time = time.time()
          _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
              [train_op, global_step, loss, predictions, labels])
          seconds_per_batch = time.time() - batch_start_time
          examples_per_second = labels_val.shape[0] / seconds_per_batch

          if self.max_steps and self.max_steps <= global_step_val:
            self.max_steps_reached = True

          if self.is_master and global_step_val % 50 == 0 and self.train_dir:
            eval_start_time = time.time()
            hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
            perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                      labels_val)
            gap = eval_util.calculate_gap(predictions_val, labels_val)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time

            logging.info("training step " + str(global_step_val) + " |  Loss: " + ("%.2f" % loss_val) +
              " | Hit@1: " + ("%.2f" % hit_at_one) + "  PERR: " + ("%.2f" % perr) +
              "  GAP: " + ("%.4f" % gap))

            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Hit@1", hit_at_one),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_Perr", perr), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Training_GAP", gap), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("global_step/Examples/Second",
                                  examples_per_second), global_step_val)
            
            #also do eval
            e_loss_val, e_predictions_val, e_labels_val = sess.run(
              [e_loss, e_predictions, e_labels])
            e_hit_at_one = eval_util.calculate_hit_at_one(e_predictions_val, e_labels_val)
            e_perr = eval_util.calculate_precision_at_equal_recall_rate(e_predictions_val,
                                                                      e_labels_val)
            e_gap = eval_util.calculate_gap(e_predictions_val, e_labels_val)
            logging.info("training step " + str(global_step_val) + " | eLoss: " + ("%.2f" % e_loss_val) +
              " |eHit@1: " + ("%.2f" % e_hit_at_one) + " ePERR: " + ("%.2f" % e_perr) +
              " eGAP: " + ("%.4f" % e_gap))

            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Eval_Hit@1", e_hit_at_one),
                global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Eval_Perr", e_perr), global_step_val)
            sv.summary_writer.add_summary(
                utils.MakeSummary("model/Eval_GAP", e_gap), global_step_val)
            
            sv.summary_writer.flush()

            # Exporting the model every x steps
            time_to_export = ((self.last_model_export_step == 0) or
                (global_step_val - self.last_model_export_step
                 >= self.export_model_steps))

            if self.is_master and time_to_export:
              self.export_model(global_step_val, sv.saver, sv.save_path, sess)
              self.last_model_export_step = global_step_val
          else:
            logging.info("training step " + str(global_step_val) + " | Loss: " +
              ("%.2f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))
      except tf.errors.OutOfRangeError:
        logging.info("%s: Done training -- epoch limit reached.",
                     task_as_string(self.task))

    logging.info("%s: Exited training loop.", task_as_string(self.task))
    sv.Stop()

  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir,
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint
        )

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(reader=reader,
                 model=model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 train_data_pattern2=FLAGS.train_data_pattern2,
                 train_data_pattern3=FLAGS.train_data_pattern3,
                 eval_data_pattern=FLAGS.eval_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs, 
                 l2_penalty=FLAGS.l2_penalty, 
                 gpu_only=FLAGS.gpu_only
                 )

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.5)


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        num_classes = FLAGS.truncated_num_classes,
        feature_names=feature_names, feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        num_classes = FLAGS.truncated_num_classes,
        decode_zlib = FLAGS.decode_zlib,
        feature_names=feature_names, feature_sizes=feature_sizes, feature_calcs=FLAGS.c_vars, feature_remove=FLAGS.r_vars)

  return reader


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))

  #Store flags in file
  flag_lines = []
  flag_lines.append("START FLAGS =====================================")
  flag_lines.append("train_dir: " + str(FLAGS.train_dir))
  flag_lines.append("train_data_pattern: "  + str(FLAGS.train_data_pattern))
  flag_lines.append("train_data_pattern2: " + str(FLAGS.train_data_pattern2))
  flag_lines.append("train_data_pattern3: " + str(FLAGS.train_data_pattern3))
  flag_lines.append("eval_data_pattern: " + str(FLAGS.eval_data_pattern))
  flag_lines.append("feature_names: " + str(FLAGS.feature_names))
  flag_lines.append("feature_sizes: " + str(FLAGS.feature_sizes))
  flag_lines.append("frame_features: " + str(FLAGS.frame_features))
  flag_lines.append("model: " + str(FLAGS.model))
  flag_lines.append("start_new_model: " + str(FLAGS.start_new_model))
  flag_lines.append("l2_penalty: " + str(FLAGS.l2_penalty))
  flag_lines.append("c_vars: " + str(FLAGS.c_vars))
  flag_lines.append("r_vars: " + str(FLAGS.r_vars))
  flag_lines.append("loss_epsilon: " + str(FLAGS.loss_epsilon))
  flag_lines.append("batch_size: " + str(FLAGS.batch_size))
  flag_lines.append("label_loss: " + str(FLAGS.label_loss))
  flag_lines.append("regularization_penalty: " + str(FLAGS.regularization_penalty))
  flag_lines.append("base_learning_rate: " + str(FLAGS.base_learning_rate))
  flag_lines.append("learning_rate_decay: " + str(FLAGS.learning_rate_decay))
  flag_lines.append("learning_rate_decay_examples: " + str(FLAGS.learning_rate_decay_examples))
  flag_lines.append("num_epochs: " + str(FLAGS.num_epochs))
  flag_lines.append("max_steps: " + str(FLAGS.max_steps))
  flag_lines.append("export_model_steps: " + str(FLAGS.export_model_steps))
  flag_lines.append("save_model_minutes: " + str(FLAGS.save_model_minutes))
  flag_lines.append("gpu_only: " + str(FLAGS.gpu_only))
  flag_lines.append("num_readers: " + str(FLAGS.num_readers))
  flag_lines.append("optimizer: " + str(FLAGS.optimizer))
  flag_lines.append("clip_gradient_norm: " + str(FLAGS.clip_gradient_norm))
  flag_lines.append("log_device_placement: " + str(FLAGS.log_device_placement))
  flag_lines.append("apply_global_normalization: " + str(FLAGS.apply_global_normalization))
  flag_lines.append("apply_batch_l2_normalization: " + str(FLAGS.apply_batch_l2_normalization))
  flag_lines.append("restart_learning_rate: " + str(FLAGS.restart_learning_rate))
  flag_lines.append("model_export_gpu_only: " + str(FLAGS.model_export_gpu_only))
  flag_lines.append("layers_keep_probs: " + str(FLAGS.layers_keep_probs))
  flag_lines.append("truncated_num_classes: " + str(FLAGS.truncated_num_classes))
  flag_lines.append("decode_zlib: " + str(FLAGS.decode_zlib))
  flag_lines.append("fold: " + str(FLAGS.fold))
  flag_lines.append("ema_halflife: " + str(FLAGS.ema_halflife))
  flag_lines.append("use_ema: " + str(FLAGS.use_ema))
  flag_lines.append("END FLAGS =======================================")

  if not os.path.exists(FLAGS.train_dir):
      os.makedirs(FLAGS.train_dir)
  with open(os.path.join(FLAGS.train_dir, 'flags.cfg'),'a') as ff:
      for line in flag_lines:
          ff.write(line+'\n')
          print(line)

  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    model = find_class_by_name(FLAGS.model,
        [frame_level_models, video_level_models,
         xp_frame_level_models, xp_video_level_models])()

    reader = get_reader()

    # Exporter only needs layer_keep_probs for predictions, so those should always be 1.0
    model_exporter = export_model.ModelExporter(
        frame_features=FLAGS.frame_features,
        model=model,
        reader=reader, 
        gpu_only=FLAGS.model_export_gpu_only)

    Trainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter,
            FLAGS.log_device_placement, FLAGS.max_steps,
            FLAGS.export_model_steps, FLAGS.gpu_only).run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == "__main__":
  app.run()
