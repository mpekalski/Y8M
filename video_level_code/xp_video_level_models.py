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

"""Contains model definitions."""
import math
import numpy as np

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "MoNN_num_experts", 4,
    "The number of mixtures (excluding the dummy 'expert') used for MoNNs.")
#%% helper functions
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=1.0/np.sqrt(2*shape[0]))
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1/shape[0], shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses relu to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
            regularizer = tf.nn.l2_loss(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation')
            tf.summary.histogram('activations', activations)
    return activations, regularizer
  

#
# First part contains models we have used, 
# later there are some models we have tried/experimented with
#
# MoNN3L
# MoNN2Lw
# MoNN3Lw
# MoNN4Ln
#

class MoNN3L(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    a1Units = 4096
    A1 = slim.fully_connected(
        model_input, a1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA1')
    a2Units = 4096
    A2 = slim.fully_connected(
        A1, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA2')
    a2Units = 4096
    A3 = slim.fully_connected(
        A2, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA3')

    expert_activations = slim.fully_connected(
        A3,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


# a wide model hoping to memorize rare labels better
class MoNN2Lw(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="MoN2w_gates")
    
    h1Units = 2305 * 6
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='MoN2w_H1')
    h2Units = 2305 * 3
    A2 = slim.fully_connected(
        A1, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='MoN2w_H2')
#    
    expert_activations = slim.fully_connected(
        A2,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="MoN2_experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}



class MoNN3Lw(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    a1Units = 2305*8
    A1 = slim.fully_connected(
        model_input, a1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA1')
    a2Units = 2305
    A2 = slim.fully_connected(
        A1, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA2')
    a2Units = 2305*4
    A3 = slim.fully_connected(
        A2, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA3')

    expert_activations = slim.fully_connected(
        A3,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class MoNN4Ln(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-6,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        a1Units = 2048
        A1 = slim.fully_connected(
            model_input, a1Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA1')
        a2Units = 2048
        A2 = slim.fully_connected(
            A1, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA2')
        a2Units = 2048
        A3 = slim.fully_connected(
            A2, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA3')
        a2Units = 2048
        A4 = slim.fully_connected(
            A3, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA4')
        expert_activations = slim.fully_connected(
            A4,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return {"predictions": final_probabilities}

#
# Abandoned Experiments
#

#%%
class MyNNModel0(models.BaseModel):
  """Logistic model with L2 regularization."""

  def create_model(self, model_input, vocab_size, l2_penalty=1e-4, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    with tf.name_scope('MyNNModel0'):
        h1Units = 2400
        a1 = slim.fully_connected(
                model_input, h1Units, activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC1')
        output = slim.fully_connected(
                a1, vocab_size, activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC2')
    return {"predictions": output}

#%%
class MyNNModel1(models.BaseModel):
  """A simple NN models (with L2 regularization)."""

  
  def create_model(self, model_input, vocab_size, l2_penalty=1e-4, 
                   is_train=True, **unused_params):
    """Creates a logistic model.

    Args:
      model_input: 'batch' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes."""
    with tf.name_scope('MyNNModel1'):
        h1Units = 1152
        h2Units = 2248
        h3Units = 3096
        keep_prob = 0.90
        A1 = slim.fully_connected(
                model_input, h1Units, activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC_H1')
        A2 = slim.fully_connected(
                A1, h2Units, activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC_H2')
        A3 = slim.fully_connected(
                A2, h3Units, activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC_H3')
        #A4 = tf.nn.dropout(A3, keep_prob)
        output = slim.fully_connected(
                A3, vocab_size, activation_fn=tf.nn.sigmoid,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                scope='FC_P')

    return {"predictions": output} 
    
#%%
class MyNNModel2(models.BaseModel):
  """A simple NN models (with L2 regularization)."""

  
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-4,
                   **unused_params):
    """Creates a simple one-hidden-layer Neural Network model.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
   
    #A1 = slim.fully_connected(
    #    model_input, 800, activation_fn=tf.nn.sigmoid,
    #    weights_regularizer=slim.l2_regularizer(l2_penalty),
    #    scope='hidden1')
    
#    output = slim.fully_connected(
#        A1, vocab_size, activation_fn=tf.nn.sigmoid,
#        weights_regularizer=slim.l2_regularizer(l2_penalty))
    h1Units = 3600
    A1, reg1 = nn_layer(model_input, 1024+128, h1Units, 'Hidden1', act=tf.nn.relu)
    h2Units = 3600
    A2, reg2 = nn_layer(A1, h1Units, h2Units, 'Hidden2', act=tf.nn.relu)
    output, reg3 = nn_layer(A2, h2Units, vocab_size, 'Pred', act=tf.nn.sigmoid)
    
    return {"predictions": output, 
            "regularization_loss":l2_penalty*(reg1+reg2+reg3)}

#%%
def nn_layer2( input_tensor, input_dim, output_dim, var_scope, act=tf.nn.relu):
    with tf.variable_scope(var_scope):
       weights = weight_variable([input_dim, output_dim])
       regularizer = tf.nn.l2_loss(weights)
       biases = bias_variable([output_dim])
       preactivate = tf.matmul(input_tensor, weights) + biases
       activations = act(preactivate, name='activation')
    return activations, regularizer
  

class MyNNModel3(models.BaseModel):
  """A simple NN models (with L2 regularization)."""

  
  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-4,
                   **unused_params):
    """Creates a simple one-hidden-layer Neural Network model.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
   
    #A1 = slim.fully_connected(
    #    model_input, 800, activation_fn=tf.nn.sigmoid,
    #    weights_regularizer=slim.l2_regularizer(l2_penalty),
    #    scope='hidden1')
    
#    output = slim.fully_connected(
#        A1, vocab_size, activation_fn=tf.nn.sigmoid,
#        weights_regularizer=slim.l2_regularizer(l2_penalty))
    with tf.variable_scope('MyNNModel3'):
      h1Units = 3600
      A1,reg1 = nn_layer2(model_input, 1024+128, h1Units, 'Hidden1', act=tf.nn.relu)
      h2Units = 2400
      A2, reg2 = nn_layer2(A1, h1Units, h2Units, 'Hidden2', act=tf.nn.relu)
      h3Units = 2400
      A3, reg3 = nn_layer2(A2, h2Units, h3Units, 'Hidden3', act=tf.nn.relu)
      output, reg4 = nn_layer(A3, h3Units, vocab_size, 'ProdictionLayer', act=tf.nn.sigmoid)
    
    return {"predictions": output,
            "regularization_loss":l2_penalty*(reg1+reg2+reg3+reg4)}

#%%
class MoNN2L(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    """Creates a Mixture of (Logistic) Experts model.

     The model consists of a per-class softmax distribution over a
     configurable number of logistic classifiers. One of the classifiers in the
     mixture is not trained, and always predicts 0.

    Args:
      model_input: 'batch_size' x 'num_features' matrix of input features.
      vocab_size: The number of classes in the dataset.
      num_mixtures: The number of mixtures (excluding a dummy 'expert' that
        always predicts the non-existence of an entity).
      l2_penalty: How much to penalize the squared magnitudes of parameter
        values.
    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      batch_size x num_classes.
    """
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")
    
    h1Units = 4096
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')
    h2Units = 4096
    A2 = slim.fully_connected(
        A1, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
#    
    expert_activations = slim.fully_connected(
        A2,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}





class MoNN2L_L1(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    h1Units = 4096
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')
    h2Units = 4096
    A2 = slim.fully_connected(
        A1, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l1_l2_regularizer(l2_penalty),
        scope='FC_H2')
#    
    expert_activations = slim.fully_connected(
        A2,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

from tensorflow import logging 
class MoNN2Drop(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def create_model(self,
                   model_input,
                   vocab_size,
                   layers_keep_probs,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
    logging.info("MoNN2Drop " + str(layers_keep_probs))
    drop_out = tf.nn.dropout(model_input, layers_keep_probs[0],name="var_dropout")
    
    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    h1Units = 4096
    
    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')

    h2Units = 4096 
    A1a = tf.nn.dropout(A1, layers_keep_probs[1])
    A2 = slim.fully_connected(
        A1a, h2Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
    A2a = tf.nn.dropout(A2, layers_keep_probs[2])

    expert_activations = slim.fully_connected(
        A2a,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class MoNN2DropBNorm(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def create_model(self,
                   model_input,
                   vocab_size,
                   layers_keep_probs,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   is_training=True,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
    logging.info("MoNN2Drop " + str(layers_keep_probs))

    drop_out = tf.nn.dropout(model_input, layers_keep_probs[0],name="input/dropout")

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    model_input_norm = slim.batch_norm(
        model_input,
        center=True,
        scale=True,
        is_training=is_training,
        scope='input/batch_norm')

    h1Units = 4096

    A1 = slim.fully_connected(
        model_input_norm, h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')

    h2Units = 4096
    A1a = tf.nn.dropout(A1, layers_keep_probs[1], name='layer1/dropout')
    A1b = slim.batch_norm(
          A1a, 
          center=True,
          scale=True,
          is_training=is_training,
          scope='layer1/batch_norm')

    A2 = slim.fully_connected(
        A1b, h2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
    A2a = tf.nn.dropout(A2, layers_keep_probs[2], name='layer2/dropout')
    
    expert_activations = slim.fully_connected(
        A2a,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

class MoNN2DropBNorm1Crelu(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def create_model(self,
                   model_input,
                   vocab_size,
                   layers_keep_probs,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   is_training=True,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
    logging.info("MoNN2Drop " + str(layers_keep_probs))

    drop_out = tf.nn.dropout(model_input, layers_keep_probs[0],name="input/dropout")

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    model_input_norm = slim.batch_norm(
        model_input,
        center=True,
        scale=True,
        is_training=is_training,
        scope='input/batch_norm')

    h1Units = 4096

    A1 = slim.fully_connected(
        model_input, h1Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1')

    h2Units = 4096
    A1a = tf.nn.dropout(A1, layers_keep_probs[1], name='layer1/dropout')
    A1b = slim.batch_norm(
          A1a,
          center=True,
          scale=True,
          is_training=is_training,
          scope='layer1/batch_norm')

    A2 = slim.fully_connected(
        A1b, h2Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2')
    A2a = tf.nn.dropout(A2, layers_keep_probs[2], name='layer2/dropout')

    expert_activations = slim.fully_connected(
        A2a,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}



class MoNN4L(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-6,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")

        a1Units = 4096
        A1 = slim.fully_connected(
            model_input, a1Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA1')
        a2Units = 4096
        A2 = slim.fully_connected(
            A1, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA2')
        a2Units = 4096
        A3 = slim.fully_connected(
            A2, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA3')
        a2Units = 4096
        A4 = slim.fully_connected(
            A3, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA4')
        expert_activations = slim.fully_connected(
            A4,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return {"predictions": final_probabilities}



class MoNN4LDropG2L(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     layers_keep_probs,
                     num_mixtures=None,
                     l2_penalty=1e-6,
                     is_training=True,
                     **unused_params):

        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """

        num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
        logging.info("MoNN4LDrop " + str(layers_keep_probs))

        drop_model_input = tf.nn.dropout(model_input, layers_keep_probs[0])

        #
        # Added one more layer to gate
        #
        X1 = slim.fully_connected(
            drop_model_input,
            vocab_size ,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_l1")

        gate_activations = slim.fully_connected(
            X1,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_l2")

        a1Units = 4096
        A1 = slim.fully_connected(
            drop_model_input, a1Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA1')


        a2Units = 4096
        A1d = tf.nn.dropout(A1, layers_keep_probs[1])
        A2 = slim.fully_connected(
            A1d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA2')

        a2Units = 4096
        A2d = tf.nn.dropout(A1, layers_keep_probs[2])
        A3 = slim.fully_connected(
            A2d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA3')

        a2Units = 4096
        A3d = tf.nn.dropout(A3, layers_keep_probs[3])
        A4 = slim.fully_connected(
            A3d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA4')

        expert_activations = slim.fully_connected(
            A4,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return {"predictions": final_probabilities}



class MoNN4LDropG3L(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     layers_keep_probs,
                     num_mixtures=None,
                     l2_penalty=1e-6,
                     is_training=True,
                     **unused_params):

        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """

        num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
        logging.info("MoNN4LDrop " + str(layers_keep_probs))

        drop_model_input = tf.nn.dropout(model_input, layers_keep_probs[0])

        #
        # Added one more layer to gate
        #
        X1 = slim.fully_connected(
            drop_model_input,
            vocab_size ,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_l1")

        X2 = slim.fully_connected(
            X1,
            vocab_size,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_l2")

        gate_activations = slim.fully_connected(
            X2,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_activation")

        a1Units = 4096
        A1 = slim.fully_connected(
            drop_model_input, a1Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA1')


        a2Units = 4096
        A1d = tf.nn.dropout(A1, layers_keep_probs[1])
        A2 = slim.fully_connected(
            A1d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA2')

        a2Units = 4096
        A2d = tf.nn.dropout(A2, layers_keep_probs[2])
        A3 = slim.fully_connected(
            A2d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA3')

        a2Units = 4096
        A3d = tf.nn.dropout(A3, layers_keep_probs[3])
        A4 = slim.fully_connected(
            A3d, a2Units, activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope='FC_HA4')

        expert_activations = slim.fully_connected(
            A4,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return {"predictions": final_probabilities}




from tensorflow import logging
class MoNN2a128r1024G1L(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""
  def create_model(self,
                   model_input,
                   vocab_size,
                   layers_keep_probs,
                   num_mixtures=None,
                   l2_penalty=1e-6,
                   is_training=True,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts
    logging.info("MoNN2Drop " + str(layers_keep_probs))
    drop_out = tf.nn.dropout(model_input, layers_keep_probs[0],name="var_dropout")

    logging.info(model_input.shape)
    inputA = model_input[:,0:128]
    inputB = model_input[:,128:1152]

    inputAd = tf.nn.dropout(inputA, layers_keep_probs[0])
    inputBd = tf.nn.dropout(inputB, layers_keep_probs[0])
    inputAdn = slim.batch_norm(
        inputAd,
        center=True,
        scale=True,
        is_training=is_training,
        scope='inputAd/batch_norm')

    inputBdn = slim.batch_norm(
        inputBd,
        center=True,
        scale=True,
        is_training=is_training,
        scope='inputBd/batch_norm')

    X1 = slim.fully_connected(
            tf.concat([inputAdn,inputBdn],1),
            vocab_size ,
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates_l1")

    gate_activations = slim.fully_connected(
        X1,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    a_h1Units = 512
    A1 = slim.fully_connected(
        inputAdn, a_h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1_audio')

    a_h2Units = 512

    A1n = slim.batch_norm(
        A1,
        center=True,
        scale=True,
        is_training=is_training,
        scope='A1/batch_norm')
    A1a = tf.nn.dropout(A1n, layers_keep_probs[1])
    logging.info("A1a")
    logging.info(A1a.shape)
    A2 = slim.fully_connected(
        A1a, a_h2Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2_audio')
    A2n = slim.batch_norm(
        A2,
        center=True,
        scale=True,
        is_training=is_training,
        scope='A2/batch_norm')

    logging.info("A2")
    logging.info(A2.shape)

    b_h1Units = 2048
    B1 = slim.fully_connected(
        inputBdn, b_h1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H1_rgb')

    B1n = slim.batch_norm(
        B1,
        center=True,
        scale=True,
        is_training=is_training,
        scope='B1/batch_norm')
    b_h2Units = 2048
    B1a = tf.nn.dropout(B1n, layers_keep_probs[1])
    logging.info("B1a")
    logging.info(B1a.shape)
    B2 = slim.fully_connected(
        B1a, b_h2Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H2_rgb')

    B2n = slim.batch_norm(
        B2,
        center=True,
        scale=True,
        is_training=is_training,
        scope='B2/batch_norm')

    A2na = tf.nn.dropout(A2n, layers_keep_probs[2])
    B2na = tf.nn.dropout(B2n, layers_keep_probs[2])
    
    logging.info(A2.shape)
    logging.info(B2.shape) 
    C3 = tf.concat([inputAdn, inputBdn, A2na, B2na], 1)

    h3Units = 4096
    C3a = slim.fully_connected(
        C3, h3Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H3_concat')

    C3ad = tf.nn.dropout(C3a, layers_keep_probs[3])

    h4Units = 4096
    C4a = slim.fully_connected(
        C3a, h4Units, activation_fn=tf.nn.crelu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_H4_concat')

    expert_activations = slim.fully_connected(
        C4a,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}






class MoNN4Lw(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    a1Units = 2305*8
    A1 = slim.fully_connected(
        model_input, a1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA1')
    a2Units = 2305
    A2 = slim.fully_connected(
        A1, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA2')
    a2Units = 2305*4
    A3 = slim.fully_connected(
        A2, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA3')

    a2Units = 2305*2
    A4 = slim.fully_connected(
        A3, a2Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA4')


    expert_activations = slim.fully_connected(
        A4,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}


class MoNN1Lvw(models.BaseModel):
  """A softmax over a mixture of logistic models (with L2 regularization)."""

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_mixtures=None,
                   l2_penalty=1e-8,
                   **unused_params):
    num_mixtures = num_mixtures or FLAGS.MoNN_num_experts

    gate_activations = slim.fully_connected(
        model_input,
        vocab_size * (num_mixtures + 1),
        activation_fn=None,
        biases_initializer=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="gates")

    a1Units = 2305*64
    A1 = slim.fully_connected(
        model_input, a1Units, activation_fn=tf.nn.relu,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope='FC_HA1')

    expert_activations = slim.fully_connected(
        A1,
        vocab_size * num_mixtures,
        activation_fn=None,
        weights_regularizer=slim.l2_regularizer(l2_penalty),
        scope="experts")

    gating_distribution = tf.nn.softmax(tf.reshape(
        gate_activations,
        [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
    expert_distribution = tf.nn.sigmoid(tf.reshape(
        expert_activations,
        [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

    final_probabilities_by_class_and_batch = tf.reduce_sum(
        gating_distribution[:, :num_mixtures] * expert_distribution, 1)
    final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                     [-1, vocab_size])
    return {"predictions": final_probabilities}

