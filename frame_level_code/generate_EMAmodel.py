# Copyright 2017 You8M team.
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
"""Script for initializing EMA weights."""

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow import logging
import sys

src_cp = sys.argv[1]  # the checkpoint with the weights
latest_checkpoint = sys.argv[2]  # the checkpoint with the graph
destination = sys.argv[3]  # the model destination

meta_graph_location = latest_checkpoint + ".meta"
sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph(meta_graph_location)
saver.restore(sess, latest_checkpoint)

ckpt_reader = pywrap_tensorflow.NewCheckpointReader(src_cp)

# update all variables
for tv in tf.global_variables():
    if ("ExponentialMovingAverage" not in tv.name) and ("layers_keep_probs" not in tv.name):
        srct = ckpt_reader.get_tensor(tv.name.split(":")[0])
        logging.info("Replacing tensor: {} with new values".format(tv.name))
        sess.run(tf.assign(tv, srct))

# initialize the moving average
for tv in tf.global_variables():
    if ("ExponentialMovingAverage" in tv.name):
        logging.info("Placing EMA tensor: {} with new values".format(tv.name))
        fetch_tensor = ckpt_reader.get_tensor("/".join(tv.name.split(":")[0].split("/")[1:-1]))
        sess.run(tf.assign(tv, fetch_tensor))


saver = tf.train.Saver(max_to_keep=0)
saver.save(sess, destination)