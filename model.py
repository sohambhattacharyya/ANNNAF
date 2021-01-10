# -*- coding: utf-8 -*-
"""Artifiial Neural Network for Nonclassical Adaptive Filtering"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

LOGDIR=os.getcwd()
tf.logging.set_verbosity(tf.logging.INFO)


def filter_model_fn(features, labels, mode):
  """Model function for Nonclassical Adaptive Filter."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 1])  
  
  # Dense Layer
  # Densely connected layer with 1024 neurons
  dense1 = tf.layers.dense(inputs=input_layer, units=1024, activation=tf.nn.relu)

  # Dense Layer 2
  # Densely connected layer with 1024 neurons
  dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
  
  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=1)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  #eval_metric_ops = {
  #    "accuracy": tf.metrics.accuracy(
  #        labels=labels, predictions=predictions["classes"])}
  #return tf.estimator.EstimatorSpec(
  #    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
  return logits


def main(unused_argv):
  # Load training and eval data
  t = np.arange(20000)
  x_t=np.float32(5*t)
  y_t=2*t**3+3*t**2
  x_t_eval=x_t[:5000]
  y_t_eval=y_t[:5000]
  train_data = x_t  # Returns np.array
  train_labels = y_t
  eval_data = x_t_eval  # Returns np.array
  eval_labels = y_t_eval

  # Create the Estimator
  filter_classifier = tf.estimator.Estimator(
      model_fn=filter_model_fn, model_dir=LOGDIR)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x":train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  filter_classifier.train(
      input_fn=train_input_fn,
      steps=10000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = filter_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()