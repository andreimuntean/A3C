"""Defines policy networks for asynchronous advantage actor-critic architectures.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import math
import numpy as np
import tensorflow as tf


def _convolutional_layer(x, shape, stride, activation_fn):
    if len(shape) != 4:
        raise ValueError('Shape "{}" is invalid. Must have length 4.'.format(shape))

    num_input_params = shape[0] * shape[1] * shape[2]
    num_output_params = shape[0] * shape[1] * shape[3]
    maxval = math.sqrt(6 / (num_input_params + num_output_params))
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0, tf.float32, [shape[3]]), name='Bias')
    conv = tf.nn.conv2d(x, W, [1, stride, stride, 1], 'VALID')

    return activation_fn(tf.nn.bias_add(conv, b))


def _fully_connected_layer(x, shape, activation_fn):
    if len(shape) != 2:
        raise ValueError('Shape "{}" is invalid. Must have length 2.'.format(shape))

    maxval = math.sqrt(6 / (shape[0] + shape[1]))
    W = tf.Variable(tf.random_uniform(shape, -maxval, maxval), name='Weights')
    b = tf.Variable(tf.constant(0, tf.float32, [shape[1]]), name='Bias')

    return activation_fn(tf.matmul(x, W) + b)


class PolicyNetwork():
    def __init__(self, num_actions, state_shape):
        """Defines a policy network implemented as a convolutional recurrent neural network.

        Args:
            num_actions: Number of possible actions.
            state_shape: A vector with three values, representing the width, height and depth of
                input states. For example, the shape of 100x80 RGB images is [100, 80, 3].
        """

        width, height, depth = state_shape
        self.x = tf.placeholder(tf.float32, [None, width, height, depth], name='Input_States')

        with tf.name_scope('Convolutional_Layer_1'):
            h_conv1 = _convolutional_layer(self.x, [3, 3, depth, 32], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_2'):
            h_conv2 = _convolutional_layer(h_conv1, [3, 3, 32, 32], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_3'):
            h_conv3 = _convolutional_layer(h_conv2, [3, 3, 32, 32], 2, tf.nn.relu)

        with tf.name_scope('Convolutional_Layer_4'):
            h_conv4 = _convolutional_layer(h_conv3, [3, 3, 32, 32], 2, tf.nn.relu)

        # Flatten the output to feed it into the LSTM layer.
        num_params = np.prod(h_conv4.get_shape().as_list()[1:])
        h_flat = tf.reshape(h_conv4, [-1, num_params])

        with tf.name_scope('LSTM_Layer'):
            self.lstm_state = (tf.placeholder(tf.float32, [1, 256]),
                               tf.placeholder(tf.float32, [1, 256]))

            self.initial_lstm_state = (np.zeros([1, 256], np.float32),
                                       np.zeros([1, 256], np.float32))

            lstm_state = tf.contrib.rnn.LSTMStateTuple(*self.lstm_state)
            lstm = tf.contrib.rnn.BasicLSTMCell(256)

            # tf.nn.dynamic_rnn expects inputs of shape [batch_size, time, features], but the shape
            # of h_flat is [batch_size, features]. We want the batch_size dimension to be treated as
            # the time dimension, so the input is redundantly expanded to [1, batch_size, features].
            # The LSTM layer will assume it has 1 batch with a time dimension of length batch_size.
            batch_size = tf.shape(h_flat)[:1]  # [:1] is a trick to correctly get the dynamic shape.
            lstm_input = tf.expand_dims(h_flat, [0])
            lstm_output, self.new_lstm_state = tf.nn.dynamic_rnn(lstm,
                                                                 lstm_input,
                                                                 batch_size,
                                                                 lstm_state)
            # Delete the fake batch dimension.
            lstm_output = tf.squeeze(lstm_output, [0])

        self.action_logits = _fully_connected_layer(lstm_output, [256, num_actions], tf.identity)
        self.value = tf.squeeze(_fully_connected_layer(lstm_output, [256, 1], tf.identity))
        self.action = tf.squeeze(tf.multinomial(
            self.action_logits - tf.reduce_max(self.action_logits, 1, keep_dims=True), 1))
        self.parameters = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            tf.get_variable_scope().name)

    def get_initial_lstm_state(self):
        """Returns a value that can be used as the initial state of the LSTM unit of the network."""

        return self.initial_lstm_state

    def sample_action(self, state, lstm_state):
        """Samples an action for the specified state from the learned mixed strategy.

        Args:
            state: State of the environment.
            lstm_state: The state of the long short-term memory unit of the network. Use the
                get_initial_lstm_state method when unknown.

        Returns:
            An action, the value of the specified state and the new state of the LSTM unit.
        """

        sess = tf.get_default_session()
        feed_dict = {self.x: [state], self.lstm_state: lstm_state}
        return sess.run((self.action, self.value, self.new_lstm_state), feed_dict)

    def estimate_value(self, state, lstm_state):
        """Estimates the value of the specified state.

        Args:
            state: State of the environment.

        Returns:
            The value of the specified state.
        """

        sess = tf.get_default_session()
        return sess.run(self.value, {self.x: [state], self.lstm_state: lstm_state})
