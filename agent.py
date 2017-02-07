"""Defines an agent that learns to play Atari games using an A3C architecture.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import a3c
import numpy as np
import tensorflow as tf


class Agent():
    def __init__(self,
                 worker_index,
                 env,
                 render,
                 num_local_steps,
                 learning_rate,
                 entropy_regularization,
                 max_gradient,
                 discount):
        """An agent that learns to play Atari games using an A3C architecture.

        Args:
            worker_index: Index of the worker thread that is running this agent.
            env: An AtariWrapper object (see 'environment.py') that wraps over an OpenAI Gym Atari
                environment.
            render: Determines whether to display the game screen.
            num_local_steps: Number of experiences used per worker when updating the model.
            learning_rate: The speed with which the network learns from new examples.
            entropy_regularization: The strength of the entropy regularization term.
            max_gradient: Maximum value allowed for gradients during backpropagation. Gradients that
                would otherwise surpass this value are reduced to it.
            discount: Discount factor for future rewards.
        """

        self.worker_index = worker_index
        self.env = env
        self.render = render
        self.num_local_steps = num_local_steps
        self.discount = discount
        self.local_step = 0

        worker_device = '/job:thread/task:{}/cpu:0'.format(worker_index)

        with tf.device(tf.train.replica_device_setter(1, '/job:master', worker_device)):
            with tf.variable_scope('global'):
                self.global_network = a3c.PolicyNetwork(env.state_space, len(env.action_space))
                self.global_step = tf.get_variable('global_step',
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0, tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.local_network = a3c.PolicyNetwork(env.state_space, env.action_space
                self.local_network.global_step = self.global_step

        self.action = tf.placeholder(tf.int32, [None, len(env.action_space)], 'Action')
        self.advantages = tf.placeholder(tf.float32, [None], 'Advantages')
        
        probabilities = tf.nn.softmax(self.local_network.logits)
        log_probabilities = tf.nn.log_softmax(self.local_network.logits)
        policy_gradient_loss = -tf.reduce_sum(
            self.advantages * tf.reduce_sum(log_probabilities * self.action, 1))

        self.reward = tf.placeholder(tf.float32, [None], 'Reward')
        value_loss = tf.nn.l2_loss(self.local_network.value - self.reward)
        entropy = -tf.reduce_sum(probabilities * log_probabilities)

        self.loss = policy_gradient_loss + 0.5 * value_loss - self.entropy_regularization * entropy

        # Fetch and clip the gradients of the local network.
        gradients = tf.gradients(self.loss, self.local_network.parameters)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient)

        # Update the global network using the clipped gradients.
        grads_and_vars = list(zip(clipped_gradients, self.global_network.parameters))
        self.train_step = [tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars),
                           self.global_step.assign_add(num_local_steps)]

        # Synchronize the local network with the global network.
        self.reset_local_network = [local_p.assign(global_p) 
                                    for local_p, global_p in zip(self.local_network.parameters,
                                                                 self.global_network.parameters)]
        
        tf.summary.image('model/state', self.local_network.x)
        tf.summary.scalar('model/policy_gradient_loss', policy_gradient_loss / num_local_steps)
        tf.summary.scalar('model/value_loss', value_loss / num_local_steps)
        tf.summary.scalar('model/entropy', entropy / num_local_steps)
        tf.summary.scalar('model/global_norm', tf.global_norm(self.local_network.parameters))
        tf.summary.scalar('model/gradient_global_norm', tf.global_norm(gradients))

        self.thread = None

    def train(self):
        """Performs a single learning step."""

        pass
