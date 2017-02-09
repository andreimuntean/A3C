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
                 max_gradient_norm,
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
            max_gradient_norm: Maximum value allowed for the L2-norms of gradients. Gradients with
                norms that would otherwise surpass this value are scaled down.
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
                self.global_network = a3c.PolicyNetwork(len(env.action_space), env.state_space)
                self.global_step = tf.get_variable('global_step',
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(0, tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.local_network = a3c.PolicyNetwork(len(env.action_space), env.state_space)
                self.local_network.global_step = self.global_step

        self.action = tf.placeholder(tf.int32, [None, len(env.action_space)], 'Action')
        self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
        self.discounted_reward = tf.placeholder(tf.float32, [None], 'Discounted_Reward')

        # Estimate the policy loss using the cross-entropy loss function.
        action_logits = self.local_network.action_logits
        policy_loss = tf.reduce_sum(
            self.advantage * -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits,
                                                                             labels=self.action))

        # Regularize the policy loss by summing it with its entropy. High entropy means the agent is
        # uncertain (meaning, it assigns similar probabilities to multiple actions). Low entropy
        # means the agent is sure of which action it should take next.
        entropy = -tf.reduce_sum(tf.nn.softmax(action_logits) * tf.nn.log_softmax(action_logits))
        policy_loss += entropy_regularization * entropy

        # Estimate the value loss using the sum of squared errors.
        value_loss = tf.nn.l2_loss(self.local_network.value - self.discounted_reward)

        # Estimate the final loss.
        self.loss = policy_loss + 0.5 * value_loss

        # Fetch and clip the gradients of the local network.
        gradients = tf.gradients(self.loss, self.local_network.parameters)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

        # Update the global network using the clipped gradients.
        batch_size = tf.shape(self.local_network.x)[0]
        grads_and_vars = list(zip(clipped_gradients, self.global_network.parameters))
        self.train_step = [tf.train.AdamOptimizer(learning_rate).apply_gradients(grads_and_vars),
                           self.global_step.assign_add(batch_size)]

        # Synchronize the local network with the global network.
        self.reset_local_network = [local_p.assign(global_p)
                                    for local_p, global_p in zip(self.local_network.parameters,
                                                                 self.global_network.parameters)]

        tf.summary.image('model/state', self.local_network.x)
        tf.summary.scalar('model/policy_loss', policy_loss / batch_size)
        tf.summary.scalar('model/value_loss', value_loss / batch_size)
        tf.summary.scalar('model/entropy', entropy / batch_size)
        tf.summary.scalar('model/global_norm', tf.global_norm(self.local_network.parameters))
        tf.summary.scalar('model/gradient_global_norm', tf.global_norm(gradients))

        self.thread = None

    def train(self):
        """Performs a single learning step."""

        pass
