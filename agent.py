"""Defines an agent that learns to play Atari games using an A3C architecture.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import a3c
import logging
import numpy as np
import tensorflow as tf

from scipy import signal


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def _apply_discount(rewards, discount):
    """Discounts the specified rewards exponentially.

    Given rewards = [r0, r1, r2, r3] and discount = 0.99, the result is:
        [r0 + 0.99 * (r1 + 0.99 * (r2 + 0.99 * r3)),
         r1 + 0.99 * (r2 + 0.99 * r3),
         r2 + 0.99 * r3,
         r3]

    Example: rewards = [10, 20, 30, 40] and discount = 0.99 -> [98.01496, 88.904, 69.6, 40].

    Returns:
        The discounted rewards.
    """

    return signal.lfilter([1], [1, -discount], rewards[::-1])[::-1]


class Agent():
    def __init__(self,
                 worker_index,
                 env,
                 render,
                 num_local_steps,
                 learning_rate,
                 entropy_regularization,
                 max_gradient_norm,
                 discount,
                 summary_writer,
                 summary_update_interval):
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
            summary_writer: A TensorFlow object that writes summaries.
            summary_update_interval: Number of training steps needed to update the summary data.
        """

        self.worker_index = worker_index
        self.env = env
        self.render = render
        self.num_local_steps = num_local_steps
        self.discount = discount
        self.summary_writer = summary_writer
        self.summary_update_interval = summary_update_interval
        self.num_times_trained = 0

        worker_device = '/job:thread/task:{}/cpu:0'.format(worker_index)

        with tf.device(tf.train.replica_device_setter(1, '/job:master', worker_device)):
            with tf.variable_scope('global'):
                self.global_network = a3c.PolicyNetwork(len(env.action_space),
                                                        env.observation_space)
                self.global_step = tf.get_variable('global_step',
                                                   [],
                                                   tf.int32,
                                                   tf.constant_initializer(0, tf.int32),
                                                   trainable=False)
        with tf.device(worker_device):
            with tf.variable_scope('local'):
                self.local_network = a3c.PolicyNetwork(len(env.action_space), env.observation_space)
                self.local_network.global_step = self.global_step

        self.action = tf.placeholder(tf.int32, [None], 'Action')
        self.advantage = tf.placeholder(tf.float32, [None], 'Advantage')
        self.discounted_reward = tf.placeholder(tf.float32, [None], 'Discounted_Reward')

        # Estimate the policy loss using the cross-entropy loss function.
        action_logits = self.local_network.action_logits
        policy_loss = tf.reduce_sum(
            self.advantage * tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits,
                                                                            labels=self.action))

        # Regularize the policy loss by adding uncertainty (subtracting entropy). High entropy means
        # the agent is uncertain (meaning, it assigns similar probabilities to multiple actions).
        # Low entropy means the agent is sure of which action it should perform next.
        entropy = -tf.reduce_sum(tf.nn.softmax(action_logits) * tf.nn.log_softmax(action_logits))
        policy_loss -= entropy_regularization * entropy

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

        tf.summary.scalar('model/loss', self.loss / tf.to_float(batch_size))
        tf.summary.scalar('model/policy_loss', policy_loss / tf.to_float(batch_size))
        tf.summary.scalar('model/value_loss', value_loss / tf.to_float(batch_size))
        tf.summary.scalar('model/entropy', entropy / tf.to_float(batch_size))
        tf.summary.scalar('model/global_norm', tf.global_norm(self.local_network.parameters))
        tf.summary.scalar('model/gradient_global_norm', tf.global_norm(gradients))
        self.summary_step = tf.summary.merge_all()

    def _get_experiences(self):
        states = []
        actions = []
        rewards = []
        values = []

        if self.env.done:
            self.env.reset()

        lstm_state = self.local_network.get_initial_lstm_state()

        for _ in range(self.num_local_steps):
            state = self.env.get_state()
            action, value, lstm_state = self.local_network.sample_action(state, lstm_state)
            reward = self.env.step(self.env.action_space[action])

            if self.render:
                self.env.render()

            # Store this experience.
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

            if self.env.done:
                LOGGER.info('Finished episode. Total reward: %d. Length: %d.',
                            self.env.episode_reward,
                            self.env.episode_length)

                summary = tf.Summary()
                summary.value.add(tag='environment/episode_length',
                                  simple_value=self.env.episode_length)
                summary.value.add(tag='environment/episode_reward',
                                  simple_value=self.env.episode_reward)
                summary.value.add(tag='environment/fps',
                                  simple_value=self.env.episode_length / self.env.episode_run_time)

                self.summary_writer.add_summary(summary, self.global_step.eval())
                self.summary_writer.flush()
                break

        # Estimate discounted rewards.
        rewards = np.array(rewards)
        next_value = 0 if self.env.done else self.local_network.estimate_value(self.env.get_state(),
                                                                               lstm_state)
        discounted_rewards = _apply_discount(np.append(rewards, next_value), self.discount)[:-1]

        # Estimate advantages.
        values = np.array(values + [next_value])
        advantages = _apply_discount(rewards + self.discount * values[1:] - values[:-1],
                                     self.discount)

        return np.array(states), np.array(actions), advantages, discounted_rewards

    def train(self, sess):
        """Performs a single learning step.

        Args:
            sess: A TensorFlow session.
        """

        sess.run(self.reset_local_network)
        states, actions, advantages, discounted_rewards = self._get_experiences()
        feed_dict = {self.local_network.x: states,
                     self.action: actions,
                     self.advantage: advantages,
                     self.discounted_reward: discounted_rewards,
                     self.local_network.lstm_state: self.local_network.get_initial_lstm_state()}

        # Occasionally write summaries.
        if self.worker_index == 0 and self.num_times_trained % self.summary_update_interval == 0:
            _, global_step, summary = sess.run(
                [self.train_step, self.global_step, self.summary_step], feed_dict)
            self.summary_writer.add_summary(tf.Summary.FromString(summary), global_step)
            self.summary_writer.flush()
        else:
            _, global_step = sess.run([self.train_step, self.global_step], feed_dict)

        self.num_times_trained += 1

        return global_step
