"""Augments OpenAI Gym Atari environments by preprocessing observations.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import gym
import numpy as np
import time

from scipy import misc


# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'Breakout-v0': [1, 2, 3]}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.

TESTING = 0
TRAINING = 1


def _preprocess_observation(observation):
    """Transforms the specified observation into a 47x47x1 grayscale image.

    Returns:
        A 47x47x1 tensor with float32 values between 0 and 1.
    """

    # Transform the observation into a grayscale image with values between 0 and 1. Use the simple
    # np.mean method instead of sophisticated luminance extraction techniques since they do not seem
    # to improve training.
    grayscale_observation = observation.mean(2)

    # Resize grayscale frame to a 47x47 matrix of 32-bit floats.
    resized_observation = misc.imresize(grayscale_observation, (47, 47)).astype(np.float32)

    return np.expand_dims(resized_observation, 2)


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and preprocesses observations."""

    def __init__(self, env_name, mode, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
            mode: The context in which the environment is used. Can be either environment.TESTING or
                environment.TRAINING.
        """

        if mode is not TESTING and mode is not TRAINING:
            raise ValueError(('Mode is invalid. Must be either environment.TESTING or '
                              'environment.TRAINING.'))

        self.env = gym.make(env_name)
        self.mode = mode
        self.observation_space = [47, 47, 1]
        self.reset()

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.env.action_space.n))

    def reset(self):
        """Resets the environment."""

        self.done = False
        self.episode_reward = 0
        self.episode_length = 0
        self.state = _preprocess_observation(self.env.reset())
        self.episode_start_time = time.time()
        self.episode_run_time = 0
        self.lives = None

    def step(self, action):
        """Performs the specified action.

        Returns:
            A reward signal which is either -1, 0 or 1.

        Raises:
            Exception: If the game ended.
            ValueError: If the action is not valid.
        """

        if self.done:
            raise Exception('Game finished.')

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        observation, reward, self.done, info = self.env.step(action)

        if self.mode is TRAINING and self.lives is not None and info['ale.lives'] < self.lives:
            # While training, treat loss of life as end of episode.
            self.done = True

        self.episode_reward += reward
        self.episode_length += 1
        self.state = _preprocess_observation(observation)
        self.episode_run_time = time.time() - self.episode_start_time
        self.lives = info['ale.lives']

        return -1 if reward < 0 else 1 if reward > 0 else 0

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def get_state(self):
        """Gets the current state.

        Returns:
            An observation (47x47x1 tensor with float32 values between 0 and 1).
        """

        return self.state
