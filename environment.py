"""Augments OpenAI Gym Atari environments by preprocessing observations.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import gym
import numpy as np

from scipy import misc


# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': {0, 2, 3},  # NONE, UP and DOWN.
                'Breakout-v0': {1, 2, 3}}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and preprocesses observations."""

    def __init__(self, env_name, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
        """

        self.env = gym.make(env_name)
        self.done = False
        self.observation_space = [80, 80, 1]

        if action_space:
            self.action_space = set(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = set(range(self.env.action_space.n))

    def reset(self):
        """Resets the environment."""

        self.env.reset()
        self.done = False

    def step(self, action):
        """Performs the specified action.

        Returns:
            An observation (80x80x1 tensor with real values between 0 and 1) and a reward.

        Raises:
            ValueError: If the action is not valid.
        """

        if self.done:
            self.reset()

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        frame, reward, self.done, _ = self.env.step(action)

        # Transform the frame into a grayscale image with values between 0 and 1. Luminance is
        # extracted using the Y = 0.299 Red + 0.587 Green + 0.114 Blue formula. Values are scaled
        # between 0 and 1 by further dividing each color channel by 255.
        grayscale_frame = (frame[..., 0] * 0.00117 +
                           frame[..., 1] * 0.0023 +
                           frame[..., 2] * 0.00045)

        # Resize grayscale frame to an 80x80 matrix of 32-bit floats.
        observation = misc.imresize(grayscale_frame, (80, 80)).astype(np.float32)

        return np.expand_dims(observation, 1), reward

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)
