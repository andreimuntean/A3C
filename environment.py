"""Augments OpenAI Gym Atari environments by preprocessing observations.

Heavily influenced by DeepMind's seminal paper 'Playing Atari with Deep Reinforcement Learning'
(Mnih et al., 2013) and 'Human-level control through deep reinforcement learning' (Mnih et al.,
2015).
"""

import cv2
import gym
import numpy as np


# Specifies restricted action spaces. For games not in this dictionary, all actions are enabled.
ACTION_SPACE = {'Pong-v0': [0, 2, 3],  # NONE, UP and DOWN.
                'Breakout-v0': [1, 2, 3]}  # FIRE (respawn ball, otherwise NOOP), UP and DOWN.


class AtariWrapper:
    """Wraps over an Atari environment from OpenAI Gym and preprocesses observations."""

    def __init__(self, env_name, observations_per_state, action_space=None):
        """Creates the wrapper.

        Args:
            env_name: Name of an OpenAI Gym Atari environment.
            observations_per_state: Number of consecutive observations within a state. Provides some
                short-term memory for the learner. Useful in games like Pong where the trajectory of
                the ball can't be inferred from a single image.
            action_space: A list of possible actions. If 'action_space' is 'None' and no default
                configuration exists for this environment, all actions will be allowed.
        """

        self.env = gym.make(env_name)
        self.state_length = observations_per_state
        self.done = False
        self.state_space = [80, 80, observations_per_state]
        self.state = np.empty(self.state_space, np.float32)

        if action_space:
            self.action_space = list(action_space)
        elif env_name in ACTION_SPACE:
            self.action_space = ACTION_SPACE[env_name]
        else:
            self.action_space = list(range(self.env.action_space.n))

        # Create the initial state by performing random actions.
        for i in range(observations_per_state):
            self.state[..., i], _ = self._step(self.sample_action())

    def reset(self):
        """Resets the environment."""

        self.env.reset()
        self.done = False

    def step(self, action):
        """Performs the specified action.

        Returns:
            The current state and a reward.

        Raises:
            ValueError: If the action is not valid.
        """

        if self.done:
            self.reset()

        if action not in self.action_space:
            raise ValueError('Action "{}" is invalid. Valid actions: {}.'.format(action,
                                                                                 self.action_space))

        self.state[..., :-1] = self.state[..., 1:]
        self.state[..., -1], reward = self._step(action)

        return self.state, reward

    def render(self):
        """Draws the environment."""

        self.env.render()

    def sample_action(self):
        """Samples a random action."""

        return np.random.choice(self.action_space)

    def _step(self, action):
        """Performs the specified action and preprocesses the observation.

        Args:
            action: An action that will be repeated self.frame_skip times.

        Returns:
            An observation (80x80 tensor with real values between 0 and 1) and the accumulated
            reward.
        """

        frame, reward, self.done, _ = self.env.step(action)

        # Transform the frame into a grayscale image with values between 0 and 1. Luminance is
        # extracted using the Y = 0.299 Red + 0.587 Green + 0.114 Blue formula. Values are scaled
        # between 0 and 1 by further dividing each color channel by 255.
        grayscale_frame = (frame[..., 0] * 0.00117 +
                           frame[..., 1] * 0.0023 +
                           frame[..., 2] * 0.00045)

        # Resize grayscale frame to an 80x80 matrix of 32-bit floats.
        observation = cv2.resize(
            grayscale_frame, (80, 80), interpolation=cv2.INTER_NEAREST).astype(np.float32)

        return observation, reward
