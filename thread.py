"""Trains an agent to play Atari games from OpenAI gym.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import argparse
import multiprocessing


PARSER = argparse.ArgumentParser(description='Train an agent to play Atari games.')

PARSER.add_argument('--env_name',
                    metavar='ENVIRONMENT',
                    help='name of an OpenAI Gym Atari environment on which to train',
                    default='Pong-v0')

PARSER.add_argument('--action_space',
                    nargs='+',
                    help='restricts the number of possible actions',
                    type=int)

PARSER.add_argument('--load_path',
                    metavar='PATH',
                    help='loads a trained model from the specified path')

PARSER.add_argument('--log_dir',
                    metavar='PATH',
                    help='directory in which summaries will be saved',
                    default='models/tmp')

PARSER.add_argument('--num_threads',
                    metavar='THREADS',
                    help='number of learning threads',
                    type=int,
                    default=multiprocessing.cpu_count())

PARSER.add_argument('--train_time',
                    metavar='TIME STEPS',
                    help='number of time steps that each thread will train for',
                    type=int,
                    default=20000000)

PARSER.add_argument('--learning_rate',
                    metavar='LAMBDA',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=1e-4)

PARSER.add_argument('--max_gradient',
                    metavar='DELTA',
                    help='maximum value allowed for gradients during backpropagation',
                    type=float,
                    default=10)

PARSER.add_argument('--discount',
                    metavar='GAMMA',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

PARSER.add_argument('--observations_per_state',
                    metavar='FRAMES',
                    help='number of consecutive frames within a state',
                    type=int,
                    default=3)


def main(args):
    """Trains an agent to play Atari games."""

    pass


if __name__ == '__main__':
    main(PARSER.parse_args())
