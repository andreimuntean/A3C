"""Trains an agent to play Atari games from OpenAI Gym.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import argparse
import environment
import multiprocessing
import signal
import sys
import tensorflow as tf
import time

PARSER = argparse.ArgumentParser(description='Train an agent to play Atari games.')

PARSER.add_argument('--env_name',
                    metavar='ENVIRONMENT',
                    help='name of an OpenAI Gym Atari environment on which to train',
                    default='Pong-v0')

PARSER.add_argument('--worker_index',
                    help='the index of this worker thread (if it is the master, leave it None)',
                    type=int,
                    default=None)

PARSER.add_argument('--render',
                    help='determines whether to display the game screen of each agent',
                    type=bool,
                    default=False)

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


def get_cluster_def(num_threads):
    """Creates a cluster definition with 1 master (parameter server) and num_threads workers."""

    port = 14000
    localhost = '127.0.0.1'
    cluster = {'master': ['{}:{}'.format(localhost, port)],
               'worker': []}

    for _ in range(num_threads):
        port += 1
        cluster['worker'].append('{}:{}'.format(localhost, port))

    return tf.train.ClusterSpec(cluster).as_cluster_def()


def run_worker(args):
    """Starts a worker thread that learns how to play the specified Atari game."""

    cluster_def = get_cluster_def(args.num_threads)
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
    server = tf.train.Server(cluster_def, 'worker', args.worker_index, config=config)

    # Initialize the environment.
    env = environment.AtariWrapper(args.env_name, args.observations_per_state, args.action_space)


def main(args):
    """Trains an agent to play Atari games."""

    # Ensure that threads are terminated gracefully.
    shutdown_thread = lambda signal, stack_frame: sys.exit(signal + 128)
    signal.signal(signal.SIGHUP, shutdown_thread)

    is_master = args.worker_index is None

    if is_master:
        cluster_def = get_cluster_def(args.num_threads)
        config = tf.ConfigProto(device_filters=['/job:master'])
        server = tf.train.Server(cluster_def, 'master', config=config)

        # Keep master thread running since worker threads depend on it.
        while True:
            time.sleep(1000)

    # Otherwise, this is a worker.
    run_worker(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
