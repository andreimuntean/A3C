"""Trains an agent to play Atari games from OpenAI Gym.

Heavily influenced by DeepMind's seminal paper 'Asynchronous Methods for Deep Reinforcement
Learning' (Mnih et al., 2016).
"""

import agent
import argparse
import environment
import logging
import multiprocessing
import os
import signal
import sys
import tensorflow as tf
import time

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
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

PARSER.add_argument('--log_dir',
                    metavar='PATH',
                    help='path to a directory where to save & restore the model and log events',
                    default='models/tmp')

PARSER.add_argument('--num_threads',
                    metavar='THREADS',
                    help='number of learning threads',
                    type=int,
                    default=multiprocessing.cpu_count())

PARSER.add_argument('--num_local_steps',
                    metavar='TIME STEPS',
                    help='number of experiences used per worker when updating the model',
                    type=int,
                    default=20)

PARSER.add_argument('--num_global_steps',
                    metavar='TIME STEPS',
                    help='number of time steps trained for in total',
                    type=int,
                    default=50000000)

PARSER.add_argument('--learning_rate',
                    metavar='LAMBDA',
                    help='rate at which the network learns from new examples',
                    type=float,
                    default=1e-4)

PARSER.add_argument('--entropy_regularization',
                    metavar='BETA',
                    help='the strength of the entropy regularization term',
                    type=float,
                    default=0.01)

PARSER.add_argument('--max_gradient_norm',
                    metavar='DELTA',
                    help='maximum value allowed for the L2-norms of gradients',
                    type=float,
                    default=40)

PARSER.add_argument('--discount',
                    metavar='GAMMA',
                    help='discount factor for future rewards',
                    type=float,
                    default=0.99)

PARSER.add_argument('--summary_update_interval',
                    metavar='TRAINING STEPS',
                    help='frequency at which summary data is updated when training',
                    type=int,
                    default=10)


def get_cluster_def(num_threads):
    """Creates a cluster definition for 1 master (parameter server) and num_threads workers."""

    port = 14000
    localhost = '127.0.0.1'
    cluster = {'master': ['{}:{}'.format(localhost, port)],
               'thread': []}

    for _ in range(num_threads):
        port += 1
        cluster['thread'].append('{}:{}'.format(localhost, port))

    return tf.train.ClusterSpec(cluster).as_cluster_def()


def run_worker(args):
    """Starts a worker thread that learns how to play the specified Atari game."""

    cluster_def = get_cluster_def(args.num_threads)
    config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2)
    server = tf.train.Server(cluster_def, 'thread', args.worker_index, config=config)

    # Configure the supervisor.
    is_chief = args.worker_index == 0
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoint')
    thread_dir = os.path.join(args.log_dir, 'thread-{}'.format(args.worker_index))
    summary_writer = tf.summary.FileWriter(thread_dir)
    global_variables_initializer = tf.global_variables_initializer()
    init_fn = lambda sess: sess.run(global_variables_initializer)

    # Initialize the model.
    env = environment.AtariWrapper(args.env_name, environment.TRAINING, args.action_space)
    player = agent.Agent(args.worker_index,
                         env,
                         args.render,
                         args.num_local_steps,
                         args.learning_rate,
                         args.entropy_regularization,
                         args.max_gradient_norm,
                         args.discount,
                         summary_writer,
                         args.summary_update_interval)

    # Local copies of the model will not be saved.
    model_variables = [var for var in tf.global_variables() if not var.name.startswith('local')]

    supervisor = tf.train.Supervisor(ready_op=tf.report_uninitialized_variables(model_variables),
                                     is_chief=is_chief,
                                     init_op=tf.variables_initializer(model_variables),
                                     logdir=checkpoint_dir,
                                     summary_op=None,
                                     saver=tf.train.Saver(model_variables),
                                     global_step=player.global_step,
                                     save_summaries_secs=30,
                                     save_model_secs=30,
                                     summary_writer=summary_writer,
                                     init_fn=init_fn)

    config = tf.ConfigProto(device_filters=['/job:master',
                                            '/job:thread/task:{}/cpu:0'.format(args.worker_index)])

    LOGGER.info('Starting worker. This may take a while.')
    with supervisor.managed_session(server.target, config=config) as sess, sess.as_default():
        global_step = 0
        while not supervisor.should_stop() and global_step < args.num_global_steps:
            global_step = player.train(sess)

    supervisor.stop()
    LOGGER.info('Stopped after %d global steps.', player.global_step)


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
        LOGGER.info('Started master thread.')

        # Keep master thread running since worker threads depend on it.
        while True:
            time.sleep(1000)

    # Otherwise, this is a worker.
    run_worker(args)


if __name__ == '__main__':
    main(PARSER.parse_args())
