#!/bin/sh

ENV_NAME="Pong-v0"
LOG_DIR="./models/$ENV_NAME"
TMUX_SESSION_NAME="a3c"
NUM_THREADS=$(nproc --all)
TENSORBOARD_PORT=15000

# Hide GPUs from TensorFlow. The model is designed to run only on CPUs.
export CUDA_VISIBLE_DEVICES=''

# Create the log directory.
mkdir -p $LOG_DIR

# Kill previous tmux session. Ignore potential "can't find session" messages.
tmux kill-session -t $TMUX_SESSION_NAME 2> /dev/null

# Initialize a new tmux session.
tmux new-session -s $TMUX_SESSION_NAME -n master -d

# Create a window for each learning thread.
for thread_id in $(seq 0 $(($NUM_THREADS - 1))); do
    tmux new-window -t $TMUX_SESSION_NAME -n thread-$thread_id
done

# Create a window for TensorBoard.
tmux new-window -t $TMUX_SESSION_NAME -n tensorboard

# Create a window for observing hardware usage.
tmux new-window -t $TMUX_SESSION_NAME -n htop

# Wait for tmux to finish setting up.
sleep 1

# Start the master thread, which synchronizes worker threads.
tmux send-keys -t a3c:master "/usr/bin/python3 thread.py" \
                             " --env_name=$ENV_NAME" \
                             " --log_dir=$LOG_DIR" \
                             " --num_threads=$NUM_THREADS" \
                             " $@" Enter

# Start worker threads.
for thread_id in $(seq 0 $(($NUM_THREADS - 1))); do
    tmux send-keys -t a3c:thread-$thread_id "/usr/bin/python3 thread.py" \
                                            " --env_name=$ENV_NAME" \
                                            " --log_dir=$LOG_DIR" \
                                            " --num_threads=$NUM_THREADS" \
                                            " --worker_index=$thread_id" \
                                            " $@" Enter
done

# Start TensorBoard.
tmux send-keys -t a3c:tensorboard "tensorboard" \
                                  " --port $TENSORBOARD_PORT" \
                                  " --logdir $LOG_DIR" Enter

# Start htop.
tmux send-keys -t a3c:htop htop Enter

echo "Started the learning session."
echo "Started TensorBoard at localhost:$TENSORBOARD_PORT."
echo "Use 'tmux attach -t $TMUX_SESSION_NAME' to connect to the session."
echo "Use 'tmux kill-session -t $TMUX_SESSION_NAME' to end the session."