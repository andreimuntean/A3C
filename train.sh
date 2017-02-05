#!/bin/sh

ENV_NAME="Pong-v0"
SAVE_DIR="./models/$ENV_NAME"
TMUX_SESSION_NAME="a3c"
NUM_THREADS=$(nproc --all)

# Create the save directory.
mkdir -p $SAVE_DIR

# Kill previous tmux session. Ignore potential "can't find session" messages.
tmux kill-session -t $TMUX_SESSION_NAME &> /dev/null

# Initialize a new tmux session.
tmux new-session -s $TMUX_SESSION_NAME -n master -d

# Create a window for each learning thread.
for thread_id in $(seq 1 $NUM_THREADS); do
    tmux new-window -t $TMUX_SESSION_NAME -n thread-$thread_id
done

# Create a window for TensorBoard.
tmux new-window -t $TMUX_SESSION_NAME -n tensorboard

# Create a window for observing hardware usage.
tmux new-window -t $TMUX_SESSION_NAME -n htop