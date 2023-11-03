#!/bin/bash

# This script will create N tmux windows and run a specified command in each.

# check if the argument N is provided
if [ $# -eq 0 ]; then
    echo "No arguments supplied. Setting default value to 1."
    N=1
else
    N=$1
fi

# create a new detached tmux session
tmux new-session -d -s "run_batch"

# create N-1 additional windows, since a new session already has one window
for ((i=1; i<N; i++)); do
    tmux new-window -t "run_batch:$i" -n "run_batch_$i"
done

# run the command in each window
for ((i=0; i<N; i++)); do
    tmux send-keys -t "run_batch:$i" "source ./venv/bin/activate; python run.py" C-m
    # Optional: wait a brief period of time between commands if necessary
    # sleep 5
done

# attach to the tmux session
tmux attach-session -t "run_batch"
