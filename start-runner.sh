#!/bin/bash

# Define the actions-runner directory
RUNNER_DIR="/Users/soutrikmaiti/Documents/actions-runner"

# Check if the directory exists
if [ ! -d "$RUNNER_DIR" ]; then
    echo "Error: Actions runner directory not found at $RUNNER_DIR"
    exit 1
fi

# Navigate to the actions-runner directory
cd "$RUNNER_DIR"

# Check if the runner is already running in a tmux session
if tmux has-session -t github-runner 2>/dev/null; then
    echo "GitHub Actions runner is already running in tmux session 'github-runner'"
    echo "To view the runner output, use: tmux attach -t github-runner"
    echo "To detach from the session without stopping it, press Ctrl+B then D"
    exit 0
fi

# Check if run.sh exists and is executable
if [ ! -x "./run.sh" ]; then
    echo "Error: run.sh not found or not executable in $RUNNER_DIR"
    echo "Making run.sh executable..."
    chmod +x ./run.sh
fi

# Create a new tmux session named 'github-runner'
echo "Starting GitHub Actions runner in a new tmux session..."
tmux new-session -d -s github-runner "./run.sh"

# Check if the session was created successfully
if tmux has-session -t github-runner 2>/dev/null; then
    echo "GitHub Actions runner started in tmux session 'github-runner'"
    echo "To view the runner output, use: tmux attach -t github-runner"
    echo "To detach from the session without stopping it, press Ctrl+B then D"
    exit 0
else
    echo "Error: Failed to start GitHub Actions runner in tmux session"
    exit 1
fi
