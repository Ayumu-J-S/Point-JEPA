#!/bin/bash

# Check for the presence of an argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/wandb/directory"
    exit 1
fi

# Use the first argument as the wandb directory path
WANDB_DIR="$1"

# Check if the wandb directory exists
if [ ! -d "$WANDB_DIR" ]; then
    echo "The specified wandb directory does not exist: $WANDB_DIR"
    exit 1
fi

# Navigate to the wandb directory
cd "$WANDB_DIR"

# Iterate over each run directory and run wandb sync
for RUN_DIR in */ ; do
    echo "Syncing $RUN_DIR..."
    wandb sync "$RUN_DIR"
done

echo "All runs synced."