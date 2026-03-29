#!/bin/bash
# Script to launch TensorBoard for all data folders in each HW directory

BASE_DIR="$(dirname "$0")"

# Recursively find all 'data' folders under each HW directory and add non-empty folders or subfolders
LOGDIRS=""
for hw in "$BASE_DIR"/HW*; do
    echo "Searching for data folders in $hw"
    for data_dir in $(find "$hw" -type d -name 'data' 2>/dev/null); do
        # Add the data_dir itself if non-empty
        if [ "$(ls -A "$data_dir")" ]; then
            LOGDIRS+="$data_dir," 
        fi
        # Also add any non-empty subfolders
        for subdir in "$data_dir"/*/; do
            if [ -d "$subdir" ] && [ "$(ls -A "$subdir")" ]; then
                LOGDIRS+="$subdir," 
            fi
        done
    done
done

# Remove trailing comma
LOGDIRS=${LOGDIRS%,}

if [ -z "$LOGDIRS" ]; then
    echo "Warning: No non-empty data folders found. Exiting."
    exit 1
fi

echo "Launching TensorBoard with logdirs: $LOGDIRS"

# Activate venv if it exists (look for .venv in CS224R_Spring_2025)
VENV_PATH="$BASE_DIR/.venv/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "Activating virtual environment at $VENV_PATH"
    source "$VENV_PATH"
else
    echo "No venv found at $VENV_PATH, running with system Python."
fi

tensorboard --logdir_spec="$LOGDIRS" --port 6006
