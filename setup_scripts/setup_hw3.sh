#!/bin/bash
umask 0002
set -e

HW3_DIR=/workspace/CS224R_Spring_2025/HW3

export PATH="/root/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

echo "==> Installing HW3 requirements..."
micromamba run -n cs224r pip install -r "$HW3_DIR/requirements.txt"
micromamba run -n cs224r pip install -e "$HW3_DIR"

echo ""
echo "==> Setup complete!"
echo ""
echo "To get started:"
echo "  micromamba activate cs224r"
echo "  cd /workspace/CS224R_Spring_2025/HW3"
echo "  # See homework PDF for TODOs"
