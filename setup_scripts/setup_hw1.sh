#!/bin/bash
umask 0002
set -e

HW1_DIR=/workspace/CS224R_Spring_2025/HW1

echo "==> Installing HW1 requirements..."
eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
micromamba run -n cs224r pip install -r "$HW1_DIR/requirements.txt"
micromamba run -n cs224r pip install -e "$HW1_DIR"

echo ""
echo "==> Setup complete!"
echo ""
echo "To get started:"
echo "  micromamba activate cs224r"
echo "  cd /workspace/CS224R_Spring_2025/HW1"
