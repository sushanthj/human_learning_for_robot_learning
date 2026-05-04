#!/bin/bash
umask 0002
set -e

HW4_DIR=/workspace/CS224R_Spring_2025/HW4

export PATH="/root/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

echo "==> Installing goal_conditioned_rl requirements..."
micromamba run -n hw4_goal pip install -r "$HW4_DIR/goal_conditioned_rl/requirements.txt"

echo "==> Installing meta_rl requirements..."
micromamba run -n hw4_meta pip install -r "$HW4_DIR/meta_rl/requirements.txt"

echo ""
echo "==> Setup complete!"
echo ""
echo "Goal-conditioned RL:"
echo "  micromamba activate hw4_goal"
echo "  cd /workspace/CS224R_Spring_2025/HW4/goal_conditioned_rl"
echo ""
echo "Meta RL:"
echo "  micromamba activate hw4_meta"
echo "  cd /workspace/CS224R_Spring_2025/HW4/meta_rl"
echo "  python dream.py exp_name -b environment=\"map\""
echo "  python rl2.py exp_name -b environment=\"map\""
