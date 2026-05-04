#!/bin/bash
umask 0002
set -e

echo "==> Installing pre-reqs dependencies..."
pip install -r /workspace/pre-reqs/MDPs_Q_learning/mountaincar/requirements.txt

echo ""
echo "==> Setup complete!"
echo ""
echo "To train the mountaincar agent:"
echo "  python3 train.py --agent value-iteration"
echo "  python3 train.py --agent tabular"
echo "  python3 train.py --agent function-approximation"
echo "  python3 train.py --agent constrained"
echo ""
echo "To visualize:"
echo "  python3 mountaincar.py --agent <agent-type>"
