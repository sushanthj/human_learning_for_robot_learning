#!/usr/bin/env bash
# setup.sh — spin up a per-homework Docker environment and drop
# into a ready-to-use shell. Each HW has its own Dockerfile/image.
#
# Usage:  ./setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$REPO_DIR/setup_scripts"

log() { echo "[launch] $*"; }

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' not found. Please install Docker (with Compose v2)." >&2
    exit 1
  fi
}

require_cmd docker

# ── Ask which homework ────────────────────────────────────────────────────────
echo ""
echo "Which assignment do you want to work on?"
echo "  0) Pre-reqs (MDPs & Q-learning)"
echo "  1) HW1"
echo "  2) HW2"
echo "  3) HW3"
echo "  4) HW4"
echo ""
read -rp "Enter choice [0-4]: " HW_CHOICE

case "$HW_CHOICE" in
  0) SERVICE="prereqs"; CONTAINER="cs224r-prereqs" ;;
  1) SERVICE="hw1";     CONTAINER="cs224r-hw1" ;;
  2) SERVICE="hw2";     CONTAINER="cs224r-hw2" ;;
  3) SERVICE="hw3";     CONTAINER="cs224r-hw3" ;;
  4) SERVICE="hw4";     CONTAINER="cs224r-hw4" ;;
  *) echo "Invalid choice." >&2; exit 1 ;;
esac

# ── Build image if needed (cached after first build) ─────────────────────────
log "Ensuring image is built..."
docker compose -f "$SCRIPT_DIR/docker-compose.yml" build "$SERVICE"

# ── Start container if not running ────────────────────────────────────────────
STATUS="$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || true)"

if [[ "$STATUS" != "running" ]]; then
  log "Starting container '$CONTAINER'..."
  docker compose -f "$SCRIPT_DIR/docker-compose.yml" up -d "$SERVICE"
else
  log "Container '$CONTAINER' is already running."
fi

# ── Run repo-dependent setup if not already done ─────────────────────────────
if ! docker exec "$CONTAINER" test -f /root/.setup_done; then
  log "Running first-time repo setup..."
  docker exec -it "$CONTAINER" bash "/workspace/setup_scripts/setup_${SERVICE}.sh"
  docker exec "$CONTAINER" touch /root/.setup_done
else
  log "Repo setup already done — skipping."
fi

# ── Enter the container ──────────────────────────────────────────────────────
case "$HW_CHOICE" in
  0)
    log "Entering pre-reqs environment."
    docker exec -it "$CONTAINER" bash -c '
      cd /workspace/pre-reqs/MDPs_Q_learning/mountaincar
      exec bash
    '
    ;;
  1)
    log "Entering HW1 environment."
    docker exec -it "$CONTAINER" bash -c '
      eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
      micromamba activate cs224r
      cd /workspace/CS224R_Spring_2025/HW1
      exec bash
    '
    ;;
  2)
    log "Entering HW2 environment."
    docker exec -it "$CONTAINER" bash -c '
      eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
      micromamba activate AC
      cd /workspace/CS224R_Spring_2025/HW2
      echo ""
      echo "Ready! Run simulation with:  run_mjpc"
      echo ""
      run_mjpc() {
        echo "Use default parameters? (horizon=0.35, w0-w3=0.0)"
        read -rp "[Y/n]: " choice
        if [[ "$choice" =~ ^[Nn] ]]; then
          read -rp "task [Quadruped Flat]: " task
          task="${task:-Quadruped Flat}"
          read -rp "steps [100]: " steps
          steps="${steps:-100}"
          read -rp "horizon [0.35]: " horizon
          horizon="${horizon:-0.35}"
          read -rp "w0 [0.0]: " w0; w0="${w0:-0.0}"
          read -rp "w1 [0.0]: " w1; w1="${w1:-0.0}"
          read -rp "w2 [0.0]: " w2; w2="${w2:-0.0}"
          read -rp "w3 [0.0]: " w3; w3="${w3:-0.0}"
        else
          task="Quadruped Flat"
          steps=100
          horizon=0.35
          w0=0.0; w1=0.0; w2=0.0; w3=0.0
        fi
        xvfb-run -a -s "-screen 0 1400x900x24" ./mujoco_mpc/build/bin/mjpc \
          --task="$task" --steps="$steps" --horizon="$horizon" \
          --w0="$w0" --w1="$w1" --w2="$w2" --w3="$w3"
        for f in videos/*.avi; do
          [ -f "$f" ] || continue
          ffmpeg -y -loglevel warning -i "$f" -c:v libx264 -pix_fmt yuv420p "${f%.avi}.mp4" && rm "$f"
          echo "Converted: ${f%.avi}.mp4"
        done
      }
      export -f run_mjpc
      exec bash
    '
    ;;
  3)
    log "Entering HW3 environment."
    docker exec -it "$CONTAINER" bash -c '
      eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
      micromamba activate cs224r
      cd /workspace/CS224R_Spring_2025/HW3
      exec bash
    '
    ;;
  4)
    log "Entering HW4 environment."
    echo ""
    echo "HW4 has two parts:"
    echo "  a) Goal-conditioned RL"
    echo "  b) Meta RL"
    echo ""
    read -rp "Enter choice [a/b]: " HW4_PART
    case "$HW4_PART" in
      a|A)
        docker exec -it "$CONTAINER" bash -c '
          eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
          micromamba activate hw4_goal
          cd /workspace/CS224R_Spring_2025/HW4/goal_conditioned_rl
          exec bash
        '
        ;;
      b|B)
        docker exec -it "$CONTAINER" bash -c '
          eval "$(/root/.local/bin/micromamba shell hook --shell bash)"
          micromamba activate hw4_meta
          cd /workspace/CS224R_Spring_2025/HW4/meta_rl
          exec bash
        '
        ;;
      *) echo "Invalid choice." >&2; exit 1 ;;
    esac
    ;;
esac
