#!/bin/bash
umask 0002
# Repo-dependent setup for HW2 (run inside the cs224r-hw2 container).
# System packages, MuJoCo, micromamba, and OpenCV are baked into the Docker image.
set -e

HW2_DIR=/workspace/CS224R_Spring_2025/HW2

export PATH="/root/.local/bin:$PATH"
eval "$(micromamba shell hook --shell bash)"

# ---------------------------------------------------------------------------
# 1. AC conda environment (needs conda_env.yml from repo)
# ---------------------------------------------------------------------------
echo "==> Creating AC conda environment..."
micromamba env create --file="$HW2_DIR/ac/conda_env.yml" -y || \
    micromamba env update --file="$HW2_DIR/ac/conda_env.yml" -n AC -y

echo "==> Installing extra AC packages..."
micromamba run -n AC pip install \
    "metaworld@git+https://github.com/Farama-Foundation/Metaworld.git@a98086ababc81560772e27e7f63fe5d120c4cc50" \
    "cython<3"

# ---------------------------------------------------------------------------
# 2. Build mujoco_mpc (needs source from repo)
# ---------------------------------------------------------------------------
echo "==> Building mujoco_mpc..."
mkdir -p "$HW2_DIR/videos"
cmake \
    -B "$HW2_DIR/mujoco_mpc/build" \
    -S "$HW2_DIR/mujoco_mpc/mjpc" \
    -D CMAKE_C_COMPILER=/usr/bin/clang-14 \
    -D CMAKE_CXX_COMPILER=/usr/bin/clang++-14
xvfb-run -a cmake --build "$HW2_DIR/mujoco_mpc/build" \
    --config Release --target mjpc -j"$(nproc)"

echo ""
echo "==> Setup complete!"
echo ""
echo "To run the simulation, use the run_mjpc helper or run manually:"
echo '  cd /workspace/CS224R_Spring_2025/HW2'
echo '  xvfb-run -a -s "-screen 0 1400x900x24" ./mujoco_mpc/build/bin/mjpc --task="Quadruped Flat" --steps=100 --horizon=0.35 --w0=0.0 --w1=0.0 --w2=0.0 --w3=0.0'
