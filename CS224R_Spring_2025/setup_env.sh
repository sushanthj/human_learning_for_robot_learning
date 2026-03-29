#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/../.venv}"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

log() {
  echo "[setup_hw1_env] $*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: required command '$cmd' not found." >&2
    exit 1
  fi
}

need_apt_package() {
  local pkg="$1"
  dpkg -s "$pkg" >/dev/null 2>&1 || return 0
  return 1
}

install_python_and_system_deps() {
  require_cmd apt-get

  local needs_python=false
  if ! command -v python3.11 >/dev/null 2>&1; then
    needs_python=true
  fi

  local pkgs=()
  if $needs_python; then
    log "python3.11 not found; installing from apt repositories."
    pkgs+=(python3.11 python3.11-venv python3.11-dev)
  fi

  if need_apt_package swig; then
    pkgs+=(swig)
  fi

  if need_apt_package build-essential; then
    pkgs+=(build-essential)
  fi

  if ((${#pkgs[@]} == 0)); then
    log "System dependencies already available."
    return
  fi

  log "Installing required apt packages: ${pkgs[*]}"
  sudo apt-get update

  if $needs_python; then
    sudo apt-get install -y software-properties-common
    if ! apt-cache show python3.11 >/dev/null 2>&1; then
      log "Adding deadsnakes PPA for python3.11 packages."
      sudo add-apt-repository -y ppa:deadsnakes/ppa
      sudo apt-get update
    fi
  fi

  sudo apt-get install -y "${pkgs[@]}"
}

create_or_refresh_venv() {
  if [[ -x "$VENV_DIR/bin/python" ]]; then
    local venv_version
    venv_version="$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' || true)"
    if [[ "$venv_version" != "3.11" ]]; then
      log "Existing venv at $VENV_DIR uses Python $venv_version; recreating with Python 3.11."
      rm -rf "$VENV_DIR"
    fi
  fi

  if [[ ! -d "$VENV_DIR" ]]; then
    log "Creating virtual environment at $VENV_DIR"
    python3.11 -m venv "$VENV_DIR"
  else
    log "Using existing virtual environment at $VENV_DIR"
  fi
}

install_python_deps() {
  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  log "Upgrading pip/setuptools/wheel"
  pip install --upgrade pip setuptools wheel

  log "Installing Python requirements from $REQ_FILE"
  pip install -r "$REQ_FILE"

  log "Installing HW1 package in editable mode"
  pip install -e "$SCRIPT_DIR"
}

main() {
  require_cmd dpkg
  install_python_and_system_deps
  create_or_refresh_venv
  install_python_deps
  log "Done. Activate with: source $VENV_DIR/bin/activate"
}

main "$@"
