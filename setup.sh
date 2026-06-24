#!/bin/bash
set -euo pipefail

# GB10 Demo Suite Setup Script
# Installs cuDNN 9.x, TensorRT 10.x, creates isolated venvs, initializes model downloads
# Safe to run multiple times (idempotent with state tracking)

PROJECT_DIR="${HOME}/gb10-demo"
STATE_FILE="${PROJECT_DIR}/.setup-state"
LOG_FILE="${PROJECT_DIR}/logs/setup.log"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Utility functions
log() {
  local msg="$1"
  local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
  echo "[${timestamp}] ${msg}" | tee -a "${LOG_FILE}"
}

warn() {
  echo -e "${YELLOW}WARNING: $1${NC}" | tee -a "${LOG_FILE}"
}

err() {
  echo -e "${RED}ERROR: $1${NC}" | tee -a "${LOG_FILE}"
}

success() {
  echo -e "${GREEN}✓ $1${NC}" | tee -a "${LOG_FILE}"
}

state_has() {
  [[ -f "${STATE_FILE}" ]] && grep -q "^$1=" "${STATE_FILE}"
}

state_set() {
  local key="$1"
  local value="$2"
  if state_has "$key"; then
    sed -i "s/^${key}=.*/${key}=${value}/" "${STATE_FILE}"
  else
    echo "${key}=${value}" >> "${STATE_FILE}"
  fi
}

state_get() {
  local key="$1"
  if state_has "$key"; then
    grep "^${key}=" "${STATE_FILE}" | cut -d'=' -f2
  fi
}

# NVIDIA Libraries - CUDA 13.0 aarch64
# Note: NVIDIA requires login for direct downloads. Using conda/pip alternatives.
CUDNN_VERSION="9.0"
TENSORRT_VERSION="10.0"

# Installation method: conda (most reliable for NVIDIA packages)
# If conda not available, script will suggest manual steps
USE_CONDA=${USE_CONDA:-1}

WORKLOADS=("hpc" "llm" "vlm" "cnn")

# ============================================================================
# Phase 1: Initialize
# ============================================================================

phase_init() {
  log "Initializing GB10 Demo Suite..."

  mkdir -p "${PROJECT_DIR}"/{env,models,workloads,config,scripts,logs}

  for workload in "${WORKLOADS[@]}"; do
    mkdir -p "${PROJECT_DIR}/models/${workload}-models"
    mkdir -p "${PROJECT_DIR}/workloads/${workload}"
  done

  success "Project directories created"
}

# ============================================================================
# Phase 2: System Checks
# ============================================================================

check_system() {
  log "Checking system requirements..."

  # Check CUDA
  if ! command -v nvcc &> /dev/null; then
    err "CUDA toolkit not found. Please install CUDA 13.0"
    return 1
  fi

  local cuda_version=$(nvcc --version | grep -oP 'release \K[0-9.]+')
  if [[ ! "$cuda_version" =~ ^13\.0 ]]; then
    warn "CUDA version is $cuda_version (expected 13.0.x)"
  fi
  success "CUDA check passed (version: $cuda_version)"

  # Check Python
  if ! command -v python3.12 &> /dev/null; then
    err "Python 3.12 not found"
    return 1
  fi
  success "Python 3.12 found"

  # Check Docker
  if ! command -v docker &> /dev/null; then
    warn "Docker not found (optional, but recommended)"
  else
    success "Docker found"
  fi

  # Check disk space
  local free_space=$(df "${PROJECT_DIR}" | awk 'NR==2 {print $4}')
  if (( free_space < 1000000 )); then
    warn "Low disk space: ${free_space}KB available (recommend >1GB)"
  fi

  state_set "system_checked" "$(date '+%Y-%m-%d %H:%M:%S')"
}

# ============================================================================
# Phase 3: Install cuDNN 9.x
# ============================================================================

install_cudnn() {
  log "Setting up cuDNN ${CUDNN_VERSION} for CUDA 13.0 aarch64..."

  if state_has "cudnn_installed" && python3 -c "import cudnn" 2>/dev/null; then
    local installed_version=$(state_get "cudnn_version")
    log "cuDNN already installed (version: $installed_version), skipping..."
    return 0
  fi

  log "Installing cuDNN via pip..."

  # Try pip install first (works on most systems)
  if pip install nvidia-cudnn-cu13==9.0 2>/dev/null; then
    success "cuDNN installed via pip"
    state_set "cudnn_installed" "yes"
    state_set "cudnn_version" "${CUDNN_VERSION}"
    return 0
  fi

  # Fallback: conda
  if command -v conda &> /dev/null; then
    log "Trying conda install..."
    if conda install -y -c conda-forge cudnn=9.0 2>/dev/null; then
      success "cuDNN installed via conda"
      state_set "cudnn_installed" "yes"
      state_set "cudnn_version" "${CUDNN_VERSION}"
      return 0
    fi
  fi

  # Provide manual download instructions
  warn "cuDNN installation failed via automated methods."
  log "Manual installation steps:"
  log "  1. Download from: https://developer.nvidia.com/cudnn"
  log "  2. Select: cuDNN 9.x for CUDA 13.0, Linux aarch64"
  log "  3. Extract and copy to /usr/local/cudnn"
  log "     tar -xzf cudnn-*.tar.xz"
  log "     sudo cp -r cudnn-*/* /usr/local/cudnn/"
  log "  4. Update LD_LIBRARY_PATH in venv activation scripts"
  log ""
  log "Continuing setup without cuDNN (workloads may not require it)..."
  state_set "cudnn_installed" "manual-pending"
  return 0
}

# ============================================================================
# Phase 4: Install TensorRT 10.x
# ============================================================================

install_tensorrt() {
  log "Setting up TensorRT ${TENSORRT_VERSION} for CUDA 13.0 aarch64..."

  if state_has "tensorrt_installed" && python3 -c "import tensorrt" 2>/dev/null; then
    local installed_version=$(state_get "tensorrt_version")
    log "TensorRT already installed (version: $installed_version), skipping..."
    return 0
  fi

  log "Installing TensorRT via pip..."

  # Try pip install first
  if pip install tensorrt==10.0 2>/dev/null; then
    success "TensorRT installed via pip"
    state_set "tensorrt_installed" "yes"
    state_set "tensorrt_version" "${TENSORRT_VERSION}"
    return 0
  fi

  # Fallback: conda
  if command -v conda &> /dev/null; then
    log "Trying conda install..."
    if conda install -y -c conda-forge tensorrt=10.0 2>/dev/null; then
      success "TensorRT installed via conda"
      state_set "tensorrt_installed" "yes"
      state_set "tensorrt_version" "${TENSORRT_VERSION}"
      return 0
    fi
  fi

  # Provide manual download instructions
  warn "TensorRT installation failed via automated methods."
  log "Manual installation steps:"
  log "  1. Download from: https://developer.nvidia.com/tensorrt"
  log "  2. Select: TensorRT 10.x for CUDA 13.0, Linux aarch64"
  log "  3. Extract to /usr/local/tensorrt"
  log "     tar -xzf tensorrt-*.tar.gz"
  log "     sudo cp -r tensorrt-*/* /usr/local/tensorrt/"
  log "  4. Install Python bindings:"
  log "     cd /usr/local/tensorrt/python && pip install ."
  log "  5. Update LD_LIBRARY_PATH and PYTHONPATH in venv scripts"
  log ""
  log "Continuing setup without TensorRT (workloads may not require it)..."
  state_set "tensorrt_installed" "manual-pending"
  return 0
}

# ============================================================================
# Phase 5: Create Virtual Environments
# ============================================================================

create_venvs() {
  log "Creating isolated Python virtual environments..."

  for workload in "${WORKLOADS[@]}"; do
    local venv_path="${PROJECT_DIR}/env/${workload}"
    local marker_file="${PROJECT_DIR}/env/.${workload}-created"

    if [[ -f "${marker_file}" ]]; then
      log "venv for ${workload} already exists, skipping..."
      continue
    fi

    log "Creating venv for ${workload}..."
    python3.12 -m venv "${venv_path}"

    # Extend activation script with CUDA paths
    local activate_script="${venv_path}/bin/activate"
    cat >> "${activate_script}" << 'VENV_EOF'

# CUDA library paths (for pip-installed cuDNN/TensorRT)
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"
VENV_EOF

    # Create marker
    touch "${marker_file}"
    success "venv created for ${workload}"
  done
}

# ============================================================================
# Phase 6: Install Requirements
# ============================================================================

install_requirements() {
  log "Installing Python requirements for each workload..."

  for workload in "${WORKLOADS[@]}"; do
    local venv_path="${PROJECT_DIR}/env/${workload}"
    local requirements="${PROJECT_DIR}/config/requirements-${workload}.txt"

    if [[ ! -f "${requirements}" ]]; then
      log "Skipping ${workload}: requirements file not found (${requirements})"
      continue
    fi

    log "Installing requirements for ${workload}..."

    # Use venv's pip directly
    if [[ ! -f "${venv_path}/bin/pip" ]]; then
      warn "pip not found in ${workload} venv, skipping..."
      continue
    fi

    # Upgrade pip
    "${venv_path}/bin/pip" install --upgrade pip setuptools wheel > /dev/null 2>&1 || true

    # Install requirements
    "${venv_path}/bin/pip" install -r "${requirements}" > /dev/null 2>&1 || {
      warn "Some packages failed to install for ${workload} (may be OK if optional)"
    }

    success "Requirements processed for ${workload}"
  done
}

# ============================================================================
# Phase 7: Start Model Downloader
# ============================================================================

start_model_downloader() {
  log "Starting model downloader queue..."

  local model_script="${PROJECT_DIR}/scripts/model-downloader.py"

  if [[ ! -f "${model_script}" ]]; then
    warn "Model downloader script not found (${model_script}), skipping..."
    return 0
  fi

  # Run downloader in background with nohup so it survives script exit
  source "${PROJECT_DIR}/env/llm/bin/activate" 2>/dev/null || true

  nohup python3 "${model_script}" >> "${PROJECT_DIR}/logs/model-download.log" 2>&1 &
  local downloader_pid=$!

  success "Model downloader started (PID: $downloader_pid)"
  state_set "model_downloader_pid" "${downloader_pid}"
}

# ============================================================================
# Phase 8: Validation
# ============================================================================

validate_setup() {
  log "Validating setup..."

  local errors=0

  # Check venvs
  for workload in "${WORKLOADS[@]}"; do
    local venv_path="${PROJECT_DIR}/env/${workload}"
    if [[ ! -f "${venv_path}/bin/activate" ]]; then
      err "venv not found for ${workload}"
      ((errors++))
    fi
  done

  # Test venv activation and imports
  for workload in "${WORKLOADS[@]}"; do
    local venv_path="${PROJECT_DIR}/env/${workload}"
    if source "${venv_path}/bin/activate" 2>/dev/null; then
      if python3 -c "import torch" 2>/dev/null; then
        success "Venv ${workload}: torch importable"
      else
        warn "Venv ${workload}: torch not yet installed (will install from requirements)"
      fi
      deactivate 2>/dev/null || true
    fi
  done

  if (( errors > 0 )); then
    err "Validation failed with ${errors} errors"
    return 1
  fi

  success "Validation passed"
}

# ============================================================================
# Main Execution Flow
# ============================================================================

main() {
  log "=========================================="
  log "GB10 Demo Suite Setup"
  log "=========================================="

  phase_init

  if ! check_system; then
    err "System checks failed"
    exit 1
  fi

  if ! install_cudnn; then
    warn "cuDNN installation encountered issues (will continue with manual steps)"
  fi

  if ! install_tensorrt; then
    warn "TensorRT installation encountered issues (will continue with manual steps)"
  fi

  if ! create_venvs; then
    err "venv creation failed"
    exit 1
  fi

  if ! install_requirements; then
    warn "Requirements installation encountered issues (may be OK if config files incomplete)"
  fi

  start_model_downloader

  if ! validate_setup; then
    warn "Setup validation completed with warnings (some components may not be fully installed yet)"
  fi

  state_set "last_run" "$(date '+%Y-%m-%d %H:%M:%S')"

  log "=========================================="
  success "Setup completed!"
  log "=========================================="
  log "Next steps:"
  log "  1. Update config/requirements-*.txt with workload dependencies"
  log "  2. Update config/model-lists.json with HuggingFace model IDs"
  log "  3. Monitor model downloads: tail -f logs/model-download.log"
  log "  4. Test venvs:"
  for workload in "${WORKLOADS[@]}"; do
    log "     source env/${workload}/bin/activate && python -c 'import torch; print(torch.cuda.is_available())'"
  done
  log "  5. Run workloads from workloads/<name>/ directories"
  log "=========================================="
  log "Setup log: ${LOG_FILE}"
  log "State file: ${STATE_FILE}"
}

main "$@"
