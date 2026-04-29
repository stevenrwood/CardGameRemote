#!/bin/bash
#
# Card Game Remote — Setup and Run Script for Neo
#
# Clones/pulls the repo, installs dependencies, and launches
# the overhead camera test harness.
#
# Usage:
#   ./scripts/setup_and_run.sh              # default: camera index 1
#   ./scripts/setup_and_run.sh --camera 0   # specify camera index
#

set -e

# --- Configuration ---
REPO_URL="https://github.com/stevenrwood/CardGameRemote.git"
REPO_DIR="$HOME/Documents/GitHub/CardGameRemote"
VENV_DIR="$REPO_DIR/.venv"
CONFIG_FILE="$REPO_DIR/local/config.json"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# --- Clone or Pull ---
if [ -d "$REPO_DIR/.git" ]; then
    info "Pulling latest changes..."
    cd "$REPO_DIR"
    git pull
else
    if [ -d "$REPO_DIR" ]; then
        # Directory exists but no git repo — initialize it
        info "Initializing git repo in existing directory..."
        cd "$REPO_DIR"
        if git remote get-url origin 2>/dev/null; then
            git pull
        else
            warn "No git remote configured. Skipping pull."
            warn "To set up: git init && git remote add origin $REPO_URL"
        fi
    else
        info "Cloning repository..."
        git clone "$REPO_URL" "$REPO_DIR"
        cd "$REPO_DIR"
    fi
fi

# --- Create local config if it doesn't exist ---
mkdir -p "$REPO_DIR/local"
if [ ! -f "$CONFIG_FILE" ]; then
    info "Creating local/config.json — please add your Anthropic API key"
    cat > "$CONFIG_FILE" << 'EOF'
{
    "anthropic_api_key": "YOUR_KEY_HERE"
}
EOF
    warn "Edit $CONFIG_FILE and replace YOUR_KEY_HERE with your actual API key"
fi

# Check if API key is set
if grep -q "YOUR_KEY_HERE" "$CONFIG_FILE" 2>/dev/null; then
    warn "Anthropic API key not set in $CONFIG_FILE"
    warn "Card recognition will not work until you add your key"
    warn "The app will still run for calibration and change detection"
fi

# --- Ensure Homebrew is in PATH (Apple Silicon) ---
if [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# --- Install Homebrew if needed ---
if ! command -v brew &>/dev/null; then
    info "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# --- Ensure Python 3.10+ ---
# The codebase uses PEP 604 union syntax (`int | None`) in several
# places, which requires Python 3.10+. Apple's CommandLineTools
# python3 is pinned to 3.9 — silently importing under it crashes the
# speech listener at startup. Detect and upgrade via Homebrew when
# the active python3 is too old.
NEED_MIN_MINOR=10
DESIRED_MINOR=13   # current stable when we need to install fresh

py_minor() {
    "$1" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0
}

PYTHON_BIN="$(command -v python3 || true)"
CUR_MINOR=$(py_minor "$PYTHON_BIN")

if [ "$CUR_MINOR" -lt "$NEED_MIN_MINOR" ]; then
    warn "Detected python3 = 3.${CUR_MINOR}; need 3.${NEED_MIN_MINOR}+"
    info "Installing python@${DESIRED_MINOR} via Homebrew..."
    brew install "python@${DESIRED_MINOR}"
    BREW_PYTHON_PREFIX="$(brew --prefix "python@${DESIRED_MINOR}")"
    # Homebrew exposes the unversioned `python3` under libexec/bin
    # so it doesn't conflict with system python on PATH.
    if [ -x "${BREW_PYTHON_PREFIX}/libexec/bin/python3" ]; then
        PYTHON_BIN="${BREW_PYTHON_PREFIX}/libexec/bin/python3"
    elif [ -x "${BREW_PYTHON_PREFIX}/bin/python3.${DESIRED_MINOR}" ]; then
        PYTHON_BIN="${BREW_PYTHON_PREFIX}/bin/python3.${DESIRED_MINOR}"
    else
        error "Could not locate Homebrew python@${DESIRED_MINOR} after install"
    fi
    CUR_MINOR=$(py_minor "$PYTHON_BIN")
fi

info "Using $PYTHON_BIN (Python 3.${CUR_MINOR})"

# --- Install ffmpeg if needed ---
if ! command -v ffmpeg &>/dev/null; then
    info "Installing ffmpeg..."
    brew install ffmpeg
fi

# --- Set up Python virtual environment ---
# If the existing venv was built against an old Python (we just
# upgraded out from under it), rebuild rather than silently using
# the stale interpreter.
if [ -d "$VENV_DIR" ]; then
    VENV_MINOR=$(py_minor "$VENV_DIR/bin/python")
    if [ "$VENV_MINOR" -lt "$NEED_MIN_MINOR" ]; then
        warn "Existing venv at ${VENV_DIR} is Python 3.${VENV_MINOR}; rebuilding with 3.${CUR_MINOR}"
        rm -rf "$VENV_DIR"
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# --- Install Python dependencies ---
info "Installing/updating dependencies..."
pip install --quiet --upgrade pip
pip install --quiet opencv-python numpy anthropic SpeechRecognition pyaudio
# mlx-whisper is the on-device speech-to-text used by the optional
# --listen flag. Apple Silicon only — installs fine on Intel Macs
# but errors at runtime; --listen stays off if it fails to import.
pip install --quiet mlx-whisper
# pyobjc-framework-AVFoundation gives us the macOS camera-name lookup so the
# overhead capture opens the Logitech Brio rather than whichever other 4K
# webcam OpenCV happens to enumerate first.
pip install --quiet pyobjc-framework-AVFoundation
# ultralytics drives YOLO-based up-card recognition from the Brio. Without
# it, host/zone_monitor.py logs "YOLO load failed" at startup and every
# up-card falls through to the Claude API path (slower + costs credits).
# Idempotent — skip when already importable so re-running the launcher
# doesn't redo the ~200MB torch+ultralytics install.
#
# Unlike the Pi side (which has to force --index-url for CPU-only torch
# to avoid pulling 6GB of NVIDIA CUDA wheels on Linux), default PyPI
# torch wheels on macOS are already CPU/MPS — no special index needed.
if ! python -c "import ultralytics" 2>/dev/null; then
    info "Installing ultralytics (first run pulls torch + torchvision)..."
    pip install --quiet ultralytics
else
    info "ultralytics already installed"
fi

# --- Create training data directory ---
mkdir -p "$REPO_DIR/host/training_data"

# --- Run the test harness ---
info "Starting overhead camera test harness..."
info "Controls: c=calibrate, r=reset baselines, s=snapshot, q=quit"
echo ""

cd "$REPO_DIR"
"$VENV_DIR/bin/python" host/main.py "$@"
