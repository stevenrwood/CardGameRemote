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

# --- Set up Python virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

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

# --- Install ffmpeg if needed ---
if ! command -v ffmpeg &>/dev/null; then
    info "Installing ffmpeg..."
    brew install ffmpeg
fi

# --- Install Python dependencies ---
info "Installing/updating dependencies..."
pip install --quiet --upgrade pip
pip install --quiet opencv-python numpy anthropic SpeechRecognition pyaudio

# --- Create training data directory ---
mkdir -p "$REPO_DIR/host/training_data"

# --- Run the test harness ---
info "Starting overhead camera test harness..."
info "Controls: c=calibrate, r=reset baselines, s=snapshot, q=quit"
echo ""

cd "$REPO_DIR"
"$VENV_DIR/bin/python" host/overhead_test.py "$@"
