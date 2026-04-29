#!/bin/bash
#
# Build and install uvc-util — Neo-side helper for PTZ control of the
# EMEET PIXY camera that Teams uses. Used by host/ptz_camera.py via
# the /table PTZ panel.
#
# uvc-util is jtfrey's macOS UVC control utility:
#   https://github.com/jtfrey/uvc-util
#
# Not in Homebrew. macOS only — needs the Xcode Command Line Tools for
# the IOKit + Foundation frameworks.
#
# Idempotent: clones into ~/src/uvc-util on first run, pulls + rebuilds
# on subsequent runs. Installs the binary to ~/bin/uvc-util (which the
# host's PTZ_BINARY_PATHS already searches).
#

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

if [[ "$(uname)" != "Darwin" ]]; then
    error "uvc-util is macOS-only — skipping"
fi

REPO_URL="https://github.com/jtfrey/uvc-util.git"
SRC_DIR="$HOME/src/uvc-util"
BIN_DIR="$HOME/bin"
BIN_PATH="$BIN_DIR/uvc-util"

# Xcode CLT provides /usr/bin/gcc + the system frameworks we link.
if ! xcode-select -p >/dev/null 2>&1; then
    info "Installing Xcode Command Line Tools..."
    xcode-select --install
    warn "Re-run this script once the CLT installer finishes."
    exit 1
fi

mkdir -p "$BIN_DIR"
mkdir -p "$(dirname "$SRC_DIR")"

if [ -d "$SRC_DIR/.git" ]; then
    info "Updating uvc-util source..."
    cd "$SRC_DIR"
    git pull
else
    info "Cloning uvc-util..."
    git clone "$REPO_URL" "$SRC_DIR"
    cd "$SRC_DIR"
fi

# Build command per upstream README / source layout — single gcc
# invocation against the four .m files, linking IOKit + Foundation.
cd "$SRC_DIR/src"
info "Building uvc-util..."
gcc -o uvc-util \
    -framework IOKit -framework Foundation \
    uvc-util.m UVCController.m UVCType.m UVCValue.m

if [ ! -x "$SRC_DIR/src/uvc-util" ]; then
    error "Build did not produce $SRC_DIR/src/uvc-util"
fi

info "Installing to $BIN_PATH"
cp "$SRC_DIR/src/uvc-util" "$BIN_PATH"
chmod +x "$BIN_PATH"

# PATH check — ~/bin isn't on PATH by default on macOS. host/ptz_camera.py
# looks at $BIN_PATH directly so this only matters for shell use, but
# it's a useful nudge.
case ":$PATH:" in
    *":$BIN_DIR:"*) : ;;
    *) warn "$BIN_DIR is not on your PATH — host/ptz_camera.py finds the binary directly, but add 'export PATH=\"$BIN_DIR:\$PATH\"' to ~/.zshrc if you want to run uvc-util from the terminal." ;;
esac

info "Done. Run '$BIN_PATH -d' to list UVC devices."
