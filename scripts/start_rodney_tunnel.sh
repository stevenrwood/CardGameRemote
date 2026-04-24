#!/usr/bin/env bash
#
# start_rodney_tunnel.sh — spin up a Cloudflare Quick Tunnel that
# exposes the local host app to the public internet via a throwaway
# https://*.trycloudflare.com URL. Intended for a remote player
# (Rodney) to open /table in his own browser so no screen-share or
# remote-control grant is needed.
#
# Usage (in a second terminal, alongside the one running the host):
#     scripts/start_rodney_tunnel.sh
#
# On first run it brew-installs cloudflared if missing. Once the
# tunnel is up it:
#   - parses the URL out of cloudflared's banner
#   - appends /table
#   - copies the full URL to the macOS clipboard (pbcopy)
#   - prints it prominently in green so it's easy to grab for a
#     Teams chat / SMS
#
# Ctrl+C ends the tunnel. Every run gets a fresh URL.

set -u

PORT=${PORT:-8888}
APP_PATH=${APP_PATH:-/table}

info()  { printf '[INFO]  %s\n' "$*"; }
warn()  { printf '[WARN]  %s\n' "$*" >&2; }

# --- 1. Make sure cloudflared is installed --------------------------------

if ! command -v cloudflared >/dev/null 2>&1; then
    warn "cloudflared not found on PATH."
    if command -v brew >/dev/null 2>&1; then
        read -rp "Install via 'brew install cloudflared' now? [y/N] " answer
        case "$answer" in
            y|Y|yes|Yes)
                brew install cloudflared || {
                    warn "brew install failed. Install cloudflared manually and retry."
                    exit 1
                }
                ;;
            *)
                warn "Skipping install. Run 'brew install cloudflared' then rerun this script."
                exit 1
                ;;
        esac
    else
        warn "Homebrew not found either. Install Homebrew first, then 'brew install cloudflared'."
        exit 1
    fi
fi

info "cloudflared: $(cloudflared --version 2>&1 | head -1)"

# --- 2. Sanity-check the host app ------------------------------------------

if ! curl -sf -o /dev/null -m 2 "http://localhost:${PORT}/ping" \
        && ! curl -sf -o /dev/null -m 2 "http://localhost:${PORT}/" ; then
    warn "Nothing seems to be answering on http://localhost:${PORT} — is the host app running?"
    warn "Starting the tunnel anyway, but you'll get a 502 from Cloudflare until the app is up."
fi

# --- 3. Launch cloudflared and watch for the URL --------------------------

info "Starting Cloudflare Quick Tunnel → http://localhost:${PORT}"
info "(Ctrl+C to stop)"
echo

FOUND=0

# Line-buffer cloudflared so we see its output in real time. Regex
# pulls the first https://*.trycloudflare.com URL out of the banner,
# which is the stable shape for Quick Tunnels.
cloudflared tunnel --url "http://localhost:${PORT}" 2>&1 \
    | while IFS= read -r line; do
        printf '%s\n' "$line"
        if [[ $FOUND -eq 0 && "$line" =~ (https://[a-z0-9-]+\.trycloudflare\.com) ]]; then
            url="${BASH_REMATCH[1]}"
            full="${url}${APP_PATH}"
            printf '%s' "$full" | pbcopy 2>/dev/null || true
            printf '\n\033[1;32m=========================================================\n'
            printf ' Send Rodney this URL (already copied to clipboard):\n'
            printf '\n   %s\n\n' "$full"
            printf ' URL stays live until you Ctrl+C this script.\n'
            printf '=========================================================\033[0m\n\n'
            FOUND=1
        fi
    done
