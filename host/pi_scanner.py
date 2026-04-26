"""
Raspberry Pi scanner-box HTTP client.

All HTTP calls to the Pi (the scanner tube and slot rack that Rodney's
down cards go through) live here. The Pi offers:

- GET /ping                     — liveness check
- GET /slots?max_slot=N         — bulk scan of N slots
- POST /flash/hold | /release   — deal-mode flash LEDs
- POST /slots/<n>/led           — per-slot LED: on | off | blink
- POST /slots/<n>/scan          — force a single-slot scan

Each function takes the host-side ``s`` state object so it can read
``s.pi_base_url`` and update ``s.pi_offline`` / ``s.pi_flash_held``
tracking. They never touch any other host state; callers own the
table/game bookkeeping that drives these calls.

``HOST_CONFIG_PATH`` and ``_load_host_config`` / ``_save_host_config``
live here too — the host config file is where the Pi URL + scanner
tunables persist, so keeping load/save alongside the Pi wrappers means
one import for the whole Pi integration.
"""

import json
from pathlib import Path

from log_buffer import log


HOST_CONFIG_PATH = Path.home() / ".cardgame_host.json"


def _load_host_config() -> dict:
    """Read persisted host tunables. Returns {} if file is missing/bad."""
    try:
        return json.loads(HOST_CONFIG_PATH.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        log.log(f"[CONFIG] Could not read {HOST_CONFIG_PATH}: {e}")
        return {}


def _save_host_config(updates: dict) -> None:
    """Merge updates into the persisted host config and write back to disk."""
    cfg = _load_host_config()
    cfg.update(updates)
    try:
        HOST_CONFIG_PATH.write_text(json.dumps(cfg, indent=2) + "\n")
    except OSError as e:
        log.log(f"[CONFIG] Could not write {HOST_CONFIG_PATH}: {e}")


def _pi_ping(s, timeout_s: float = 1.5) -> bool:
    """One quick GET /ping to test Pi reachability. True on success."""
    import urllib.request
    try:
        url = f"{s.pi_base_url.rstrip('/')}/ping"
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            resp.read()
        return True
    except Exception:
        return False


def _pi_fetch_slots(s):
    """Fetch /slots from the Pi, limiting to the slots our game uses.

    Passes max_slot so the Pi skips capturing + matching the unused ones.
    Returns the parsed dict or None on error. If s.pi_offline is set (Deal
    determined the Pi was unreachable) this returns None without making a
    network call, so simulation kicks in immediately.
    """
    if s.pi_offline:
        return None
    import urllib.request
    # Late import avoids a circular dep: overhead_test imports this module,
    # and _total_downs_in_pattern is a small helper that reads game phases.
    from overhead_test import _total_downs_in_pattern
    max_slot = _total_downs_in_pattern(s.game_engine)
    if max_slot <= 0:
        return {"slots": []}
    try:
        url = f"{s.pi_base_url.rstrip('/')}/slots?max_slot={max_slot}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.log(f"[PI] /slots error: {e}")
        return None


def _pi_flash(s, hold):
    """Hold or release the Pi's flash LEDs. Tracks state to avoid redundant calls."""
    if s.pi_offline:
        return
    if s.pi_flash_held == hold:
        return
    import urllib.request
    path = "/flash/hold" if hold else "/flash/release"
    try:
        url = f"{s.pi_base_url.rstrip('/')}{path}"
        req = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
        s.pi_flash_held = hold
        log.log(f"[PI] flash {'held' if hold else 'released'}")
    except Exception as e:
        # Bare repr(e) catches empty-string errors (URLError / HTTPError).
        detail = repr(e) if not str(e) else str(e)
        log.log(f"[PI] {path} error: {type(e).__name__}: {detail}")


def _pi_buzz(s, n: int = 2, on_time: float = 0.12, off_time: float = 0.12):
    """POST /buzz — beep the Pi's piezo n times. Used as a quiet nag
    when cards are left in the scanner after a hand has ended."""
    if s.pi_offline:
        return
    import urllib.request
    url = f"{s.pi_base_url.rstrip('/')}/buzz"
    body = json.dumps({"n": n, "on_time": on_time, "off_time": off_time}).encode()
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as e:
        log.log(f"[PI] buzz failed: {type(e).__name__}: {e}")


def _pi_slot_led(s, slot_num: int, state: str):
    """POST /slots/<n>/led with state = on | off | blink."""
    if s.pi_offline:
        return
    import urllib.request
    url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/led"
    body = json.dumps({"state": state}).encode()
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=2).read()
    except Exception as e:
        log.log(f"[PI] LED slot {slot_num} {state} failed: "
                f"{type(e).__name__}: {e}")


def _pi_slot_scan(s, slot_num: int):
    """POST /slots/<n>/scan — returns dict (present/card/...) or None on error."""
    if s.pi_offline:
        return None
    import urllib.request
    url = f"{s.pi_base_url.rstrip('/')}/slots/{slot_num}/scan"
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        log.log(f"[PI] slot_scan {slot_num} failed: "
                f"{type(e).__name__}: {e}")
        return None
