# Card Game Remote — Software Components

## Component Names

| Name | Runs On | Description |
|------|---------|-------------|
| **Scan Controller** | Pi (Scanner) | Camera capture, card recognition, LED/buzzer control |
| **Game Controller** | Neo | Game engine, serves web UIs, speech recognition, overhead camera |
| **Remote Player Web UI** | External Monitor (via Neo) | Hand display, discard/challenge — shared to remote player via Teams |
| **Dealer Web UI** | Phone | Game controls, status — optimized for mobile |

## Overview

Two deployment targets, one shared codebase where possible:

```
┌─────────────────────────────────────────────────────────────────┐
│                     MacBook (Host)                              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Host App    │  │ Speech       │  │ Overhead Camera        │ │
│  │ (FastAPI)   │  │ Recognition  │  │ AI Vision              │ │
│  │             │  │ (Whisper)    │  │ (Claude API or YOLO)   │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘ │
│         │                │                      │              │
│         └────────────────┴──────────────────────┘              │
│                          │                                     │
│         Neo screen (lid closed): /remote                       │
│         Remote player sees/controls via Teams screen share     │
│         Dealer phone on WiFi: /host                            │
└──────────────────────────┼─────────────────────────────────────┘
           WiFi LAN        │
                │          │
┌───────────────┴──┐       │
│   Raspberry Pi   │       │
│                  │       │
│  ┌────────────┐  │       │
│  │ Pi Scanner │  │       │
│  │ Server     │  │       │
│  └────────────┘  │       │
└──────────────────┘       │
```

---

## Scan Controller

**Runs on:** Pi (Scanner)
**Purpose:** Camera capture, card recognition, LED/buzzer control
**Language:** Python

### Files

| File | Status | Description |
|------|--------|-------------|
| `pi/scan_controller.py` | **TO BUILD** | Main server — HTTP + WebSocket API, ties everything together |
| `pi/card_recognition/detector.py` | **BUILT** | Card identification via corner template matching (100% accuracy) |
| `pi/card_recognition/train.py` | **BUILT** | Generate reference templates from card photos |
| `pi/card_recognition/test_detector.py` | **BUILT** | Test harness with debug visualization |
| `pi/card_recognition/generate_cards.py` | **BUILT** | Synthetic card image generator (fallback) |
| `pi/hardware/camera.py` | **TO BUILD** | Pi Camera v3 capture — frame grab, LED flash coordination |
| `pi/hardware/leds.py` | **TO BUILD** | Flash LED control (MOSFET on/off) + green slot LEDs (individual GPIO) |
| `pi/hardware/buzzer.py` | **TO BUILD** | Piezo buzzer — beep patterns (success, warning, error) |
| `pi/slot_monitor.py` | **TO BUILD** | Continuous monitoring — detect card changes across 7 slots |

### scan_controller.py — Main Scan Controller

Responsibilities:
- Run HTTP + WebSocket server (Flask or FastAPI)
- Continuous monitoring loop: every ~2 seconds, flash LEDs, capture frame, check for card changes
- When new card detected in a slot: identify it, beep, notify host via WebSocket
- Accept commands from host: set green LEDs, reset state
- Report status (connected, slot states)

**API (Host → Pi):**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Health check, current slot states |
| `/leds/green` | POST | Set green LED states `{"slots": [2, 4]}` (on) or `{"slots": []}` (all off) |
| `/mode` | POST | `{"mode": "monitoring"}` or `{"mode": "idle"}` |
| `/reset` | POST | Clear all slot states, turn off all LEDs |

**WebSocket Events (Pi → Host):**

| Event | Description |
|-------|-------------|
| `card_detected` | New card found in slot `{"slot": 3, "card": {"rank": "K", "suit": "hearts"}, "confidence": 0.97}` |
| `card_removed` | Card removed from slot `{"slot": 3}` |
| `slot_change` | Bulk update of all slot states (for sync) |
| `error` | Hardware or recognition error |

### camera.py — Camera + LED Flash Coordination

```python
class ScannerCamera:
    def capture_with_flash(self) -> np.ndarray:
        """Flash LEDs, capture frame, LEDs off. Returns image."""

    def capture_raw(self) -> np.ndarray:
        """Capture without flash (for ambient light comparison)."""
```

### leds.py — LED Control

```python
class LEDController:
    def flash_on(self):
        """Turn on all 10 flash LEDs (via MOSFET)."""

    def flash_off(self):
        """Turn off all 10 flash LEDs."""

    def set_green_leds(self, slots: list[int]):
        """Turn on green LEDs for specified slots, off for all others."""

    def all_green_off(self):
        """Turn off all green LEDs."""
```

### buzzer.py — Buzzer Control

```python
class Buzzer:
    def beep(self, duration=0.15):
        """Single beep — card recognized."""

    def double_beep(self):
        """Double beep — low confidence."""

    def error_tone(self):
        """Error tone — card not recognized."""
```

### slot_monitor.py — Continuous Card Monitoring

The core loop that watches for card changes across all 7 slots.

```python
class SlotMonitor:
    def __init__(self, camera, detector, leds, buzzer):
        self.slots = [None] * 7  # Current card in each slot (or None)

    async def monitor_loop(self):
        """Every ~2 seconds: flash, capture, check each slot window."""

    def _extract_slot_windows(self, frame) -> list[np.ndarray]:
        """Crop the 7 slot windows from the full camera frame."""

    def _detect_changes(self, new_cards) -> list[dict]:
        """Compare new scan against current state, return changes."""
```

**Key logic:**
- Extract 7 regions from one camera frame (one per slot window)
- For each region, determine: empty, same card as before, or new card
- Empty detection: check if region has enough "ink" content vs blank
- New card: run detector, if confidence > threshold → notify host
- Card removed: region goes from card to empty → notify host

### Multi-Card Detection Changes to detector.py

The current detector finds one card in one image. For the platen design,
it needs to identify a card from a small cropped window (10x25mm of the corner).

**Changes needed:**
- The slot_monitor crops individual windows from the full frame
- Each window is passed to the detector as if it were a single card corner
- The detector's `_extract_card` step is skipped — the window IS the corner
- Template matching runs directly on the cropped window
- Add a new method: `identify_corner(corner_image) -> CardResult`

---

## Game Controller

**Runs on:** Neo (MacBook M4)
**Purpose:** Game flow control, scan controller communication, serve both web UIs, speech recognition, overhead camera AI vision
**Language:** Python (FastAPI backend) + HTML/CSS/JS (frontend)

### Files

| File | Status | Description |
|------|--------|-------------|
| `host/app.py` | **BUILT** | FastAPI server — routes, WebSocket, scan controller connection |
| `host/game_engine.py` | **BUILT** | Game state, 9 templates, phase management, wild card tracking, player/dealer rotation |
| `host/pi_client.py` | **BUILT** | Async client for scan controller API + mock for dev |
| `host/speech.py` | **TO BUILD** | Whisper-based speech recognition for game/card announcements |
| `host/overhead_cam.py` | **TO BUILD** | Brio 4K capture + AI vision for table card recognition |
| `host/static/dealer.js` | **BUILT** | Dealer web UI logic (optimized for phone) |
| `host/static/remote.js` | **BUILT** | Remote player web UI logic |
| `host/static/style.css` | **BUILT** | Shared styles |
| `host/templates/dealer.html` | **BUILT** | Dealer web UI page (optimized for phone) |
| `host/templates/remote.html` | **BUILT** | Remote player web UI page |

### Updates Needed to Existing Files

**app.py:**
- Handle `card_detected` and `card_removed` events from Pi (not just `card_scanned`)
- Green LED control: when remote player selects discards/challenge, send LED command to Pi
- Integrate speech recognition events
- Add overhead camera API endpoints
- Add challenge comparison flow (track multiple challengers)

**game_engine.py:**
- Track slot contents (which card is in which physical slot)
- Detect card swaps (for challenge flow — different card appears in same slot)
- `card_removed(slot_number)` method
- `card_replaced(slot_number, rank, suit)` method for challenge scanning
- Challenge state machine: remember remote player's cards, accumulate challenger hands

**pi_client.py:**
- Add `set_green_leds(slots)` method
- Handle new event types: `card_detected`, `card_removed`, `slot_change`
- Update mock client to simulate multi-slot behavior

**host.js:**
- Display slot-based hand (cards mapped to physical slot positions)
- Show Pi scanner box status with slot occupancy
- Challenge flow UI (show challenger hands as they're scanned)

**remote.js:**
- Challenge flow: show accumulating challenger hands
- Improved discard UI with green LED feedback confirmation

### speech.py — Speech Recognition

Uses OpenAI Whisper running locally on M4.

```python
class SpeechRecognizer:
    def __init__(self):
        """Load Whisper model (base or small — M4 handles easily)."""

    async def listen_loop(self, callback):
        """Continuous listening from MacBook microphone."""

    def parse_game_name(self, text: str) -> str | None:
        """Match spoken text to a game template name."""
        # "we're playing follow the queen" → "Follow the Queen"

    def parse_card_call(self, text: str) -> dict | None:
        """Match spoken card announcement."""
        # "king of hearts to Steve" → {"rank": "K", "suit": "hearts", "player": "Steve"}
```

**Integration with game engine:**
- Game name detected → auto-select template, start new hand
- Up card called → inject as `card_dealt` with `card_type: "up"` (no scanner needed)
- Dealer rotation tracked → know when called card is for remote player vs others

### overhead_cam.py — AI Vision for Table Cards

**Phase 1: Vision API**

```python
class OverheadCamera:
    def __init__(self, camera_index=1):
        """Initialize Brio 4K capture."""

    def capture_table(self) -> np.ndarray:
        """Capture current table image."""

    def get_player_zones(self) -> list[np.ndarray]:
        """Crop frame into 5 pre-calibrated player zones."""

    async def read_table_cards(self) -> dict:
        """Send zone crops to Claude/GPT-4o, return identified cards per player."""
        # Returns: {"player_1": ["Kh", "Qs"], "player_2": ["Ac"], ...}

    def calibrate_zones(self):
        """Interactive calibration: click to define player zone boundaries."""
```

**Phase 2: Local YOLO (if needed)**

```python
class YOLOCardDetector:
    def __init__(self, model_path="models/poker_cards.pt"):
        """Load trained YOLOv8 model."""

    def detect(self, image: np.ndarray) -> list[dict]:
        """Run inference, return detected cards with positions."""

    def map_to_players(self, detections, zones) -> dict:
        """Map card positions to player zones."""
```

---

## Remote Player Web UI

**Displayed on:** External monitor (face-down on table), shared to remote player via Teams
**Purpose:** Display hand, select discards/challenges, show table cards
**Technology:** HTML/CSS/JS served by Game Controller — browser only

### Files

| File | Status | Description |
|------|--------|-------------|
| `host/static/remote.js` | **BUILT** | WebSocket client, hand rendering, discard/challenge UI |
| `host/templates/remote.html` | **BUILT** | Page structure |
| `host/static/style.css` | **BUILT** | Shared styles |

### Updates Needed

- Challenge comparison display (show multiple challenger hands)
- Wild card display improvements
- Table cards display (from overhead camera AI vision)
- Connection recovery (auto-reconnect, state replay)
- Sound notifications (beep when card dealt, alert when action needed)

---

## Build Order (Recommended)

### Phase 1: Core System (Minimum Viable)

Get the basic deal-scan-display loop working end to end.

| # | Task | Component | Dependencies |
|---|------|-----------|-------------|
| 1 | Update detector for corner-only identification | Scan Controller | None |
| 2 | Build slot_monitor.py (multi-slot continuous scanning) | Scan Controller | #1 |
| 3 | Build hardware control (camera.py, leds.py, buzzer.py) | Scan Controller | None |
| 4 | Build scan_controller.py (HTTP + WebSocket) | Scan Controller | #2, #3 |
| 5 | Update pi_client.py for new events + green LEDs | Game Controller | #4 |
| 6 | Update app.py for slot-based flow + green LED commands | Game Controller | #5 |
| 7 | Update game_engine.py for slot tracking + card removal | Game Controller | None |
| 8 | Update dealer.js + remote.js for slot-based display | Web UIs | #6, #7 |
| 9 | End-to-end test with mock scan controller (no hardware) | All | #1-8 |

**Result:** Full deal → scan → display → discard → replace cycle working in browser.

### Phase 2: Speech Recognition

| # | Task | Component | Dependencies |
|---|------|-----------|-------------|
| 10 | Build speech.py (Whisper integration) | Game Controller | None |
| 11 | Integrate speech events into app.py + game_engine | Game Controller | #10, Phase 1 |
| 12 | Test game selection by voice | Game Controller | #11 |
| 13 | Test up card recognition by voice | Game Controller | #11 |

**Result:** Dealer can announce games and up cards verbally.

### Phase 3: Hardware Integration

| # | Task | Component | Dependencies |
|---|------|-----------|-------------|
| 14 | Assemble Scanner + wiring | Hardware | — |
| 15 | Train card recognition on actual Bicycle deck | Scan Controller | #14 |
| 16 | Calibrate slot window positions in camera frame | Scan Controller | #14, #15 |
| 17 | End-to-end test with real Scanner hardware | All | Phase 1, #14-16 |

**Result:** Physical scanner box working with real cards.

### Phase 4: Overhead Camera + AI Vision

| # | Task | Component | Dependencies |
|---|------|-----------|-------------|
| 18 | Build overhead_cam.py (Brio capture + zone cropping) | Game Controller | None |
| 19 | Integrate Claude/GPT-4o vision API | Game Controller | #18 |
| 20 | Zone calibration tool | Game Controller | #18 |
| 21 | Display table cards in remote player web UI | Remote Player Web UI | #19 |
| 22 | Test with live game | All | Phase 3, #18-21 |

**Result:** Remote player sees other players' up cards identified digitally.

### Phase 5: Polish

| # | Task | Component | Dependencies |
|---|------|-----------|-------------|
| 24 | Challenge comparison flow (multi-challenger) | Game Controller | Phase 1 |
| 25 | Connection recovery + state replay on reconnect | Game Controller | Phase 1 |
| 26 | Sound notifications in remote player web UI | Remote Player Web UI | Phase 1 |
| 27 | YOLO training (if vision API insufficient) | Game Controller | Phase 4 |

---

## Technology Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| Scan Controller | Python + Flask/FastAPI | Matches camera/GPIO libraries |
| Pi camera | picamera2 | Official Pi camera library |
| Pi GPIO | RPi.GPIO or gpiozero | Standard Pi GPIO control |
| Card recognition | OpenCV + template matching | Simple, proven, 100% accuracy |
| Game Controller | Python + FastAPI | Async, WebSocket support, serves both UIs |
| Web UIs | Vanilla HTML/CSS/JS | No build step, works in any browser |
| Real-time | WebSocket | Built into FastAPI and browsers |
| Speech | OpenAI Whisper (local) | Free, offline, runs well on M4 |
| AI vision (Phase 1) | Claude/GPT-4o Vision API | Zero training, immediate results |
| AI vision (Phase 2) | YOLOv8 (Ultralytics) | Fast local inference if API insufficient |
| Remote access | Teams screen share | No VPN, no networking config |
| Video/audio | Microsoft Teams | Already using it |
