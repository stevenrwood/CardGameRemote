# Card Game Remote — Project Plan

## Overview

Hardware/software system enabling a remote poker player to participate in a monthly
home game via Microsoft Teams. One of five players in a 35-year poker group is moving
to Texas. This system lets them continue playing remotely.

**Philosophy:** Keep it simple. Start with the minimum viable system, iterate only
when something proves inadequate.

---

## Component Names

### Hardware

| Name | Description |
|------|-------------|
| **Scanner** | Raspberry Pi, 2 cameras, 10 illumination LEDs, 7 green card-state LEDs, mirror, buzzer, enclosure with 7 card trays |
| **Neo** | MacBook Pro M4 — runs game controller, Teams, overhead camera AI vision |
| **External Monitor** | Connected to Neo, face-down on table — displays remote player web UI |
| **Overhead Camera** | Logitech Brio 4K, mounted on chandelier — AI vision for table cards |
| **Phone** | Dealer's mobile phone — displays dealer web UI |

### Software

| Name | Description |
|------|-------------|
| **Scan Controller** | Python script running on Pi — camera capture, card recognition, LED/buzzer control, continuous monitoring |
| **Game Controller** | Python script running on Neo — game engine, templates, speech recognition, overhead camera AI vision, serves both web UIs |
| **Remote Player Web UI** | Browser-based UI displayed on external monitor, shared with remote player via Teams screen share |
| **Dealer Web UI** | Browser-based UI optimized for display on mobile phone |

---

## System Architecture

```
┌─────────────────┐   WiFi    ┌──────────────────────────────────────────┐
│   Pi Scanner     │◄────────►│   MacBook Neo (M4) — lid open             │
│   Box            │  LAN     │                                          │
│                  │          │  ┌─────────────┐  ┌────────────────────┐ │
│ - 1-2 Pi Cam v3 │          │  │ Host App    │  │ Teams              │ │
│ - LEDs (flash)   │          │  │ (FastAPI)   │  │ - Remote player    │ │
│ - 7 green LEDs  │          │  │ localhost   │  │   face on Neo      │ │
│ - Mirror        │          │  │             │  │   screen (visible  │ │
│ - Buzzer        │          │  └─────────────┘  │   to table)        │ │
│                  │          │                   │ - Screen share     │ │
└─────────────────┘          │  ┌─────────────┐  │   ext monitor to   │ │
                             │  │ Brio 4K     │  │   remote player    │ │
                             │  │ + AI vision │  └────────────────────┘ │
                             │  └─────────────┘                         │
                             │  /host → dealer's phone                  │
                             └──────────────────────────────────────────┘
                                        │
                             ┌──────────┴──────────┐
                             │  External Monitor    │
                             │  (face down on table)│
                             │  /remote browser     │
                             │  Remote player sees  │
                             │  + controls via      │
                             │  Teams screen share  │
                             └─────────────────────┘

   Neo screen: Remote player's face via Teams (visible to table)
   External monitor (face down): /remote with full hand (only remote player sees)
   Dealer's phone: /host dealer controls
```

---

## Part 1: Scanner Box (Pi)

### Hardware Design — Glass-Free Platen with Mirror

A low-profile box sits on the poker table next to the laptop.

**Enclosure:** 2" tall × ~20" wide × 14" deep
- Bottom: 1/4" plywood
- Walls: plywood
- Interior painted flat black
- No glass platen — 7 individual 3D-printed card trays on top

**Card Trays (x7):**
- Each tray has 2 side walls and a back wall to guide the card to a fixed position
- Card placed face-down — face visible through a small 10x25mm window in the tray floor
- Window exposes exactly the card corner (rank + suit symbol)
- Green indicator LED per tray (visible from above)
- Removable/replaceable — printed in PLA

**Mirror:**
- ~500mm wide × ~65mm tall (face dimension)
- Bent from single piece of 5052 aluminum (1/16" / 0.062"), mirror mylar film
  applied to the 45° face — acts as a first-surface mirror (no ghosting)
- Apply mylar before bending for smoothest result
- Bend creates both the mirror face and mounting base — no separate bracket needed
- Angle doesn't need to be precisely 45° — a few degrees off just shifts the
  image in the camera frame; autofocus and card detection handle it
- Mounted near the front of box, under the card trays
- Redirects camera view from horizontal to vertical (looking up at card windows)

**Camera:**
- 1x Pi Camera Module v3 (autofocus, ordered)
- Mounted on inside of back wall, pointing forward at mirror
- Total folded optical path: ~400mm (camera → mirror → card)
- Single camera covers all 7 windows at ~9+ px/mm resolution
- Second camera can be added if needed — start with one
- Software applies `cv2.flip(frame, 1)` to correct mirror reflection

**Illumination:**
- 10x 5mm diffused flat-top cool white LEDs
- Mounted in 5x 3D-printed brackets, 2 LEDs each (one high, one low relative to camera lens)
- Brackets spaced evenly across the width of the mirror, on the same plane as the camera
- Flash only: on for ~50ms during capture, imperceptible to players
- Driven by single MOSFET (IRLZ44N) from one GPIO pin
- 39Ω resistor per LED for ~40mA flash current
- 330Ω gate resistor + 10K pull-down on MOSFET gate

**Green Slot Indicator LEDs (x7):**
- Pre-wired with inline resistor, one per card tray
- Each on its own GPIO pin for individual control
- Light up when remote player selects cards to discard or challenge with
- Turn off when software detects new/original cards in those slots
- Wiring connections at front of box, between front wall and back of mirror
- 8-conductor ribbon cable (7 signal + 1 GND) runs along box bottom to Pi
- IDC screw terminal breakout boards on each end for clean disconnect

**Buzzer:**
- Passive piezo buzzer on one GPIO pin
- Single beep = card recognized
- Double beep = low confidence
- Error tone = not recognized

**Raspberry Pi Compute Module 4 (CM4):**
- CM4 with 2GB RAM, WiFi, no eMMC (boots from SD card)
- CM4 base board with 2x CSI camera ribbon connectors (one per camera)
- Mounted vertically behind the camera and LED brackets
- Headless — no display
- USB-C panel mount power input and external antenna connector on back wall
- WiFi connection to Neo
- 3D-printed camera mount bracket

**Interior Layout (side view):**
```
    ┌─card trays──────────────────────────────────┐
    │          ╲ mirror    │ LED  │         │      │
    │           ╲          │mounts│ camera  │  Pi  │
    │            ╲         │      │         │      │
    └──green LED──╲────────┴──────┴─────────┴──────┘
      wiring       ╲                          USB-C
    (front)     (front-mid)    (mid-back)    (back)
```

**Pi GPIO Pin Assignments (BCM numbering):**

All outputs use GPIO 9-27 (default pull-down at boot — nothing activates during startup).
GPIO 2, 3 reserved for Camera v3 I2C. GPIO 14, 15 avoided (UART).

| Signal | BCM GPIO | Physical Pin |
|--------|----------|-------------|
| Green LED - Slot 1 | GPIO 17 | Pin 11 |
| Green LED - Slot 2 | GPIO 27 | Pin 13 |
| Green LED - Slot 3 | GPIO 22 | Pin 15 |
| Green LED - Slot 4 | GPIO 23 | Pin 16 |
| Green LED - Slot 5 | GPIO 24 | Pin 18 |
| Green LED - Slot 6 | GPIO 25 | Pin 22 |
| Green LED - Slot 7 | GPIO 12 | Pin 32 |
| Flash MOSFET (10 white LEDs) | GPIO 16 | Pin 36 |
| Piezo Buzzer | GPIO 26 | Pin 37 |
| Camera v3 (#1) | CSI-0 ribbon + I2C | Reserved |
| Camera v3 (#2) | CSI-1 ribbon + I2C | Reserved |
| **Total** | **9 GPIO + 2x CSI** | |

Free for future use: GPIO 9, 10, 11, 13, 18, 19, 20, 21

**No motors, no gears, no moving parts.**

### Card Flow

```
1. Dealer places card face-down in next tray slot
2. Card settles against back wall, corner visible through window
3. Continuous monitoring: every ~2 seconds, LEDs flash, camera captures
4. Software detects new card in a slot via image comparison
5. Card identified by corner template matching (100% accuracy achieved)
6. Beep confirms recognition
7. Card identity + slot number sent to remote player via host app
```

### Discard Flow

```
1. Remote player selects cards to discard in browser UI
2. Corresponding green LEDs light up on scanner box
3. Dealer removes those cards from lit slots
4. Dealer places new cards in those same slots (or next available)
5. Software detects new cards, sends to remote player
6. Green LEDs turn off
```

### Challenge Flow (High/Low/High Challenge)

```
1. Remote player selects cards to challenge with → green LEDs on
2. Dealer removes remote player's cards from lit slots, sets aside face-down
3. Challenger 1's cards placed in those slots → scanned → shown to remote
4. Challenger 1's cards removed
5. ... repeat for up to 3 more challengers ...
6. Remote player's original cards returned to slots
7. Software recognizes originals → green LEDs off
8. Remote display shows all challenger hands until next deal
```

---

## Part 2: Host Application (MacBook)

### Technology

- **Backend:** Python / FastAPI
- **Frontend:** HTML/CSS/JS served by FastAPI (no separate build)
- **Two web views:**
  - `localhost:8000/host` — dealer control panel
  - `http://<tailscale-ip>:8000/remote` — remote player view (browser on Windows PC)
- **Real-time:** WebSocket connections for instant updates

### Game Engine

Phase-based template system supporting 9 poker variants:

| Game | Deal Pattern | Special |
|------|-------------|---------|
| 5 Card Draw | 5 down → draw(3) | |
| 3 Toed Pete | 3 down → draw(3) → draw(2) → draw(1) | |
| 7 Card Stud | d,d,u,u,u,u,d | |
| 7 Stud Deuces Wild | d,d,u,u,u,u,d | 2s always wild |
| Follow the Queen | d,d,u,u,u,u,d | Dynamic wilds tracked by software |
| High Chicago | d,d,u,u,u,u,d | Split pot |
| High/Low/High Challenge | 3d → challenge(2) → 2d → challenge(3) → 2d → challenge(5) | Repeatable |
| 7/27 | 2 down → hit rounds | Open-ended |
| Texas Hold'em | 2 down → community cards | Rare |

- Templates are user-editable JSON
- Host can override any template decision during play
- Dealer rotation tracked (rotates each hand among 5 players)

### Speech Recognition

Whisper running locally on the M4 MacBook:

1. **Game selection:** Dealer announces "We're playing Follow the Queen" →
   software auto-selects template, starts new hand

Up card recognition is handled by the overhead camera landing zones
(see Part 3), not speech recognition. Voice recognition for card names
proved unreliable — not all dealers call cards clearly enough.

### Host UI (Dealer View)

- Game selector dropdown + New Hand / End Hand buttons
- Current phase indicator (dealing / betting / draw / challenge)
- Remote player's hand display (all slots)
- Pi scanner status
- Dev panel: simulate card scans without hardware
- Manual card entry (rank/suit picker) as fallback

### Remote Player UI (Browser Only — No App to Install)

The remote player sees and controls the `/remote` view on an external
monitor connected to the Neo. The monitor is placed face-down on the
table so no one at the table can see the remote player's down cards.
The remote player accesses it via Teams screen sharing. No app to install.

- Current hand display with card images
- Cards clickable for discard/challenge selection
- Draw prompt: "Select up to N cards to discard" + Submit / Stand Pat
- Challenge prompt: "Select N cards" + Submit / Pass
- Wild card indicator (e.g., "8s are wild - follows Queen")
- Peek cards section (challenger hands during High/Low/High)
- Connection status indicator

### Communication

- **Pi ↔ Host:** WiFi (LAN), REST + WebSocket
- **Host ↔ Remote player:** Teams screen share of Neo's `/remote` browser window
- **Dealer controls:** Phone browser on same WiFi → `<neo-ip>:8000/host`
- **No VPN, no Tailscale, no port forwarding** — everything on localhost or LAN

### Physical Setup at Table

- **Neo laptop (open):** Shows remote player's face via Teams. Faces table so
  everyone can see the remote player — like they're sitting at the table.
- **External monitor (face down):** Shows `/remote` with full hand including down
  cards. Shared to remote player via Teams screen share. Face-down so only the
  remote player can see it through the screen share.
- **Scanner box:** Next to Neo, card trays accessible to dealer.
- **Dealer's phone:** Shows `/console` dealer control panel on same WiFi.
- **Overhead camera (Brio):** Mounted on chandelier, connected via Thunderbolt hub.

### USB-C / Thunderbolt Cabling

Neo has two USB-C ports. The main port is Thunderbolt (USB 3.2). The second
port is USB 2.0 only. The Brio requires USB 3.2 for 4K resolution and the
Ananta external monitor requires Thunderbolt for DisplayPort Alt Mode. Both
need the main port — solved with a Thunderbolt hub.

**Amazon Basics Thunderbolt 4 Hub** (ordered April 2026, $145):
- 1 upstream Thunderbolt port, 3 downstream Thunderbolt ports
- Includes 100W (20V/5A) power brick

```
100W PD Supply ──→ Ananta 17" display PD input (powers display)

Amazon Basics Thunderbolt Hub:
  Hub power brick ──→ Hub (powers hub + PD charges Neo)
  Hub upstream    ──→ Neo main USB-C port (Thunderbolt)
  Hub downstream 1 ──→ Brio 4K camera (USB 3.2 → 4K resolution)
  Hub downstream 2 ──→ Ananta Thunderbolt port (display signal)
  Hub downstream 3    (spare)

Neo second USB-C port: unused
```

This gives Neo power, 4K camera, and external display all through one port.

---

## Part 3: Overhead Table Camera (Future Enhancement)

### Purpose

AI vision reads other players' face-up cards from the overhead camera,
displaying them digitally to the remote player. Supplements the Teams
video feed with precise card identification.

### Hardware

- **Camera:** Logitech Brio 4K, 90° FOV
- **Mounting:** Chandelier above poker table, 37" above surface
- **Coverage:** 37.75" diameter black felt circle — fits within Brio's FOV at 37"
- **Used by AI vision app only** — not streamed to Teams.
  AI vision identifies table cards and sends card data to `/remote` UI.
  A periodic table snapshot (JPEG) is also shown in the remote UI for context.
  No live video streaming — minimal bandwidth.

### Landing Zones

Each player (including Rodney) has a defined landing zone on the felt where
up cards are placed for scanning. Zones are calibrated once as pixel bounding
boxes in the Brio frame.

**Up card scanning flow:**
1. Dealer deals up cards normally, making an effort to hit landing zones
2. Overhead camera continuously monitors all zones
3. Card recognized → voice announcement: "Bill, Ace of Hearts"
4. Player hears confirmation, moves card out of zone whenever they want
5. If card missed the zone or isn't recognized — no announcement
6. Player notices no callout and nudges their card into the zone
7. Recognition triggers → "Joe, Seven of Clubs"
8. If recognition still fails → "Joe, try repositioning upcard"

**Pacing:** Dealer never stops dealing to wait for recognition. Deals
all up cards around the table in normal rhythm. Players are responsible
for getting their card into the zone if the dealer missed. Only potential
slowdown is if the deal comes back around to a player still trying to
get their last card recognized.

Voice announcements use macOS built-in text-to-speech (`say` command).
Cards can be placed in any orientation — the AI vision handles rotation.

### Frame Capture for Remote Player

The Brio frame is cropped to the black felt circle before sending to the
remote player UI. One-time calibration defines the circle center and radius.
Remote player sees a clean circular table image — no background clutter.
Snapshots refresh periodically, not live video.

### AI Vision Approach (in order of simplicity)

**Phase 1 — Vision API (start here):**
- Continuous monitoring of landing zones
- Crop each zone, send to Claude/GPT-4o vision API when change detected
- "Identify the playing card in this image"
- ~2-3 seconds per recognition, ~$0.02 per card
- Zero training, works immediately
- Handles any card orientation
- Needs internet

**Phase 2 — Local YOLO (if API isn't reliable enough):**
- Download pre-trained card dataset from Roboflow
- Fine-tune on ~200 photos from actual table setup
- Train: 1-2 hours on M4
- Inference: 30+ FPS, fully local, no internet
- Precise bounding boxes → map to player zones

### Bet Tracking

Handled verbally over Teams. Remote player announces bets,
dealer manages chips physically. No software tracking needed.

---

## Part 4: What's NOT in the Software

These are handled externally, not by our system:

| Function | Handled By |
|----------|-----------|
| Video/audio | Microsoft Teams |
| Overhead table view | Second Teams camera or AI vision |
| Betting amounts | Verbal over Teams |
| Chip management | Dealer handles physically |
| Game rules enforcement | Players (as always) |
| Hand evaluation/winner | Players determine |

---

## Current Status

### Completed

- [x] Communication protocol designed (docs/protocol.md)
- [x] Game engine with 9 templates, phase management, wild card tracking
- [x] Card recognition prototype — **100% accuracy** on 52-card test set
- [x] Host app backend (FastAPI, WebSocket, mock Pi client)
- [x] Host dealer UI (web-based control panel)
- [x] Remote player UI (hand view, discard, challenge selection)
- [x] Pi scanner client + mock mode for development

### In Progress

- [ ] Scanner box enclosure (plywood)
- [ ] 3D model: card trays (x7)
- [ ] Bend aluminum mirror with mylar film applied
- [ ] 3D model: camera mount bracket
- [ ] 3D model: LED mount brackets (x5, 2 LEDs each)

### To Do — Hardware

- [ ] Order: Logitech Brio 4K camera
- [ ] Order: 10x cool white diffused flat-top LEDs (ordered)
- [ ] Order: 7x pre-wired green LEDs with inline resistors
- [ ] Order: Passive piezo buzzer
- [ ] Order: IRLZ44N MOSFET
- [ ] Order: Resistors (39Ω x10, 330Ω x1, 10K x1)
- [ ] Order: IDC 8-pin screw terminal breakout boards (x2)
- [ ] Order: 8-conductor ribbon cable
- [ ] Order: USB-C panel mount connector
- [ ] Order/have: 5052 aluminum sheet (1/16") for mirror substrate
- [ ] Order: Mirror mylar film
- [ ] Order/have: Pi Camera Module v3 (ordered)
- [ ] Order/have: Raspberry Pi (model TBD)
- [ ] Build: Scanner box enclosure
- [ ] Build: 7 card trays
- [ ] Build: Mirror mount
- [ ] Wiring: Flash LEDs + MOSFET circuit
- [ ] Wiring: Green slot LEDs + ribbon cable
- [ ] Wiring: Buzzer
- [ ] Wiring: Camera ribbon cable
- [ ] Assembly and testing

### To Do — Software

- [ ] Update card detector for multi-card recognition (7 windows in one frame)
- [ ] Map card positions to slot numbers based on X coordinate
- [ ] Continuous monitoring mode (detect card changes between scans)
- [ ] Pi scanner server (HTTP + WebSocket API)
- [ ] Green LED control from game engine
- [ ] Flash LED + camera capture coordination
- [ ] Speech recognition integration (Whisper)
- [ ] Update protocol.md for new platen design
- [ ] Challenge comparison flow (scan multiple challengers sequentially)
- [ ] Overhead camera integration (Phase 1: vision API)

---

## Bill of Materials (Estimated)

| Item | Est. Cost |
|------|----------|
| Raspberry Pi 4/5 | $35-80 |
| Pi Camera Module v3 | $25 |
| Logitech Brio 4K | $130 |
| LEDs, resistors, MOSFET, buzzer | $15 |
| Mirror (~500x65mm) | $10-20 |
| IDC breakout boards + ribbon cable | $10 |
| USB-C panel mount | $5 |
| 3D printing filament (PLA) | $10 |
| 1/4" plywood (base) | $5 |
| Amazon Basics Thunderbolt 4 Hub | $145 |
| Misc (wires, connectors, screws) | $10 |
| **Total (excl. laptop + Brio)** | **~$290-325** |
| **Total (incl. Brio)** | **~$420-455** |

---

## Key Design Decisions Log

1. **One-way card flow only** — up cards handled by speech recognition, not scanner
2. **No motors/moving parts** — glass-free platen with individual card trays replaced
   the entire scanner tube + roller + elevator + slot rack design
3. **Mirror fold** — reduces box height from 9" to 2" for minimal table presence
4. **Flash LEDs** — 50ms pulse, imperceptible, no bright light on table during game
5. **Green indicator LEDs** — physical feedback on scanner box for discard/challenge
6. **One camera first** — single Pi Camera v3 covers all 7 slots; add second if needed
7. **Vision API before YOLO** — start with Claude/GPT-4o for overhead cam, train YOLO only if needed
8. **Verbal betting** — no bet tracking in software, keeps it simple
9. **Web-based apps** — one server, two browser views; no Windows app to install
10. **Teams screen share** — remote player accesses Neo screen directly, no VPN needed
11. **Thunderbolt hub** — Neo's second USB-C port is USB 2.0 only, can't drive Brio at 4K or external display. Hub multiplexes everything through the single Thunderbolt port: 4K camera, external monitor (DisplayPort Alt Mode), and PD power to Neo
