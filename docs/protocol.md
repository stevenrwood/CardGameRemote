# Card Game Remote — Communication Protocol

## System Components

```
┌─────────────┐     WiFi (LAN)      ┌─────────────┐    Tailscale VPN     ┌─────────────┐
│  Raspberry   │ ◄────────────────► │    Host      │ ◄─────────────────► │   Remote     │
│  Pi Scanner  │   REST/WebSocket   │  App (Mac)   │     WebSocket       │ App (Windows)│
│              │                    │              │                     │              │
│ - Camera     │                    │ - Game flow  │                     │ - Hand view  │
│ - Motors     │                    │ - Card relay │                     │ - Discards   │
│ - LEDs       │                    │ - Pi control │                     │ - Challenges │
└─────────────┘                    └─────────────┘                     └─────────────┘
```

**Not part of this software:**
- Video/audio: Teams
- Overhead table camera: second Teams video source
- Betting: verbal over Teams, host manages chips physically

---

## Data Model

### Card
```json
{
  "rank": "A",          // "2"-"10", "J", "Q", "K", "A"
  "suit": "spades"      // "hearts", "diamonds", "clubs", "spades"
}
```

### Slot
```json
{
  "slot_number": 1,
  "card": { "rank": "A", "suit": "spades" },
  "card_type": "down",  // "down" or "up"
  "status": "active"    // "active", "discarded", or "challenged"
}
```

### Hand State (maintained by host, mirrored to remote)
```json
{
  "game_name": "7 Card Stud",
  "slots": [],
  "next_slot": 1,
  "wild_cards": [],             // list of ranks that are currently wild (e.g., ["2"])
  "wild_label": "Deuces Wild"   // human-readable wild card description
}
```

---

## Pi ↔ Host Protocol (WiFi LAN)

The Pi runs a lightweight HTTP + WebSocket server. The host connects to it.

### Operational Modes

**Monitoring mode (default on boot):** Pi continuously monitors for a card on the scanner
ledge using its sensor (IR break-beam or camera-based detection). When a card is detected,
it auto-captures and identifies the card, then reports to the host via WebSocket. The card
remains on the scanner until the host sends an eject command.

**Idle mode:** Pi stops monitoring. Used between games or during setup.

### Pi → Host Events (WebSocket)

**card_scanned**
Sent automatically when a card is placed on the scanner ledge. Pi detects the card,
captures an image, runs recognition, and reports the result — all without host intervention.
```json
{
  "event": "card_scanned",
  "card": { "rank": "K", "suit": "hearts" },
  "confidence": 0.97,
  "image_url": "/captures/latest.jpg",
  "timestamp": "2026-04-05T20:15:30Z"
}
```

**eject_complete**
Sent after a card has been ejected and (if slot-side) the elevator has advanced.
```json
{
  "event": "eject_complete",
  "direction": "slot",
  "slot_number": 3
}
```

**error**
```json
{
  "event": "error",
  "message": "Motor stall detected"
}
```

### Host → Pi Commands (HTTP POST)

**POST /eject**
Eject the card currently on the scanner. Called by host after receiving `card_scanned`.
```json
// Request
{
  "direction": "slot"   // "slot" or "table"
}
```
- `"slot"`: Pinch rollers push card into slot rack, then elevator advances one position.
- `"table"`: Pinch rollers push card out table-side.
```json
// Response
{
  "success": true
}
```

**POST /reset**
Reset elevator to home position (bottom). Used when manually clearing slots between games.
```json
// Response
{
  "success": true,
  "slot_position": 0
}
```

**POST /mode**
Switch between monitoring and idle modes.
```json
// Request
{
  "mode": "monitoring"   // "monitoring" or "idle"
}
```

**GET /status**
Health check and current state.
```json
{
  "connected": true,
  "mode": "monitoring",
  "slot_position": 3,
  "card_on_scanner": false
}
```

---

## Host ↔ Remote Protocol (Tailscale WebSocket)

Persistent WebSocket connection. Host is the server, remote is the client.

### Host → Remote Messages

**new_hand**
Start a new hand. Clears the remote player's display.
```json
{
  "type": "new_hand",
  "game_name": "Follow the Queen"
}
```

**card_dealt**
A card has been scanned and dealt to the remote player.
```json
{
  "type": "card_dealt",
  "slot_number": 3,
  "card": { "rank": "K", "suit": "hearts" },
  "card_type": "down"
}
```
- Down cards: shown in remote player's hand view (private)
- Up cards: shown in remote player's hand view, marked as visible to all

**peek_card**
A card scanned in peek mode — for showing other players' cards to the remote player
(e.g., during High/Low/High Challenge showdowns). Not part of the remote player's hand.
```json
{
  "type": "peek_card",
  "card": { "rank": "J", "suit": "diamonds" },
  "label": "Player 3 challenge card"
}
```

**draw_prompt**
Signals that a draw round is active. Enables discard selection in the remote UI.
```json
{
  "type": "draw_prompt",
  "draw_round": 1,
  "max_draw": 3
}
```

**challenge_prompt**
Signals that a challenge round is active (High/Low/High Challenge). Enables card
selection in the remote UI.
```json
{
  "type": "challenge_prompt",
  "challenge_round": 1,
  "select_cards": 2,
  "label": "Best 2-card high hand"
}
```

**discard_acknowledged**
Confirms the remote player's discard request.
```json
{
  "type": "discard_acknowledged",
  "slot_numbers": [2, 4]
}
```

**challenge_acknowledged**
Confirms the remote player's challenge selection.
```json
{
  "type": "challenge_acknowledged",
  "slot_numbers": [1, 3]
}
```

**wild_card_update**
Notifies the remote player of a change in wild cards (e.g., Follow the Queen).
```json
{
  "type": "wild_card_update",
  "wild_ranks": ["8"],
  "label": "8s are wild (follows Queen)"
}
```

**hand_over**
Signals end of current hand.
```json
{
  "type": "hand_over"
}
```

### Remote → Host Messages

**discard_request**
Remote player wants to turn in cards from the specified slots.
```json
{
  "type": "discard_request",
  "slot_numbers": [2, 4]
}
```

**challenge_request**
Remote player selects cards to challenge with (High/Low/High Challenge).
```json
{
  "type": "challenge_request",
  "slot_numbers": [1, 3]
}
```

**pass_challenge**
Remote player passes on the challenge round.
```json
{
  "type": "pass_challenge"
}
```

---

## Scan Modes

### Deal Scan (automated, template-driven)
The normal card dealing flow. Each card is scanned, routed per the game template
(slot or table), and sent to the remote player as part of their hand.

### Peek Scan (manual, host-triggered)
Used to show arbitrary cards to the remote player without affecting hand state.
Host clicks "Peek Scan" in the UI, then places card(s) on the scanner.

Use cases:
- Showdown comparisons (e.g., High/Low/High Challenge — scanning a local
  player's challenge cards so the remote player can see them)
- Any time the remote player needs to see a card that isn't theirs

Peek-scanned cards always eject to the table side and are sent as `peek_card`
messages (not `card_dealt`). They don't occupy slots or affect hand state.

---

## Automated Flow

The system is fully template-driven. Once the host selects a game and starts a hand,
the deal sequence is automated. The dealer just places cards — the software knows
whether each card is up or down and routes accordingly.

### Per-Card Cycle (fully automatic)

```
DEALER places card ──► Pi auto-detects & scans ──► Host receives card_scanned
                                                          │
                                                   Game template says
                                                   card #N is "down" or "up"
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                       Eject to SLOT            Eject to TABLE
                                       (advance elevator)       (falls on table)
                                              │                       │
                                              ▼                       ▼
                                       card_dealt to remote    card_dealt to remote
                                       (card_type: "down")     (card_type: "up")
                                                                      │
                                                               If Follow the Queen:
                                                               check for wild card
                                                               update and notify
```

If confidence is below threshold, the flow pauses for host to confirm/correct the card.
Otherwise, no host interaction needed — just keep placing cards.

### Host Interactions During a Hand

The host only needs to interact with the app for:
- **"Next Street" / "Continue"** — resume dealing after a betting round (stud games)
- **"Hand Over"** — end the current hand
- **Card correction** — if scan confidence is low, confirm or override
- **"Peek Scan"** — switch to peek mode for showdown comparisons
- **Override** — force a card up/down against the template (rare edge cases)

### Follow the Queen — Wild Card Tracking

During Follow the Queen, the host app watches for Queens in up-card positions:
1. Card scanned and identified as a Queen
2. Template says this card is "up"
3. Host app flags: next up card's rank becomes the new wild
4. Next up card scanned → that rank is now wild
5. Host sends `wild_card_update` to remote with the new wild rank
6. If another Queen comes up later, the wild changes to whatever follows it
7. Remote app displays current wild prominently (e.g., "8s are wild")

Queens themselves are always wild in Follow the Queen.

---

## Game Templates

Phase-based templates define the complete flow for each game. The host selects a game
before each hand and the template drives the automated flow.

### Phase Types

| Phase | Description | Remote Interaction |
|-------|-------------|--------------------|
| `deal` | Deal cards in a fixed up/down pattern | None (just receives cards) |
| `betting` | Pause for verbal betting round | None (verbal over Teams) |
| `draw` | Draw round — player discards and gets replacements | Select cards to discard |
| `challenge` | Challenge round — player selects cards to play | Select cards or pass |
| `hit_round` | Open-ended — each player can take or refuse a card | None (verbal over Teams) |

### Template Definitions

```json
{
  "5 Card Draw": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down", "down", "down", "down"] },
      { "type": "betting" },
      { "type": "draw", "max_draw": 3 },
      { "type": "betting" }
    ]
  },

  "3 Toed Pete": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down", "down"] },
      { "type": "betting" },
      { "type": "draw", "max_draw": 3 },
      { "type": "betting" },
      { "type": "draw", "max_draw": 2 },
      { "type": "betting" },
      { "type": "draw", "max_draw": 1 },
      { "type": "betting" }
    ]
  },

  "7 Card Stud": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down", "up"] },
      { "type": "betting" },
      { "type": "deal", "pattern": ["up"] },
      { "type": "betting" },
      { "type": "deal", "pattern": ["up"] },
      { "type": "betting" },
      { "type": "deal", "pattern": ["up"] },
      { "type": "betting" },
      { "type": "deal", "pattern": ["down"] },
      { "type": "betting" }
    ]
  },

  "7 Stud Deuces Wild": {
    "extends": "7 Card Stud",
    "wild_cards": { "ranks": ["2"], "label": "Deuces Wild" }
  },

  "Follow the Queen": {
    "extends": "7 Card Stud",
    "wild_cards": { "ranks": ["Q"], "label": "Queens wild" },
    "dynamic_wild": "follow_the_queen"
  },

  "High Chicago": {
    "extends": "7 Card Stud",
    "notes": "Split pot: best poker hand + highest spade in the hole"
  },

  "High/Low/High Challenge": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down", "down"] },
      { "type": "betting" },
      { "type": "challenge", "select_cards": 2, "label": "Best 2-card high hand" },
      { "type": "deal", "pattern": ["down", "down"] },
      { "type": "betting" },
      { "type": "challenge", "select_cards": 3, "label": "Best 3-card low hand" },
      { "type": "deal", "pattern": ["down", "down"] },
      { "type": "betting" },
      { "type": "challenge", "select_cards": 5, "label": "Best 5-card poker hand" }
    ],
    "repeatable": true
  },

  "7/27": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down"] },
      { "type": "betting" },
      { "type": "hit_round", "card_type": "up" }
    ],
    "notes": "Hit rounds repeat: betting → optional face-up card per player, until all frozen/folded"
  },

  "Texas Hold'em": {
    "phases": [
      { "type": "deal", "pattern": ["down", "down"] },
      { "type": "betting" },
      { "type": "community", "pattern": ["up", "up", "up"], "label": "Flop" },
      { "type": "betting" },
      { "type": "community", "pattern": ["up"], "label": "Turn" },
      { "type": "betting" },
      { "type": "community", "pattern": ["up"], "label": "River" },
      { "type": "betting" }
    ],
    "notes": "Community cards dealt via overhead camera / peek scan, not through slot rack"
  }
}
```

Templates are stored as user-editable JSON. New games can be added at any time.

---

## Error Handling

- **Low confidence scan**: If card recognition confidence < 0.85, host app shows
  the captured image for manual identification. Host can override the detected card.
- **Connection lost (Pi)**: Host app shows disconnected state. Auto-reconnects.
- **Connection lost (Remote)**: Host app shows remote disconnected. Full hand state
  is replayed on reconnect so remote player catches up.
- **Card not recognized**: Host can manually enter the card (rank + suit picker).

## Security

- Pi only accessible on local WiFi (no internet exposure)
- Host ↔ Remote secured by Tailscale (WireGuard encryption)
- No authentication layer needed beyond Tailscale network membership
