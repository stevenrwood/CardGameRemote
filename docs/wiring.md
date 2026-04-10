# Card Game Remote — Electrical Wiring Guide

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Scanner Box                                   │
│                                                                        │
│  ┌──────────┐    CSI ribbon x2    ┌──────────────────────────────────┐ │
│  │ CM4 +    │◄───────────────────►│ Camera v3 #1 + Camera v3 #2     │ │
│  │ Base     │                     └──────────────────────────────────┘ │
│  │ Board    │    GPIO pins         ┌────────────────────────────────┐  │
│  │          │◄────────────────────►│ Flash LEDs, Green LEDs, Buzzer │  │
│  └────┬─────┘                     └────────────────────────────────┘  │
│       │                                                                │
│  USB-C panel mount                                                     │
│  + Ext antenna                                                         │
└───────┼────────────────────────────────────────────────────────────────┘
        │
   5V USB-C Power Supply
```

---

## Raspberry Pi CM4 + Base Board

### Power Input
- USB-C panel mount connector on back wall of enclosure
- Female USB-C (panel mount) → short internal USB-C cable → CM4 base board USB-C input
- Power supply: 5V / 3A USB-C (standard phone charger is sufficient)

### WiFi
- External antenna connector on back wall
- SMA or U.FL pigtail from CM4 → SMA panel mount bulkhead on back wall
- External antenna attached outside the enclosure for better signal

### Camera Connections
- Camera v3 #1 → CSI-0 ribbon cable → CM4 base board CAM0 connector
- Camera v3 #2 → CSI-1 ribbon cable → CM4 base board CAM1 connector
- 15-pin FFC (flat flexible cable), 16mm wide
- Ribbon cables route from cameras through slots in brackets to CM4

### SD Card
- CM4 has no eMMC — boots from SD card on the base board
- SD card slot must remain accessible (back or bottom of enclosure)

---

## GPIO Pin Assignments (BCM Numbering)

All output pins use GPIO 9-27 range (default pull-down at boot — nothing
activates during power-up).

| Signal | BCM GPIO | Physical Pin | Direction |
|--------|----------|-------------|-----------|
| Green LED - Slot 1 | GPIO 17 | Pin 11 | Output |
| Green LED - Slot 2 | GPIO 27 | Pin 13 | Output |
| Green LED - Slot 3 | GPIO 22 | Pin 15 | Output |
| Green LED - Slot 4 | GPIO 23 | Pin 16 | Output |
| Green LED - Slot 5 | GPIO 24 | Pin 18 | Output |
| Green LED - Slot 6 | GPIO 25 | Pin 22 | Output |
| Green LED - Slot 7 | GPIO 12 | Pin 32 | Output |
| Flash MOSFET | GPIO 16 | Pin 36 | Output |
| Piezo Buzzer | GPIO 26 | Pin 37 | Output |

### Reserved Pins (do not use)

| Pin | Reason |
|-----|--------|
| GPIO 2 (SDA1) | Camera v3 I2C control |
| GPIO 3 (SCL1) | Camera v3 I2C control |
| GPIO 14 | UART TX (serial console) |
| GPIO 15 | UART RX (serial console) |
| GPIO 0, 1 | I2C-0 / HAT EEPROM |

### Available for Future Use

GPIO 9, 10, 11, 13, 18, 19, 20, 21

---

## Flash Illumination LEDs (12x Cool White)

### Components
- 12x 5mm diffused flat-top cool white LEDs (20mA rated)
- 12x 39Ω 1/4W resistors (one per LED)
- 1x IRLZ44N N-channel MOSFET (logic-level, TO-220)
- 1x 3KΩ resistor (MOSFET gate series)
- 1x 10KΩ resistor (MOSFET gate pull-down)

### Circuit

```
Pi 5V (Pin 2 or 4)
    │
    ├──[39Ω]──LED 1 anode (+)──LED 1 cathode (-)──┐
    ├──[39Ω]──LED 2 anode (+)──LED 2 cathode (-)──┤
    ├──[39Ω]──LED 3 anode (+)──LED 3 cathode (-)──┤
    ├──[39Ω]──LED 4 anode (+)──LED 4 cathode (-)──┤
    ├──[39Ω]──LED 5 anode (+)──LED 5 cathode (-)──┤
    ├──[39Ω]──LED 6 anode (+)──LED 6 cathode (-)──┤
    ├──[39Ω]──LED 7 anode (+)──LED 7 cathode (-)──┤
    ├──[39Ω]──LED 8 anode (+)──LED 8 cathode (-)──┤
    ├──[39Ω]──LED 9 anode (+)──LED 9 cathode (-)──┤
    ├──[39Ω]──LED 10 anode (+)──LED 10 cathode (-)─┤
    ├──[39Ω]──LED 11 anode (+)──LED 11 cathode (-)─┤
    └──[39Ω]──LED 12 anode (+)──LED 12 cathode (-)─┤
                                                    │
                                                 DRAIN (center pin)
                                                    │
                                                IRLZ44N
                                                    │
                                                 SOURCE (right pin)
                                                    │
                                                   GND ── Pi GND (Pin 6, 9, or 14)


    GPIO 16 (Pin 36) ──[3KΩ]── GATE (left pin)
                                  │
                                [10KΩ]
                                  │
                                 GND
```

### IRLZ44N Pinout (facing front, text readable, legs down)

```
    ┌─────────┐
    │ IRLZ44N │
    │         │
    └─┬──┬──┬─┘
      G  D  S
     left    right
     center
```

- **Gate (G)** — left pin: GPIO 16 through 3KΩ series resistor
- **Drain (D)** — center pin: all LED cathodes connect here
- **Source (S)** — right pin: ground

### LED Identification

- **Anode (+)**: longer leg
- **Cathode (-)**: shorter leg, flat spot on lens housing

### Physical Layout

LEDs mounted in 2x camera+LED combo brackets + no spacer needed:

```
Back wall (facing mirror):

Bracket 1 (left, 0-166mm):
  ● ●    [Camera #1]    ● ●         ● ●
  0mm      103mm                   166mm

Bracket 1 (right, 291-394mm):
  ● ●    [Camera #2]    ● ●
  291mm    322mm        394mm
```

Each ● pair = one LED high, one LED low relative to camera lens center.
12 LEDs total across 6 positions.

### Wiring Approach

Run two bare copper bus wires across each bracket:
- **5V bus**: connects to resistor → anode of each LED
- **GND bus (drain)**: connects to cathode of each LED

Solder LED legs and resistors directly to bus wires. Two wires
per bracket run back to the MOSFET/power connections.

### Operating Parameters

| Parameter | Value |
|-----------|-------|
| Flash current per LED | ~40mA (pulsed) |
| Total peak current | ~480mA (12 LEDs) |
| Flash duration | ~50ms |
| Duty cycle | ~2.5% (50ms every 2s) |
| Power source | Pi 5V rail (ok for pulsed load) |

---

## Green Slot Indicator LEDs (7x)

### Components
- 7x pre-wired green LEDs with inline resistor
- 1x 8-conductor ribbon cable (length: ~400mm, from front to back of box)
- 2x IDC 8-pin screw terminal breakout boards

### Circuit

Each green LED is directly driven by a GPIO pin — no MOSFET needed
(single LED current ~20mA is within GPIO pin limits).

```
GPIO 17 (Pin 11) ──── inline resistor ──── LED 1 (Slot 1) ──── GND
GPIO 27 (Pin 13) ──── inline resistor ──── LED 2 (Slot 2) ──── GND
GPIO 22 (Pin 15) ──── inline resistor ──── LED 3 (Slot 3) ──── GND
GPIO 23 (Pin 16) ──── inline resistor ──── LED 4 (Slot 4) ──── GND
GPIO 24 (Pin 18) ──── inline resistor ──── LED 5 (Slot 5) ──── GND
GPIO 25 (Pin 22) ──── inline resistor ──── LED 6 (Slot 6) ──── GND
GPIO 12 (Pin 32) ──── inline resistor ──── LED 7 (Slot 7) ──── GND
                                                                 │
                                                          All share
                                                          common GND
```

The inline resistor is already part of the pre-wired LED assembly:
- Lead with resistor → GPIO pin
- Other lead → GND

### Physical Wiring

```
Card Trays (top of box)
┌──┬──┬──┬──┬──┬──┬──┐
│G1│G2│G3│G4│G5│G6│G7│  ← green LED in each tray
└──┴──┴──┴──┴──┴──┴──┘
   │  wires run along bottom of trays
   │
   ▼ Behind mirror (front of box)
┌─────────────────────┐
│ IDC Screw Terminal   │  ← 7 signal + 1 GND screw terminals
│ Breakout Board #1    │     LED wires land here
└─────────┬───────────┘
          │ 8-conductor ribbon cable
          │ runs along bottom of box
          │ against far wall
          │
┌─────────┴───────────┐
│ IDC Screw Terminal   │  ← connects to Pi GPIO pins
│ Breakout Board #2    │
└─────────────────────┘
          │
          ▼ Pi GPIO header
```

### Ribbon Cable Pinout

| Wire # | Signal | Color (suggested) |
|--------|--------|-------------------|
| 1 | GND (shared) | Black |
| 2 | Slot 1 (GPIO 17) | — |
| 3 | Slot 2 (GPIO 27) | — |
| 4 | Slot 3 (GPIO 22) | — |
| 5 | Slot 4 (GPIO 23) | — |
| 6 | Slot 5 (GPIO 24) | — |
| 7 | Slot 6 (GPIO 25) | — |
| 8 | Slot 7 (GPIO 12) | — |

---

## Piezo Buzzer

### Components
- 1x passive piezo buzzer (2-pin)

### Circuit

```
GPIO 26 (Pin 37) ──── Buzzer (+) ──── Buzzer (-) ──── GND
```

No resistor needed for a passive piezo. The GPIO pin drives the
buzzer directly with a PWM signal to produce tones.

### Tones

| Pattern | Meaning |
|---------|---------|
| Single 1kHz beep (150ms) | Card recognized |
| Double beep | Low confidence |
| Descending tone | Not recognized |

### Physical Location

Mounted inside box, behind mirror with the green LED wiring.
Connected to Pi via the ribbon cable (add 2 more conductors)
or with separate 2-wire run to Pi GPIO.

---

## Camera Modules (2x Pi Camera v3)

### Specifications

| Parameter | Value |
|-----------|-------|
| Sensor | Sony IMX708 |
| Resolution | 4608 x 2592 (11.9 MP) |
| Autofocus | Yes (5cm to infinity) |
| Connection | 15-pin CSI-2 ribbon cable |
| Board size | 25mm x 24mm |
| Mounting holes | 4x M2.5, 21mm x 12.5mm spacing |
| Total thickness | ~11.5mm (incl. autofocus mechanism) |

### Mounting

- Each camera on 4x M2.5 standoff posts (5-6mm tall) in bracket
- Ribbon cable exits bottom of board, bends 90° through slot in bracket
- Ribbon routes horizontally to CM4 base board

### Camera Position (from left side of box)

| Camera | Position | Covers Slots |
|--------|----------|-------------|
| #1 (CSI-0) | 103mm | 1-4 |
| #2 (CSI-1) | 322mm | 5-7 |

---

## Complete Connection Summary

### To Pi GPIO Header

| Physical Pin | BCM GPIO | Connects To |
|-------------|----------|-------------|
| Pin 2 | 5V | Flash LED 5V bus (through resistors) |
| Pin 6 | GND | Flash LED GND (MOSFET source), Green LED GND, Buzzer GND |
| Pin 11 | GPIO 17 | Green LED Slot 1 |
| Pin 13 | GPIO 27 | Green LED Slot 2 |
| Pin 15 | GPIO 22 | Green LED Slot 3 |
| Pin 16 | GPIO 23 | Green LED Slot 4 |
| Pin 18 | GPIO 24 | Green LED Slot 5 |
| Pin 22 | GPIO 25 | Green LED Slot 6 |
| Pin 32 | GPIO 12 | Green LED Slot 7 |
| Pin 36 | GPIO 16 | MOSFET gate (through 3KΩ) |
| Pin 37 | GPIO 26 | Piezo buzzer |

### To CM4 Base Board

| Connector | Connects To |
|-----------|-------------|
| CAM0 (CSI-0) | Camera v3 #1 (15-pin ribbon) |
| CAM1 (CSI-1) | Camera v3 #2 (15-pin ribbon) |
| USB-C | Panel mount USB-C (power input) |
| Antenna | SMA panel mount (external antenna) |
| SD Card | MicroSD card (boot drive) |

---

## Wire Gauge Summary

| Connection | Gauge | Reason |
|-----------|-------|--------|
| 5V + GND bus (flash LEDs) | 22 AWG | Carries up to 480mA peak |
| LED-to-bus jumpers | 30 AWG | Individual LED, 40mA each |
| Green LED ribbon cable | 28 AWG (ribbon) | 20mA per line |
| Buzzer wires | 28-30 AWG | Minimal current |
| MOSFET gate wiring | 30 AWG | Signal only |

---

## Physical Wiring Zones Inside Box

```
Side view:

    ┌─card trays (green LEDs mounted in trays)────────────────┐
    │ green LED    ╲          │ LED    │         │             │
    │ wiring       ╲ mirror  │brackets│ cameras │ CM4 + base  │
    │ + buzzer      ╲        │+ LEDs  │         │ board       │
    │ + IDC board    ╲       │        │         │ + USB-C     │
    └────────────────╲───────┴────────┴─────────┴─── antenna──┘
      FRONT                                            BACK
      
    Zone 1: Green LED wiring, buzzer, IDC breakout board #1
    Zone 2: Mirror (no wiring)
    Zone 3: Flash LED brackets + cameras, 5V/GND bus wires
    Zone 4: CM4, base board, IDC breakout board #2, power, antenna
    
    Ribbon cable runs along bottom from Zone 1 to Zone 4
```
