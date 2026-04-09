"""
Card Game Remote — Host Application

FastAPI server that:
- Serves the dealer control panel at /host
- Serves the remote player view at /remote
- Manages WebSocket connections for real-time updates
- Connects to the Pi scanner (or mock for dev)
- Drives game flow via the game engine

Usage:
    # Development mode (no Pi hardware):
    python app.py

    # With Pi scanner:
    python app.py --pi-host raspberrypi.local
"""

import argparse
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from game_engine import GameEngine
from pi_client import PiClient, MockPiClient

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# --- Global state ---
engine = GameEngine()
pi: PiClient = MockPiClient()

# Connected WebSocket clients
host_connections: list[WebSocket] = []
remote_connections: list[WebSocket] = []


# --- App lifecycle ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    await pi.connect()

    # Register Pi event handlers
    pi.on_card_scanned(handle_card_scanned)
    pi.on_eject_complete(handle_eject_complete)
    pi.on_error(handle_pi_error)

    yield
    await pi.disconnect()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Pi event handlers ---

async def handle_card_scanned(card: dict, confidence: float, image_url: str):
    """Called when Pi auto-scans a card."""
    log.info(f"Card scanned: {card} (confidence: {confidence})")

    if confidence < 0.85:
        # Low confidence — notify host for manual confirmation
        await broadcast_host({
            "type": "scan_confirm_needed",
            "card": card,
            "confidence": confidence,
            "image_url": image_url,
        })
        return

    await process_scanned_card(card)


async def process_scanned_card(card: dict):
    """Process a confirmed card scan through the game engine."""
    result = engine.card_scanned(card["rank"], card["suit"])

    # Tell Pi to eject
    await pi.eject(result["direction"])

    # Send messages to remote player
    for msg in result["messages"]:
        await broadcast_remote(msg)

    # Update host with current state
    await broadcast_host({
        "type": "hand_update",
        "state": engine.get_hand_state(),
    })


async def handle_eject_complete(direction: str, slot_number: int | None):
    """Called when Pi confirms card ejection."""
    log.info(f"Eject complete: {direction}, slot {slot_number}")


async def handle_pi_error(message: str):
    """Called when Pi reports an error."""
    log.error(f"Pi error: {message}")
    await broadcast_host({"type": "pi_error", "message": message})


# --- WebSocket broadcast helpers ---

async def broadcast_host(message: dict):
    """Send a message to all connected host clients."""
    data = json.dumps(message)
    disconnected = []
    for ws in host_connections:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        host_connections.remove(ws)


async def broadcast_remote(message: dict):
    """Send a message to all connected remote clients."""
    data = json.dumps(message)
    disconnected = []
    for ws in remote_connections:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        remote_connections.remove(ws)


# --- Page routes ---

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "host.html")


@app.get("/host", response_class=HTMLResponse)
async def host_page(request: Request):
    return templates.TemplateResponse(request, "host.html")


@app.get("/remote", response_class=HTMLResponse)
async def remote_page(request: Request):
    return templates.TemplateResponse(request, "remote.html")


# --- API routes ---

@app.get("/api/games")
async def get_games():
    return {"games": engine.get_game_list()}


@app.get("/api/players")
async def get_players():
    return {"players": engine.get_players_info(), "dealer": engine.get_dealer().name}


@app.get("/api/state")
async def get_state():
    return engine.get_hand_state()


@app.get("/api/pi/status")
async def get_pi_status():
    return await pi.get_status()


# --- Host WebSocket ---

@app.websocket("/ws/host")
async def host_websocket(websocket: WebSocket):
    await websocket.accept()
    host_connections.append(websocket)
    log.info("Host client connected")

    # Send current state on connect
    await websocket.send_text(json.dumps({
        "type": "hand_update",
        "state": engine.get_hand_state(),
    }))

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            await handle_host_message(data)
    except WebSocketDisconnect:
        host_connections.remove(websocket)
        log.info("Host client disconnected")


async def handle_host_message(data: dict):
    """Handle messages from the host/dealer UI."""
    msg_type = data.get("type")

    if msg_type == "new_hand":
        game_name = data.get("game_name")
        result = engine.new_hand(game_name)
        await broadcast_remote(result)
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })

    elif msg_type == "continue_betting":
        messages = engine.continue_after_betting()
        for msg in messages:
            await broadcast_remote(msg)
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })

    elif msg_type == "end_hand":
        result = engine.end_hand()
        await broadcast_remote(result)
        await pi.reset()
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })

    elif msg_type == "scan_confirm":
        # Host confirmed/corrected a low-confidence scan
        card = data.get("card")
        await process_scanned_card(card)

    elif msg_type == "manual_card":
        # Host manually enters a card (scan failed entirely)
        card = data.get("card")
        await process_scanned_card(card)

    elif msg_type == "peek_scan_start":
        # Switch to peek scan mode
        await broadcast_host({"type": "mode_update", "mode": "peek"})

    elif msg_type == "peek_card":
        # Process a peek-scanned card
        card = data.get("card")
        label = data.get("label", "")
        msg = engine.peek_card(card["rank"], card["suit"], label)
        await broadcast_remote(msg)

    elif msg_type == "reset_elevator":
        await pi.reset()

    elif msg_type == "simulate_scan":
        # Dev mode: simulate a card scan
        card = data.get("card")
        await process_scanned_card(card)


# --- Remote WebSocket ---

@app.websocket("/ws/remote")
async def remote_websocket(websocket: WebSocket):
    await websocket.accept()
    remote_connections.append(websocket)
    log.info("Remote player connected")

    # Send current hand state so they can catch up
    state = engine.get_hand_state()
    if state["game_name"]:
        # Replay all cards dealt so far
        await websocket.send_text(json.dumps({
            "type": "new_hand",
            "game_name": state["game_name"],
            "wild_ranks": state["wild_ranks"],
            "wild_label": state["wild_label"],
        }))
        for slot in state["slots"]:
            await websocket.send_text(json.dumps({
                "type": "card_dealt",
                "slot_number": slot["slot_number"],
                "card": slot["card"],
                "card_type": slot["card_type"],
            }))

    try:
        while True:
            data = json.loads(await websocket.receive_text())
            await handle_remote_message(data)
    except WebSocketDisconnect:
        remote_connections.remove(websocket)
        log.info("Remote player disconnected")


async def handle_remote_message(data: dict):
    """Handle messages from the remote player UI."""
    msg_type = data.get("type")

    if msg_type == "discard_request":
        slot_numbers = data.get("slot_numbers", [])
        result = engine.process_discard(slot_numbers)
        await broadcast_remote(result)
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })

    elif msg_type == "challenge_request":
        slot_numbers = data.get("slot_numbers", [])
        result = engine.process_challenge(slot_numbers)
        await broadcast_remote(result)
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })

    elif msg_type == "pass_challenge":
        result = engine.process_pass_challenge()
        await broadcast_remote(result)
        await broadcast_host({
            "type": "hand_update",
            "state": engine.get_hand_state(),
        })


# --- Main ---

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Card Game Remote Host")
    parser.add_argument("--pi-host", type=str, default=None,
                        help="Pi hostname (default: mock mode)")
    parser.add_argument("--pi-port", type=int, default=5000)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.pi_host:
        pi = PiClient(args.pi_host, args.pi_port)
    else:
        pi = MockPiClient()
        log.info("Running in mock mode (no Pi hardware)")

    uvicorn.run(app, host="0.0.0.0", port=args.port)
