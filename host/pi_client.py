"""
Async client for the Raspberry Pi scanner API.

Connects to the Pi over WiFi (LAN). Sends eject/reset commands via HTTP
and receives card_scanned events via WebSocket.
"""

import asyncio
import json
import logging

import httpx
import websockets

log = logging.getLogger(__name__)


class PiClient:
    """Client for the Pi scanner's HTTP + WebSocket API."""

    def __init__(self, pi_host: str = "raspberrypi.local", pi_port: int = 5000):
        self.base_url = f"http://{pi_host}:{pi_port}"
        self.ws_url = f"ws://{pi_host}:{pi_port}/ws"
        self._ws = None
        self._connected = False
        self._on_card_scanned = None
        self._on_eject_complete = None
        self._on_error = None

    @property
    def connected(self) -> bool:
        return self._connected

    def on_card_scanned(self, callback):
        """Register callback for card_scanned events. callback(card, confidence, image_url)"""
        self._on_card_scanned = callback

    def on_eject_complete(self, callback):
        """Register callback for eject_complete events. callback(direction, slot_number)"""
        self._on_eject_complete = callback

    def on_error(self, callback):
        """Register callback for error events. callback(message)"""
        self._on_error = callback

    async def connect(self):
        """Connect to the Pi's WebSocket for events."""
        try:
            self._ws = await websockets.connect(self.ws_url)
            self._connected = True
            log.info(f"Connected to Pi at {self.ws_url}")
            asyncio.create_task(self._listen())
        except Exception as e:
            log.warning(f"Could not connect to Pi: {e}")
            self._connected = False

    async def _listen(self):
        """Listen for WebSocket events from the Pi."""
        try:
            async for message in self._ws:
                data = json.loads(message)
                event = data.get("event")

                if event == "card_scanned" and self._on_card_scanned:
                    await self._on_card_scanned(
                        card=data.get("card"),
                        confidence=data.get("confidence", 0),
                        image_url=data.get("image_url", ""),
                    )
                elif event == "eject_complete" and self._on_eject_complete:
                    await self._on_eject_complete(
                        direction=data.get("direction"),
                        slot_number=data.get("slot_number"),
                    )
                elif event == "error" and self._on_error:
                    await self._on_error(data.get("message", "Unknown error"))

        except websockets.ConnectionClosed:
            log.warning("Pi WebSocket connection closed")
            self._connected = False
        except Exception as e:
            log.error(f"Pi WebSocket error: {e}")
            self._connected = False

    async def eject(self, direction: str) -> dict:
        """Send eject command. direction: 'slot' or 'table'."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/eject",
                json={"direction": direction},
            )
            return resp.json()

    async def reset(self) -> dict:
        """Reset elevator to home position."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self.base_url}/reset")
            return resp.json()

    async def set_mode(self, mode: str) -> dict:
        """Set Pi mode: 'monitoring' or 'idle'."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/mode",
                json={"mode": mode},
            )
            return resp.json()

    async def get_status(self) -> dict:
        """Get Pi health/status."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{self.base_url}/status", timeout=3.0)
                return resp.json()
        except Exception:
            return {"connected": False}

    async def disconnect(self):
        """Close WebSocket connection."""
        if self._ws:
            await self._ws.close()
            self._connected = False


class MockPiClient(PiClient):
    """
    Mock Pi client for development/testing without hardware.
    Simulates card scanning and ejection.
    """

    def __init__(self):
        super().__init__()
        self._connected = True
        self._slot_position = 0

    async def connect(self):
        self._connected = True
        log.info("Mock Pi client connected (no hardware)")

    async def eject(self, direction: str) -> dict:
        if direction == "slot":
            self._slot_position += 1
            log.info(f"Mock eject to slot (position {self._slot_position})")
            if self._on_eject_complete:
                await self._on_eject_complete(
                    direction="slot",
                    slot_number=self._slot_position,
                )
        else:
            log.info("Mock eject to table")
            if self._on_eject_complete:
                await self._on_eject_complete(direction="table", slot_number=None)
        return {"success": True}

    async def reset(self) -> dict:
        self._slot_position = 0
        log.info("Mock elevator reset")
        return {"success": True, "slot_position": 0}

    async def set_mode(self, mode: str) -> dict:
        log.info(f"Mock mode set to {mode}")
        return {"success": True}

    async def get_status(self) -> dict:
        return {
            "connected": True,
            "mode": "monitoring",
            "slot_position": self._slot_position,
            "card_on_scanner": False,
            "mock": True,
        }

    async def disconnect(self):
        self._connected = False
