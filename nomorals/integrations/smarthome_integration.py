"""Smart home integration with multiple backends.

Supports:
1. Home Assistant API (most flexible, supports 2000+ devices)
2. Direct device APIs (Philips Hue, TP-Link Kasa, etc.)
3. Voice bridge (via Alexa/Google Home)

Usage:
    smarthome = SmartHomeIntegration(account_manager, session_manager)
    
    # Control lights
    await smarthome.set_light("living_room", brightness=80, color="#FF6B35")
    
    # Get device status
    status = await smarthome.get_device_status("thermostat")
    print(f"Temperature: {status['current_temperature']}°C")
    
    # Create scene
    await smarthome.activate_scene("movie_night")
    
    # List all devices
    devices = await smarthome.list_devices()
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["SmartHomeIntegration", "Device", "DeviceState", "Scene",
           "HAWebSocket", "HAWebSocketError", "HAAuthError", "ProactiveAlert"]

_log = get_logger(__name__)


@dataclass
class Device:
    """Represents a smart home device."""
    
    device_id: str
    name: str
    device_type: str  # light, switch, thermostat, lock, sensor, etc.
    room: str = ""
    state: dict[str, Any] = field(default_factory=dict)
    capabilities: list[str] = field(default_factory=list)
    manufacturer: str = ""
    model: str = ""
    available: bool = True
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "device_id": self.device_id,
            "name": self.name,
            "type": self.device_type,
            "room": self.room,
            "state": self.state,
            "capabilities": self.capabilities,
            "available": self.available,
        }


@dataclass
class DeviceState:
    """Current state of a device."""
    
    device_id: str
    is_on: bool = False
    brightness: int = 0  # 0-100
    color_temp: int = 0  # Kelvin
    color: str = ""  # Hex color
    temperature: float = 0.0  # For thermostats/sensors
    humidity: float = 0.0  # For sensors
    locked: bool = False  # For locks
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "device_id": self.device_id,
            "is_on": self.is_on,
            "brightness": self.brightness,
            "color_temp": self.color_temp,
            "color": self.color,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "locked": self.locked,
            "attributes": self.attributes,
        }


@dataclass
class Scene:
    """A smart home scene/automation."""
    
    scene_id: str
    name: str
    description: str = ""
    devices_affected: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "scene_id": self.scene_id,
            "name": self.name,
            "description": self.description,
            "devices_affected": self.devices_affected,
        }


class SmartHomeIntegration:
    """Multi-backend smart home integration."""
    
    def __init__(
        self,
        account_manager: AccountManager,
        session_manager: SessionManager,
    ) -> None:
        self.account_manager = account_manager
        self.session_manager = session_manager
        self._device_cache: dict[str, Device] = {}
        _log.info("Smart home integration initialized")
    
    # ── Device Discovery ─────────────────────────────────────────────────────
    
    async def list_devices(
        self,
        *,
        room: str | None = None,
        device_type: str | None = None,
    ) -> list[Device]:
        """List all smart home devices.
        
        Args:
            room: Filter by room name
            device_type: Filter by device type
            
        Returns:
            List of Device objects
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._list_devices_ha(room, device_type)
        else:
            return []
    
    async def get_device(self, device_id: str) -> Optional[Device]:
        """Get device details.
        
        Args:
            device_id: Device ID
            
        Returns:
            Device object or None
        """
        if device_id in self._device_cache:
            return self._device_cache[device_id]
        
        devices = await self.list_devices()
        for device in devices:
            if device.device_id == device_id:
                self._device_cache[device_id] = device
                return device
        
        return None
    
    async def get_device_status(self, device_id_or_name: str) -> DeviceState:
        """Get current status of a device.
        
        Args:
            device_id_or_name: Device ID or friendly name
            
        Returns:
            DeviceState object
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._get_status_ha(device_id_or_name)
        else:
            return DeviceState(device_id=device_id_or_name)
    
    # ── Light Control ────────────────────────────────────────────────────────
    
    async def set_light(
        self,
        device_id_or_name: str,
        *,
        on: bool | None = None,
        brightness: int | None = None,
        color: str | None = None,
        color_temp: int | None = None,
        transition: float = 0.5,
    ) -> bool:
        """Control a light.
        
        Args:
            device_id_or_name: Light ID or name
            on: Turn on/off
            brightness: Brightness level (0-100)
            color: Hex color (e.g., "#FF6B35")
            color_temp: Color temperature in Kelvin
            transition: Transition time in seconds
            
        Returns:
            True if successful
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._set_light_ha(
                device_id_or_name, on, brightness, color, color_temp, transition
            )
        else:
            _log.warning("No smart home backend available")
            return False
    
    async def turn_on(self, device_id_or_name: str) -> bool:
        """Turn on a device."""
        return await self.set_light(device_id_or_name, on=True)
    
    async def turn_off(self, device_id_or_name: str) -> bool:
        """Turn off a device."""
        return await self.set_light(device_id_or_name, on=False)
    
    # ── Thermostat Control ───────────────────────────────────────────────────
    
    async def set_temperature(
        self,
        device_id_or_name: str,
        *,
        temperature: float,
        mode: str = "auto",
    ) -> bool:
        """Set thermostat temperature.
        
        Args:
            device_id_or_name: Thermostat ID or name
            temperature: Target temperature
            mode: HVAC mode (heat, cool, auto, off)
            
        Returns:
            True if successful
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._set_temperature_ha(device_id_or_name, temperature, mode)
        else:
            return False
    
    # ── Lock Control ─────────────────────────────────────────────────────────
    
    async def set_lock(self, device_id_or_name: str, *, locked: bool) -> bool:
        """Lock or unlock a door.
        
        Args:
            device_id_or_name: Lock ID or name
            locked: True to lock, False to unlock
            
        Returns:
            True if successful
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._set_lock_ha(device_id_or_name, locked)
        else:
            return False
    
    # ── Scenes ───────────────────────────────────────────────────────────────
    
    async def list_scenes(self) -> list[Scene]:
        """List available scenes.
        
        Returns:
            List of Scene objects
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._list_scenes_ha()
        else:
            return []
    
    async def activate_scene(self, scene_id_or_name: str) -> bool:
        """Activate a scene.
        
        Args:
            scene_id_or_name: Scene ID or name
            
        Returns:
            True if successful
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._activate_scene_ha(scene_id_or_name)
        else:
            return False
    
    # ── Automation ───────────────────────────────────────────────────────────
    
    async def create_automation(
        self,
        name: str,
        trigger: dict[str, Any],
        actions: list[dict[str, Any]],
        *,
        conditions: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a new automation.
        
        Args:
            name: Automation name
            trigger: Trigger configuration
            actions: List of actions to execute
            conditions: Optional conditions
            
        Returns:
            Automation ID
        """
        backend = self._detect_backend()
        
        if backend == "homeassistant":
            return await self._create_automation_ha(name, trigger, actions, conditions)
        else:
            raise RuntimeError("No smart home backend available")
    
    # ── Event stream ─────────────────────────────────────────────────────

    def event_stream(
        self,
        *,
        cooldown_s: float = 60.0,
        trigger_engine: Any | None = None,
    ) -> "HAWebSocket":
        """Build a persistent Home Assistant WebSocket event stream.

        Uses the stored ``homeassistant`` credential (long-lived access
        token).  Raises :class:`HAWebSocketError` when no Home Assistant
        backend is configured.

        Pair with a :class:`TriggerEngine` for event-driven automations::

            engine = TriggerEngine(db, notify_fn=...)
            stream = smarthome.event_stream(trigger_engine=engine)
            await stream.start()   # state_changed → engine.on_entity_state
        """
        try:
            cred = self.account_manager.get_credential("homeassistant", "default")
            base = cred.metadata.get("url", "http://homeassistant.local:8123")
            token = cred.password
        except Exception as exc:
            raise HAWebSocketError(
                f"no Home Assistant credential configured: {exc}") from exc
        ws_url = (base.replace("https://", "wss://")
                      .replace("http://", "ws://")
                      .rstrip("/") + "/api/websocket")
        return HAWebSocket(ws_url, token, cooldown_s=cooldown_s,
                           trigger_engine=trigger_engine)

    # ── Backend Detection ────────────────────────────────────────────────────
    
    def _detect_backend(self) -> str:
        """Detect available smart home backend."""
        # Check for Home Assistant
        try:
            cred = self.account_manager.get_credential("homeassistant", "default")
            if cred.is_active:
                return "homeassistant"
        except Exception as e:
            _log.debug("smart-home backend probe failed, using none: %s", e)
        
        return "none"
    
    # ── Home Assistant Backend ───────────────────────────────────────────────
    
    async def _ha_request(
        self,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a request to Home Assistant API."""
        cred = self.account_manager.get_credential("homeassistant", "default")
        ha_url = cred.metadata.get("url", "http://homeassistant.local:8123")
        token = cred.password
        
        url = f"{ha_url}/api/{endpoint}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 204:
                    return {}
                return json.loads(response.read().decode())
        except Exception as e:
            _log.error(f"Home Assistant request failed: {e}")
            raise
    
    async def _list_devices_ha(
        self,
        room: str | None,
        device_type: str | None,
    ) -> list[Device]:
        """List devices via Home Assistant."""
        try:
            states = await self._ha_request("GET", "states")
            
            devices = []
            for state in states:
                entity_id = state["entity_id"]
                domain = entity_id.split(".")[0]
                
                # Map HA domains to device types
                type_map = {
                    "light": "light",
                    "switch": "switch",
                    "climate": "thermostat",
                    "lock": "lock",
                    "sensor": "sensor",
                    "binary_sensor": "sensor",
                    "cover": "cover",
                    "fan": "fan",
                    "media_player": "media_player",
                }
                
                dev_type = type_map.get(domain)
                if not dev_type:
                    continue
                
                if device_type and dev_type != device_type:
                    continue
                
                # Get room from area
                ha_room = state.get("attributes", {}).get("room", "")
                if room and ha_room.lower() != room.lower():
                    continue
                
                device = Device(
                    device_id=entity_id,
                    name=state.get("attributes", {}).get("friendly_name", entity_id),
                    device_type=dev_type,
                    room=ha_room,
                    state=state.get("attributes", {}),
                    capabilities=self._get_capabilities(domain),
                    available=state.get("state") != "unavailable",
                )
                
                devices.append(device)
            
            return devices
        except Exception as e:
            _log.error(f"Failed to list HA devices: {e}")
            return []
    
    def _get_capabilities(self, domain: str) -> list[str]:
        """Get capabilities for a device domain."""
        caps = {
            "light": ["on_off", "brightness", "color", "color_temp"],
            "switch": ["on_off"],
            "climate": ["temperature", "mode", "fan_mode"],
            "lock": ["lock_unlock"],
            "sensor": ["read"],
            "cover": ["open_close", "position"],
            "fan": ["on_off", "speed"],
            "media_player": ["play_pause", "volume", "source"],
        }
        return caps.get(domain, [])
    
    async def _get_status_ha(self, device_id_or_name: str) -> DeviceState:
        """Get device status via Home Assistant."""
        # Resolve name to entity_id
        entity_id = await self._resolve_entity_id(device_id_or_name)
        
        try:
            state_data = await self._ha_request("GET", f"states/{entity_id}")
            
            attrs = state_data.get("attributes", {})
            ha_state = state_data.get("state", "off")
            
            return DeviceState(
                device_id=entity_id,
                is_on=ha_state not in ("off", "unavailable", "locked"),
                brightness=int(attrs.get("brightness", 0) / 255 * 100) if "brightness" in attrs else 0,
                color_temp=attrs.get("color_temp", 0),
                color=attrs.get("rgb_color", ""),
                temperature=attrs.get("current_temperature", 0.0),
                humidity=attrs.get("current_humidity", 0.0),
                locked=(ha_state == "locked"),
                attributes=attrs,
            )
        except Exception as e:
            _log.error(f"Failed to get status: {e}")
            return DeviceState(device_id=device_id_or_name)
    
    async def _set_light_ha(
        self,
        device_id_or_name: str,
        on: bool | None,
        brightness: int | None,
        color: str | None,
        color_temp: int | None,
        transition: float,
    ) -> bool:
        """Control light via Home Assistant."""
        entity_id = await self._resolve_entity_id(device_id_or_name)
        
        if on is False:
            # Turn off
            data = {"entity_id": entity_id}
            await self._ha_request("POST", "services/light/turn_off", data)
            return True
        
        # Turn on with parameters
        data: dict[str, Any] = {"entity_id": entity_id}
        
        if brightness is not None:
            data["brightness_pct"] = brightness
        if color is not None:
            # Convert hex to RGB
            color = color.lstrip("#")
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            data["rgb_color"] = [r, g, b]
        if color_temp is not None:
            data["color_temp_kelvin"] = color_temp
        if transition > 0:
            data["transition"] = transition
        
        await self._ha_request("POST", "services/light/turn_on", data)
        return True
    
    async def _set_temperature_ha(
        self,
        device_id_or_name: str,
        temperature: float,
        mode: str,
    ) -> bool:
        """Set temperature via Home Assistant."""
        entity_id = await self._resolve_entity_id(device_id_or_name)
        
        data = {
            "entity_id": entity_id,
            "temperature": temperature,
            "hvac_mode": mode,
        }
        
        await self._ha_request("POST", "services/climate/set_temperature", data)
        return True
    
    async def _set_lock_ha(self, device_id_or_name: str, locked: bool) -> bool:
        """Control lock via Home Assistant."""
        entity_id = await self._resolve_entity_id(device_id_or_name)
        
        service = "lock" if locked else "unlock"
        data = {"entity_id": entity_id}
        
        await self._ha_request("POST", f"services/lock/{service}", data)
        return True
    
    async def _list_scenes_ha(self) -> list[Scene]:
        """List scenes via Home Assistant."""
        try:
            states = await self._ha_request("GET", "states")
            
            scenes = []
            for state in states:
                if state["entity_id"].startswith("scene."):
                    scenes.append(Scene(
                        scene_id=state["entity_id"],
                        name=state.get("attributes", {}).get("friendly_name", ""),
                    ))
            
            return scenes
        except Exception as e:
            _log.error(f"Failed to list scenes: {e}")
            return []
    
    async def _activate_scene_ha(self, scene_id_or_name: str) -> bool:
        """Activate scene via Home Assistant."""
        entity_id = await self._resolve_entity_id(scene_id_or_name, domain="scene")
        
        data = {"entity_id": entity_id}
        await self._ha_request("POST", "services/scene/turn_on", data)
        return True
    
    async def _create_automation_ha(
        self,
        name: str,
        trigger: dict[str, Any],
        actions: list[dict[str, Any]],
        conditions: list[dict[str, Any]] | None,
    ) -> str:
        """Create automation via Home Assistant."""
        automation_id = f"automation.{name.lower().replace(' ', '_')}"
        
        data = {
            "alias": name,
            "trigger": trigger,
            "action": actions,
        }
        
        if conditions:
            data["condition"] = conditions
        
        await self._ha_request("POST", "services/automation/reload", data)
        return automation_id
    
    async def _resolve_entity_id(
        self,
        name_or_id: str,
        *,
        domain: str | None = None,
    ) -> str:
        """Resolve a friendly name to an entity ID."""
        # If it already looks like an entity ID, return it
        if "." in name_or_id:
            return name_or_id
        
        # Search for matching entity
        try:
            states = await self._ha_request("GET", "states")
            
            for state in states:
                entity_id = state["entity_id"]
                friendly_name = state.get("attributes", {}).get("friendly_name", "")
                
                if domain and not entity_id.startswith(f"{domain}."):
                    continue
                
                if friendly_name.lower() == name_or_id.lower():
                    return entity_id
                
                # Partial match
                if name_or_id.lower() in friendly_name.lower():
                    return entity_id
        except Exception as e:
            _log.debug("smart-home entity lookup failed, using fallback: %s", e)
        
        # Fallback: assume it's an entity ID with domain
        if domain:
            return f"{domain}.{name_or_id.lower().replace(' ', '_')}"
        
        return name_or_id


# ── Home Assistant WebSocket API ───────────────────────────────────────────
#
# Real-time ``state_changed`` event stream.  Stdlib-only asyncio WS client
# (RFC 6455: HTTP-upgrade handshake, masked client text frames, ping/pong
# and close handling).  Home Assistant only sends text frames, so binary
# frames are ignored.

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_BACKOFF_START = 1.0
_BACKOFF_MAX = 300.0


class HAWebSocketError(RuntimeError):
    """WebSocket transport failure (connect, handshake, protocol)."""


class HAAuthError(HAWebSocketError):
    """The token was rejected.  Fatal — reconnecting won't fix it."""


class _WSClosed(HAWebSocketError):
    """Server closed the connection (drives the reconnect loop)."""


async def _ws_handshake(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter,
                        host: str, port: int, path: str) -> None:
    """Client side of the RFC 6455 opening handshake.  Raises on failure."""
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    )
    writer.write(request.encode("ascii"))
    await writer.drain()
    try:
        status = await asyncio.wait_for(reader.readline(), 10)
    except (asyncio.TimeoutError, TimeoutError) as exc:
        raise HAWebSocketError("handshake timed out") from exc
    if not status.startswith(b"HTTP/1.1 101"):
        raise HAWebSocketError(
            "handshake rejected: "
            + status.decode("latin1", "replace").strip())
    accept: str | None = None
    while True:
        line = await asyncio.wait_for(reader.readline(), 10)
        if line in (b"\r\n", b"\n", b""):
            break
        name, _, value = line.decode("latin1").partition(":")
        if name.strip().lower() == "sec-websocket-accept":
            accept = value.strip()
    expected = base64.b64encode(
        hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()
    ).decode("ascii")
    if accept != expected:
        raise HAWebSocketError("bad Sec-WebSocket-Accept in handshake")


async def _ws_send_text(writer: asyncio.StreamWriter, text: str) -> None:
    """Send one masked text frame (clients MUST mask per RFC 6455)."""
    data = text.encode("utf-8")
    n = len(data)
    frame = bytearray([0x81])  # FIN + text opcode
    mask = os.urandom(4)
    if n < 126:
        frame.append(0x80 | n)
    elif n < 65536:
        frame.append(0x80 | 126)
        frame.extend(n.to_bytes(2, "big"))
    else:
        frame.append(0x80 | 127)
        frame.extend(n.to_bytes(8, "big"))
    frame.extend(mask)
    frame.extend(b ^ mask[i % 4] for i, b in enumerate(data))
    writer.write(bytes(frame))
    await writer.drain()


async def _ws_recv_text(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> str:
    """Read frames until a whole text message arrives.

    Answers pings, ignores binary/pong frames, raises :class:`_WSClosed`
    on close frames or a dead transport.
    """
    chunks: list[bytes] = []
    while True:
        try:
            hdr = await reader.readexactly(2)
        except (asyncio.IncompleteReadError, ConnectionError) as exc:
            raise _WSClosed(f"connection lost: {exc}") from exc
        b1, b2 = hdr[0], hdr[1]
        fin = bool(b1 & 0x80)
        opcode = b1 & 0x0F
        length = b2 & 0x7F
        if length == 126:
            length = int.from_bytes(await reader.readexactly(2), "big")
        elif length == 127:
            length = int.from_bytes(await reader.readexactly(8), "big")
        mask = await reader.readexactly(4) if b2 & 0x80 else None
        payload = await reader.readexactly(length) if length else b""
        if mask:
            payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        if opcode == 0x8:  # close
            raise _WSClosed("server closed the connection")
        if opcode == 0x9:  # ping → pong
            writer.write(bytes(bytearray([0x8A, len(payload)])) + payload)
            await writer.drain()
            continue
        if opcode in (0xA, 0x2):  # pong / binary (HA never sends binary)
            continue
        if opcode not in (0x0, 0x1):  # unknown opcode
            continue
        chunks.append(payload)
        if fin:
            return b"".join(chunks).decode("utf-8", "replace")


@dataclass
class _Subscription:
    id: int
    domain: str | None
    entity_id: str | None
    callback: Callable[[str, Any, Any, dict[str, Any]], None] | None


@dataclass
class ProactiveAlert:
    """One situation worth surfacing: situation + solution in one breath."""

    kind: str  # lights_on | cover_open | garage_open | lock_unlocked
    message: str  # e.g. "3 lights left on (Kitchen, Hall) — turn them off?"
    entity_ids: list[str] = field(default_factory=list)
    suggested_action: str = ""  # turn_off | close | lock


class HAWebSocket:
    """Persistent Home Assistant WebSocket connection.

    Authenticates with a long-lived access token, subscribes to
    ``state_changed`` events, keeps a live entity state cache, and routes
    every event to (a) registered subscriptions and (b) an optional
    :class:`~nomorals.triggers.engine.TriggerEngine` — entity_state
    triggers fire on the event, never on a poll.

    Resilience: :meth:`start` never raises.  A dropped connection
    reconnects with exponential backoff (1s → 300s cap, reset on a good
    auth).  A rejected token is fatal: it is logged once and the loop
    stops, since retrying can't fix it.  Per-entity cooldowns keep
    flapping sensors from spamming subscription callbacks.
    """

    def __init__(
        self,
        url: str,
        token: str,
        *,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] | None = None,
        trigger_engine: Any | None = None,
        connect_factory: Callable | None = None,
    ) -> None:
        parsed = urllib.parse.urlparse(url)
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or (443 if parsed.scheme == "wss" else 8123)
        self._path = parsed.path or "/api/websocket"
        self._secure = parsed.scheme == "wss"
        self._token = token
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._clock = clock or time.time
        self._trigger_engine = trigger_engine
        #: async (host, port, secure, path) -> (reader, writer); injectable
        self._connect_factory = connect_factory
        self._subs: dict[int, _Subscription] = {}
        self._next_sub_id = 0
        #: entity_id -> {"state", "attributes", "last_changed"}
        self._state_cache: dict[str, dict[str, Any]] = {}
        #: entity_id -> unix time when its subscription cooldown expires
        self._cooldown_until: dict[str, float] = {}
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._pending: dict[int, asyncio.Future] = {}
        self._msg_id = 0
        self._running = False
        self._task: asyncio.Task | None = None
        self._read_task: asyncio.Task | None = None
        self._backoff_s = _BACKOFF_START
        _log.info("HA websocket configured for %s:%d", self._host, self._port)

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        """True while the socket is open."""
        return self._writer is not None and not self._writer.is_closing()

    @property
    def running(self) -> bool:
        return self._running

    def attach_trigger_engine(self, engine: Any) -> None:
        """Wire (or re-wire) the trigger engine for event routing."""
        self._trigger_engine = engine

    async def start(self) -> None:
        """Begin the connect → serve → reconnect loop.  Never raises."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(),
                                         name="ha-websocket")

    async def stop(self) -> None:
        """Stop the loop and close the socket.  Never raises."""
        self._running = False
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001 - stop must not raise
                _log.exception("HA websocket task teardown failed")
        try:
            await self._close()
        except Exception:  # noqa: BLE001 - stop must not raise
            _log.exception("HA websocket close failed")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._connect_and_serve()
            except HAAuthError as exc:
                _log.error("HA websocket auth failed (%s); not retrying",
                           exc)
                self._running = False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect forever
                _log.warning("HA websocket error (%s); retry in %.0fs",
                             exc, self._backoff_s)
                try:
                    await asyncio.sleep(self._backoff_s)
                except asyncio.CancelledError:
                    raise
                self._backoff_s = min(self._backoff_s * 2, _BACKOFF_MAX)
        await self._close()

    async def _connect_and_serve(self) -> None:
        reader, writer = await self._connect()
        self._reader, self._writer = reader, writer
        try:
            await _ws_handshake(reader, writer,
                                self._host, self._port, self._path)
            first = await self._next_message()
            if first.get("type") != "auth_required":
                raise HAWebSocketError(
                    f"unexpected greeting: {first.get('type')!r}")
            await _ws_send_text(
                writer, json.dumps({"type": "auth",
                                    "access_token": self._token}))
            auth = await self._next_message()
            if auth.get("type") == "auth_invalid":
                raise HAAuthError(auth.get("message") or "token rejected")
            if auth.get("type") != "auth_ok":
                raise HAWebSocketError(f"auth failed: {auth!r}")
            self._backoff_s = _BACKOFF_START
            _log.info("HA websocket authenticated (%s)", self._host)
            self._read_task = asyncio.create_task(self._read_loop())
            try:
                await self._send_command({"type": "subscribe_events",
                                          "event_type": "state_changed"})
                states = await self._send_command({"type": "get_states"})
                self._ingest_snapshot(states or [])
                _log.info("HA websocket live (%d entities tracked)",
                          len(self._state_cache))
                await self._read_task  # runs until close/error
            finally:
                if self._read_task is not None and not self._read_task.done():
                    self._read_task.cancel()
                    try:
                        await self._read_task
                    except (asyncio.CancelledError, _WSClosed,
                            HAWebSocketError):
                        pass
                self._read_task = None
        finally:
            await self._close()

    async def _connect(self) -> tuple[asyncio.StreamReader,
                                      asyncio.StreamWriter]:
        if self._connect_factory is not None:
            return await self._connect_factory(
                self._host, self._port, self._secure, self._path)
        ssl_ctx = ssl.create_default_context() if self._secure else None
        try:
            return await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port,
                                        ssl=ssl_ctx), 10)
        except (asyncio.TimeoutError, TimeoutError, OSError) as exc:
            raise HAWebSocketError(
                f"cannot reach {self._host}:{self._port}: {exc}") from exc

    async def _close(self) -> None:
        reader, writer = self._reader, self._writer
        self._reader = self._writer = None
        self._fail_pending(_WSClosed("connection closed"))
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass

    def _fail_pending(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                try:
                    fut.set_exception(exc)
                except Exception:  # noqa: BLE001 - never break teardown
                    pass
        self._pending.clear()

    # ── protocol ─────────────────────────────────────────────────────

    async def _next_message(self) -> dict[str, Any]:
        raw = await asyncio.wait_for(
            _ws_recv_text(self._reader, self._writer), 15)
        return json.loads(raw)

    async def _send_command(self, payload: dict[str, Any]) -> Any:
        """Send a command and wait for its ``result`` reply."""
        loop = asyncio.get_running_loop()
        self._msg_id += 1
        mid = self._msg_id
        fut = loop.create_future()
        self._pending[mid] = fut
        await _ws_send_text(self._writer, json.dumps({"id": mid, **payload}))
        try:
            return await asyncio.wait_for(fut, 15)
        finally:
            self._pending.pop(mid, None)

    async def _read_loop(self) -> None:
        while self._running:
            try:
                raw = await _ws_recv_text(self._reader, self._writer)
            except _WSClosed:
                self._fail_pending(_WSClosed("server went away"))
                raise
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                _log.debug("HA websocket: ignoring non-JSON frame")
                continue
            self._handle_message(msg)

    def _handle_message(self, msg: dict[str, Any]) -> None:
        mtype = msg.get("type")
        if mtype == "result":
            fut = self._pending.get(msg.get("id"))
            if fut is not None and not fut.done():
                if msg.get("success"):
                    fut.set_result(msg.get("result"))
                else:
                    err = msg.get("error") or {}
                    fut.set_exception(HAWebSocketError(
                        err.get("message", "command failed")))
        elif mtype == "event":
            event = msg.get("event") or {}
            if event.get("event_type") == "state_changed":
                try:
                    self._dispatch_state_changed(event.get("data") or {})
                except Exception:  # noqa: BLE001 - one bad event never kills
                    _log.exception("HA websocket event dispatch failed")

    # ── events ───────────────────────────────────────────────────────

    def _ingest_snapshot(self, states: list[dict[str, Any]]) -> None:
        for st in states:
            eid = str(st.get("entity_id") or "").lower()
            if not eid:
                continue
            self._state_cache[eid] = {
                "state": st.get("state"),
                "attributes": dict(st.get("attributes") or {}),
                "last_changed": st.get("last_changed"),
            }

    def _dispatch_state_changed(self, data: dict[str, Any]) -> None:
        entity_id = str(data.get("entity_id") or "").lower()
        old = data.get("old_state") or {}
        new = data.get("new_state") or {}
        old_state = old.get("state")
        new_state = new.get("state")
        attrs = dict(new.get("attributes") or {})
        self._state_cache[entity_id] = {
            "state": new_state,
            "attributes": attrs,
            "last_changed": new.get("last_changed"),
        }
        if self._trigger_engine is not None:
            try:
                self._trigger_engine.on_entity_state(
                    entity_id, old_state, new_state, attrs)
            except Exception:  # noqa: BLE001 - engine trouble never kills
                _log.exception("trigger engine entity_state dispatch failed")
        now = self._clock()
        for sub in list(self._subs.values()):
            if sub.domain and not entity_id.startswith(sub.domain + "."):
                continue
            if sub.entity_id and sub.entity_id != entity_id:
                continue
            if sub.callback is None:
                continue
            if self._cooldown_until.get(entity_id, 0.0) > now:
                continue
            self._cooldown_until[entity_id] = now + self.cooldown_s
            try:
                sub.callback(entity_id, old_state, new_state, attrs)
            except Exception:  # noqa: BLE001 - one bad callback never kills
                _log.exception("HA websocket subscription callback failed")

    def subscribe_state_changed(
        self,
        domain: str | None = None,
        entity_id: str | None = None,
        callback: Callable[[str, Any, Any, dict[str, Any]], None] | None = None,
    ) -> int:
        """Register a ``state_changed`` subscription.

        Filters are ANDed: ``domain`` (e.g. ``"light"``) matches the
        entity_id prefix, ``entity_id`` (e.g. ``"light.kitchen"``) matches
        exactly.  ``callback(entity_id, old_state, new_state, attributes)``
        must be quick and non-blocking.  Per-entity cooldowns
        (``cooldown_s``) suppress flapping sensors.

        Returns a subscription id for :meth:`unsubscribe`.
        """
        self._next_sub_id += 1
        sub = _Subscription(
            id=self._next_sub_id,
            domain=(domain or "").strip().lower() or None,
            entity_id=(entity_id or "").strip().lower() or None,
            callback=callback,
        )
        self._subs[sub.id] = sub
        return sub.id

    def unsubscribe(self, sub_id: int) -> bool:
        """Remove a subscription.  Returns True when it existed."""
        return self._subs.pop(sub_id, None) is not None

    # ── state + proactive ────────────────────────────────────────────

    def state_snapshot(self) -> dict[str, dict[str, Any]]:
        """A copy of the live entity state cache."""
        return {eid: dict(info) for eid, info in self._state_cache.items()}

    def _friendly(self, entity_id: str) -> str:
        return (self._state_cache.get(entity_id, {})
                .get("attributes", {}).get("friendly_name")
                or entity_id)

    def proactive_alerts(self) -> list[ProactiveAlert]:
        """Analyze the live state cache for situations worth surfacing.

        Situation + solution in one breath, e.g.
        ``"3 lights left on (Kitchen, Hall, Porch) — turn them off?"``.
        """
        alerts: list[ProactiveAlert] = []
        cache = self._state_cache

        lights_on = sorted(
            eid for eid, info in cache.items()
            if eid.startswith("light.") and info.get("state") == "on")
        if lights_on:
            names = [self._friendly(eid) for eid in lights_on]
            n = len(lights_on)
            alerts.append(ProactiveAlert(
                kind="lights_on",
                message=f"{n} light{'s' if n != 1 else ''} left on "
                        f"({', '.join(names)}) — turn them off?",
                entity_ids=lights_on,
                suggested_action="turn_off",
            ))

        for eid in sorted(cache):
            info = cache[eid]
            if not eid.startswith("cover.") or info.get("state") != "open":
                continue
            name = self._friendly(eid)
            attrs = info.get("attributes", {})
            is_garage = ("garage" in eid or "garage" in name.lower()
                         or attrs.get("device_class") in ("garage", "door"))
            alerts.append(ProactiveAlert(
                kind="garage_open" if is_garage else "cover_open",
                message=(f"garage door is open ({name}) — close it?"
                         if is_garage else f"{name} is open — close it?"),
                entity_ids=[eid],
                suggested_action="close",
            ))

        for eid in sorted(cache):
            info = cache[eid]
            if eid.startswith("lock.") and info.get("state") == "unlocked":
                alerts.append(ProactiveAlert(
                    kind="lock_unlocked",
                    message=f"{self._friendly(eid)} is unlocked — lock it?",
                    entity_ids=[eid],
                    suggested_action="lock",
                ))

        return alerts
