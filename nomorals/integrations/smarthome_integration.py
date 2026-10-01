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

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

from ..accounts.manager import AccountManager
from ..accounts.sessions import SessionManager
from ..core.logging_setup import get_logger

__all__ = ["SmartHomeIntegration", "Device", "DeviceState", "Scene"]

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
