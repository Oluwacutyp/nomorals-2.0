"""Paired devices - phone as sensor/command surface.

Treats the user's phone as a remote sensor and command surface:
- Location tracking (GPS)
- Camera access (take photos)
- Sensors (accelerometer, gyroscope, light, proximity)
- Battery status
- Network info (WiFi SSID, signal strength)
- Device info (model, OS version)
- Notifications (push to phone)
- Clipboard (read/write)
- Vibration/haptics
- Screen brightness
- Volume control

The phone runs a lightweight companion app that exposes these capabilities
via a REST API or WebSocket connection.

Usage:
    devices = PairedDeviceManager(db)
    
    # Pair a new device
    device = await devices.pair(
        device_id="phone123",
        name="User's iPhone",
        api_url="http://192.168.1.100:8080",
    )
    
    # Get location
    location = await devices.get_location("phone123")
    
    # Take a photo
    photo_path = await devices.take_photo("phone123", camera="back")
    
    # Get battery status
    battery = await devices.get_battery("phone123")
    
    # Push notification
    await devices.push_notification("phone123", "Meeting in 5 minutes!")
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from ..storage.db import Database

__all__ = [
    "PairedDeviceManager",
    "PairedDevice",
    "DeviceLocation",
    "DeviceBattery",
    "DeviceSensor",
]

_log = get_logger(__name__)


@dataclass
class PairedDevice:
    """A paired device (phone, tablet, etc.)."""
    
    device_id: str
    name: str
    api_url: str
    api_key: str = ""
    device_type: str = "phone"  # phone, tablet, watch
    os: str = ""  # iOS, Android
    os_version: str = ""
    model: str = ""
    is_online: bool = False
    last_seen: float = 0.0
    paired_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "device_type": self.device_type,
            "os": self.os,
            "model": self.model,
            "is_online": self.is_online,
            "last_seen": self.last_seen,
        }


@dataclass
class DeviceLocation:
    """Device location (GPS)."""
    
    latitude: float
    longitude: float
    accuracy: float = 0.0  # meters
    altitude: float = 0.0
    speed: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy": self.accuracy,
            "altitude": self.altitude,
            "timestamp": self.timestamp,
        }


@dataclass
class DeviceBattery:
    """Device battery status."""
    
    level: int  # 0-100
    is_charging: bool
    battery_health: str = "good"  # good, fair, poor
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "is_charging": self.is_charging,
            "battery_health": self.battery_health,
        }


@dataclass
class DeviceSensor:
    """Device sensor reading."""
    
    sensor_type: str  # accelerometer, gyroscope, light, proximity, etc.
    values: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "sensor_type": self.sensor_type,
            "values": self.values,
            "timestamp": self.timestamp,
        }


class PairedDeviceManager:
    """Manages paired devices and their capabilities."""
    
    def __init__(self, db: Database) -> None:
        self.db = db
        self._ensure_schema()
        _log.info("Paired device manager initialized")
    
    def _ensure_schema(self) -> None:
        with self.db.transaction():
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS paired_devices (
                    device_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    api_url TEXT NOT NULL,
                    api_key TEXT NOT NULL DEFAULT '',
                    device_type TEXT NOT NULL DEFAULT 'phone',
                    os TEXT NOT NULL DEFAULT '',
                    os_version TEXT NOT NULL DEFAULT '',
                    model TEXT NOT NULL DEFAULT '',
                    is_online INTEGER NOT NULL DEFAULT 0,
                    last_seen REAL NOT NULL DEFAULT 0,
                    paired_at REAL NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)
    
    async def pair(
        self,
        device_id: str,
        name: str,
        api_url: str,
        *,
        api_key: str = "",
    ) -> PairedDevice:
        """Pair a new device.
        
        Args:
            device_id: Unique device identifier
            name: Human-readable name
            api_url: Device API URL
            api_key: Optional API key for authentication
            
        Returns:
            PairedDevice object
        """
        device = PairedDevice(
            device_id=device_id,
            name=name,
            api_url=api_url,
            api_key=api_key,
        )
        
        # Verify connectivity
        try:
            info = await self._api_request(device, "GET", "/info")
            device.os = info.get("os", "")
            device.os_version = info.get("os_version", "")
            device.model = info.get("model", "")
            device.is_online = True
            device.last_seen = time.time()
        except Exception as e:
            _log.warning(f"Device paired but unreachable: {e}")
        
        with self.db.transaction():
            self.db.execute("""
                INSERT OR REPLACE INTO paired_devices
                (device_id, name, api_url, api_key, device_type, os, os_version,
                 model, is_online, last_seen, paired_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                device.device_id, device.name, device.api_url, device.api_key,
                device.device_type, device.os, device.os_version, device.model,
                int(device.is_online), device.last_seen, device.paired_at,
                json.dumps(device.metadata),
            ))
        
        _log.info(f"Paired device: {name} ({device_id})")
        return device
    
    async def list_devices(self, *, online_only: bool = False) -> list[PairedDevice]:
        """List all paired devices."""
        query = "SELECT * FROM paired_devices"
        if online_only:
            query += " WHERE is_online = 1"
        query += " ORDER BY last_seen DESC"
        
        rows = self.db.query(query)
        return [self._row_to_device(r) for r in rows]
    
    async def get_device(self, device_id: str) -> Optional[PairedDevice]:
        """Get a device by ID."""
        row = self.db.query_one(
            "SELECT * FROM paired_devices WHERE device_id = ?",
            (device_id,)
        )
        return self._row_to_device(row) if row else None
    
    async def unpair(self, device_id: str) -> bool:
        """Unpair a device."""
        with self.db.transaction():
            self.db.execute("DELETE FROM paired_devices WHERE device_id = ?", (device_id,))
        _log.info(f"Unpaired device: {device_id}")
        return True
    
    # ── Location ─────────────────────────────────────────────────────────────
    
    async def get_location(self, device_id: str) -> Optional[DeviceLocation]:
        """Get device location (GPS)."""
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            data = await self._api_request(device, "GET", "/location")
            return DeviceLocation(
                latitude=data["latitude"],
                longitude=data["longitude"],
                accuracy=data.get("accuracy", 0),
                altitude=data.get("altitude", 0),
                speed=data.get("speed", 0),
            )
        except Exception as e:
            _log.error(f"Failed to get location: {e}")
            return None
    
    # ── Camera ───────────────────────────────────────────────────────────────
    
    async def take_photo(
        self,
        device_id: str,
        *,
        camera: str = "back",
        flash: bool = False,
        save_path: str = "",
    ) -> Optional[str]:
        """Take a photo with device camera.
        
        Args:
            device_id: Device ID
            camera: "back" or "front"
            flash: Use flash
            save_path: Where to save the photo (default: temp file)
            
        Returns:
            Path to saved photo, or None if failed
        """
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            data = await self._api_request(device, "POST", "/camera/photo", {
                "camera": camera,
                "flash": flash,
            })
            
            # Download photo
            photo_url = data.get("photo_url")
            if not photo_url:
                return None
            
            if not save_path:
                save_path = f"/tmp/photo_{device_id}_{int(time.time())}.jpg"
            
            urllib.request.urlretrieve(photo_url, save_path)
            return save_path
        except Exception as e:
            _log.error(f"Failed to take photo: {e}")
            return None
    
    # ── Battery ──────────────────────────────────────────────────────────────
    
    async def get_battery(self, device_id: str) -> Optional[DeviceBattery]:
        """Get device battery status."""
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            data = await self._api_request(device, "GET", "/battery")
            return DeviceBattery(
                level=data["level"],
                is_charging=data.get("is_charging", False),
                battery_health=data.get("health", "good"),
            )
        except Exception as e:
            _log.error(f"Failed to get battery: {e}")
            return None
    
    # ── Sensors ──────────────────────────────────────────────────────────────
    
    async def get_sensor(self, device_id: str, sensor_type: str) -> Optional[DeviceSensor]:
        """Get sensor reading.
        
        Args:
            device_id: Device ID
            sensor_type: accelerometer, gyroscope, light, proximity, etc.
        """
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            data = await self._api_request(device, "GET", f"/sensor/{sensor_type}")
            return DeviceSensor(
                sensor_type=sensor_type,
                values=data.get("values", {}),
            )
        except Exception as e:
            _log.error(f"Failed to get sensor: {e}")
            return None
    
    # ── Notifications ────────────────────────────────────────────────────────
    
    async def push_notification(
        self,
        device_id: str,
        message: str,
        *,
        title: str = "NoMorals AI",
        sound: bool = True,
    ) -> bool:
        """Push a notification to the device."""
        device = await self.get_device(device_id)
        if not device:
            return False
        
        try:
            await self._api_request(device, "POST", "/notification", {
                "title": title,
                "message": message,
                "sound": sound,
            })
            return True
        except Exception as e:
            _log.error(f"Failed to push notification: {e}")
            return False
    
    # ── Clipboard ────────────────────────────────────────────────────────────
    
    async def get_clipboard(self, device_id: str) -> Optional[str]:
        """Read device clipboard."""
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            data = await self._api_request(device, "GET", "/clipboard")
            return data.get("text", "")
        except Exception as e:
            _log.error(f"Failed to get clipboard: {e}")
            return None
    
    async def set_clipboard(self, device_id: str, text: str) -> bool:
        """Write to device clipboard."""
        device = await self.get_device(device_id)
        if not device:
            return False
        
        try:
            await self._api_request(device, "POST", "/clipboard", {"text": text})
            return True
        except Exception as e:
            _log.error(f"Failed to set clipboard: {e}")
            return False
    
    # ── Haptics ──────────────────────────────────────────────────────────────
    
    async def vibrate(self, device_id: str, *, pattern: str = "short") -> bool:
        """Trigger device vibration/haptics.
        
        Args:
            device_id: Device ID
            pattern: "short", "long", "double", "triple"
        """
        device = await self.get_device(device_id)
        if not device:
            return False
        
        try:
            await self._api_request(device, "POST", "/vibrate", {"pattern": pattern})
            return True
        except Exception as e:
            _log.error(f"Failed to vibrate: {e}")
            return False
    
    # ── Network ──────────────────────────────────────────────────────────────
    
    async def get_network_info(self, device_id: str) -> Optional[dict[str, Any]]:
        """Get device network info (WiFi SSID, signal strength, etc.)."""
        device = await self.get_device(device_id)
        if not device:
            return None
        
        try:
            return await self._api_request(device, "GET", "/network")
        except Exception as e:
            _log.error(f"Failed to get network info: {e}")
            return None
    
    # ── Helpers ──────────────────────────────────────────────────────────────
    
    async def _api_request(
        self,
        device: PairedDevice,
        method: str,
        endpoint: str,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make API request to device."""
        url = f"{device.api_url}{endpoint}"
        
        headers = {"Content-Type": "application/json"}
        if device.api_key:
            headers["Authorization"] = f"Bearer {device.api_key}"
        
        body = json.dumps(data).encode() if data else None
        
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode())
                
                # Update last_seen
                with self.db.transaction():
                    self.db.execute("""
                        UPDATE paired_devices SET is_online = 1, last_seen = ?
                        WHERE device_id = ?
                    """, (time.time(), device.device_id))
                
                return result
        except Exception as e:
            # Mark offline
            with self.db.transaction():
                self.db.execute("""
                    UPDATE paired_devices SET is_online = 0 WHERE device_id = ?
                """, (device.device_id,))
            raise
    
    def _row_to_device(self, row: dict[str, Any]) -> PairedDevice:
        """Convert DB row to PairedDevice."""
        return PairedDevice(
            device_id=row["device_id"],
            name=row["name"],
            api_url=row["api_url"],
            api_key=row["api_key"],
            device_type=row["device_type"],
            os=row["os"],
            os_version=row["os_version"],
            model=row["model"],
            is_online=bool(row["is_online"]),
            last_seen=row["last_seen"],
            paired_at=row["paired_at"],
            metadata=json.loads(row["metadata"]),
        )
