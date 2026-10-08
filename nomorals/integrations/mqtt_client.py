"""Native MQTT client for Devon — one client, thousands of devices.

Build-map #63. Zigbee2MQTT topic grammar is Devon's device
abstraction::

    zigbee2mqtt/<device>              device state (JSON or plain)
    zigbee2mqtt/<device>/set          commands (Devon → device)
    zigbee2mqtt/<device>/get          state poll requests
    zigbee2mqtt/<device>/availability  online / offline

Device state updates route into the trigger engine through the
*same* ``entity_state`` source #61's Home Assistant stream uses
(``entity_id = "zigbee2mqtt.<device>"``) — one source, one
validation path, one history.  ``set_state()`` pairs with the
routine builder (#62): routines can target MQTT devices.

Resilience mirrors :class:`HAWebSocket`: :meth:`start` never
raises, a dropped broker reconnects with exponential backoff
(1s → 300s cap), per-device cooldowns keep flapping sensors
from spamming triggers, and a retained-message state cache
gives last-known state without a broker round-trip.

``paho-mqtt`` is the only dependency (``pip install paho-mqtt``).
When it isn't installed the bridge fails closed: :meth:`start`
logs the install recipe and stays down, ``publish``/``set_state``
return ``False`` — never a fake "sent".
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: pip recipe shown when paho-mqtt is missing.
PAHO_RECIPE = "pip install paho-mqtt  (then restart Devon)"

#: default Mosquitto broker.
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 1883

#: Zigbee2MQTT topic root — Devon's device abstraction.
Z2M_PREFIX = "zigbee2mqtt"

#: reconnect backoff bounds (mirrors HAWebSocket).
_BACKOFF_START = 1.0
_BACKOFF_CAP = 300.0


class MQTTError(RuntimeError):
    """Base MQTT failure."""


class MQTTUnavailable(MQTTError):
    """paho-mqtt isn't installed — the bridge can't run.  Fail closed."""


# ─────────────────────────── topic grammar ──────────────────────────────


def state_topic(device: str, *, prefix: str = Z2M_PREFIX) -> str:
    """``zigbee2mqtt/<device>`` — device state (inbound)."""
    return f"{prefix}/{device}"


def set_topic(device: str, *, prefix: str = Z2M_PREFIX) -> str:
    """``zigbee2mqtt/<device>/set`` — commands (Devon → device)."""
    return f"{prefix}/{device}/set"


def get_topic(device: str, *, prefix: str = Z2M_PREFIX) -> str:
    """``zigbee2mqtt/<device>/get`` — state poll requests."""
    return f"{prefix}/{device}/get"


def availability_topic(device: str, *, prefix: str = Z2M_PREFIX) -> str:
    """``zigbee2mqtt/<device>/availability`` — online / offline."""
    return f"{prefix}/{device}/availability"


def _entity_id(device: str, *, prefix: str = Z2M_PREFIX) -> str:
    """Map a Zigbee2MQTT device name onto a trigger-engine entity_id.

    ``zigbee2mqtt.<device>`` — sanitized so domain matching in
    ``match_entity_state`` works (domain == the prefix).
    """
    clean = re.sub(r"[^a-z0-9]+", "_", str(device or "").lower()).strip("_")
    return f"{prefix}.{clean or 'unknown'}"


def _decode_payload(raw: bytes | str) -> Any:
    """JSON when possible, otherwise the raw text."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return text


def _have_paho() -> bool:
    try:
        import paho.mqtt.client  # noqa: F401
        return True
    except ImportError:
        return False


class MQTTBridge:
    """Persistent MQTT broker connection with Zigbee2MQTT grammar.

    :meth:`start` never raises.  Everything else degrades to
    ``False``/``None`` when the broker is unreachable.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        username: str | None = None,
        password: str | None = None,
        client_id: str | None = None,
        keepalive: int = 60,
        prefix: str = Z2M_PREFIX,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] | None = None,
        trigger_engine: Any | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._host = host or DEFAULT_HOST
        self._port = int(port or DEFAULT_PORT)
        self._username = username
        self._password = password
        self._client_id = client_id or f"devon-mqtt-{int(time.time()) % 100000}"
        self._keepalive = keepalive
        self._prefix = prefix or Z2M_PREFIX
        self.cooldown_s = max(0.0, float(cooldown_s))
        self._clock = clock or time.time
        self._trigger_engine = trigger_engine
        #: () -> paho-like client; injectable so tests never need a broker.
        self._client_factory = client_factory

        self._subs: dict[int, tuple[str, Callable]] = {}
        self._next_sub_id = 0
        #: device -> {"state", "attributes", "last_changed", "raw"}
        self._state_cache: dict[str, dict[str, Any]] = {}
        #: device -> "online" | "offline"
        self._availability: dict[str, str] = {}
        #: entity_id -> unix time when its trigger cooldown expires
        self._cooldown_until: dict[str, float] = {}

        self._client: Any | None = None
        self._running = False
        self._connected = False
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()
        self._backoff_s = _BACKOFF_START
        _log.info("MQTT bridge configured for %s:%d (prefix %s)",
                  self._host, self._port, self._prefix)

    # ── lifecycle ────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def running(self) -> bool:
        return self._running

    def attach_trigger_engine(self, engine: Any) -> None:
        """Wire (or re-wire) the trigger engine for state routing."""
        self._trigger_engine = engine

    def start(self) -> None:
        """Begin the connect → serve → reconnect loop.  Never raises."""
        if self._running:
            return
        self._running = True
        self._wake.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="mqtt-bridge")
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and disconnect.  Never raises."""
        self._running = False
        self._wake.set()
        try:
            client, self._client = self._client, None
            if client is not None:
                try:
                    client.loop_stop()
                except Exception:  # noqa: BLE001 - best effort
                    pass
                try:
                    client.disconnect()
                except Exception:  # noqa: BLE001 - best effort
                    pass
        finally:
            self._connected = False

    # ── pub/sub ──────────────────────────────────────────────────────

    def subscribe(self, topic: str, callback: Callable[[str, Any], None]) -> int:
        """Subscribe to a raw topic; returns a subscription id."""
        sub_id = self._next_sub_id
        self._next_sub_id += 1
        self._subs[sub_id] = (topic, callback)
        client = self._client
        if client is not None and self._connected:
            try:
                client.subscribe(topic)
            except Exception:  # noqa: BLE001 - resubscribed on reconnect
                _log.debug("mqtt subscribe failed (will retry)", exc_info=True)
        return sub_id

    def unsubscribe(self, sub_id: int) -> None:
        self._subs.pop(sub_id, None)

    def publish(self, topic: str, payload: Any,
                *, retain: bool = False, qos: int = 0) -> bool:
        """Publish; ``False`` when not connected (never raises)."""
        client = self._client
        if client is None or not self._connected:
            return False
        if not isinstance(payload, (bytes, str)):
            payload = json.dumps(payload)
        try:
            client.publish(topic, payload, qos=qos, retain=retain)
            return True
        except Exception:  # noqa: BLE001 - broker hiccup, not fatal
            _log.debug("mqtt publish failed", exc_info=True)
            return False

    # ── device commands (Zigbee2MQTT grammar) ────────────────────────

    def set_state(self, device: str, state: Any) -> bool:
        """Command a device: publish to ``zigbee2mqtt/<device>/set``.

        Pairs with the routine builder (#62) — routines can target
        MQTT devices.  Returns ``False`` when the broker is down.
        """
        return self.publish(set_topic(device, prefix=self._prefix), state)

    def request_state(self, device: str) -> bool:
        """Poll a device: publish to ``zigbee2mqtt/<device>/get``."""
        return self.publish(get_topic(device, prefix=self._prefix), "")

    def device_state(self, device: str) -> dict[str, Any] | None:
        """Last-known state from the retained cache (no broker needed)."""
        return self._state_cache.get(str(device))

    def availability(self, device: str) -> str | None:
        """``"online"`` / ``"offline"`` / ``None`` (never seen)."""
        return self._availability.get(str(device))

    # ── internals ────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Supervisor: connect → serve → backoff → reconnect. Never raises."""
        while self._running:
            try:
                if self._client_factory is None and not _have_paho():
                    _log.warning("paho-mqtt not installed — MQTT bridge "
                                 "stays down. %s", PAHO_RECIPE)
                    self._running = False
                    return
                self._serve_once()
            except Exception:  # noqa: BLE001 - the loop must survive
                _log.debug("mqtt serve cycle failed", exc_info=True)
            if not self._running:
                break
            delay = self._backoff_s
            self._backoff_s = min(self._backoff_s * 2, _BACKOFF_CAP)
            _log.info("MQTT reconnecting in %.0fs", delay)
            self._wake.wait(delay)
        self._connected = False

    def _serve_once(self) -> None:
        """One connect → serve-until-disconnect cycle."""
        client = self._make_client()
        self._client = client
        disconnected = threading.Event()

        def _on_connect(_c, _u, _f, rc, _p=None):
            if rc == 0:
                self._connected = True
                self._backoff_s = _BACKOFF_START
                _log.info("MQTT connected to %s:%d", self._host, self._port)
                try:
                    client.subscribe(f"{self._prefix}/#")
                    for topic, _cb in self._subs.values():
                        client.subscribe(topic)
                except Exception:  # noqa: BLE001
                    _log.debug("mqtt resubscribe failed", exc_info=True)
            else:
                _log.warning("MQTT broker refused connection (rc=%s)", rc)

        def _on_disconnect(_c, _u, _f, rc, _p=None):
            self._connected = False
            _log.info("MQTT disconnected (rc=%s)", rc)
            disconnected.set()

        def _on_message(_c, _u, msg):
            try:
                self._handle_message(str(msg.topic),
                                     msg.payload if hasattr(msg, "payload")
                                     else msg)
            except Exception:  # noqa: BLE001 - one bad message never kills
                _log.debug("mqtt message handling failed", exc_info=True)

        client.on_connect = _on_connect
        client.on_disconnect = _on_disconnect
        client.on_message = _on_message
        try:
            client.connect(self._host, self._port, self._keepalive)
        except Exception as exc:
            _log.info("MQTT connect to %s:%d failed: %s",
                      self._host, self._port, exc)
            self._connected = False
            return
        try:
            client.loop_start()
        except Exception:  # noqa: BLE001
            pass
        # serve until disconnect or stop()
        while self._running and not disconnected.is_set():
            if self._wake.wait(0.5):
                break
        try:
            client.loop_stop()
        except Exception:  # noqa: BLE001
            pass
        self._client = None
        self._connected = False

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        import paho.mqtt.client as _paho
        client = _paho.Client(client_id=self._client_id,
                             callback_api_version=_paho.CallbackAPIVersion.VERSION2)
        if self._username:
            client.username_pw_set(self._username, self._password)
        return client

    def _handle_message(self, topic: str, raw: Any) -> None:
        prefix = self._prefix + "/"
        if not topic.startswith(prefix):
            self._dispatch_raw(topic, raw)
            return
        rest = topic[len(prefix):]
        parts = rest.split("/")
        device = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        payload = _decode_payload(raw)

        if suffix == "availability":
            self._availability[device] = str(payload).strip().lower()
            self._route_trigger(device, None, self._availability[device],
                                {"kind": "availability"})
            return
        if suffix in ("set", "get"):
            return  # our own outbound traffic echoed back — ignore

        # state message
        if isinstance(payload, dict):
            new_state = payload.get("state", payload)
            attributes = {k: v for k, v in payload.items() if k != "state"}
        else:
            new_state = payload
            attributes = {}
        old = self._state_cache.get(device, {})
        old_state = old.get("state")
        self._state_cache[device] = {
            "state": new_state,
            "attributes": attributes,
            "raw": payload,
            "last_changed": self._clock(),
        }
        self._route_trigger(device, old_state, new_state, attributes)
        self._dispatch_raw(topic, payload)

    def _dispatch_raw(self, topic: str, payload: Any) -> None:
        for sub_topic, callback in list(self._subs.values()):
            if self._topic_matches(sub_topic, topic):
                try:
                    callback(topic, payload)
                except Exception:  # noqa: BLE001 - one bad callback never kills
                    _log.debug("mqtt subscriber failed", exc_info=True)

    @staticmethod
    def _topic_matches(sub: str, topic: str) -> bool:
        """MQTT wildcards: ``+`` one level, ``#`` rest."""
        if sub == topic:
            return True
        sp, tp = sub.split("/"), topic.split("/")
        i = 0
        while i < len(sp):
            if sp[i] == "#":
                return True
            if i >= len(tp) or (sp[i] != "+" and sp[i] != tp[i]):
                return False
            i += 1
        return i == len(tp)

    def _route_trigger(self, device: str, old_state: Any, new_state: Any,
                       attributes: dict[str, Any]) -> None:
        engine = self._trigger_engine
        if engine is None:
            return
        entity_id = _entity_id(device, prefix=self._prefix)
        now = self._clock()
        if now < self._cooldown_until.get(entity_id, 0):
            return  # flapping sensor — cooldown suppresses
        self._cooldown_until[entity_id] = now + self.cooldown_s
        try:
            engine.on_entity_state(entity_id, old_state, new_state, attributes)
        except Exception:  # noqa: BLE001 - triggers never break the bridge
            _log.debug("mqtt trigger routing failed", exc_info=True)


def bridge(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, **kwargs: Any
           ) -> MQTTBridge:
    """Convenience constructor: ``bridge("192.168.1.10")``."""
    return MQTTBridge(host, port, **kwargs)
