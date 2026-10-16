"""Browser profiles: identity management, NOT deception.

A :class:`BrowserProfile` holds one browsing identity the owner actually
owns — their login cookies for a site, their preferred user agent, their
proxy choice, their storage state. It never forges a different device's
fingerprint, never spoofs hardware/canvas/WebGL signals, and never
pretends to be someone else. Header rotation cycles through a small fixed
list of ordinary, real-browser User-Agent strings plus matching
Accept-Language values — the same variety any real user gets from using
two browsers — and records every rotation so the identity stays
auditable.

Cookie *values* are never logged: :meth:`BrowserProfile.masked_summary`
shows names/domains only. Profiles persist as JSON under a directory
(:class:`ProfileStore`); loading a profile never executes anything in it.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "BrowserProfile",
    "ProfileError",
    "ProfileStore",
    "DEFAULT_USER_AGENTS",
]

_log = get_logger(__name__)


class ProfileError(ValueError):
    """A profile is malformed, or a store operation failed."""


#: Ordinary real-browser User-Agents. Fixed list, no fabricated devices:
#: rotation picks among browsers a real user could actually be running.
DEFAULT_USER_AGENTS: tuple[str, ...] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:133.0) Gecko/20100101 Firefox/133.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
    "Gecko/20100101 Firefox/133.0",
)

#: Accept-Language values paired with the UA list by index (same order).
DEFAULT_LANGUAGES: tuple[str, ...] = (
    "en-US,en;q=0.9",
    "en-US,en;q=0.9",
    "en-US,en;q=0.5",
    "en-GB,en;q=0.9",
)

#: Header names a profile may carry. Anything else is refused at
#: validation — a profile must not smuggle fingerprint-spoofing headers.
_ALLOWED_HEADERS = frozenset({
    "user-agent", "accept", "accept-language", "accept-encoding",
    "referer", "dnt", "upgrade-insecure-requests",
})


@dataclass
class BrowserProfile:
    """One owned browsing identity."""

    name: str = ""
    cookies: list[dict[str, Any]] = field(default_factory=list)
    headers: dict[str, str] = field(default_factory=dict)
    user_agent: str = ""
    proxy: str = ""  # "" = direct; otherwise an http(s)/socks URL
    storage_state_path: str = ""  # playwright storage_state file, if any
    notes: str = ""
    rotation_index: int = 0  # position in the UA rotation; persisted

    def __post_init__(self) -> None:
        self.name = (self.name or "").strip()
        self.user_agent = (self.user_agent or "").strip()
        self.proxy = (self.proxy or "").strip()
        if self.user_agent and self.user_agent not in DEFAULT_USER_AGENTS:
            # A custom UA is allowed (the owner's real browser may differ)
            # but it must look like a real UA, not a fabrication.
            if len(self.user_agent) < 20 or "Mozilla" not in self.user_agent:
                raise ProfileError(
                    "custom user_agent must be a full real-browser UA string")

    # -- validation ---------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name:
            errs.append("name is required")
        elif "/" in self.name or "\\" in self.name or self.name in {".", ".."}:
            errs.append(f"name {self.name!r} is not a safe profile name (no slashes)")
        for key in self.headers:
            if key.lower() not in _ALLOWED_HEADERS:
                errs.append(f"header {key!r} is not allowed in a profile "
                            f"(allowed: {sorted(_ALLOWED_HEADERS)})")
        for i, cookie in enumerate(self.cookies):
            if not isinstance(cookie, dict) or not cookie.get("name") or not cookie.get("domain"):
                errs.append(f"cookies[{i}] needs at least name + domain")
        if self.proxy:
            parsed = urllib.parse.urlparse(
                self.proxy if "://" in self.proxy else f"http://{self.proxy}")
            if parsed.scheme not in {"http", "https", "socks5", "socks4", "socks"}:
                errs.append(f"proxy scheme {parsed.scheme!r} unsupported (want http/https/socks5/socks4)")
            elif not parsed.hostname:
                errs.append(f"proxy {self.proxy!r} has no hostname")
        if self.rotation_index < 0:
            errs.append("rotation_index must be >= 0")
        return errs

    @property
    def is_valid(self) -> bool:
        return not self.validate()

    # -- identity use ---------------------------------------------------------
    def headers_for_request(self) -> dict[str, str]:
        """Headers for an outgoing request: stored headers plus the
        profile UA (the UA wins — it is the identity's browser)."""
        out = {str(k): str(v) for k, v in self.headers.items()}
        ua = self.user_agent or DEFAULT_USER_AGENTS[0]
        out["User-Agent"] = ua
        return out

    def rotate_headers(self) -> dict[str, str]:
        """Advance to the next UA in the fixed rotation and update
        Accept-Language to match. Safe by construction: the rotation only
        cycles real-browser strings, never fabricates a device. Returns
        the new headers."""
        nxt = (int(self.rotation_index) + 1) % len(DEFAULT_USER_AGENTS)
        self.rotation_index = nxt
        self.user_agent = DEFAULT_USER_AGENTS[nxt]
        self.headers = {k: v for k, v in self.headers.items()
                        if k.lower() != "accept-language"}
        self.headers["Accept-Language"] = DEFAULT_LANGUAGES[nxt]
        _log.info("profile %r: rotated UA to index %d", self.name, nxt)
        return self.headers_for_request()

    def masked_proxy(self) -> str:
        """Proxy URL with any password hidden (for logs/events)."""
        if not self.proxy:
            return "direct"
        try:
            parsed = urllib.parse.urlparse(self.proxy)
            if parsed.password:
                netloc = parsed.hostname or ""
                if parsed.username:
                    netloc = f"{parsed.username}:***@{netloc}"
                if parsed.port:
                    netloc += f":{parsed.port}"
                return urllib.parse.urlunparse(
                    (parsed.scheme, netloc, parsed.path, "", "", ""))
        except Exception:  # noqa: BLE001 - masking must never raise
            pass
        return self.proxy

    def masked_summary(self) -> dict[str, Any]:
        """Log-safe summary: cookie names/domains only — never values."""
        return {
            "name": self.name,
            "cookies": [{"name": c.get("name"), "domain": c.get("domain")}
                        for c in self.cookies],
            "user_agent": (self.user_agent or "")[:60],
            "proxy": self.masked_proxy(),
            "storage_state_path": self.storage_state_path,
            "rotation_index": self.rotation_index,
        }

    def cookie_header_for(self, url: str) -> str:
        """A ``Cookie:`` header for ``url`` from this profile's jar
        (domain + path matched, expired cookies skipped). Values stay in
        the header — callers must not log it."""
        import time

        try:
            parsed = urllib.parse.urlparse(url)
            host = (parsed.hostname or "").lower()
            path = parsed.path or "/"
        except Exception:  # noqa: BLE001 - a bad URL yields no cookies
            return ""
        parts: list[str] = []
        now = time.time()
        for cookie in self.cookies:
            domain = str(cookie.get("domain") or "").lower().lstrip(".")
            if not domain or not (host == domain or host.endswith("." + domain)):
                continue
            cpath = str(cookie.get("path") or "/")
            if not path.startswith(cpath):
                continue
            expires = cookie.get("expires")
            if expires:
                try:
                    if float(expires) < now:
                        continue
                except (TypeError, ValueError):
                    pass
            parts.append(f"{cookie.get('name')}={cookie.get('value')}")
        return "; ".join(parts)

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BrowserProfile":
        raw = dict(raw or {})
        cookies = raw.get("cookies") or []
        headers = raw.get("headers") or {}
        try:
            rotation = int(raw.get("rotation_index") or 0)
        except (TypeError, ValueError):
            rotation = 0
        return cls(
            name=str(raw.get("name") or ""),
            cookies=[dict(c) for c in cookies if isinstance(c, dict)],
            headers={str(k): str(v) for k, v in headers.items()},
            user_agent=str(raw.get("user_agent") or ""),
            proxy=str(raw.get("proxy") or ""),
            storage_state_path=str(raw.get("storage_state_path") or ""),
            notes=str(raw.get("notes") or ""),
            rotation_index=rotation,
        )


class ProfileStore:
    """One JSON file per profile under a directory (atomic writes)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    def _path_for(self, name: str) -> Path:
        safe = (name or "").strip()
        if not safe or "/" in safe or "\\" in safe or safe in {".", ".."}:
            raise ProfileError(f"unsafe profile name {name!r}")
        return self.directory / f"{safe}.json"

    def save(self, profile: BrowserProfile) -> Path:
        errs = profile.validate()
        if errs:
            raise ProfileError(
                f"profile {profile.name!r} is invalid: " + "; ".join(errs))
        path = self._path_for(profile.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(profile.to_dict(), indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(path)
        return path

    def load(self, name: str) -> BrowserProfile:
        path = self._path_for(name)
        if not path.is_file():
            raise ProfileError(f"no such profile: {name!r} (looked in {self.directory})")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProfileError(f"corrupt profile {name!r} at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProfileError(f"corrupt profile {name!r}: top level must be an object")
        profile = BrowserProfile.from_dict(raw)
        errs = profile.validate()
        if errs:
            raise ProfileError(f"stored profile {name!r} is invalid: " + "; ".join(errs))
        return profile

    def list_profiles(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob("*.json")
                      if p.is_file() and not p.name.endswith(".tmp"))

    def delete(self, name: str) -> bool:
        path = self._path_for(name)
        if not path.is_file():
            return False
        path.unlink()
        return True
