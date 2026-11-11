"""Her source registry — the media source list that LEARNS.

The picture pools (``sources.IMAGE_POOLS``) and the expression
provider chain (``expression.default_providers``) are code-time
SEED lists: rich, but frozen at whatever the last ship tabled.
This registry is the runtime half of the open-sources law: a
persisted JSON document in her data dir holding extra HTTP
sources in the existing ``_PoolSource`` / provider shapes.
She (or the owner, through the same file the settings hook
feeds) can register a source at runtime; every pool walk and
every expression chain built afterwards reads seeds +
registry as ONE list.

Failure discipline mirrors ``expression.ProviderHealth`` /
``agency/sourcehealth``: consecutive failures are JOURNALED
into the file (never silent), 3 in a row stands the source
down for a spell, 6 disables it — a registered source that
keeps failing is skipped by the walks, never retried forever,
and re-registering the same name revives it with a clean
record.  Success resets the streak.

Stdlib-only on purpose: this module sits at the shared layer
so the anime organ (pool walks), the expression organ (GIF
chains) and the runtimes can all read the same registry
without an upward import anywhere.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

#: Shapes a registered source may declare — the extractors
#: that already exist across the two walks: the pool's
#: ``ImagePoolClient._extract`` shapes, plus the two
#: list-shaped answers (a bare JSON list of rows / of URLs).
KNOWN_SHAPES = (
    "waifu.pics",     # {"url": …}
    "nekos.best",     # {"results": [{"url": …, "anime_name": …}]}
    "results-url",    # {"results": [{"url": …}, …]}
    "nekosapi",       # [{"url": …, "rating": "safe"}] (safe rows only)
    "cataas",         # /api/cats rows: gif rows → cataas.com/cat/<id>
    "randomfox",      # {"image": …}
    "tenor",          # Tenor search envelope (keyed)
    "giphy",          # Giphy search envelope (keyed)
    "url-list",       # ["https://…", …] or [{"url": …}, …]
)

#: The chains a registered source may join: the pool kinds
#: plus ``gif`` — the expression family's search chain.
KNOWN_KINDS = (
    "gif", "waifu", "neko", "kitsune", "husbando", "charpic",
    "couplepp",
)

STAND_DOWN_AFTER = 3      # consecutive failures → a spell out
DISABLE_AFTER = 6         # consecutive failures → off until revived
STAND_DOWN_S = 300.0
_JOURNAL_CAP = 200

_NAME_OK = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def default_registry_path(home: Any = None) -> Path:
    """The registry's home: her data dir — an explicit home
    hint (settings.home_path) wins, then NM_HOME, then the
    standing ~/.nomorals layout."""
    if home:
        return Path(str(home)) / "source_registry.json"
    env = os.environ.get("NM_HOME") or ""
    if env:
        return Path(env) / "source_registry.json"
    return Path.home() / ".nomorals" / "source_registry.json"


def open_registry(home: Any = None) -> SourceRegistry:
    """The registry at the default path for ``home``."""
    return SourceRegistry(default_registry_path(home))


def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Normalize + validate one registered-source spec.
    Raises ValueError naming the exact problem — a bad spec
    is refused at the door, never half-stored."""
    if not isinstance(spec, dict):
        raise ValueError("a source spec is an object")
    name = str(spec.get("name") or "").strip().lower()
    if not _NAME_OK.match(name):
        raise ValueError(
            f"source name {name!r} is not a slug "
            "(a-z 0-9 . _ -, 1-64 chars)")
    url = str(spec.get("url") or "").strip()
    if not (url.startswith("https://") or url.startswith("http://")):
        raise ValueError(f"source {name}: url must be http(s)")
    shape = str(spec.get("shape") or "").strip()
    if shape not in KNOWN_SHAPES:
        raise ValueError(
            f"source {name}: shape {shape!r} is not one of "
            f"{', '.join(KNOWN_SHAPES)}")
    kinds_raw = spec.get("kinds") or ()
    if isinstance(kinds_raw, str):
        kinds_raw = [kinds_raw]
    kinds = sorted({str(k).strip().lower()
                    for k in kinds_raw if str(k).strip()})
    unknown = [k for k in kinds if k not in KNOWN_KINDS]
    if unknown:
        raise ValueError(
            f"source {name}: kinds {unknown} are not chains "
            f"this registry feeds ({', '.join(KNOWN_KINDS)})")
    if not kinds:
        kinds = ["gif"]
    return {
        "name": name,
        "url": url,
        "shape": shape,
        "kinds": kinds,
        "source_class": str(spec.get("source_class") or "general"),
        "weight": max(1, int(spec.get("weight") or 1)),
        "animated": bool(spec.get("animated", True)),
        "key": str(spec.get("key") or ""),
        "key_param": str(spec.get("key_param") or ""),
        "query": str(spec.get("query") or ""),
    }


class SourceRegistry:
    """The persisted registry: registered specs + their
    health record + the journal, one JSON document saved
    atomically (tmp + replace), beside her other state."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.data: dict[str, Any] = {"version": 1, "sources": [],
                                     "journal": []}
        self.load()

    # -- persistence -----------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("sources"), list):
                raise ValueError("registry root is not an object")
            self.data = raw
        except (OSError, ValueError):
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            with contextlib.suppress(OSError):
                self.path.replace(quarantine)
            self.data = {"version": 1, "sources": [], "journal": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def _journal(self, event: str, name: str, detail: str = "") -> None:
        entries = self.data.setdefault("journal", [])
        entries.append({"at": time.time(), "event": event,
                        "source": str(name), "detail": str(detail)})
        del entries[:-_JOURNAL_CAP]

    def journal(self, *, limit: int = 50) -> list[dict[str, Any]]:
        entries = list(self.data.get("journal") or [])
        return entries[-max(1, int(limit)):]

    # -- registration ------------------------------------------
    def register(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Add (or replace) one source.  Re-registering a name
        revives it: the health record starts clean, and the
        revival is journaled like everything else here."""
        clean = validate_spec(spec)
        rows = [r for r in self.data.get("sources") or []
                if isinstance(r, dict)
                and r.get("name") != clean["name"]]
        clean.update({"fails": 0, "stood_down_until": 0.0,
                      "disabled": False, "last_error": "",
                      "last_fail_at": 0.0,
                      "registered_at": time.time()})
        rows.append(clean)
        self.data["sources"] = rows
        self._journal("registered", clean["name"],
                      f"kinds={','.join(clean['kinds'])} "
                      f"shape={clean['shape']}")
        self.save()
        return dict(clean)

    def unregister(self, name: str) -> bool:
        name = str(name or "").strip().lower()
        rows = self.data.get("sources") or []
        kept = [r for r in rows
                if not (isinstance(r, dict)
                        and r.get("name") == name)]
        if len(kept) == len(rows):
            return False
        self.data["sources"] = kept
        self._journal("unregistered", name)
        self.save()
        return True

    def registered(self) -> list[dict[str, Any]]:
        """Every registered spec, health fields included."""
        return [dict(r) for r in self.data.get("sources") or []
                if isinstance(r, dict)]

    # -- what the walks read ------------------------------------
    def specs(self, kind: str = "", *,
              now: float | None = None) -> list[dict[str, Any]]:
        """The specs a walk may use right now: enabled, not
        stood down, and (when ``kind`` is named) serving that
        chain.  Stood-down and disabled sources are simply
        absent — the walk never sees them."""
        now = time.time() if now is None else now
        out: list[dict[str, Any]] = []
        for row in self.data.get("sources") or []:
            if not isinstance(row, dict):
                continue
            if row.get("disabled"):
                continue
            if float(row.get("stood_down_until") or 0.0) > now:
                continue
            if kind and kind not in (row.get("kinds") or []):
                continue
            out.append(dict(row))
        return out

    # -- health: the journal keeps what the walks learn ---------
    def note_result(self, name: str, ok: bool, *,
                    error: str = "",
                    now: float | None = None) -> None:
        """Record one walk outcome for a registered source.
        A success clears the streak; failures accumulate in
        the file — 3 stand the source down for a spell, 6
        disable it — and every transition is journaled, so a
        source's death is a readable record, not a silence."""
        now = time.time() if now is None else now
        for row in self.data.get("sources") or []:
            if not isinstance(row, dict) \
                    or row.get("name") != name:
                continue
            if ok:
                if row.get("fails") or row.get("disabled"):
                    self._journal("recovered", name)
                row.update({"fails": 0, "stood_down_until": 0.0,
                            "disabled": False, "last_error": ""})
                self.save()
                return
            row["fails"] = int(row.get("fails") or 0) + 1
            row["last_error"] = str(error or "")[:200]
            row["last_fail_at"] = now
            self._journal("failure", name,
                          f"streak {row['fails']}: "
                          f"{row['last_error']}")
            if int(row["fails"]) >= DISABLE_AFTER \
                    and not row.get("disabled"):
                row["disabled"] = True
                self._journal(
                    "disabled", name,
                    f"{row['fails']} consecutive failures — "
                    "walks skip it until it is re-registered")
            elif int(row["fails"]) >= STAND_DOWN_AFTER:
                row["stood_down_until"] = now + STAND_DOWN_S
                self._journal(
                    "stood_down", name,
                    f"{row['fails']} consecutive failures — "
                    f"out for {int(STAND_DOWN_S)}s")
            self.save()
            return
