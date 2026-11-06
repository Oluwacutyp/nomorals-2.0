"""Fold the legal music registry into the ONE media source registry.

AA-57 (assembly audit, owner's open-sources law): the legal music
registry (``music_sources/registry.py``) was a closed in-code list
whose health lived in a separate in-memory prober the playback
path never consulted.  This bridge folds every row into the ONE
media source registry (``media/source_registry.py``) as a source
class — kinds ``("music", "legal-music")`` — so music sources get
the registry's persisted health, learned scores and stand-down
machinery, and :func:`play.play_via_registry` consults that health
when it selects a source and records each real download outcome
back into it.

The music registry module itself is Layer-0 data (stdlib only, by
design) and stays untouched: this bridge is the layer that may
import the media registry.  Rows carry no URL/host — a source here
is a service/catalogue, not a fetch host — so each row registers
under a synthetic single-label domain ``music-source:<id>`` (the
media registry keys descriptors by host-shaped strings; the colon
form keeps every row distinct and greppable).  There is no honest
probe URL for such a descriptor, so its health is judged by REAL
play outcomes only — the same posture the registry takes for
keyed APIs whose root verdicts are unverifiable.  The music
registry's public accessors (``all_sources`` / ``by_id`` /
``by_class`` / ``by_pack`` / ``by_licence`` / ``health_due``) keep
working exactly as before; this module only ADDS the shared-
registry view.
"""

from __future__ import annotations

import logging
from typing import Any

from .registry import all_sources

_log = logging.getLogger(__name__)

__all__ = [
    "MUSIC_KINDS",
    "music_domain",
    "music_registry_for_context",
    "ensure_music_sources",
    "music_source_desc",
    "source_health",
    "is_source_usable",
    "record_play_outcome",
]

#: The registry kinds the legal-music class carries.
MUSIC_KINDS = ("music", "legal-music")

#: Synthetic-domain prefix for music rows (see module docstring).
_DOMAIN_PREFIX = "music-source:"


def music_domain(source_id: str) -> str:
    """The media-registry domain a music row registers under."""
    return f"{_DOMAIN_PREFIX}{source_id}"


def music_registry_for_context(context: Any = None,
                               injected: Any = None) -> Any:
    """The ONE source registry, resolved the #88–#94 way: an
    injected registry wins; otherwise the shared ``DEFAULT_REGISTRY``
    bound to the context's KV store — and only when the context
    carries a database, so db-less callers keep the pre-AA-57
    behaviour exactly.  Never raises."""
    if injected is not None:
        return injected
    try:
        db = getattr(context, "db", None) if context is not None else None
        if db is None:
            return None
        from ..media.source_registry import DEFAULT_REGISTRY
        reg = DEFAULT_REGISTRY
        try:
            from ..storage.kv import KVStore
            reg.bind_store(KVStore(db))
        except Exception:  # noqa: BLE001 - persistence is an
            # enhancement; its absence never sinks playback
            pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def ensure_music_sources(reg: Any) -> int:
    """Register every legal-music row in the ONE registry
    (idempotent — the registry's ``register`` is first-wins and
    refuses duplicate keys).  Returns how many rows were NEW.
    Never raises."""
    if reg is None:
        return 0
    try:
        from ..media.source_registry import Licence, SourceDescriptor
        added = 0
        for src in all_sources():
            pack = src.pack or {}
            nation = str(pack.get("nation") or "*")
            desc = SourceDescriptor(
                # the registry's word for a worldwide row is GLOBAL
                nation="GLOBAL" if nation == "*" else nation,
                language=str(pack.get("language") or "*"),
                domain=music_domain(src.id),
                name=src.name,
                kinds=MUSIC_KINDS,
                auth_needed=str(src.auth or "none") not in ("", "none"),
                # the row's licence is free text in the legal
                # registry (the licence authority); the enum stays
                # honestly UNKNOWN here and the text rides in notes
                licence=Licence.UNKNOWN,
                downloadable=src.source_class != "metadata",
                notes=(
                    f"legal music registry class {src.source_class}; "
                    f"licence: {src.licence}; "
                    f"volatility {src.volatility}; quality up to "
                    f"{src.quality_ceiling or '—'}; health judged by "
                    "real play outcomes (AA-57)"),
            )
            if reg.register(desc):
                added += 1
        return added
    except Exception:  # noqa: BLE001
        return 0


def music_source_desc(reg: Any, source_id: str) -> Any:
    """The registry descriptor for one music row, or None (row not
    folded in yet, or no registry).  Resolved by EXACT domain scan:
    ``for_domain`` canonicalizes through ``_host_of``, whose port
    stripping would mangle the synthetic ``music-source:<id>``
    domain.  Never raises."""
    try:
        if reg is None:
            return None
        want = music_domain(source_id).lower()
        for desc in reg.all():
            if desc.domain.lower() == want:
                return desc
        return None
    except Exception:  # noqa: BLE001
        return None


def source_health(reg: Any, source_id: str) -> dict[str, Any] | None:
    """The registry's health view of one music source — persisted
    health state, learned score and outcome counts — or None when
    the row has no descriptor.  Never raises."""
    try:
        desc = music_source_desc(reg, source_id)
        if desc is None:
            return None
        return {
            "health": reg.health_for(desc),
            "active": bool(reg.is_active(desc)),
            "score": reg.score(desc),
            "stats": reg.stats_for(desc),
        }
    except Exception:  # noqa: BLE001
        return None


def is_source_usable(reg: Any, source_id: str) -> bool:
    """True when play selection may use this source: no descriptor
    (the fold has not run / no registry) keeps the pre-AA-57
    behaviour; a descriptor the registry has stood down, disabled
    or marked dead is NOT usable.  Never raises."""
    try:
        desc = music_source_desc(reg, source_id)
        if desc is None:
            return True
        return bool(reg.is_active(desc))
    except Exception:  # noqa: BLE001
        return True


def record_play_outcome(reg: Any, source_id: str, ok: bool, *,
                        error_class: str = "") -> None:
    """Record one REAL play/download outcome against the source's
    registry descriptor (AA-57) — the evidence its persisted health
    is judged by.  Best-effort; never raises."""
    try:
        desc = music_source_desc(reg, source_id)
        if desc is None:
            return
        reg.record_outcome(desc, bool(ok),
                           error_class=error_class or "")
    except Exception as exc:  # noqa: BLE001
        _log.debug("music registry outcome record failed: %s", exc)
