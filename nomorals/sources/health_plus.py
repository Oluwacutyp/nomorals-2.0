"""New sources + source-health, together.

Skip-list correction: source-health is NOT a gate on adding sources —
health ships as part of adding them. This module adds 5 new source
descriptors and the health layer in one file.

AA-62 (assembly audit): the 5 descriptors are media-registry SEEDS
(``media/source_registry.py`` — punchng.com and audiomack.com were
already seeded there under identical keys; techcrunch.com,
espn.com and yorubaculture.example joined the seed list so all
five resolve through the ONE registry), and :class:`SourceHealth`
DELEGATES to that registry's persisted health whenever one is
injected (``SourceHealth(registry=…)``): recorded probes land in
the registry as outcomes, and ``success_rate`` / ``is_healthy`` /
``fallback_chain`` prefer the registry's truth for sources it
knows.  Unbound, the class keeps its in-memory behaviour exactly.
This module stays L0: the registry is duck-typed, never imported.

The 5 new sources (descriptors are data; fetching stays with the
existing polite-fetch / registry machinery):

* ``punchng.com`` — Nigerian news (NG, en, news). One pack among N.
* ``audiomack.com`` — African music (pan-African music discovery;
  seeded with NG/GH/KE coverage, not a Nigeria-only source).
* ``techcrunch.com`` — global tech news (US, en, tech).
* ``espn.com`` — global sports (US, en, sports).
* ``yorubaculture`` — Yoruba culture (language ``yo``, culture pack;
  one language pack among N — the registry already keys by
  ``(nation, language, domain)`` and accepts any language).

Universal rule applied: these are 5 descriptors added to a
per-nation registry. Nothing here hardcodes Nigeria/Africa as the
axis — the registry, the health checks, and the fallback chain are
nation-agnostic; a Brazilian, Japanese, German, or Indian source
registers and is health-checked by the exact same code path.

Source health (:class:`SourceHealth`):

* ``probe(source, probe_fn)`` — run an injected probe (callable
  returning ``(ok, latency_s)``; no network here, the caller owns
  fetching). Outcomes are recorded per source.
* ``success_rate(source)`` — fraction of successful probes (None
  when never probed: unknown, not failed).
* ``is_stale(source, max_age_s)`` — no successful probe within the window.
* ``fallback_chain(candidates)`` — candidates ordered best-first by
  (healthy first, success rate, latency). Callers try the chain in
  order instead of hard-failing on one dead source.
* ``health_report()`` — one row per known source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "NEW_SOURCE_DESCRIPTORS", "new_source_descriptors", "SourceHealth",
    "SourceProbeResult",
]

#: The 5 new source descriptors (plain data — see module docstring).
#: Keys match :class:`nomorals.media.source_registry.SourceDescriptor`
#: fields so :func:`new_source_descriptors` can convert them.
NEW_SOURCE_DESCRIPTORS: list[dict[str, Any]] = [
    {"nation": "NG", "language": "en", "domain": "punchng.com",
     "name": "Punch", "kinds": ("news",),
     "notes": "Nigerian news — one nation's news pack among N."},
    {"nation": "NG", "language": "en", "domain": "audiomack.com",
     "name": "Audiomack", "kinds": ("music",),
     "notes": "African music discovery (NG/GH/KE coverage) — pan-African, not Nigeria-only."},
    {"nation": "US", "language": "en", "domain": "techcrunch.com",
     "name": "TechCrunch", "kinds": ("news", "tech"),
     "notes": "Global tech news."},
    {"nation": "US", "language": "en", "domain": "espn.com",
     "name": "ESPN", "kinds": ("news", "sports"),
     "notes": "Global sports coverage."},
    {"nation": "NG", "language": "yo", "domain": "yorubaculture.example",
     "name": "Yoruba Culture", "kinds": ("culture",),
     "notes": "Yoruba-language culture pack — one language pack among N."},
]


def new_source_descriptors() -> list[Any]:
    """Return the 5 descriptors as plain dicts.

    Layering: this L0 module must not import the L4 media registry.
    Callers that own ``SourceDescriptor`` convert dicts with one line.
    """
    return [dict(d) for d in NEW_SOURCE_DESCRIPTORS]


@dataclass
class SourceProbeResult:
    source: str
    ok: bool
    latency_s: float
    at: float = 0.0
    error: str = ""


@dataclass
class SourceHealth:
    """Per-source health: probe outcomes, success rates, fallback chains.

    AA-62: when ``registry`` (the ONE media source registry,
    duck-typed) is injected, this class CONSOLIDATES onto it —
    records mirror into the registry as outcomes and reads prefer
    the registry's persisted health for sources it knows.  The
    in-memory history is kept either way (it is this class's own
    contract and its tests' surface)."""
    max_history: int = 200
    history: dict[str, list[SourceProbeResult]] = field(default_factory=dict)
    registry: Any = field(default=None, repr=False, compare=False)

    def _desc(self, source: str) -> Any:
        """The registry descriptor for ``source`` (a domain), or
        None — unbound, unknown, or the registry hiccuped."""
        if self.registry is None:
            return None
        try:
            return self.registry.for_domain(source)
        except Exception:  # noqa: BLE001 - delegation never breaks
            return None

    def _registry_rate(self, source: str) -> float | None:
        """The registry's success rate for ``source`` when it has
        recorded attempts, else None."""
        desc = self._desc(source)
        if desc is None:
            return None
        try:
            stats = self.registry.stats_for(desc)
            total = int(stats.get("ok", 0)) + int(stats.get("fail", 0)) \
                + int(stats.get("blocked", 0)) \
                + int(stats.get("rate_limited", 0)) \
                + int(stats.get("no_result", 0))
            if total <= 0:
                return None
            return float(stats.get("success_rate", 0.0))
        except Exception:  # noqa: BLE001
            return None

    def record(self, result: SourceProbeResult) -> None:
        hist = self.history.setdefault(result.source, [])
        hist.append(result)
        self.history[result.source] = hist[-self.max_history:]
        # AA-62: the same outcome lands in the ONE registry, judged
        # by its persisted health machinery from then on.
        desc = self._desc(result.source)
        if desc is not None:
            try:
                self.registry.record_outcome(
                    desc, bool(result.ok),
                    latency_s=float(result.latency_s or 0.0),
                    error_class="" if result.ok else "probe_failed")
            except Exception:  # noqa: BLE001 - delegation never
                # breaks the local record; the mirror is retried on
                # the next probe
                self._registry_mirror_failed = True

    def probe(self, source: str,
              probe_fn: Callable[[str], tuple[bool, float]]) -> SourceProbeResult:
        """Run an injected probe for ``source`` and record the outcome.

        ``probe_fn`` returns ``(ok, latency_s)``; it owns all I/O.
        Probe failures are recorded, never raised.
        """
        started = time.time()
        try:
            ok, latency = probe_fn(source)
            result = SourceProbeResult(source=source, ok=bool(ok),
                                       latency_s=float(latency), at=time.time())
        except Exception as exc:  # probe errors are data, not exceptions
            result = SourceProbeResult(source=source, ok=False,
                                       latency_s=time.time() - started,
                                       at=time.time(),
                                       error=f"{type(exc).__name__}: {exc}")
        self.record(result)
        return result

    def success_rate(self, source: str) -> float | None:
        # AA-62: the registry's persisted rate wins when it has
        # evidence for this source — one health truth, not two.
        rate = self._registry_rate(source)
        if rate is not None:
            return rate
        hist = self.history.get(source) or []
        if not hist:
            return None
        return sum(1 for r in hist if r.ok) / len(hist)

    def avg_latency(self, source: str) -> float | None:
        hist = [r for r in (self.history.get(source) or []) if r.ok]
        if not hist:
            return None
        return sum(r.latency_s for r in hist) / len(hist)

    def last_success(self, source: str) -> float | None:
        ok_times = [r.at for r in (self.history.get(source) or []) if r.ok]
        return max(ok_times) if ok_times else None

    def is_stale(self, source: str, max_age_s: float = 86400.0,
                 now: float | None = None) -> bool:
        """True when no successful probe within ``max_age_s``.

        Never probed counts as stale — unknown is not healthy.
        """
        now = now if now is not None else time.time()
        last = self.last_success(source)
        return last is None or (now - last) > max_age_s

    def is_healthy(self, source: str, *, min_rate: float = 0.5) -> bool:
        # AA-62: a registry-known source is healthy when the
        # registry still ACTIVELY serves it and its persisted rate
        # clears the bar — the registry's stand-down/dead states
        # are verdicts this class no longer keeps a second copy of.
        desc = self._desc(source)
        if desc is not None and self._registry_rate(source) is not None:
            try:
                return bool(self.registry.is_active(desc)) \
                    and (self.success_rate(source) or 0.0) >= min_rate
            except Exception:  # noqa: BLE001 - fall through to local
                pass
        rate = self.success_rate(source)
        return rate is not None and rate >= min_rate and not self.is_stale(source)

    def fallback_chain(self, candidates: list[str]) -> list[str]:
        """Order candidates best-first: healthy, then success rate, then latency.

        Unknown sources (never probed) sort after known-healthy ones
        but before known-dead ones — worth trying once before giving up.
        With a registry bound (AA-62), a candidate the registry
        knows is ordered by the registry's own standing (active
        first, learned score as the rate) — one ordering truth.
        """
        def key(source: str) -> tuple[int, float, float]:
            desc = self._desc(source)
            if desc is not None:
                try:
                    active = bool(self.registry.is_active(desc))
                    score = float(self.registry.score(desc))
                    return (0 if active else 2, -score, 0.0)
                except Exception:  # noqa: BLE001 - fall through
                    pass
            rate = self.success_rate(source)
            latency = self.avg_latency(source)
            if rate is None:
                return (1, 0.0, 0.0)
            return (0 if rate >= 0.5 else 2, -rate, latency or 0.0)
        return sorted(candidates, key=key)

    def health_report(self) -> list[dict[str, Any]]:
        rows = []
        for source in sorted(self.history):
            hist = self.history[source]
            row = {"source": source, "probes": len(hist),
                   "success_rate": self.success_rate(source),
                   "avg_latency_s": self.avg_latency(source),
                   "stale": self.is_stale(source),
                   "healthy": self.is_healthy(source)}
            desc = self._desc(source)
            if desc is not None:
                try:
                    row["registry_health"] = self.registry.health_for(desc)
                except Exception:  # noqa: BLE001 - reported, not hidden
                    row["registry_health"] = "unknown"
            rows.append(row)
        return rows
