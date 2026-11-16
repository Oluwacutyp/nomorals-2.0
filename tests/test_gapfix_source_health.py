"""Gap R1-16: ``nomorals.sources.health_plus`` is deleted — the media
registry is the ONE health truth.

The dormant organ (``sources/health_plus.py``: ``SourceHealth``) never
ran in production: zero importers. Every method it offered duplicates
a live, persisted, already-consumed mechanism:

* probe/record  ->  ``SourceRegistry.record_outcome`` (betdesk, pumpdesk
  already feed it) + the health loop ``SourceRegistry.verify_health``
  (re-verifies every registered source + candidate, circuit breaker,
  persists; called from 8+ live tool paths).
* fallback_chain -> ``SourceRegistry.ordered_for`` — which additionally
  keeps the nation-first product rule that ``fallback_chain`` drops.
* success_rate/is_healthy/health_report -> ``stats_for`` / ``health_for``
  / ``is_active`` (+ betdesk/pumpdesk status surfaces).

Wiring the class in would have added a THIRD health layer, double-counted
probes (``record()`` mirrors via ``record_outcome`` AND ``verify_health``
records outcomes), and regressed fallback ordering. Deletion is the fix
the audit offers, proven by the evidence above. These tests pin the end
state: the package is gone, the five descriptors live only as registry
seeds, and the registry's health machinery is the single truth.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

import nomorals

#: The five descriptors, as the registry seeds them (nation, language,
#: domain, kinds). Kept literal here: the test owns the expectation so
#: no second copy of the descriptor data survives anywhere.
EXPECTED_FIVE = (
    ("NG", "en", "punchng.com", ("news",)),
    ("NG", "en", "audiomack.com", ("music",)),
    ("US", "en", "techcrunch.com", ("news", "tech")),
    ("US", "en", "espn.com", ("news", "sports")),
    ("NG", "yo", "yorubaculture.example", ("culture",)),
)


def _kv(tmp_path):
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _registry(tmp_path):
    from nomorals.media.source_registry import _seed

    store, _db = _kv(tmp_path)
    return _seed(store=store)


def test_health_plus_module_is_gone():
    """The dormant package is deleted: no files, no import."""
    root = Path(nomorals.__file__).resolve().parent
    assert not (root / "sources" / "health_plus.py").exists()
    assert not (root / "sources").exists()
    for mod in [m for m in list(sys.modules)
                if m == "nomorals.sources"
                or m.startswith("nomorals.sources.")]:
        del sys.modules[mod]
    with pytest.raises(ImportError):
        importlib.import_module("nomorals.sources.health_plus")
    with pytest.raises(ImportError):
        importlib.import_module("nomorals.sources")


def test_five_descriptors_are_registry_seeds(tmp_path):
    """The five sources resolve through the ONE registry, with the
    kinds the (deleted) descriptors declared. No second copy of the
    descriptor data survives anywhere."""
    reg = _registry(tmp_path)
    seen_domains = set()
    for nation, language, domain, kinds in EXPECTED_FIVE:
        desc = reg.for_domain(domain)
        assert desc is not None, domain
        assert desc.nation == nation and desc.language == language
        assert set(kinds) <= set(desc.kinds), domain
        seen_domains.add(domain)
    assert len(seen_domains) == 5


def test_unknown_source_is_unknown_not_failed(tmp_path):
    """Zero evidence reads 'unknown' — never 'failed' or 'dead' before
    any attempt (the registry's zero-evidence rule)."""
    reg = _registry(tmp_path)
    desc = reg.for_domain("punchng.com")
    stats = reg.stats_for(desc)
    assert stats["ok"] == 0 and stats["fail"] == 0
    assert reg.health_for(desc) == "unknown"
    assert reg.is_active(desc)


def test_outcomes_shape_rate_and_ordering(tmp_path):
    """Outcomes feed the one ledger; the fallback order follows it:
    a healthy source serves, a stood-down source leaves the ACTIVE
    ordering — never silently."""
    reg = _registry(tmp_path)
    punch = reg.for_domain("punchng.com")
    espn = reg.for_domain("espn.com")
    reg.record_outcome(punch, True, latency_s=0.2)
    assert reg.stats_for(punch)["success_rate"] == 1.0
    assert reg.health_for(punch) == "verified-live"
    assert reg.is_active(punch)
    for _ in range(3):  # STAND_DOWN_AFTER
        reg.record_outcome(espn, False, error_class="fetch")
    assert reg.stats_for(espn)["success_rate"] == 0.0
    assert reg.health_for(espn) == "stood-down"
    assert not reg.is_active(espn)
    ordered = [d.domain for d in reg.ordered_for(kind="news")]
    assert "punchng.com" in ordered
    assert "espn.com" not in ordered  # stood-down leaves the ordering


def test_dead_source_leaves_a_record(tmp_path):
    """A dead source leaves a health record (counts, last error,
    stood-down/disabled verdict) — it never dies silently."""
    reg = _registry(tmp_path)
    espn = reg.for_domain("espn.com")
    for _ in range(10):  # DISABLE_AFTER
        reg.record_outcome(espn, False, error_class="fetch")
    stats = reg.stats_for(espn)
    assert stats["fail"] == 10
    assert stats["last_error_class"] == "fetch"
    assert reg.health_for(espn) == "disabled"
    assert not reg.is_active(espn)


def test_health_loop_never_raises_on_probe_failure(tmp_path):
    """The health loop degrades, never crashes: a fetcher that raises
    for every URL still yields a summary."""
    reg = _registry(tmp_path)

    def _boom(url):
        raise RuntimeError("network down")

    summary = reg.verify_health(fetcher=_boom)
    assert isinstance(summary, dict)
    assert summary["checked"] > 0
