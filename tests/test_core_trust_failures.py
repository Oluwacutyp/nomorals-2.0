"""Failure pins for nomorals/core/trust.py (tail-112).

One pin per census site. All sites adjudicated EXCLUSION in-batch;
152 and 209 are additionally flagged TARGETED-FIX candidates
(data-loss shelf — see below):

- 89 _hostname: urlparse crash on garbage -> "" -> domain_tier
  returns _UNRESOLVED (0.30, "unresolved host"). Fail-closed
  direction; the noqa prints it: "garbage in, low score out".
- 152 _load: store.get crash -> {} — was a TARGETED-FIX
  candidate; CONVERTED by the data-loss targeted fix (Pass 1).
  feedback() is load-then-rewrite, so a read crash used to make
  the next feedback() overwrite the entire persisted
  fetch-history with a single-host dict (byte-proven in the
  original pin). Now the crashed/wrong-shaped load is logged at
  ERROR, recorded in the core failures ledger (a wrong-shaped
  value is also copied to ``<key>.corrupt``), and _save refuses
  to rewrite while the key is flagged unreadable — the persisted
  bytes survive untouched.
- 209 _host_set: store.get crash -> the in-memory set only —
  sibling of 152, CONVERTED in the same pass. allow()/block()/
  unallow()/unblock() still apply the pin to the in-memory set
  for the session, but the persisted allow/block list is no
  longer overwritten from the crashed load's partial image.

Recording: 152/209 now record (module logger at ERROR + core
failures ledger) — the pins below assert exactly that.
"""

import json
import logging

import nomorals.core.failures as core_fail
from nomorals.core import trust as tru


class _FakeStore:
    """KV double mirroring the get/set_raw pairing SourceTrust uses:
    set_raw stores a JSON string; get returns the parsed value."""

    def __init__(self):
        self.raw = {}

    def get(self, key):
        value = self.raw.get(key)
        return json.loads(value) if value is not None else None

    def set_raw(self, key, raw, kind="json"):
        self.raw[key] = raw


class _CrashGetStore(_FakeStore):
    def get(self, key):
        raise RuntimeError("kv read boom")


class _Recordings:
    """Capturing handler attached to the module logger."""

    def __init__(self, logger):
        self.records = []
        self._logger = logger
        self._handler = logging.Handler(level=logging.DEBUG)
        self._handler.emit = self.records.append

    def __enter__(self):
        self._logger.addHandler(self._handler)
        return self.records

    def __exit__(self, *exc):
        self._logger.removeHandler(self._handler)
        return False


# --------------------------------------------------------------------- 89
def test_hostname_garbage_yields_unresolved_tier():
    assert tru._hostname("http://[::1") == ""
    assert tru.domain_tier("http://[::1") == (0.30, "unresolved host")


def test_hostname_and_tier_controls():
    assert tru._hostname("https://EXAMPLE.com/x") == "example.com"
    ladder = {frag: t for frag, t, _n in tru._DOMAIN_LADDER}
    assert tru.domain_tier("https://github.com/x")[0] == ladder["github.com"]


# -------------------------------------------------------------------- 152
def test_load_crash_returns_empty_and_records():
    # CONVERTED (data-loss Pass 1): the crashed load still reads as
    # empty, but it is no longer silent — ERROR log + core ledger.
    trust = tru.SourceTrust(None, store=_CrashGetStore())
    with _Recordings(tru._log) as records:
        assert trust._load() == {}
        assert trust._history("https://a.com/x") == (0, 0)
    assert records != []
    assert "core.trust._load" in [
        f.site for f in core_fail.recent_failures(limit=200)]


def test_feedback_after_load_crash_preserves_history():
    # CONVERTED (data-loss Pass 1): the save that used to write back
    # only the hosts the crashed load could see is now refused —
    # the persisted history survives byte-for-byte.
    healthy = _FakeStore()
    seed = tru.SourceTrust(None, store=healthy)
    seed.feedback("https://old.com/x", ok=True)
    persisted = healthy.raw[tru.SourceTrust._KV]
    store = _CrashGetStore()
    store.raw = healthy.raw  # same backing bytes; reads crash
    trust = tru.SourceTrust(None, store=store)
    trust.feedback("https://a.com/x", ok=True)
    assert store.raw[tru.SourceTrust._KV] == persisted


def test_history_roundtrip_control():
    store = _FakeStore()
    first = tru.SourceTrust(None, store=store)
    first.feedback("https://a.com/x", ok=True)
    first.feedback("https://a.com/x", ok=False)
    first.feedback("https://b.com/y", ok=True)
    second = tru.SourceTrust(None, store=store)
    assert second._history("https://a.com/x") == (1, 1)
    assert set(second._load().keys()) == {"a.com", "b.com"}


# -------------------------------------------------------------------- 209
def test_host_set_crash_returns_memory_only_and_records():
    # CONVERTED (data-loss Pass 1): memory-only reads, but recorded.
    trust = tru.SourceTrust(None, store=_CrashGetStore())
    with _Recordings(tru._log) as records:
        assert trust._host_set(tru.SourceTrust._BLOCK_KV) == set()
        assert trust.is_blocked("https://old.com/") is False
    assert records != []
    assert "core.trust._host_set" in [
        f.site for f in core_fail.recent_failures(limit=200)]


def test_block_after_host_set_crash_preserves_persisted_list():
    # CONVERTED (data-loss Pass 1): old.com was persisted by a
    # healthy session; with reads crashing, block() must not save
    # the crashed load's partial image over it.
    store = _CrashGetStore()
    store.raw[tru.SourceTrust._BLOCK_KV] = json.dumps(["old.com"])
    trust = tru.SourceTrust(None, store=store)
    trust.block("https://new.com/")
    saved = json.loads(store.raw[tru.SourceTrust._BLOCK_KV])
    assert saved == ["old.com"]
    # The new pin still holds for this instance via the memory set.
    assert trust.is_blocked("https://new.com/") is True


def test_block_persistence_control():
    store = _FakeStore()
    first = tru.SourceTrust(None, store=store)
    first.block("https://bad.com/")
    second = tru.SourceTrust(None, store=store)
    assert second.is_blocked("https://bad.com/") is True
    assert second.score("https://bad.com/") == 0.0
    assert second.blocked() == ["bad.com"]
