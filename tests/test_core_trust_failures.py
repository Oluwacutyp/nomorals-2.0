"""Failure pins for nomorals/core/trust.py (tail-112).

One pin per census site. All sites adjudicated EXCLUSION in-batch;
152 and 209 are additionally flagged TARGETED-FIX candidates
(data-loss shelf — see below):

- 89 _hostname: urlparse crash on garbage -> "" -> domain_tier
  returns _UNRESOLVED (0.30, "unresolved host"). Fail-closed
  direction; the noqa prints it: "garbage in, low score out".
- 152 _load: store.get crash -> {} — TARGETED-FIX candidate.
  feedback() is load-then-rewrite: it loads the history dict,
  mutates one host, and _save() writes the WHOLE dict back. A read
  crash therefore makes the next feedback() overwrite the entire
  persisted fetch-history with a single-host dict (byte-proven in
  the pin below). Excluded from batch converts per protocol
  (data-loss shapes route to the targeted shelf: the honest repair
  must distinguish "unreadable" from "empty" before any rewrite,
  which changes _load's contract for _history and every caller).
- 209 _host_set: store.get crash -> the in-memory set only —
  TARGETED-FIX candidate, sibling of 152. allow()/block()/
  unallow()/unblock() are load-then-rewrite over the same guard,
  so a read crash makes the next pin-write overwrite the persisted
  allow/block list (byte-proven below: a previously persisted
  blocked host is lost), and while the store is unreadable a
  persisted block pin silently drops out of scoring. Held as an
  exclusion rather than a posture question because the score's
  consumers are annotation/ranking only (search engine: "trust is
  a display bonus"; source_registry orders by score) — nothing
  refuses a fetch on is_blocked.

Non-recording: the module HAS a logger and the sibling save
handlers (_save, _save_host_set) debug-log their failures; the
152/209 load guards bind nothing and log nothing (pinned below).
"""

import json
import logging

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
def test_load_crash_returns_empty_and_records_nothing():
    trust = tru.SourceTrust(None, store=_CrashGetStore())
    with _Recordings(tru._log) as records:
        assert trust._load() == {}
        assert trust._history("https://a.com/x") == (0, 0)
    assert records == []


def test_feedback_after_load_crash_overwrites_history():
    # The data-loss shape, pinned to its true (bad) shape: the save
    # writes back only the hosts the crashed load could see.
    store = _CrashGetStore()
    trust = tru.SourceTrust(None, store=store)
    trust.feedback("https://a.com/x", ok=True)
    saved = json.loads(store.raw[tru.SourceTrust._KV])
    assert list(saved.keys()) == ["a.com"]
    assert saved["a.com"]["ok"] == 1


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
def test_host_set_crash_returns_memory_only_and_records_nothing():
    trust = tru.SourceTrust(None, store=_CrashGetStore())
    with _Recordings(tru._log) as records:
        assert trust._host_set(tru.SourceTrust._BLOCK_KV) == set()
        assert trust.is_blocked("https://old.com/") is False
    assert records == []


def test_block_after_host_set_crash_overwrites_persisted_list():
    # Sibling data-loss shape: old.com was persisted by a healthy
    # session; with reads crashing, block() saves only what the
    # crashed load returned (memory + the new host).
    store = _CrashGetStore()
    store.raw[tru.SourceTrust._BLOCK_KV] = json.dumps(["old.com"])
    trust = tru.SourceTrust(None, store=store)
    trust.block("https://new.com/")
    saved = json.loads(store.raw[tru.SourceTrust._BLOCK_KV])
    assert saved == ["new.com"]
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
