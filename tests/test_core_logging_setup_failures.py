"""Tail-87 pins: nomorals/core/logging_setup.py census sites.

The module's standing contract is "logging must never raise" —
every site is a defensive guard inside the logging pipeline
itself, held as exclusions:

116 — ``_ContextInjectionFilter`` stamp crash -> pass, record
flows unstamped. Correlation context is diagnostic garnish; the
record itself is intact.
143 — ``_fingerprint`` crash -> "[REDACTED]": the failure value
is MORE conservative than success (a sha256 fingerprint), i.e.
fail-closed for secrets.
256 — ``RedactionFilter`` scrub crash -> pass, record flows.
WEAKEST exclusion of the batch, tension stated plainly: if the
scrubber itself crashes mid-record the record passes UNREDACTED,
so a secret-shaped value can reach the log — the fail-open
direction for secret hygiene. Held on the module's explicit
contract ("Never raises: a broken redactor must not break
logging") and the stdlib filter contract: the alternatives
(drop the record, or raise into every log call site in the
estate) are pipeline-policy changes, not within-contract fixes.
Trigger is pathological (the scrubber crashing on a record it
was handed). A fail-closed drop-on-scrub-crash posture note is
shelved for a deliberate decision.
415 — JsonFormatter per-attribute ``json.dumps`` crash -> the
attribute is carried as ``str(value)``: visibly stringified in
the payload, never dropped or falsified.
545 — ``shutdown_logging`` listener.stop() crash -> pass:
teardown continues, the listener list is cleared and the
configured flag reset regardless.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace

from nomorals.core import logging_setup as ls


def _raise(exc):
    raise exc


def _record(msg="hello", args=None):
    return logging.LogRecord("nomorals.test", logging.INFO, __file__, 1,
                             msg, args, None)


# ── 116: context injection never raises ─────────────────────────

class _BadCtx(dict):
    def items(self):
        raise RuntimeError("boom")


def test_116_context_stamp_crash_passes_record_unstamped():
    token = ls._log_context.set(_BadCtx())
    try:
        rec = _record()
        assert ls._ContextInjectionFilter().filter(rec) is True
        assert not hasattr(rec, "trace_id")
    finally:
        ls._log_context.reset(token)
    # control: a healthy context stamps the record
    token = ls._log_context.set({"trace_id": "t-1"})
    try:
        rec = _record()
        assert ls._ContextInjectionFilter().filter(rec) is True
        assert rec.trace_id == "t-1"
    finally:
        ls._log_context.reset(token)


# ── 143: fingerprint failure is full redaction ──────────────────

class _BadStr:
    def __str__(self):
        raise RuntimeError("boom")


def test_143_fingerprint_crash_falls_back_to_redacted():
    assert ls._fingerprint(_BadStr()) == "[REDACTED]"
    # control: a normal value fingerprints
    assert ls._fingerprint("abc").startswith("sha256:")


# ── 256: redaction filter never raises ──────────────────────────

def test_256_redactor_crash_passes_record_unredacted(monkeypatch):
    rec = _record("token=hunter22x")
    monkeypatch.setattr(ls, "redact",
                        lambda text: _raise(RuntimeError("boom")))
    assert ls.RedactionFilter().filter(rec) is True
    # the failure mode, pinned honestly: the record is untouched
    assert rec.msg == "token=hunter22x"
    monkeypatch.undo()
    # control: a working redactor scrubs the secret
    rec2 = _record("token=hunter22x")
    assert ls.RedactionFilter().filter(rec2) is True
    assert "hunter22x" not in rec2.msg


# ── 415: unserializable attribute is stringified ────────────────

def test_415_unserializable_attr_stringified_not_dropped():
    fmt = ls.JsonFormatter()
    rec = _record("hello")
    cyc = []
    cyc.append(cyc)
    rec.cycle = cyc
    payload = json.loads(fmt.format(rec))
    assert isinstance(payload.get("cycle"), str)
    assert "[[" in payload["cycle"]
    # control: a serializable attribute is carried verbatim
    rec2 = _record("hello")
    rec2.answer = 42
    payload2 = json.loads(fmt.format(rec2))
    assert payload2.get("answer") == 42


# ── 545: shutdown swallows a stuck listener ─────────────────────

def test_545_listener_stop_crash_does_not_block_shutdown():
    saved_listeners = list(ls._LISTENERS)
    saved_configured = ls._CONFIGURED
    good = SimpleNamespace(stopped=False)

    def _stop():
        good.stopped = True

    good.stop = _stop
    bad = SimpleNamespace(stop=lambda: _raise(RuntimeError("boom")))
    ls._LISTENERS[:] = [bad, good]
    try:
        ls.shutdown_logging()
        assert ls._LISTENERS == []
        assert good.stopped is True
        assert ls._CONFIGURED is False
    finally:
        ls._LISTENERS[:] = saved_listeners
        ls._CONFIGURED = saved_configured
