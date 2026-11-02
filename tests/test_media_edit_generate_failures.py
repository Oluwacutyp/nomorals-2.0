"""Tail-105 pins: nomorals/media_edit/generate.py census sites.

All four sites are held as exclusions:

727 — ``NativeBackend.available()``: the import probe for the
imggen organ. An import crash returns False, which is the
probe's honest negative — "cannot confirm availability" —
and backend selection simply falls through to the next
candidate. The reason-carrying surface is the companion
``readiness()``, which binds the exception and reports it.

733 — ``available()`` again: a ``list_models()`` crash also
returns False. Same probe semantics; a registry too broken
to enumerate cannot confirm a loadable model exists.

763 — ``readiness()``: the CLIP tokenizer load crash degrades
to ``tok = None`` and the readiness message DISCLOSES it —
"tokenizer: hashing fallback (reduced prompt fidelity —
add vocab.json + merges.txt next to the checkpoint)". The
failure changes the user-facing claim instead of hiding
behind it; readiness stays True for the model itself, which
is loadable. Disclosed degradation, not a fabricated claim.

1424 — ``get_backend("auto")``: the ``huggingface_hub``
import probe. Absence is handled immediately below by
raising ``GenerativeEditError(_NO_BACKEND_MSG)`` — the
module's own honest-failure surface ("never a fake image,
never a traceback"). With every other backend unavailable
and no hub, raising is the refusal-shaped outcome.

NOT-recording is pinned with the snapshot method: generate.py
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.media.imggen as imggen
import nomorals.media.imggen.registry as registry
from nomorals.media_edit import generate as G


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


_LOADABLE_SD15 = [{"loadable": True, "name": "m1",
                   "model_class": "sd15", "path": "/ckpt",
                   "loadable_reason": ""}]


# ── 727: imggen import crash -> not available ───────────────────

def test_727_available_false_when_imggen_unimportable():
    before = _snap()
    with mock.patch.dict(sys.modules, {"nomorals.media.imggen": None}):
        assert G.NativeBackend.available() is False
    assert _snap() == before


# ── 733: registry crash -> not available; controls both ways ────

def test_733_available_false_when_registry_crashes():
    before = _snap()
    with mock.patch.object(imggen, "TORCH_AVAILABLE", True), \
            mock.patch.object(registry, "list_models", _raise):
        assert G.NativeBackend.available() is False
    # control: torch present + a loadable model -> available
    with mock.patch.object(imggen, "TORCH_AVAILABLE", True), \
            mock.patch.object(registry, "list_models",
                              lambda: _LOADABLE_SD15):
        assert G.NativeBackend.available() is True
    # control: no torch -> unavailable without touching the registry
    with mock.patch.object(imggen, "TORCH_AVAILABLE", False):
        assert G.NativeBackend.available() is False
    assert _snap() == before


# ── 763: tokenizer crash -> readiness DISCLOSES the fallback ────

def _tokenizer_stub(fn):
    stub = ModuleType("nomorals.media.imggen.tokenizers")
    stub.load_clip_tokenizer = fn
    return stub


def test_763_readiness_discloses_tokenizer_fallback():
    before = _snap()
    with mock.patch.object(imggen, "TORCH_AVAILABLE", True), \
            mock.patch.object(registry, "list_models",
                              lambda: _LOADABLE_SD15), \
            mock.patch.dict(sys.modules, {
                "nomorals.media.imggen.tokenizers":
                    _tokenizer_stub(_raise)}):
        ready, msg = G.NativeBackend.readiness()
    assert ready is True
    assert "hashing fallback (reduced prompt fidelity" in msg
    # control: a working tokenizer is reported as itself
    with mock.patch.object(imggen, "TORCH_AVAILABLE", True), \
            mock.patch.object(registry, "list_models",
                              lambda: _LOADABLE_SD15), \
            mock.patch.dict(sys.modules, {
                "nomorals.media.imggen.tokenizers":
                    _tokenizer_stub(lambda path: object())}):
        ready2, msg2 = G.NativeBackend.readiness()
    assert ready2 is True and "tokenizer: CLIP BPE" in msg2
    assert _snap() == before


# ── 1424: no hub + no local backend -> honest GenerativeEditError ─

import contextlib


@contextlib.contextmanager
def _unavailable(cls):
    """Force a backend class's static available() to False, restoring
    the exact original descriptor afterwards."""
    orig = cls.__dict__["available"]
    cls.available = staticmethod(lambda: False)
    try:
        yield
    finally:
        cls.available = orig


def test_1424_auto_without_hub_raises_generative_edit_error():
    before = _snap()
    with _unavailable(G.NativeBackend), _unavailable(G.DiffusersBackend), \
            mock.patch.dict(sys.modules, {"huggingface_hub": None}):
        with pytest.raises(G.GenerativeEditError):
            G.get_backend("auto")
    # control: hub importable -> the HF backend is selected
    hub = ModuleType("huggingface_hub")
    with _unavailable(G.NativeBackend), _unavailable(G.DiffusersBackend), \
            mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
        assert isinstance(G.get_backend("auto"),
                          G.HFInferenceBackend)
    assert _snap() == before
