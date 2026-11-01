"""Tail-87 pins: nomorals/cmdline/commands/browse.py census sites.

All five held as exclusions — the ``nm browse`` CLI's documented
best-effort startup / session semantics:

30 — ``_service`` restore crash -> pass: a corrupt sessions.json
raises BrowserError out of ``BrowserService.restore()`` by design;
the CLI still starts with an unrestored service rather than dying
on saved state. Tension, stated plainly: the outcome is
byte-identical to a first run, so lost restores are invisible at
this surface; restore()'s own per-tab containment is the designed
granularity and the remedy is observability only (shelved sketch).
41 — artifact-store attach crash -> pass: "artifacts are a bonus,
not required" is the documented contract, and every consumer of
``BrowserService._artifact_store`` guards ``is not None``, so an
unattached store simply means artifacts are not persisted.
50 — ``_session`` get->open fallback (an unreferenced helper; the
live twin is site 76): ``get_session`` raises only for an unknown
session, so the fallback fires on exactly its designed case.
76 — ``_LocalBackend._handle``: same get->open semantics, live on
every local-backend verb.
101 — ``download`` with an unreadable active_tab downloads bare
(url, no cookies): download is deliberately usable with no tab at
all, unlike the ``_active_tab`` verbs which fail loud. The
download result dict reports the outcome honestly.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from nomorals.cmdline.commands import browse as browse_mod


def _raise(exc):
    raise exc


# ── 30 / 41: _service best-effort construction ──────────────────

class _FakeSvc:
    def __init__(self, restore_exc=None):
        self._restore_exc = restore_exc
        self.attached = "unset"

    def restore(self):
        if self._restore_exc is not None:
            raise self._restore_exc
        return 2

    def attach_store(self, store):
        self.attached = store


def _patched_service(monkeypatch, svc):
    monkeypatch.setattr("nomorals.browser.BrowserService",
                        lambda data_dir=None: svc)
    monkeypatch.setattr("nomorals.browser.daemon.default_data_dir",
                        lambda: "/tmp/nm-tail87-pin")


def test_30_restore_crash_still_returns_service(monkeypatch):
    svc = _FakeSvc(restore_exc=RuntimeError("corrupt sessions.json"))
    _patched_service(monkeypatch, svc)
    out = browse_mod._service(None, SimpleNamespace(db=None))
    assert out is svc


def test_41_attach_crash_leaves_service_unattached(monkeypatch):
    svc = _FakeSvc()
    _patched_service(monkeypatch, svc)
    monkeypatch.setattr("nomorals.storage.artifacts.ArtifactStore",
                        lambda *a: _raise(RuntimeError("boom")))
    ctx = SimpleNamespace(db=SimpleNamespace(path="/tmp/x/db.sqlite"))
    out = browse_mod._service(None, ctx)
    assert out is svc and svc.attached == "unset"
    # control: a working store attaches
    svc2 = _FakeSvc()
    _patched_service(monkeypatch, svc2)
    monkeypatch.setattr("nomorals.storage.artifacts.ArtifactStore",
                        lambda *a: "STORE")
    monkeypatch.setattr("nomorals.storage.blob.BlobStore",
                        lambda *a: "BLOB")
    browse_mod._service(None, ctx)
    assert svc2.attached == "STORE"


# ── 50 / 76: get_session -> open_session fallback ───────────────

class _SessSvc:
    def __init__(self, get_exc=None):
        self._get_exc = get_exc
        self.opened = []

    def get_session(self, name):
        if self._get_exc is not None:
            raise self._get_exc
        return "HANDLE"

    def open_session(self, name):
        self.opened.append(name)
        return "FRESH"


def test_50_session_helper_opens_fresh_on_unknown():
    svc = _SessSvc(get_exc=RuntimeError("unknown session"))
    out = browse_mod._session(svc, SimpleNamespace(session="cli"))
    assert out == "FRESH" and svc.opened == ["cli"]
    # control: an existing session is returned, nothing opened
    svc2 = _SessSvc()
    out = browse_mod._session(svc2, SimpleNamespace(session="cli"))
    assert out == "HANDLE" and svc2.opened == []


def test_76_handle_opens_fresh_on_unknown():
    backend = object.__new__(browse_mod._LocalBackend)
    svc = _SessSvc(get_exc=RuntimeError("unknown session"))
    backend._svc = svc
    backend.session_name = "cli"
    assert backend._handle() == "FRESH" and svc.opened == ["cli"]
    # control: an existing session is returned, nothing opened
    backend2 = object.__new__(browse_mod._LocalBackend)
    svc2 = _SessSvc()
    backend2._svc = svc2
    backend2.session_name = "cli"
    assert backend2._handle() == "HANDLE" and svc2.opened == []


# ── 101: tab-less download ──────────────────────────────────────

class _DLResult:
    def to_dict(self):
        return {"ok": True}


class _DLHandle:
    def __init__(self, exc=None, tab=None):
        self._exc = exc
        self._tab = tab

    @property
    def active_tab(self):
        if self._exc is not None:
            raise self._exc
        return self._tab


class _DLSvc:
    def __init__(self, handle):
        self._handle = handle
        self.calls = []

    def get_session(self, name):
        return self._handle

    def download(self, a, b, organize=False):
        self.calls.append((a, b, organize))
        return _DLResult()


def test_101_download_without_tab_goes_bare():
    backend = object.__new__(browse_mod._LocalBackend)
    svc = _DLSvc(_DLHandle(exc=RuntimeError("no tabs")))
    backend._svc = svc
    backend.session_name = "cli"
    out = backend.download("http://x/f.bin")
    assert svc.calls == [("http://x/f.bin", "", False)]
    assert out == {"ok": True}
    # control: with a tab, the download goes through the tab
    backend2 = object.__new__(browse_mod._LocalBackend)
    tab = object()
    svc2 = _DLSvc(_DLHandle(tab=tab))
    backend2._svc = svc2
    backend2.session_name = "cli"
    backend2.download("http://x/f.bin")
    assert svc2.calls == [(tab, "http://x/f.bin", False)]
