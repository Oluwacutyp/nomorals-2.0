"""Tail-94 pins: nomorals/tools/games.py census sites.

All five sites guard OPTIONAL roster enrichment around one
deliverable: ``_engine`` returning a working GameEngine. The
core engine is constructed BEFORE every guarded block and is
returned on every path; a failure degrades only the enrichment
it wraps, and downstream the absence stays honest — an
unregistered game is an explicit "unknown game" error at start
time, never a fake room. Held as exclusions:

47 — one invented design fails to register -> skipped; the
loop continues ("one bad design can't block" is the site's own
comment and the correct roster semantic: designs are files on
disk, any one may be stale against a newer engine).
49 — ``load_designs()`` itself fails -> the engine proceeds
with the built-in roster only; invented games are an add-on,
not the roster.
55 — character seating (``register_with_engine``) fails -> the
engine proceeds without character seats; human play is whole.
59 — an exotic registry refuses attribute assignment -> the
engine is returned uncached. Every call on such a registry
rebuilds (rooms cannot persist across calls on a registry
that cannot hold state BY CONSTRUCTION), but this call's
engine is real and the failure never masquerades as a cached
one. Spine registries accept attrs; this is a duck-typing
guard, pinned with a frozen double below.
252 — ``_invented_names`` load failure -> empty set, so the
``list`` action labels every game ``invented: False``. See the
adjudication note in this pass's report: this is the batch's
weakest exclusion (a wrong label, not an empty value) — held
because the flag is catalog metadata nothing branches on, and
for it to be wrong load_designs must fail at list time after
succeeding at engine-build time.

No recording is claimed or possible: nomorals/tools has no
failures ledger (no tools/failures.py exists) and the module
imports none; the pins assert behavior only. The silence is
an observability gap — shelved sketch (bind + _log.debug at
the five sites; the module's _log is currently unused in
_engine), not a convert.

Each test pins the degraded behavior, with a healthy control
where the same path succeeds. No bare ``except: pass``
anywhere; no sys.path hacks.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.characters as characters
import nomorals.games.engine as engine_mod
import nomorals.games.invent as invent_mod
from nomorals.tools import games as G


class _FakeEngine:
    """Stands in for GameEngine: records register() calls."""

    def __init__(self, context=None, send=None, suggest=None):
        self.context = context
        self.registered = []

    def register(self, game):
        self.registered.append(game)


def _engine_env(load_designs, invented_game, register_with_engine):
    """Patch the three seams _engine reaches, yield nothing."""
    return mock.patch.multiple(
        engine_mod, GameEngine=_FakeEngine), mock.patch.multiple(
        invent_mod, load_designs=load_designs,
        InventedGame=invented_game), mock.patch.multiple(
        characters, CharacterStore=lambda: object(),
        register_with_engine=register_with_engine)


def _quiet_engine(load_designs=lambda: [],
                  invented_game=lambda d, suggest=None: ("game", d),
                  register_with_engine=lambda *a, **k: None):
    patches = _engine_env(load_designs, invented_game,
                          register_with_engine)
    for p in patches:
        p.start()
    return patches


def _stop(patches):
    for p in patches:
        p.stop()


# ── 47: one bad design skipped, the rest register ──────────────

def test_47_one_bad_design_cannot_block_the_roster():
    def invent(design, suggest=None):
        if design == "bad":
            raise RuntimeError("stale design")
        return ("game", design)

    patches = _quiet_engine(load_designs=lambda: ["good", "bad", "good2"],
                            invented_game=invent)
    try:
        eng = G._engine(SimpleNamespace())
        assert eng.registered == [("game", "good"), ("game", "good2")]
    finally:
        _stop(patches)


# ── 49: load_designs failure -> built-in roster only ───────────

def test_49_load_designs_failure_still_returns_engine():
    def boom():
        raise RuntimeError("designs unreadable")

    patches = _quiet_engine(load_designs=boom)
    try:
        eng = G._engine(SimpleNamespace())
        assert isinstance(eng, _FakeEngine)
        assert eng.registered == []
    finally:
        _stop(patches)


# ── 55: character seating failure -> engine without seats ──────

def test_55_character_seating_failure_still_returns_engine():
    def boom(*a, **k):
        raise RuntimeError("character store down")

    patches = _quiet_engine(register_with_engine=boom)
    try:
        eng = G._engine(SimpleNamespace())
        assert isinstance(eng, _FakeEngine)
    finally:
        _stop(patches)


# ── 59: attr-refusing registry -> engine returned uncached ─────

class _FrozenRegistry:
    def __setattr__(self, key, value):
        raise RuntimeError("frozen registry")


def test_59_attr_refusing_registry_returns_uncached_engine():
    patches = _quiet_engine()
    try:
        reg = _FrozenRegistry()
        eng = G._engine(reg)
        assert isinstance(eng, _FakeEngine)
        # honest consequence: nothing cached, so a second call
        # builds a second engine rather than reusing a phantom
        assert G._engine(reg) is not eng
        # control: a normal registry caches and reuses
        reg2 = SimpleNamespace()
        assert G._engine(reg2) is G._engine(reg2)
    finally:
        _stop(patches)


# ── 252: invented-names load failure -> empty set ──────────────

def test_252_invented_names_failure_returns_empty_set():
    with mock.patch.object(invent_mod, "load_designs",
                           lambda: [{"name": "zz"}, {"name": "aa"}]):
        assert G._invented_names(SimpleNamespace()) == {"aa", "zz"}

    def boom():
        raise RuntimeError("designs unreadable")

    with mock.patch.object(invent_mod, "load_designs", boom):
        assert G._invented_names(SimpleNamespace()) == set()
