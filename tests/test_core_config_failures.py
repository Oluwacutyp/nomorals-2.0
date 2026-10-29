"""Tail-46 pins for nomorals/core/config.py — 0 converts /
8 exclusions, 0 flagged.

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the guarded path
was actually exercised, asserting via the double's/wrapper's
own call log that the failing operation itself was attempted.
The file contains no exception handlers, so it adds zero
census sites.

config.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure / recent_failures reference),
so NOT-recording is pinned with the tail-8 snapshot method:
agents recent_failures() before/after each trigger must be
unchanged. For the ConfigWatcher sites the module's own
observability surface (watcher.stats()) is pinned as well.

Verdict notes (full grounds in the tail-46 SURVEY.md):
- 1281: _resolved_types failure -> raw string annotations, the
  explicitly-marked expected fallback; trigger is a code-level
  unresolvable forward ref, never routine data.
- 1543: _heal_retired_urls urlparse failure -> host "", the
  malformed URL is left exactly as-is; no claim is made.
- 1759: config_schema default_factory failure -> default None,
  the schema's no-default placeholder; real factories are
  literal constructors that cannot raise on a routine path.
- 1877: ConfigWatcher.stop join failure -> pass; stop is
  documented "Never raises", the thread ref is already cleared
  and the stop event already set before the join.
- 1884: ConfigWatcher._prime loader failure -> pass, current
  stays None; the same error surfaces on reload (counted in
  stats, emitted via on_error) — proven in the pin.
- 1892: _run watchfiles import failure -> polling, the
  documented degrade path (the routine path where watchfiles
  is not installed); polling preserves full function.
- 1898: _run watchfiles runtime failure -> pass, then the
  polling degrade runs; capability preserved, not lost.
- 1989: _emit_error on_error callback failure -> pass;
  callbacks never break reloads, and the underlying failure
  was already counted in stats before the emit.
"""
import sys
import types
import typing
import urllib.parse
from dataclasses import dataclass, fields

sys.path.insert(0, "/home/hatch/workspace/nm-tail46-tree")

from nomorals.agents import failures as ag_fail
from nomorals.core import config as cfg


def _snap():
    return list(ag_fail.recent_failures())


def test_1281_resolved_types_failure_returns_raw_annotations():
    before = _snap()
    calls = []
    orig = typing.get_type_hints

    def counting(cls, *a, **k):
        calls.append(cls)
        return orig(cls, *a, **k)

    @dataclass
    class BadHints:
        x: "NoSuchTypeAnywhere" = None

    @dataclass
    class GoodHints:
        x: int = 0
        y: str = "a"

    typing.get_type_hints = counting
    try:
        bad = cfg._resolved_types(BadHints)
        good = cfg._resolved_types(GoodHints)
    finally:
        typing.get_type_hints = orig
    assert BadHints in calls  # resolution was attempted
    assert bad == {"x": "NoSuchTypeAnywhere"}  # raw fallback
    assert good.get("x") is int and good.get("y") is str  # control
    assert _snap() == before


def test_1543_heal_retired_urls_bad_url_left_unchanged():
    before = _snap()
    calls = []
    orig = urllib.parse.urlparse

    def counting(url, *a, **k):
        calls.append(url)
        return orig(url, *a, **k)

    urllib.parse.urlparse = counting
    try:
        bad = cfg.Settings()
        bad.llm.hf_base_url = "http://[::1"  # invalid IPv6
        cfg._heal_retired_urls(bad)
        assert "http://[::1" in calls  # the parse was attempted
        assert bad.llm.hf_base_url == "http://[::1"  # left as-is
        retired = cfg.Settings()
        retired.llm.hf_base_url = "https://api-inference.huggingface.co"
        cfg._heal_retired_urls(retired)
        assert retired.llm.hf_base_url == \
            "https://router.huggingface.co/hf-inference"  # control
        other = cfg.Settings()
        other.llm.hf_base_url = "https://example.com/v1"
        cfg._heal_retired_urls(other)
        assert other.llm.hf_base_url == "https://example.com/v1"
    finally:
        urllib.parse.urlparse = orig
    assert _snap() == before


def test_1759_config_schema_factory_failure_defaults_none():
    before = _snap()
    calls = []
    fld = [f for f in fields(cfg.LLMSettings)
           if f.name == "fallback_chain"][0]
    orig_factory = fld.default_factory
    real_default = orig_factory()

    def raising_factory():
        calls.append(1)
        raise RuntimeError("factory exploded")

    fld.default_factory = raising_factory
    try:
        schema = cfg.config_schema()
    finally:
        fld.default_factory = orig_factory
    assert len(calls) == 1  # the factory was attempted
    entry = schema["sections"]["llm"]["fields"]["fallback_chain"]
    assert entry["default"] is None
    control = cfg.config_schema()
    entry2 = control["sections"]["llm"]["fields"]["fallback_chain"]
    assert entry2["default"] == real_default  # control
    assert _snap() == before


class _RaisingThread:
    def __init__(self):
        self.join_calls = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls += 1
        raise RuntimeError("join exploded")


class _GoodThread:
    def __init__(self):
        self.join_calls = 0

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.join_calls += 1


def test_1877_watcher_stop_join_failure_swallowed():
    before = _snap()
    w = cfg.ConfigWatcher(loader=lambda: cfg.Settings(), paths=[])
    rt = _RaisingThread()
    w._thread = rt
    w.stop()  # documented "Never raises" — must not propagate
    assert rt.join_calls == 1  # the join was attempted
    assert w._thread is None and w._stop.is_set()
    w2 = cfg.ConfigWatcher(loader=lambda: cfg.Settings(), paths=[])
    gt = _GoodThread()
    w2._thread = gt
    w2.stop()
    assert gt.join_calls == 1 and w2._thread is None  # control
    assert _snap() == before


def test_1884_watcher_prime_loader_failure_keeps_none():
    before = _snap()
    calls = []

    def raising_loader():
        calls.append(1)
        raise RuntimeError("loader exploded")

    w = cfg.ConfigWatcher(loader=raising_loader, paths=[])
    w._prime()
    assert len(calls) == 1  # the load was attempted
    assert w.current is None
    sentinel = cfg.Settings()
    w2 = cfg.ConfigWatcher(loader=lambda: sentinel, paths=[])
    w2._prime()
    assert w2.current is sentinel  # control
    # the documented surface: the same error DOES surface on reload
    seen = []
    w3 = cfg.ConfigWatcher(loader=raising_loader, paths=[],
                           on_error=lambda exc: seen.append(exc))
    assert w3._maybe_reload() is False
    assert len(calls) == 3  # reload retries the loader twice
    assert w3.stats()["failures"] == 1
    assert len(seen) == 1 and isinstance(seen[0], RuntimeError)
    assert _snap() == before


def _watcher_with_recorders(tmp_path):
    target = tmp_path / "cfg.toml"
    target.write_text("x = 1\n")
    poll_calls, wf_calls = [], []
    w = cfg.ConfigWatcher(loader=lambda: cfg.Settings(),
                          paths=[str(target)])
    w._run_poll = lambda: poll_calls.append(1)
    w._run_watchfiles = lambda watch: wf_calls.append(watch)
    return w, poll_calls, wf_calls


def _restore_watchfiles(saved):
    if saved == "ABSENT":
        sys.modules.pop("watchfiles", None)
    else:
        sys.modules["watchfiles"] = saved


def test_1892_run_without_watchfiles_degrades_to_polling(tmp_path):
    before = _snap()
    saved = sys.modules.get("watchfiles", "ABSENT")
    w, poll_calls, wf_calls = _watcher_with_recorders(tmp_path)
    sys.modules["watchfiles"] = None  # forces the ImportError
    try:
        w._run()
    finally:
        _restore_watchfiles(saved)
    assert poll_calls == [1]  # polling degrade ran
    assert wf_calls == []  # watchfiles runner never entered
    # control: a present watchfiles module takes that branch
    fake = types.ModuleType("watchfiles")
    fake.watch = object()
    w2, poll2, wf2 = _watcher_with_recorders(tmp_path)
    sys.modules["watchfiles"] = fake
    try:
        w2._run()
    finally:
        _restore_watchfiles(saved)
    assert wf2 == [fake.watch] and poll2 == []
    assert _snap() == before


def test_1898_run_watchfiles_failure_degrades_to_polling(tmp_path):
    before = _snap()
    saved = sys.modules.get("watchfiles", "ABSENT")
    target = tmp_path / "cfg.toml"
    target.write_text("x = 1\n")
    watch_calls, poll_calls = [], []

    def dying_watch(*a, **k):
        watch_calls.append((a, k))
        raise RuntimeError("watch died")

    fake = types.ModuleType("watchfiles")
    fake.watch = dying_watch
    w = cfg.ConfigWatcher(loader=lambda: cfg.Settings(),
                          paths=[str(target)])
    w._run_poll = lambda: poll_calls.append(1)
    sys.modules["watchfiles"] = fake
    try:
        w._run()  # real _run_watchfiles iterates the dying watch
    finally:
        _restore_watchfiles(saved)
    assert len(watch_calls) == 1  # the watch was attempted
    assert poll_calls == [1]  # polling degrade ran
    # control: a healthy empty watch returns cleanly, no polling
    fake2 = types.ModuleType("watchfiles")
    fake2.watch = lambda *a, **k: iter(())
    w2 = cfg.ConfigWatcher(loader=lambda: cfg.Settings(),
                           paths=[str(target)])
    w2._run_poll = lambda: poll_calls.append(1)
    sys.modules["watchfiles"] = fake2
    try:
        w2._run()
    finally:
        _restore_watchfiles(saved)
    assert poll_calls == [1]  # unchanged by the control run
    assert _snap() == before


def test_1989_emit_error_callback_failure_swallowed():
    before = _snap()
    calls = []

    def raising_on_error(exc):
        calls.append(exc)
        raise RuntimeError("callback exploded")

    w = cfg.ConfigWatcher(loader=lambda: cfg.Settings(), paths=[],
                          on_error=raising_on_error)
    boom = RuntimeError("reload broke")
    fails_before = w.stats()["failures"]
    w._emit_error(boom)  # callbacks never break reloads
    assert calls == [boom]  # the callback was attempted
    assert w.stats()["failures"] == fails_before  # emit counts nothing
    seen = []
    w2 = cfg.ConfigWatcher(loader=lambda: cfg.Settings(), paths=[],
                           on_error=lambda exc: seen.append(exc))
    w2._emit_error(boom)
    assert seen == [boom]  # control

    class FakeLog:
        def __init__(self):
            self.warnings = []

        def warning(self, *a, **k):
            self.warnings.append(a)

    fl = FakeLog()
    w3 = cfg.ConfigWatcher(loader=lambda: cfg.Settings(), paths=[],
                           logger=fl)
    w3._emit_error(boom)
    assert len(fl.warnings) == 1  # control: logger path warns
    assert _snap() == before
