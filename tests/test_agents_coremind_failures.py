"""Pass #173 pins: the agents failure ledger (REUSED) + coremind conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) is reused from #152:
coremind is an agents-subsystem module, so reuse is correct (unlike #161's
engine/trivia split). The 17 durability tests below mirror the #141+
shape exactly; the 12 site pins prove each conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name); the 6 NOT-recording pins prove each exclusion stays silent.
No pin hardcodes scan totals or coremind counts. Every test redirects
stores with the app's own ``NM_HOME`` override (never the real home,
never the repo tree). Restart tests use real subprocesses. Every trigger
was probed live before its pin was written (probe.py).

Pins rule (#169): no bare-except swallowing anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``.
"""
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import failures as af


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in af.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("agents")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agents").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("agents").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no agents failures recorded."
    af.record_failure("agents.site-a", ValueError("boom"))
    af.record_failure("agents.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["agents.site-b", "agents.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["agents.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "agents.site-a" in text and "agents.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("agents.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "agents.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("agents.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["agents.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agents.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.agents import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("agents").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pins ------------------------------------------------------------

from nomorals.agents.coremind import (
    CoreMind, Intent, _owner_intent_model_check, _llm_intent_interpret,
    _music_intent,
)
import nomorals.agents.coremind as cm
import nomorals.agents.notifier as notmod


class FakeSettings:
    def resolve(self, key):
        raise RuntimeError("no settings in tests")


class FakeContext:
    def __init__(self, router=None, extras=None, db=None):
        self.settings = FakeSettings()
        self.extras = {} if extras is None else extras
        self.memory = None
        self.router = router
        self.db = db


class RaisingExtras(dict):
    def get(self, *a, **k):
        raise RuntimeError("extras boom")


class FakeRuntime:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def _is_operator(self, message):
        raise RuntimeError("operator boom")

    def _game_engine(self):
        raise RuntimeError("engine boom")

    def _game_player(self, message):
        raise RuntimeError("player boom")

    def _control_game(self, tail, chat_key, player=None, kind="dm"):
        return f"game reply player={player}"

    def _control_play(self, query, chat_key=None):
        return f"queued {query}"


def _dm_msg():
    return types.SimpleNamespace(chat=types.SimpleNamespace(kind="dm"))


def _mind(**kw):
    return CoreMind(FakeContext(**kw), runtime=None)


def test_site_owner_check(monkeypatch):
    import nomorals.llm.power as power
    monkeypatch.setattr(power, "model_usable", _boom)
    r = _owner_intent_model_check("i'm the creator of you", FakeContext(router=object()))
    assert r is False
    assert _sites() == ["coremind.owner_check"]


def test_site_state_dir_setup():
    mind = _mind()
    assert mind.state_dir is None and mind.state_file is None
    assert _sites() == ["coremind.state_dir_setup"]


def test_site_tune():
    mind = CoreMind(FakeContext(extras=RaisingExtras()), runtime=None)
    af.clear_failures()
    assert mind._tune() is None
    assert _sites() == ["coremind.tune"]


def test_site_route_log():
    mind = CoreMind(FakeContext(db=MagicMock()), runtime=None)
    af.clear_failures()
    mind._route_log = MagicMock()
    mind._route_log.append.side_effect = RuntimeError("log boom")
    mind._record_route(Intent("chat", 1.0, route="brain"))
    assert _sites() == ["coremind.route_log"]


def test_site_is_owner_dm():
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()
    assert mind._is_owner_dm(_dm_msg(), "somekey") is False
    assert _sites() == ["coremind.is_owner_dm"]


def test_site_dm_kind():
    class RaisingKind:
        @property
        def kind(self):
            raise AttributeError("kind boom")

    class RaisingChat:
        chat = RaisingKind()

    mind = _mind()
    af.clear_failures()
    assert mind._dm_kind(RaisingChat()) is False
    assert _sites() == ["coremind.dm_kind"]


def test_site_live_game():
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()
    assert mind._live_game("k") is None
    assert _sites() == ["coremind.live_game"]


def test_site_music_gate(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.partner.chat_modes", None)
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()
    reply = mind._dispatch_play(Intent("music", 0.9, target="lofi beats"), "j1", "c", _dm_msg())
    assert "queued lofi beats" in reply
    assert _sites() == ["coremind.music_gate"]


def test_site_browser_session_dir():
    mind = _mind()
    af.clear_failures()
    assert mind._browser_session_dir() == ""
    assert _sites() == ["coremind.browser_session_dir"]


def test_site_directives_notifier(monkeypatch):
    class BoomNotifier:
        def __init__(self, *a, **k):
            raise RuntimeError("notifier boom")

    monkeypatch.setattr(notmod, "Notifier", BoomNotifier)
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()
    agent = mind._directives()
    assert agent.notifier is None
    assert _sites() == ["coremind.directives_notifier"]


def test_site_edit_src_media():
    bad_msg = types.SimpleNamespace()
    bad_msg.first_media = _boom
    mind = _mind()
    af.clear_failures()
    reply = mind._dispatch_image(Intent("image", 0.9, target="x", action="edit"), "j2", "c", bad_msg)
    assert "attach the image" in reply
    assert _sites() == ["coremind.edit_src_media"]


def test_site_game_player():
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()
    reply = mind._dispatch_game(Intent("game", 0.9, target="hangman", action="start"), "j3", "c", _dm_msg())
    assert "player=None" in reply
    assert _sites() == ["coremind.game_player"]


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_excl_llm_intent(monkeypatch):
    import nomorals.llm.brain as brainmod
    monkeypatch.setattr(brainmod, "brain_for", _boom)
    r = _llm_intent_interpret("research fusion reactors", FakeContext(router=object()))
    assert r is None
    assert _sites() == []


def test_excl_music_style_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.media.music", None)
    intent = _music_intent("compose a song about love")
    assert intent is not None
    assert _sites() == []


def test_excl_decide_llm_wrapper(monkeypatch):
    monkeypatch.setattr(cm, "_llm_intent_interpret", _boom)
    mind = _mind()
    af.clear_failures()
    intent = mind._decide("research fusion reactors")
    assert intent.kind == "research"
    assert _sites() == []


def test_excl_decide_owner_wrapper(monkeypatch):
    called = []

    def raising_check(*a, **k):
        called.append(True)
        raise RuntimeError("check boom")

    monkeypatch.setattr(cm, "_owner_intent_model_check", raising_check)
    mind = _mind()
    af.clear_failures()
    intent = mind._decide("hello there")
    assert called  # non-vacuous: the wrapper really ran
    assert intent.kind == "chat"
    assert _sites() == []


def test_excl_status_live_games(monkeypatch):
    rt = FakeRuntime(gateway=object())
    rt._game_engine = _boom
    mind = CoreMind(FakeContext(), runtime=rt)
    af.clear_failures()
    st = mind.status()
    assert "live games" not in st
    assert _sites() == []


def test_excl_status_queued_missions():
    mind = CoreMind(FakeContext(), runtime=FakeRuntime())
    af.clear_failures()

    class BoomDirectives:
        def list(self, n):
            raise RuntimeError("list boom")

    mind._directives = lambda: BoomDirectives()
    st = mind.status()
    assert "queued missions" not in st
    assert _sites() == []
