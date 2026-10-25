"""Pass #188 pins: the console failure ledger (REUSED) + run_chat_bot conversions.

``nomorals/console/failures.py`` (namespace ``console``) is reused:
``scripts/run_chat_bot.py`` is the entrypoint that runs the console
surface (banner, dashboard, mirror, commands), and the console ledger
already records this file's snapshot-provider failures at
``commands.ConsoleCommands._safe_snapshot``. A new ``nomorals/scripts/``
ledger was NOT created: an unmapped top-level package falls through to
layer 0 in ``tests/test_layering.py``, so a scripts ledger importing
``core`` would add a 5th layering violation (see SURVEY.md for the proof
and the parent policy question). Site names stay module-qualified
(``run_chat_bot.*``), following the console ledger's existing
``commands.*`` / ``dashboard.*`` convention.

The 17 durability tests below mirror the #141+ shape exactly (same
ledger); the 3 site pins prove each conversion records without changing
flow (fallback value unchanged AND a ledger entry under its site name);
the 13 NOT-recording pins prove the exclusions stay silent. No pin
hardcodes scan totals or run_chat_bot counts. Every test redirects
stores with the app's own ``NM_HOME`` override (never the real home,
never the repo tree). Restart tests use real subprocesses. Every trigger
was probed live before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import importlib.util
import json
import os
import subprocess
import sys
import threading
from io import StringIO
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN_CHAT_BOT = ROOT / "scripts" / "run_chat_bot.py"

from nomorals.core import failure_store as fs
from nomorals.console import failures as cf
from nomorals.console import widgets as widgets_mod


def _load_script():
    spec = importlib.util.spec_from_file_location("run_chat_bot", RUN_CHAT_BOT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MOD = _load_script()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    cf.clear_failures()
    yield
    cf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in cf.recent_failures()]


class _Boom(Exception):
    pass


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("console")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("console").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("console")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("console").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("console")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("console").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("console").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("console").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("console")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("console").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("console").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("console").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("console").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("console")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("console").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        cf.record_failure("no-home", ValueError("boom"))
        assert cf.failure_count() == 1
        assert [f.site for f in cf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []
    assert cf.format_failures() == "no console failures recorded."
    cf.record_failure("run_chat_bot.site-a", ValueError("boom"))
    cf.record_failure("run_chat_bot.site-b", RuntimeError("bang"))
    assert cf.failure_count() == 2
    assert [f.site for f in cf.recent_failures()] == ["run_chat_bot.site-b", "run_chat_bot.site-a"]
    assert [f.site for f in cf.recent_failures(1)] == ["run_chat_bot.site-b"]
    text = cf.format_failures()
    assert "2 total" in text and "run_chat_bot.site-a" in text and "run_chat_bot.site-b" in text
    cf.clear_failures()
    assert cf.failure_count() == 0
    assert cf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    cf.record_failure("run_chat_bot.disk", ValueError("boom"))
    entries, total = fs.Store("console").load()
    assert total == 1 and entries[0]["site"] == "run_chat_bot.disk"
    cf.clear_failures()
    assert fs.Store("console").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    cf.record_failure("run_chat_bot.mem", ValueError("boom"))
    assert cf.failure_count() == 1
    assert [f.site for f in cf.recent_failures()] == ["run_chat_bot.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                cf.record_failure(f"run_chat_bot.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert cf.failure_count() == 400
    assert len(cf.recent_failures(1000)) == 200
    entries, total = cf._store.load()
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
        "from nomorals.console import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.console import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.console import failures as f\n"
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
    store_path = fs.Store("console").path()
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


# ---- Shared harnesses for site + NOT-recording pins ----------------------

class _RaisingLock:
    def __enter__(self):
        raise _Boom("lock down")

    def __exit__(self, *a):
        return False


class _RaisingMessage:
    def __getattr__(self, name):
        raise _Boom("msg down")


class _PropRuntime:
    """Runtime whose _autonomy/_arena raise only when flagged, delegating
    everything else to a base runtime (plain assignment cannot raise on
    access, so properties are required to trigger the extras handlers)."""

    def __init__(self, base, raise_autonomy=False, raise_arena=False):
        self._base = base
        self._raise_autonomy = raise_autonomy
        self._raise_arena = raise_arena

    @property
    def _autonomy(self):
        if self._raise_autonomy:
            raise _Boom("autonomy prop down")
        return self._base._autonomy

    @property
    def _arena(self):
        if self._raise_arena:
            raise _Boom("arena prop down")
        return self._base._arena

    def __getattr__(self, name):
        return getattr(self._base, name)


def _base_runtime():
    rt = SimpleNamespace()
    rt.gateway = SimpleNamespace(
        status=lambda: {"telegram-bot": {"running_in_session": True,
                                         "received": 1, "sent": 2}})
    rt._stats_lock = threading.Lock()
    rt.stats = {"messages": 3, "replies": 2, "errors": 0, "controls": 1}
    rt._scheduler = None
    rt._brain = None
    rt._router = None
    rt._autonomy = True
    rt._arena = False
    return rt


def _base_context():
    ctx = SimpleNamespace()
    ctx.db = SimpleNamespace(query_one=lambda sql: {"n": 7})
    return ctx


class _Hist:
    def per_minute(self):
        return [0] * 30


def _snapshot(rt=None, ctx=None, hist=None):
    provider = MOD._build_snapshot_provider(
        rt or _base_runtime(), ctx or _base_context(), 0.0, hist or _Hist())
    return provider()


def _main_harness(runtime):
    fake_context = mock.MagicMock()
    fake_context.__enter__.return_value = fake_context
    fake_context.__exit__.return_value = False
    wait_calls = {"n": 0}

    def _fake_wait(stop):
        wait_calls["n"] += 1
        stop.set()

    with (
        mock.patch("nomorals.core.config.load_settings", return_value=mock.MagicMock()),
        mock.patch("nomorals.core.logging_setup.setup_logging"),
        mock.patch("nomorals.agents.context.build_context", return_value=fake_context),
        mock.patch("nomorals.agents.partner_runtime.PartnerRuntime", return_value=runtime),
        mock.patch.object(MOD, "_wait_for_shutdown", side_effect=_fake_wait),
        mock.patch("builtins.print") as mock_print,
    ):
        rc = MOD.main()
    printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
    return rc, wait_calls["n"], printed


def _mirror(gw=None):
    gw = gw if gw is not None else SimpleNamespace()
    MOD._install_console_mirror(
        SimpleNamespace(gateway=gw), SimpleNamespace(record=lambda: None))
    return gw.console_mirror


# ---- Site pins ------------------------------------------------------------

def test_pin_run_chat_bot_local_adapter():
    rt = mock.MagicMock()
    rt.start.return_value = ["telegram-bot"]
    rt.gateway.adapters.get.side_effect = _Boom("adapters down")
    rc, waited, _ = _main_harness(rt)
    assert rc == 0 and waited == 1
    assert "run_chat_bot.local_adapter" in _sites()


def test_pin_run_chat_bot_shutdown_stop():
    rt = mock.MagicMock()
    rt.start.return_value = ["telegram-bot"]
    rt.gateway.adapters.get.return_value = None
    rt.stop.side_effect = _Boom("stop down")
    rc, waited, printed = _main_harness(rt)
    assert rc == 0 and waited == 1
    assert "Devon stopped." in printed
    assert "run_chat_bot.shutdown_stop" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_no_record_certifi(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setitem(sys.modules, "certifi", None)
    assert MOD._ensure_ssl_cert_file() is None
    assert "SSL_CERT_FILE" not in os.environ
    assert _sites() == []


def test_pin_no_record_snapshot_adapters():
    rt = _base_runtime()
    rt.gateway = SimpleNamespace(status=lambda: (_ for _ in ()).throw(_Boom("gw down")))
    assert _snapshot(rt=rt)["adapters"] == {}
    assert _sites() == []


def test_pin_no_record_snapshot_traffic():
    rt = _base_runtime()
    rt._stats_lock = _RaisingLock()
    rt.stats = {}
    assert _snapshot(rt=rt)["traffic"] == {}
    assert _sites() == []


def test_pin_no_record_snapshot_scheduler():
    rt = _base_runtime()
    rt._scheduler = SimpleNamespace(running=lambda: (_ for _ in ()).throw(_Boom("sched down")))
    assert _snapshot(rt=rt)["scheduler"] == {}
    assert _sites() == []


def test_pin_no_record_snapshot_games():
    ctx = SimpleNamespace(db=SimpleNamespace(query_one=lambda sql: (_ for _ in ()).throw(_Boom("db down"))))
    assert _snapshot(ctx=ctx)["games"] == {"players": "—"}
    assert _sites() == []


def test_pin_run_chat_bot_snapshot_history():
    hist = SimpleNamespace(per_minute=lambda: (_ for _ in ()).throw(_Boom("hist down")))
    assert _snapshot(hist=hist)["history"] == []
    assert "run_chat_bot.snapshot_history" in _sites()


def test_pin_no_record_snapshot_brain():
    rt = _base_runtime()
    rt._brain = SimpleNamespace(router=SimpleNamespace(
        stats_snapshot=lambda: (_ for _ in ()).throw(_Boom("router down"))))
    assert _snapshot(rt=rt)["llm"] == {}
    assert _sites() == []


def test_pin_no_record_snapshot_extras_autonomy():
    rt = _PropRuntime(_base_runtime(), raise_autonomy=True)
    assert _snapshot(rt=rt)["extras"] == {"arena": "off"}
    assert _sites() == []


def test_pin_no_record_snapshot_extras_arena():
    rt = _PropRuntime(_base_runtime(), raise_arena=True)
    assert _snapshot(rt=rt)["extras"] == {"autonomy": "on"}
    assert _sites() == []


def test_pin_no_record_banner_version(monkeypatch):
    import nomorals
    saved = getattr(nomorals, "__version__", "ABSENT")
    if hasattr(nomorals, "__version__"):
        monkeypatch.delattr(nomorals, "__version__")
    captured = {}
    with mock.patch("nomorals.console.banner.render_banner",
                    side_effect=lambda a, version="", theme=None, color=True: captured.setdefault("version", version) or "BANNER"), \
         mock.patch("nomorals.console.themes.get_theme", return_value=None), \
         redirect_stdout(StringIO()):
        MOD._print_banner(["telegram-bot"])
    assert captured.get("version") == ""
    assert _sites() == []
    if saved != "ABSENT":
        nomorals.__version__ = saved


def test_pin_no_record_banner_outer():
    with mock.patch("nomorals.console.banner.render_banner", side_effect=_Boom("banner down")), \
         mock.patch("nomorals.console.themes.get_theme", return_value=None):
        buf = StringIO()
        with redirect_stdout(buf):
            MOD._print_banner(["telegram-bot"])
    assert "DEVON · chat gateway live" in buf.getvalue()
    assert _sites() == []


def test_pin_no_record_mirror_event():
    mirror = _mirror()
    mirror(_RaisingMessage())
    assert _sites() == []


def test_pin_no_record_mirror_card():
    gw = SimpleNamespace()
    with mock.patch("nomorals.console.palette.supports_color", return_value=True), \
         mock.patch.object(widgets_mod, "format_message_card", side_effect=_Boom("card down")):
        mirror = _mirror(gw)
    with mock.patch.object(widgets_mod.WatchHub, "is_active", return_value=False):
        mirror(SimpleNamespace(chat=SimpleNamespace(platform="t", title="c"),
                               sender="s", text="hi", ts=0.0, incoming=True))
    assert _sites() == []


def test_pin_no_record_mirror_outer(monkeypatch):
    gw = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "nomorals.console.palette", None)
    MOD._install_console_mirror(
        SimpleNamespace(gateway=gw), SimpleNamespace(record=lambda: None))
    assert not hasattr(gw, "console_mirror")
    assert _sites() == []
