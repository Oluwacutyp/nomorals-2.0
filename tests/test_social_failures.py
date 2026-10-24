"""Pass #150 pins: the NEW social failure ledger + conversions.

The social ledger did not exist before this pass; it is created
here, durability-wired through the shared failure_store helper (the
#141 shape: Store.save() resolves its path INSIDE the try).
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.social.failures``; restart tests use real
subprocesses (a fresh interpreter is the honest simulated restart —
no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name. The excluded handlers (LLM
personalization probes: failure yields exactly the no-enhancement
outcome) are pinned as NOT recording.
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.social import failures as sf
from nomorals.social import leads as leads_mod
from nomorals.social.leads import (
    CommentEvent, LeadStore, control_leads, dm_everyone, handle_comment,
    render_dm,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    sf.clear_failures()
    yield
    sf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in sf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _store():
    return LeadStore(db_path=os.path.join(tempfile.mkdtemp(), "leads.db"))


class _BrokenDb:
    """execute() always raises; everything else is inert."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("db gone")

    def commit(self):
        return None


def _broken(store):
    store._db = _BrokenDb()
    return store


def _armed(store, platform="instagram", post="p1", keyword="VAULT"):
    trig = store.add_trigger(platform, post, keyword)
    assert trig is not None
    return trig


def _ev(platform="instagram", post="p1", commenter="amy",
        contact="amy-c", text="I want VAULT please"):
    return CommentEvent(platform=platform, post_id=post, commenter=commenter,
                        commenter_contact=contact, text=text)


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("social")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("social").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("social")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("social").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("social")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("social").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("social").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("social").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("social")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("social").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("social").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("social").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("social").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("social")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("social").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        sf.record_failure("no-home", ValueError("boom"))
        assert sf.failure_count() == 1
        assert [f.site for f in sf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert sf.failure_count() == 0
    assert sf.recent_failures() == []
    assert sf.format_failures() == "no social failures recorded."
    sf.record_failure("social.site-a", ValueError("boom"))
    sf.record_failure("social.site-b", RuntimeError("bang"))
    assert sf.failure_count() == 2
    assert [f.site for f in sf.recent_failures()] == ["social.site-b", "social.site-a"]
    assert [f.site for f in sf.recent_failures(1)] == ["social.site-b"]
    text = sf.format_failures()
    assert "2 total" in text and "social.site-a" in text and "social.site-b" in text
    sf.clear_failures()
    assert sf.failure_count() == 0
    assert sf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    sf.record_failure("social.disk", ValueError("boom"))
    entries, total = fs.Store("social").load()
    assert total == 1 and entries[0]["site"] == "social.disk"
    sf.clear_failures()
    assert fs.Store("social").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    sf.record_failure("social.mem", ValueError("boom"))
    assert sf.failure_count() == 1
    assert [f.site for f in sf.recent_failures()] == ["social.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                sf.record_failure(f"social.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sf.failure_count() == 400
    assert len(sf.recent_failures(1000)) == 200
    entries, total = sf._store.load()
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
        "from nomorals.social import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.social import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.social import failures as f\n"
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
    store_path = fs.Store("social").path()
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_default_db_records_and_returns_memory(monkeypatch):
    from pathlib import Path as _Path

    def _no_home(*args, **kwargs):
        raise OSError("no home")

    monkeypatch.setattr(_Path, "home", _no_home)
    assert leads_mod._default_db() == ":memory:"
    assert _sites() == ["leads.default_db"]


def test_converted_store_init_records_and_dead_store(tmp_path):
    store = LeadStore(db_path=str(tmp_path))  # a directory, not a file
    assert store._db is None
    assert _sites() == ["leads.store_init"]


def test_converted_add_trigger_records_and_returns_none():
    store = _broken(_store())
    assert store.add_trigger("instagram", "p1", "VAULT") is None
    assert _sites() == ["leads.add_trigger"]


def test_converted_get_trigger_records_and_returns_none():
    store = _broken(_store())
    assert store.get_trigger("trig_x") is None
    assert _sites() == ["leads.get_trigger"]


def test_converted_list_triggers_records_and_returns_empty():
    store = _broken(_store())
    assert store.list_triggers() == []
    assert _sites() == ["leads.list_triggers"]


def test_converted_stop_trigger_records_and_returns_false():
    store = _broken(_store())
    assert store.stop_trigger("trig_x") is False
    assert _sites() == ["leads.stop_trigger"]


def test_converted_match_triggers_records_and_returns_empty():
    store = _broken(_store())
    assert store.match_triggers("instagram", "p1", "VAULT") == []
    assert _sites() == ["leads.match_triggers"]


def test_converted_log_comment_records_and_returns_false():
    store = _broken(_store())
    assert store.log_comment(_ev()) is False
    assert _sites() == ["leads.log_comment"]


def test_converted_commenters_records_and_returns_empty():
    store = _broken(_store())
    assert store.commenters("p1") == []
    assert _sites() == ["leads.commenters"]


def test_converted_already_dmed_records_and_returns_false():
    store = _broken(_store())
    assert store.already_dmed("instagram", "p1", "amy", "trig_x") is False
    assert _sites() == ["leads.already_dmed"]


def test_converted_mark_dmed_records_and_returns_none():
    store = _broken(_store())
    assert store.mark_dmed("instagram", "p1", "amy", "trig_x") is None
    assert _sites() == ["leads.mark_dmed"]


def test_converted_capture_lead_records_and_returns_none():
    store = _broken(_store())
    assert store.capture_lead("amy", "instagram", "amy-c", "x", "p1") is None
    assert _sites() == ["leads.capture_lead"]


def test_converted_get_lead_records_and_returns_none():
    store = _broken(_store())
    assert store.get_lead("lead_x") is None
    assert _sites() == ["leads.get_lead"]


def test_converted_list_leads_records_and_returns_empty():
    store = _broken(_store())
    assert store.list_leads() == []
    assert _sites() == ["leads.list_leads"]


def test_converted_set_lead_status_records_and_returns_false():
    store = _broken(_store())
    assert store.set_lead_status("lead_x", "dmed") is False
    assert _sites() == ["leads.set_lead_status"]


def test_converted_set_trigger_meta_records_and_returns_false():
    store = _broken(_store())
    assert store.set_trigger_meta("trig_x", "public_reply", "hi {name}") is False
    assert _sites() == ["leads.set_trigger_meta"]


def test_converted_get_trigger_meta_records_and_returns_empty():
    store = _broken(_store())
    assert store.get_trigger_meta("trig_x", "public_reply") == ""
    assert _sites() == ["leads.get_trigger_meta"]


def test_converted_funnel_stats_records_and_returns_empty(monkeypatch):
    store = _broken(_store())
    fake = SimpleNamespace(platform="instagram", post_id="p1", keyword="VAULT",
                           qualifier="", public_reply="")
    monkeypatch.setattr(LeadStore, "get_trigger", lambda self, tid: fake)
    assert store.funnel_stats("trig_x") == {}
    assert _sites() == ["leads.funnel_stats"]


class _BadStr:
    def __str__(self):
        raise RuntimeError("boom")


def test_converted_render_dm_records_and_returns_template():
    from nomorals.social.leads import Lead
    lead = Lead(name="amy", platform="instagram", contact="amy-c",
                interest="VAULT")
    assert render_dm("{x}", lead, {"x": _BadStr()}) == "{x}"
    assert _sites() == ["leads.render_dm"]


def test_converted_handle_comment_cost_gate_records_and_still_sends(monkeypatch):
    from nomorals.social import whatsapp_cost as wc
    monkeypatch.setattr(wc, "worth_it", _boom)
    store = _store()
    _armed(store, "whatsapp", "p1", "VAULT")
    res = handle_comment(store, _ev("whatsapp", "p1"))
    assert len(res) == 1 and res[0]["dmed"] is True
    assert _sites() == ["leads.handle_comment.cost_gate"]


def test_converted_handle_comment_dm_sender_records_and_not_sent():
    store = _store()
    _armed(store, "instagram", "p1", "VAULT")
    res = handle_comment(store, _ev(), dm_sender=_boom)
    assert len(res) == 1 and res[0]["dmed"] is False
    assert _sites() == ["leads.handle_comment.dm_sender"]


def test_converted_handle_comment_followup_records_and_still_dmed():
    store = _store()
    trig = store.add_trigger("instagram", "p1", "VAULT",
                             followup_template="later {name}")
    assert trig is not None and trig.followup_template == "later {name}"
    engine = SimpleNamespace(enqueue=_boom)
    res = handle_comment(store, _ev(), followup_engine=engine)
    assert len(res) == 1 and res[0]["dmed"] is True
    assert _sites() == ["leads.handle_comment.followup"]


def test_converted_handle_comment_outer_records_and_returns_empty():
    fake = SimpleNamespace(log_comment=lambda ev: None,
                           match_triggers=lambda *a: [object()])
    assert handle_comment(fake, _ev()) == []
    assert _sites() == ["leads.handle_comment"]


def test_converted_dm_everyone_cost_gate_records_and_still_dmed(monkeypatch):
    from nomorals.social import whatsapp_cost as wc
    monkeypatch.setattr(wc, "worth_it", _boom)
    store = _store()
    store.log_comment(_ev("whatsapp", "p1"))
    out = dm_everyone(store, "p1", platform="whatsapp")
    assert out["dmed"] == 1 and out["details"][0]["cost_ok"] is True
    assert _sites() == ["leads.dm_everyone.cost_gate"]


def test_converted_dm_everyone_dm_sender_records_and_skipped():
    store = _store()
    store.log_comment(_ev())
    out = dm_everyone(store, "p1", dm_sender=_boom)
    assert out["dmed"] == 0 and out["skipped"] == 1
    assert _sites() == ["leads.dm_everyone.dm_sender"]


def test_converted_dm_everyone_outer_records_and_returns_empty_counts():
    fake = SimpleNamespace(commenters=_boom)
    assert dm_everyone(fake, "p1") == {"dmed": 0, "skipped": 0, "details": []}
    assert _sites() == ["leads.dm_everyone"]


def test_converted_mark_lead_magnet_records_and_returns_false():
    store = _broken(_store())
    assert store is not None and leads_mod.mark_lead_magnet(store, "trig_x") is False
    assert _sites() == ["leads.mark_lead_magnet"]


def test_converted_is_lead_magnet_records_and_returns_false():
    store = _broken(_store())
    assert leads_mod.is_lead_magnet(store, "trig_x") is False
    assert _sites() == ["leads.is_lead_magnet"]


def test_converted_control_leads_outer_records_and_returns_hiccup():
    fake = SimpleNamespace(list_triggers=lambda: [object()])
    assert control_leads("triggers", store=fake) == "leads hiccup — try again."
    assert _sites() == ["leads.control_leads"]


# ---- Excluded handlers: proven NOT recording -----------------------------

def test_excluded_handle_comment_llm_probe_records_nothing():
    store = _store()
    _armed(store, "instagram", "p1", "VAULT")
    res = handle_comment(store, _ev(), llm_fn=_boom)
    assert len(res) == 1 and res[0]["dmed"] is True
    assert sf.recent_failures() == []


def test_excluded_dm_everyone_llm_probe_records_nothing():
    store = _store()
    store.log_comment(_ev())
    out = dm_everyone(store, "p1", llm_fn=_boom)
    assert out["dmed"] == 1
    assert sf.recent_failures() == []
