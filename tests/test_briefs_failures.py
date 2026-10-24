"""Pass #147 pins: the reused marketing failure ledger + briefs conversions.

The marketing ledger already exists (pass #100, durability-wired by
pass #141 through the shared failure_store helper); pass #147 REUSES
it for nomorals/marketing/briefs.py rather than duplicating it.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.marketing.failures``; restart tests use real
subprocesses (a fresh interpreter is the honest simulated restart —
no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name. The two debug-honest handlers
(``schedule_draft`` / ``run`` outer, which return the exception in
their reason) are pinned as NOT recording.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.marketing import failures as mk


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mk.clear_failures()
    yield
    mk.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in mk.recent_failures()]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("marketing")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("marketing").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("marketing")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("marketing").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("marketing")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("marketing").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("marketing").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("marketing").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("marketing")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("marketing").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("marketing").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("marketing").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    # The #141 fix-up shape, pinned for the reused ledger too: save()
    # must resolve its path INSIDE the try, so a home-resolution
    # failure degrades to a counted write failure, never raises.
    # NM_HOME must be unset: an absolute NM_HOME never calls
    # expanduser, which is how scratch-home batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("marketing").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mk.record_failure("no-home", ValueError("boom"))
        assert mk.failure_count() == 1
        assert [f.site for f in mk.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("marketing")
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
    entries, _ = fs.Store("marketing").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert mk.failure_count() == 0
    assert mk.recent_failures() == []
    assert mk.format_failures() == "no marketing failures recorded."
    mk.record_failure("mk.site-a", ValueError("boom"))
    mk.record_failure("mk.site-b", RuntimeError("bang"))
    assert mk.failure_count() == 2
    assert [f.site for f in mk.recent_failures()] == ["mk.site-b", "mk.site-a"]
    assert [f.site for f in mk.recent_failures(1)] == ["mk.site-b"]
    text = mk.format_failures()
    assert "2 total" in text and "mk.site-a" in text and "mk.site-b" in text
    mk.clear_failures()
    assert mk.failure_count() == 0
    assert mk.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mk.record_failure("mk.disk", ValueError("boom"))
    entries, total = fs.Store("marketing").load()
    assert total == 1 and entries[0]["site"] == "mk.disk"
    mk.clear_failures()
    assert fs.Store("marketing").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mk.record_failure("mk.mem", ValueError("boom"))
    assert mk.failure_count() == 1
    assert [f.site for f in mk.recent_failures()] == ["mk.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mk.record_failure(f"mk.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mk.failure_count() == 400
    assert len(mk.recent_failures(1000)) == 200
    entries, total = mk._store.load()
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
        "from nomorals.marketing import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.marketing import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.marketing import failures as f\n"
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
    store_path = home / "failures" / "marketing.json"
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

def test_converted_summary_records_and_returns_brief_label(monkeypatch):
    from nomorals.marketing import briefs as mod
    monkeypatch.setattr(mod, "brief_quality",
                        lambda b: (_ for _ in ()).throw(RuntimeError("boom")))
    brief = mod.ContentBrief(topic="rent")
    assert brief.summary() == "📋 Brief"
    assert "briefs.ContentBrief.summary" in _sites()


def test_converted_extract_keywords_records_and_returns_empty():
    from nomorals.marketing import briefs as mod
    assert mod._extract_keywords(None, "rent") == []
    assert "briefs._extract_keywords" in _sites()


def test_converted_store_init_records_and_leaves_db_none(tmp_path):
    from nomorals.marketing.briefs import BriefStore
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    store = BriefStore(str(blocker / "briefs.db"))
    assert store._db is None
    assert "briefs.BriefStore.__init__" in _sites()


def test_converted_build_search_records_and_builds_topic_only():
    from nomorals.marketing.briefs import BriefStore

    def boom_search(topic):
        raise RuntimeError("search down")

    store = BriefStore(":memory:")
    brief = store.build_brief("lagos rent", search_fn=boom_search)
    assert brief is not None
    assert brief.topic == "lagos rent"
    assert "briefs.BriefStore.build_brief.search" in _sites()


def test_converted_build_has_bands_records_and_builds_without_bands():
    from nomorals.marketing.briefs import BriefStore
    real = BriefStore(":memory:")
    # Fresh tables carry term_bands, so the no-bands INSERT only
    # succeeds on the old schema: drop the column first, then
    # fail PRAGMA. The handler must record and continue with
    # has_bands=False, building the brief on the old schema.
    real._db.execute("ALTER TABLE briefs DROP COLUMN term_bands")
    real._db.commit()

    class PragmaFails:
        def __init__(self, conn):
            self._conn = conn

        def execute(self, sql, *args):
            if "PRAGMA" in sql:
                raise RuntimeError("pragma down")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self._conn, name)

    real._db = PragmaFails(real._db)
    brief = real.build_brief("lagos rent")
    assert brief is not None
    assert "briefs.BriefStore.build_brief.has_bands" in _sites()


def test_converted_brief_from_row_term_bands_records_and_defaults():
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")
    brief = store.build_brief("lagos rent")
    assert brief is not None
    store._db.execute("UPDATE briefs SET term_bands='{not json' "
                      "WHERE brief_id=?", (brief.brief_id,))
    store._db.commit()
    got = store.get_brief(brief.brief_id)
    assert got is not None and got.term_bands == {}
    assert "briefs.BriefStore._brief_from_row.term_bands" in _sites()


def test_fixed_score_virality_returns_live_score_not_neutral():
    # Import fix: the lazy `from ..social.voice import virality_score`
    # now resolves. It was `...social.voice` — beyond the top-level
    # package — so it always failed and brand_voice stuck at the 50.0
    # fallback. brand_voice is now the live heuristic score and nothing
    # is recorded at the virality site.
    from nomorals.marketing.briefs import BriefStore, ContentBrief
    from nomorals.social.voice import virality_score
    store = BriefStore(":memory:")
    brief = ContentBrief(topic="rent", keywords=["rent"])
    text = "Rent in Lagos explained here."
    scored = store.score_draft(text, brief)
    assert scored.brand_voice == round(
        virality_score(text, platform="x").score, 1)
    assert scored.brand_voice != 50.0
    assert "briefs.BriefStore.score_draft.virality" not in _sites()


def test_converted_save_draft_records_and_never_raises():
    from nomorals.marketing.briefs import BriefStore, BriefDraft
    store = BriefStore(":memory:")

    class ExecuteFails:
        def execute(self, *args, **kwargs):
            raise RuntimeError("db down")

        def commit(self):
            raise RuntimeError("db down")

    store._db = ExecuteFails()
    store._save_draft(BriefDraft(draft_id="bd_x", topic="rent"))
    assert "briefs.BriefStore._save_draft" in _sites()


def test_converted_schedule_queue_records_and_still_reports_ok():
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")
    brief = store.build_brief("lagos rent")
    draft = store.draft_from_brief(brief)
    assert draft is not None

    class QueueFails:
        def create_draft(self, *args, **kwargs):
            raise RuntimeError("queue down")

    res = store.schedule_draft(draft.draft_id, 9999999999.0,
                               queue=QueueFails())
    assert res["ok"] is True
    assert "briefs.BriefStore.schedule_draft.queue" in _sites()


def test_converted_schedule_scheduler_records_and_still_reports_ok():
    # Import fix: `from ..social.drafts import schedule_post` now resolves
    # (it was `...social.content_pipeline` — beyond the top-level package,
    # and content_pipeline never exported the name). With scheduler=object()
    # and no queue, the step now fails on queue.schedule (None has no
    # schedule) instead of on the import — still recorded, ok stays True,
    # and no ImportError is recorded anywhere.
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")
    brief = store.build_brief("lagos rent")
    draft = store.draft_from_brief(brief)
    assert draft is not None
    res = store.schedule_draft(draft.draft_id, 9999999999.0,
                               scheduler=object())
    assert res["ok"] is True
    assert "briefs.BriefStore.schedule_draft.scheduler" in _sites()
    assert all(f.exc_type != "ImportError" for f in mk.recent_failures())


def test_converted_stale_outcome_records_and_returns_empty():
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")
    brief = store.build_brief("lagos rent")
    draft = store.draft_from_brief(brief)
    assert draft is not None
    store._db.execute(
        "UPDATE drafts SET outcome_metrics='{not json', published_at=1.0 "
        "WHERE draft_id=?", (draft.draft_id,))
    store._db.commit()
    assert store.stale_drafts() == []
    assert "briefs.BriefStore.stale_drafts.outcome" in _sites()


def test_converted_control_brief_records_and_reports_hiccup():
    from nomorals.marketing.briefs import control_brief

    class StoreFails:
        def build_brief(self, *args, **kwargs):
            raise RuntimeError("store down")

    assert control_brief("lagos rent", store=StoreFails()) == \
        "brief pipeline hiccup — try again."
    assert "briefs.control_brief" in _sites()


def test_converted_parse_when_records_and_returns_none():
    from nomorals.marketing import briefs as mod
    assert mod._parse_when(123) is None
    assert "briefs._parse_when" in _sites()


def test_debug_honest_schedule_outer_records_nothing():
    # The outer handler returns the exception in its reason: the
    # cause already surfaces, so it stays unrecorded by design.
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")

    def boom_get(*args, **kwargs):
        raise RuntimeError("lookup down")

    store.get_draft = boom_get
    res = store.schedule_draft("bd_nope", 9999999999.0)
    assert res["ok"] is False and "lookup down" in res["reason"]
    assert "briefs.BriefStore.schedule_draft" not in _sites()


def test_debug_honest_run_outer_records_nothing():
    from nomorals.marketing.briefs import BriefStore
    store = BriefStore(":memory:")

    def boom_build(*args, **kwargs):
        raise RuntimeError("build down")

    store.build_brief = boom_build
    res = store.run("lagos rent")
    assert res["ok"] is False and "build down" in res["reason"]
    assert "briefs.BriefStore.run" not in _sites()
