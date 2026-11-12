"""The active source-candidate intake (mp3.pm lesson).

The registry only grew passively — ``note_candidate`` records
hosts already encountered on the real path, so a source nobody
ever pastes is never discovered.  This suite pins the active
intake: a named host drops into the SAME candidate queue with
provenance (who named it, when, what class), is verified by
the EXISTING ``verify_health`` before promotion, and is listed
through ``nm sources``.

All fetches are stubbed — no network.  Persistence uses a real
in-memory ``KVStore`` (the pass86 pattern), never a fixed path.
"""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

from nomorals.cli import _cli_command_help, _cli_overview, _parser
from nomorals.cmdline import _COMMANDS
from nomorals.cmdline.commands.sources import _cmd_sources

# ── helpers ────────────────────────────────────────────────────────────


def _kv():
    from nomorals.storage.db import Database
    from nomorals.storage.kv import KVStore

    db = Database(":memory:")
    db.migrate()
    return KVStore(db), db


def _seeded(store=None):
    from nomorals.media.source_registry import _seed

    return _seed(store=store)


def _ok_fetcher(url):
    return 200, b"<html><body>a real page with real content</body></html>"


def _dead_fetcher(url):
    return 404, b"gone"


def _ctx(reg):
    return SimpleNamespace(source_registry=reg, db=None,
                           settings=SimpleNamespace(home="/nonexistent"))


def _run(ctx, **fields):
    base = {"sources_action": "", "host": "", "source_class": "downloads",
            "json": False}
    base.update(fields)
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _cmd_sources(argparse.Namespace(**base), ctx)
    return rc, buf.getvalue()


def _add(ctx, host, source_class="downloads", json=False):
    return _run(ctx, sources_action="add", host=host,
                source_class=source_class, json=json)


def _candidates(ctx, json=False):
    return _run(ctx, sources_action="candidates", json=json)


# ── (a) registration ───────────────────────────────────────────────────


def test_parser_and_dispatch_register_sources():
    parser = _parser()
    args = parser.parse_args(["sources", "candidates"])
    assert args.command == "sources"
    assert args.sources_action == "candidates"
    args = parser.parse_args(["sources", "add", "mp3.pm",
                              "--class", "downloads"])
    assert args.sources_action == "add"
    assert args.host == "mp3.pm"
    assert args.source_class == "downloads"
    args = parser.parse_args(["sources", "add", "https://mp3.pm/x"])
    assert args.host == "https://mp3.pm/x"
    assert "sources" in _COMMANDS
    page = _cli_overview()
    assert "sources" in page
    help_page = _cli_command_help("sources")
    assert help_page is not None
    assert "usage: nm sources" in help_page


# ── (b) registry: a named candidate records provenance ────────────────


def test_named_candidate_records_provenance():
    reg = _seeded()
    assert reg.note_named_candidate("mp3.pm", named_by="owner",
                                    source_class="downloads") is True
    # untrusted: not a source, serves nothing
    assert reg.for_domain("mp3.pm") is None
    assert reg.for_url("https://mp3.pm/search?q=x") is None
    rows = reg.candidate_list()
    assert [r["domain"] for r in rows] == ["mp3.pm"]
    row = rows[0]
    assert row["named_by"] == "owner"
    assert row["class"] == "downloads"
    assert row["kinds"] == ["music", "film"]
    assert row["via"] == "named:owner"
    assert row["health"] == "unknown"
    assert row["seen"] == 1
    assert row["first_seen"] > 0
    assert row["last_error"] == ""
    # the status surface renders the SAME rows — one shape, no drift
    assert reg.status()["candidates"] == rows


def test_named_candidate_custom_class_kinds():
    reg = _seeded()
    assert reg.note_named_candidate("maps.example", named_by="owner",
                                    source_class="maps") is True
    assert reg.note_named_candidate("tunes.example", named_by="owner",
                                    source_class="music") is True
    rows = {r["domain"]: r for r in reg.candidate_list()}
    assert rows["maps.example"]["kinds"] == ["maps"]
    assert rows["maps.example"]["class"] == "maps"
    assert rows["tunes.example"]["kinds"] == ["music"]
    assert rows["tunes.example"]["class"] == "music"


def test_named_candidate_refusals():
    reg = _seeded()
    assert reg.note_named_candidate("mp3.pm") is True
    assert reg.note_named_candidate("mp3.pm") is False  # duplicate
    # a seed is already a source — never a candidate
    assert reg.note_named_candidate("youtube.com") is False
    assert reg.note_named_candidate("not-a-host") is False
    assert reg.note_named_candidate("123.456") is False
    assert reg.note_named_candidate("mp3.pm",
                                    source_class="not a class") is False
    assert [r["domain"] for r in reg.candidate_list()] == ["mp3.pm"]


def test_passive_then_named_upgrades_provenance():
    reg = _seeded()
    assert reg.note_candidate("https://mp3.pm/search",
                              via="download path") is True
    assert reg.note_named_candidate("mp3.pm", named_by="owner",
                                    source_class="downloads") is False
    rows = reg.candidate_list()
    assert len(rows) == 1
    assert rows[0]["named_by"] == "owner"
    assert rows[0]["class"] == "downloads"
    assert rows[0]["seen"] == 2
    assert rows[0]["via"] == "download path"  # first discovery stands


def test_named_candidate_persists_and_reloads_with_provenance():
    kv, _db = _kv()
    reg = _seeded(store=kv)
    assert reg.note_named_candidate("mp3.pm", named_by="owner",
                                    source_class="downloads") is True
    reg2 = _seeded(store=kv)
    rows = reg2.candidate_list()
    assert [r["domain"] for r in rows] == ["mp3.pm"]
    row = rows[0]
    assert row["named_by"] == "owner"
    assert row["class"] == "downloads"
    assert row["via"] == "named:owner"
    assert row["seen"] == 1
    assert reg2.for_domain("mp3.pm") is None  # still untrusted


# ── (c) verification: the EXISTING health loop decides ────────────────


def test_unhealthy_candidate_stays_unpromoted_with_reason():
    reg = _seeded()
    reg.note_named_candidate("deadtunes.example", named_by="owner",
                             source_class="downloads")
    summary = reg.verify_health(fetcher=_dead_fetcher)
    assert summary["checked"] >= 1
    assert summary["promoted"] == 0
    # still a candidate, never silently promoted or dropped
    rows = reg.candidate_list()
    assert [r["domain"] for r in rows] == ["deadtunes.example"]
    assert rows[0]["health"] == "dead"
    assert rows[0]["last_error"] == "HTTP 404"
    assert reg.for_domain("deadtunes.example") is None
    # a second failed verification keeps it a candidate too
    reg.verify_health(fetcher=_dead_fetcher)
    assert [r["domain"] for r in reg.candidate_list()] == [
        "deadtunes.example"]


def test_healthy_candidate_promotes_through_verify_health():
    reg = _seeded()
    reg.note_named_candidate("mp3.pm", named_by="owner",
                             source_class="downloads")
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1, summary
    assert reg.candidate_list() == []
    desc = reg.for_domain("mp3.pm")
    assert desc is not None and desc.domain == "mp3.pm"
    assert reg.origin_for(desc) == "grown"
    assert reg.health_for(desc) == "verified-live"


def test_failed_candidate_is_shed_after_three_and_named_plainly():
    kv, _db = _kv()
    reg = _seeded(store=kv)
    reg.note_named_candidate("deadtunes.example", named_by="owner",
                             source_class="downloads")
    for _ in range(3):
        reg.verify_health(fetcher=_dead_fetcher)
    assert reg.candidate_list() == []
    assert reg.for_domain("deadtunes.example") is None
    shed = reg.status()["shed_list"]
    assert "deadtunes.example" in [s["domain"] for s in shed]


# ── (d) CLI: add → list with provenance ───────────────────────────────


def test_cli_add_then_candidates_lists_provenance():
    reg = _seeded()
    ctx = _ctx(reg)
    rc, out = _add(ctx, "mp3.pm")
    assert rc == 0, out
    assert "candidate recorded: mp3.pm" in out
    assert "class downloads" in out
    assert "named by owner" in out
    assert "untrusted until the health loop verifies it" in out
    rc, out = _candidates(ctx)
    assert rc == 0, out
    assert "mp3.pm [candidate/unknown]" in out
    assert "class=downloads" in out
    assert "named by owner" in out
    assert "via named:owner" in out
    rc, out = _candidates(ctx, json=True)
    assert rc == 0, out
    payload = json.loads(out)
    assert payload["count"] == 1
    row = payload["candidates"][0]
    assert row["domain"] == "mp3.pm"
    assert row["named_by"] == "owner"
    assert row["class"] == "downloads"


def test_cli_add_twice_is_idempotent():
    reg = _seeded()
    ctx = _ctx(reg)
    assert _add(ctx, "mp3.pm")[0] == 0
    rc, out = _add(ctx, "https://mp3.pm/search?q=x")
    assert rc == 0, out
    assert "already a candidate: mp3.pm" in out
    assert len(reg.candidate_list()) == 1


def test_cli_add_invalid_host_fails():
    reg = _seeded()
    ctx = _ctx(reg)
    rc, out = _add(ctx, "not-a-host")
    assert rc == 2
    assert reg.candidate_list() == []
    rc, out = _add(ctx, "123.456")
    assert rc == 2


def test_cli_add_known_seed_is_refused_plainly():
    reg = _seeded()
    ctx = _ctx(reg)
    rc, out = _add(ctx, "youtube.com")
    assert rc == 0, out
    assert "already a registered source: youtube.com" in out
    assert reg.candidate_list() == []


def test_cli_add_shed_window_is_named():
    reg = _seeded()
    ctx = _ctx(reg)
    assert _add(ctx, "deadtunes.example")[0] == 0
    for _ in range(3):
        reg.verify_health(fetcher=_dead_fetcher)
    rc, out = _add(ctx, "deadtunes.example")
    assert rc == 0, out
    assert "shed" in out
    assert "re-intake opens 7 days after the shed" in out


def test_cli_add_then_verify_promotes():
    reg = _seeded()
    ctx = _ctx(reg)
    assert _add(ctx, "mp3.pm")[0] == 0
    summary = reg.verify_health(fetcher=_ok_fetcher)
    assert summary["promoted"] == 1
    rc, out = _candidates(ctx)
    assert rc == 0, out
    assert "no candidates pending" in out
    assert reg.for_domain("mp3.pm") is not None


def test_cli_candidates_empty_names_the_door():
    reg = _seeded()
    ctx = _ctx(reg)
    rc, out = _candidates(ctx)
    assert rc == 0, out
    assert "no candidates pending" in out
    assert "nm sources add <host>" in out


def test_cli_persisted_flag_follows_the_store():
    kv, _db = _kv()
    reg = _seeded(store=kv)
    ctx = _ctx(reg)
    assert _add(ctx, "mp3.pm")[0] == 0
    rc, out = _candidates(ctx)
    assert rc == 0, out
    assert "registry: persisted" in out
    reg2 = _seeded(store=kv)
    ctx2 = _ctx(reg2)
    rc, out = _candidates(ctx2)
    assert rc == 0, out
    assert "mp3.pm [candidate/unknown]" in out
