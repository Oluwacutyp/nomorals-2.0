"""/pair journal — the personal vault's journal, readable in
chat (2026-11-07).  Until this subcommand the journal existed
only as a file on disk: the rows that NAME surface breaks (the
status_view_trace rows, the refused-verb entries) were
invisible to the owner they exist for.  These pins drive the
real /pair handler against a seeded vault and pin the reply
shapes: fields in words (never a raw JSON dump), newest first,
kind filter, an honest shown-of-total count, unknown-kind
discoverability, and an honest empty answer."""

from types import SimpleNamespace

from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
from nomorals.social.pairing import AccountRecord, PairingRegistry
from nomorals.social.personal import PersonalSurfaces, PersonalVault, _append_jsonl

ACC = "whatsapp:2348012345678"
OTHER = "telegram:555000111"
T0 = 1762000000.0


def _runtime(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id=ACC, platform="whatsapp", identity="2348012345678",
        status="linked", created_at=1.0))
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = SimpleNamespace(registry=registry)
    rt._pair_personal_svc = {"registry": registry,
                             "personal": personal}
    rt.context = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    return rt, personal.vault


def _row(kind, ts, **kw):
    row = {"ts": T0 + ts, "kind": kind, "account": ACC,
           "platform": "whatsapp", "counterpart_id": "",
           "poster_name": "", "media_kind": "", "source_id": "",
           "path": "", "bytes": 0, "note": ""}
    row.update(kw)
    return row


def _trace(ts, poster, toggle, attempted, verdict):
    return _row(
        "status_view_trace", ts, poster_name=poster,
        counterpart_id="111@s.whatsapp.net",
        status_viewing_toggle=toggle, view_attempted=attempted,
        bridge_verdict=verdict, seam_poster_id="111@s.whatsapp.net",
        seam_source_id="st1", seam_ts=T0 + ts,
        note="status view trace — the exact values the view "
             "chain carried for this post")


def _seeded(tmp_path):
    rt, vault = _runtime(tmp_path)
    rows = [
        _row("status_react_failed", 100, poster_name="Dara",
             note="status_react verb did not land — the bridge "
                  "refused it or is unreachable; no reaction was "
                  "placed"),
        _trace(200, "Chidi", False, False, None),
        _trace(300, "Ada", True, True, {"ok": True, "error": ""}),
        _trace(400, "Bola", True, True,
               {"ok": False, "error": "bridge exploded"}),
        _row("status_react", 500, poster_name="Ada",
             note="reacted 🔥"),
        _row("channel_posted", 600, poster_name="Gold Room",
             counterpart_id="120363@g.us", media_kind="text",
             text="evening levels",
             note="posted as the owner via /chanstory"),
    ]
    for row in rows:
        _append_jsonl(vault.journal_path, row)
    return rt, vault


def test_journal_default_newest_first_across_kinds(tmp_path):
    rt, _vault = _seeded(tmp_path)
    out = rt._control_pair("journal")
    assert out.startswith("📓 journal (latest 6 of 6):")
    # newest first: channel post (600) → react (500) → refused
    # trace (400) → landed trace (300) → idle trace (200) →
    # react failure (100)
    order = [out.index("Gold Room"), out.index("reacted 🔥"),
             out.index("Bola"), out.index("bridge: landed"),
             out.index("Chidi"), out.index("Dara")]
    assert order == sorted(order)


def test_journal_trace_rows_render_fields_in_words(tmp_path):
    rt, _vault = _seeded(tmp_path)
    out = rt._control_pair("journal status_view_trace")
    assert out.startswith(
        "📓 journal · status_view_trace (latest 3 of 3):")
    bola = next(ln for ln in out.splitlines() if "Bola" in ln)
    assert "status view trace" in bola
    assert "status_viewing on" in bola
    assert "attempted yes" in bola
    assert "bridge: refused — bridge exploded" in bola
    ada = next(ln for ln in out.splitlines() if "Ada" in ln)
    assert "bridge: landed" in ada
    chidi = next(ln for ln in out.splitlines() if "Chidi" in ln)
    assert "status_viewing off" in chidi
    assert "attempted no" in chidi
    assert "view not attempted" in chidi
    # never a raw JSON dump of the row
    assert "bridge_verdict" not in out
    assert '{"ok"' not in out
    assert "seam_poster_id" not in out


def test_journal_kind_filter_and_count_cap_are_honest(tmp_path):
    rt, vault = _runtime(tmp_path)
    for i in range(12):
        _append_jsonl(vault.journal_path, _row(
            "status_react", 100 + i, poster_name=f"P{i}",
            note="reacted ❤️"))
    out = rt._control_pair("journal status_react 5")
    assert "(latest 5 of 12)" in out
    assert out.count("reacted ❤️") == 5
    out = rt._control_pair("journal status_react")
    assert "(latest 10 of 12)" in out  # the default count
    out = rt._control_pair("journal 3 status_react")
    assert "(latest 3 of 12)" in out  # arg order does not matter


def test_journal_unknown_kind_lists_present_kinds(tmp_path):
    rt, _vault = _seeded(tmp_path)
    out = rt._control_pair("journal bogus_kind")
    assert "no bogus_kind entries in the journal yet" in out
    assert "status_view_trace (3)" in out
    assert "status_react (1)" in out
    assert "channel_posted (1)" in out


def test_journal_empty_answers_honestly(tmp_path):
    rt, _vault = _runtime(tmp_path)
    out = rt._control_pair("journal")
    assert "the journal is empty" in out
    out = rt._control_pair("journal status_view_trace")
    assert "the journal is empty" in out


def test_journal_account_scope(tmp_path):
    rt, vault = _seeded(tmp_path)
    _append_jsonl(vault.journal_path, _row(
        "status_react", 700, account=OTHER, platform="telegram",
        poster_name="Zed", note="reacted 👍"))
    out = rt._control_pair("journal")
    assert "Zed" in out  # unscoped: every account's rows
    out = rt._control_pair("journal 2348012345678")
    assert out.startswith(f"📓 journal — {ACC} (latest 6 of 6):")
    assert "Zed" not in out


def test_journal_failure_rows_show_the_reason(tmp_path):
    rt, _vault = _seeded(tmp_path)
    out = rt._control_pair("journal status_react_failed")
    assert "did not land — the bridge refused it" in out
    out = rt._control_pair("journal channel_posted")
    assert "Gold Room" in out
    assert "posted as the owner via /chanstory" in out


def test_journal_registered_in_usage_and_catalog():
    from nomorals.social.chat.control import COMMAND_DETAILS

    rt = object.__new__(RuntimeProvisionMixin)
    usage = rt._control_pair.__doc__ or ""
    assert "/pair journal [kind] [count]" in usage
    assert "journal [kind] [count]" in \
        COMMAND_DETAILS["pair"]["usage"]
