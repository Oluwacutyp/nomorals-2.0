"""Dormant product-surface organs, wired into their production paths.

Assembly audit 2026-11-05, findings AA-02, AA-19, AA-25, AA-31,
AA-33, AA-37, AA-38: seven organs that existed in code but no
production surface ever called. Every test here drives the
PRODUCTION entry point (the /focus, /learn, /legal contract,
/iddocs, /scamcheck and /health chat controls, and the books tool
registry) — never the dormant organ in isolation.

Teeth: on the pristine tree each test fails for the missing wiring
(the focus builder's own sections instead of the composer's, the
JAMB product verbs falling through to subject lookup, no precedent
citations on a contract review, /iddocs unregistered, no general
signals section on /scamcheck, the publish-extension tools absent
from the books registry, the meal-photo facade uncalled and the
costs answer unstamped).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _reach_host(tmp_path):
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _memory_host():
    from nomorals.agents.partner.runtime_memory import RuntimeMemoryMixin

    return RuntimeMemoryMixin()


def _ctx(tmp: str):
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings)


# ── AA-02: the periodic-review composer serves /focus week|month ────


def test_focus_week_uses_the_review_composer(tmp_path):
    host = _reach_host(tmp_path)
    host._control_decisions("add Ship it | ship | it lands | 60%")
    out = host._control_focus("week")
    # The composer's own sections (the focus builder renders
    # Wins/Open commitments/Decisions instead).
    assert "# Weekly Review" in out
    assert "## What slipped" in out
    assert "## What's blocked" in out
    # …built on the same real data: the open decision blocks the week.
    assert "Ship it" in out


def test_focus_month_aggregates_weekly_reviews(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_focus("month")
    assert "# Monthly Review" in out
    assert "## Reliability trend" in out or "## What happened" in out


# ── AA-19: the JAMB product pack rides /learn jamb ───────────────────


def test_learn_jamb_packs_surface():
    from nomorals.learn.chat import control_learn

    out = control_learn("jamb packs")
    assert "exam / curriculum packs" in out
    assert "JAMB" in out and "SAT" in out
    assert "descriptor_only" in out  # honest: unmapped packs say so


def test_learn_jamb_bank_and_topics_surface():
    from nomorals.learn.chat import control_learn

    bank = control_learn("jamb bank")
    assert "misconception bank:" in bank
    topics = control_learn("jamb topics Mathematics")
    assert "syllabus topics" in topics
    ekiti = control_learn("jamb ekiti")
    assert "Ekiti" in ekiti


def test_learn_jamb_subject_path_still_works():
    # Preserved behaviour: a bare subject still lists its JAMB topics.
    from nomorals.learn.chat import control_learn

    out = control_learn("jamb Mathematics")
    assert "JAMB" in out and "topics" in out


# ── AA-25: contract review runs through the legal product layer ─────

_CONTRACT = (
    "FREELANCE SERVICE AGREEMENT. The Client shall pay the Freelancer "
    "within 90 days of invoice. All intellectual property created under "
    "this agreement belongs entirely to the Client. The Freelancer may "
    "not work for any other client during the term. Liability is "
    "unlimited. This agreement is governed by the laws of Nigeria."
)


def test_contract_review_carries_product_citations():
    from nomorals.legal.chat import control_legal_full

    out = control_legal_full(f"contract review freelance {_CONTRACT}")
    # The product layer stamps disclaimer + precedent citations; the
    # bypassed path (contracts.format_review alone) never did.
    assert "Citations:" in out
    assert "DoNotPay FTC $193K" in out
    assert "not legal advice" in out.lower()


def test_contract_control_also_routes_through_product():
    from nomorals.legal.contracts import control_contract

    out = control_contract(f"review freelance {_CONTRACT}")
    assert "DoNotPay FTC $193K" in out


# ── AA-31: identity document tracker has a production surface ───────


def _iddocs_ctx(tmp_path):
    return SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))


def test_iddocs_round_trip(tmp_path):
    from nomorals.identity.chat import control_iddocs

    ctx = _iddocs_ctx(tmp_path)
    added = control_iddocs(
        "add passport 2020-01-01 vault:passport-ng-1 Nigeria", context=ctx)
    assert added.startswith("tracked")
    # The tracker persisted at the production home path.
    assert (tmp_path / "data" / "identity_documents.sqlite").exists()
    listing = control_iddocs("list", context=ctx)
    assert "passport" in listing and "EXPIRED" in listing
    assert "vault:passport-ng-1" in listing
    soon = control_iddocs("expiring 4000", context=ctx)
    assert "passport" in soon
    overdue = control_iddocs("overdue", context=ctx)
    assert "passport" in overdue
    doc_line = next(line for line in listing.splitlines()
                    if "passport" in line)
    doc_id = doc_line.strip().split(":", 1)[0]
    renew = control_iddocs(f"renew {doc_id}", context=ctx)
    assert "renewal" in renew and "Confirm expiry" in renew


def test_iddocs_rejects_raw_document_numbers(tmp_path):
    from nomorals.identity.chat import control_iddocs

    ctx = _iddocs_ctx(tmp_path)
    out = control_iddocs("add passport 2030-01-01 A01234567", context=ctx)
    assert "couldn't track" in out
    assert "vault" in out  # the refusal explains the reference shape
    assert control_iddocs("list", context=ctx).startswith(
        "no identity documents")


def test_iddocs_registered_in_the_catalog():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS, LIST_GROUPS, LIST_ONELINERS)

    assert "iddocs" in COMMAND_DETAILS
    assert "iddocs" in LIST_ONELINERS
    grouped = {kind for _title, kinds in LIST_GROUPS for kind in kinds}
    assert "iddocs" in grouped


# ── AA-33: general scam signals ride /scamcheck ──────────────────────

_SCAMMY = (
    "Unbelievably cheap flat, far below market price! Act now, "
    "limited time only! Send money upfront and pay before viewing "
    "to secure it."
)


def test_scamcheck_includes_general_signals():
    from nomorals.property.scam import control_scamcheck

    out = control_scamcheck(_SCAMMY)
    assert "General scam signals:" in out
    assert "suspicious" in out
    assert "advance_fee" in out


def test_scamcheck_general_signals_fail_honestly(monkeypatch):
    import nomorals.scam.signals as signals
    from nomorals.property.scam import control_scamcheck

    def _boom(*a, **k):
        raise RuntimeError("scorer down")

    monkeypatch.setattr(signals, "assess", _boom)
    out = control_scamcheck(_SCAMMY)
    # An unscored text is "not checked" — never dressed as safe.
    assert "General scam signals: not checked" in out


# ── AA-37: publish extension serves the books publish path ──────────


def _books_tools(tmp_path):
    from nomorals.books.tools import register as books_register

    seen = {}

    class FakeRegistry:
        def __init__(self, context):
            self.context = context

        def register(self, name, description="", capability=None,
                     parameters=None):
            def deco(fn):
                seen[name] = fn
                return fn
            return deco

    books_register(FakeRegistry(_ctx(str(tmp_path))))
    return seen


def test_books_registry_carries_publish_extension(tmp_path):
    seen = _books_tools(tmp_path)
    for name in ("book_cover_brief", "publish_arc", "book_series"):
        assert name in seen


def test_book_cover_brief_from_real_book(tmp_path):
    from nomorals.books.forge import BookForge

    seen = _books_tools(tmp_path)
    book = BookForge(_ctx(str(tmp_path))).create(
        "a tale of two markets", title="Two Markets", chapters=1,
        author="A. Writer", genre="fantasy", research=False)
    out = seen["book_cover_brief"](book.slug, culture_pack="nigeria")
    # No image generator in the registry path: brief + prompt, honestly.
    assert out["ok"] is False
    assert out["brief"]["title"] == "Two Markets"
    assert "#0B6E4F" in out["brief"]["palette"]  # nigeria pack palette
    assert "Two Markets" in out["prompt"]


def test_publish_arc_and_series_round_trip(tmp_path):
    seen = _books_tools(tmp_path)
    arc = seen["publish_arc"]
    assert arc("my-book", action="add_reader", reader="r1")["ok"]
    fb = arc("my-book", action="feedback", reader="r1", chapter=1,
             feedback="loved it", rating=5)
    assert fb["ok"]
    stats = arc("my-book", action="stats")
    assert stats["readers"] == 1 and stats["avg_rating"] == 5

    series = seen["book_series"]
    assert series("Chronicles", action="add", volume=1,
                  slug="my-book")["ok"]
    assert series("Chronicles", action="next")["next"]["volume"] == 1
    assert series("Chronicles", action="release", volume=1)["ok"]
    assert series("Chronicles", action="next")["next"] is None
    order = series("Chronicles", action="order")["volumes"]
    assert order[0]["book"] == "my-book" and order[0]["released"]


def test_publish_release_builds_notification(tmp_path):
    seen = _books_tools(tmp_path)
    story = Path(tmp_path) / "books" / "fiction" / "my-story"
    (story / "chapters").mkdir(parents=True)
    (story / "chapters" / "0001.md").write_text(
        "Chapter one text here.", encoding="utf-8")
    started = seen["publish_start"]("my-story", cadence="manual",
                                    title="My Story")
    assert started["ok"]
    seen["publish_follow"]("my-story", "reader-1")
    out = seen["publish_release"]("my-story")
    assert out["ok"] and out["chapter"] == 1
    note = out["notification"]
    assert note["chapter"] == 1
    assert note["followers"] == ["reader-1"]
    assert "My Story" in note["message"]


# ── AA-38: the health surface points at the expose facade ───────────


def test_health_costs_come_through_the_facade(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = _memory_host()._control_health("costs", chat_key="t")
    # price_transparency() stamps the tracking disclaimer onto the
    # previsit costs table; the direct formatter never did.
    assert "Tracking / coaching only" in out
    assert "not a doctor" in out


def test_health_meal_photo_path_uses_the_facade(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    import nomorals.health.expose as expose

    def _fake_log(photo_path, *, seer=None):
        return {
            "ok": True, "error": "",
            "items": [{"name": "jollof rice", "portion_hint": "1 plate",
                       "cal_low": 450, "cal_high": 650, "known": True,
                       "oily": True}],
            "photo_estimate": {"low": 450, "high": 650,
                               "text": "approximately 450-650 cal"},
            "undercount_adjusted": {
                "low": 672, "high": 970,
                "text": "approximately 672-970 cal after the ~33% "
                        "photo-undercount correction"},
            "follow_up_questions": ["was it cooked with much oil?"],
            "note": "Photos alone undercount by ~33%.",
        }

    monkeypatch.setattr(expose, "log_meal_photo", _fake_log)
    out = _memory_host()._control_health("meal /photos/lunch.jpg",
                                         chat_key="t")
    assert "jollof rice" in out
    assert "672-970" in out  # the adjusted range reaches the owner
    assert "much oil" in out


def test_health_meal_photo_failure_is_honest(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = _memory_host()._control_health(
        "meal /nonexistent/definitely-not-a-photo.jpg", chat_key="t")
    # The real facade reports the unreadable photo — it does NOT
    # silently arm the conversational flow instead.
    assert "couldn't log" in out
