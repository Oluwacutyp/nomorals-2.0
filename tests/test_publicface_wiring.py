"""Public face, wired into production (E-6).

MORE_TO_ADD gap 10: the gate (nomorals/publicface/publish.py) existed
and was unit-pinned (tests/test_more_gaps710.py) but nothing in
production reached it durably — publish_episode in shows/composer.py
is its only production importer and itself has no production callers
(only the re-export in shows/__init__.py), there is no durable store,
no chat command, no tool, and no route serves /p/<slug>. This lane
wires the owner-approved surface: draft a page, approve it through
the gate (private-data scan + owner approval, both required), list
what is public, pull a page back, remove a draft.

Every test here drives the PRODUCTION wiring: the durable
PublicPageStore and the /publicface control — never the dormant gate
in isolation.

Teeth: on the pristine base each test fails for the missing wiring
(PublicPage has no to_dict, no PublicPageStore, no _control_publicface,
no "publicface" command kind).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from nomorals.publicface.publish import (
    PublicPage,
    PublicPageStore,
)
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    help_text,
)


def _reach_host(tmp_path):
    """Production control host on a throwaway home (decisions/focus harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _draft_page(host, pack="jamb-tutor", title="JAMB Physics 101",
                content="Forces and motion."):
    out = host._control_publicface(f"draft {pack} | {title} | {content}")
    assert out.startswith("draft recorded: "), out
    return out


# ── durable store ───────────────────────────────────────────────────


def test_page_dict_round_trip():
    page = PublicPage(title="Oral Corpus Tale", content="Once upon a time...",
                      pack="oral-corpus", created=date(2026, 11, 15))
    page.status = "public"
    page.owner_approved = True
    assert PublicPage.from_dict(page.to_dict()) == page


def test_page_from_dict_refuses_bad_rows():
    try:
        PublicPage.from_dict({"title": "T", "content": "C", "pack": "p",
                              "status": "live"})
    except ValueError as exc:
        assert "status" in str(exc)
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("an unknown status must refuse to load")
    try:
        PublicPage.from_dict({"content": "C", "pack": "p"})
    except (KeyError, TypeError, ValueError):
        pass
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("a row without a title must refuse to load")


def test_store_persists_across_instances(tmp_path):
    path = str(tmp_path / "publicface.json")
    store = PublicPageStore(path)
    assert store.pages == {}
    draft = PublicPage(title="JAMB Physics 101", content="Forces.",
                       pack="jamb-tutor")
    store.add(draft)
    assert store.get(draft.id) is draft

    fresh = PublicPageStore(path)
    got = fresh.get(draft.id)
    assert got is not None and got.title == "JAMB Physics 101"
    assert got.status == "draft" and got.owner_approved is False
    assert got.created == draft.created


def test_store_tolerant_load(tmp_path):
    missing = PublicPageStore(str(tmp_path / "nope.json"))
    assert missing.pages == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert PublicPageStore(str(corrupt)).pages == {}

    mixed = tmp_path / "mixed.json"
    good = PublicPage(title="Good", content="kept", pack="jamb-tutor")
    mixed.write_text(json.dumps({
        "pages": [good.to_dict(),
                  {"title": "Bad", "content": "x", "pack": "p",
                   "status": "live"},
                  {"content": "no title", "pack": "p"}],
    }), encoding="utf-8")
    loaded = PublicPageStore(str(mixed))
    assert list(loaded.pages) == [good.id]
    assert loaded.get(good.id).title == "Good"


# ── /publicface control ─────────────────────────────────────────────


def test_publicface_empty_list(tmp_path):
    host = _reach_host(tmp_path)
    out = host._control_publicface("list")
    assert "no pages yet" in out
    assert "/publicface draft" in out


def test_publicface_draft_list_show(tmp_path):
    host = _reach_host(tmp_path)
    _draft_page(host)
    listed = host._control_publicface("list")
    assert "JAMB Physics 101" in listed and "draft" in listed
    shown = host._control_publicface("show " + _page_id(host))
    assert "JAMB Physics 101" in shown
    assert "jamb-tutor" in shown and "draft" in shown
    assert "Forces and motion." in shown
    assert "usage:" in host._control_publicface("draft jamb-tutor | only-title")
    assert "no page matching" in host._control_publicface("show nope")


def _page_id(host, title="JAMB Physics 101"):
    listed = host._control_publicface("list")
    for line in listed.splitlines():
        if title in line:
            return line.strip().split(":")[0]
    raise AssertionError("draft page missing from list:\n" + listed)


def test_publicface_approve_clean_publishes(tmp_path):
    host = _reach_host(tmp_path)
    _draft_page(host)
    page_id = _page_id(host)
    out = host._control_publicface(f"approve {page_id}")
    assert "published" in out and "/p/jamb-physics-101" in out
    assert "No route serves" in out
    listed = host._control_publicface("list")
    assert "public" in listed
    fresh = PublicPageStore(str(
        Path(tmp_path) / "data" / "publicface.json"))
    assert fresh.get(page_id).status == "public"


def test_publicface_approve_leaking_stays_draft(tmp_path):
    host = _reach_host(tmp_path)
    _draft_page(host, title="Yoruba Songs",
                content="token = 'secret-thing-here'")
    page_id = _page_id(host, "Yoruba Songs")
    out = host._control_publicface(f"approve {page_id}")
    assert "left as draft" in out
    assert "token" in out
    listed = host._control_publicface("list")
    assert "draft" in listed and "public" not in listed


def test_publicface_unpublish_and_remove(tmp_path):
    host = _reach_host(tmp_path)
    _draft_page(host)
    page_id = _page_id(host)
    assert "draft" in host._control_publicface(f"unpublish {page_id}")
    host._control_publicface(f"approve {page_id}")
    assert "public" in host._control_publicface("list")
    assert "cannot remove" in host._control_publicface(f"remove {page_id}")
    assert "back to draft" in host._control_publicface(f"unpublish {page_id}")
    assert "removed" in host._control_publicface(f"remove {page_id}")
    assert "no pages yet" in host._control_publicface("list")


def test_publicface_unknown_verb_and_bad_id(tmp_path):
    host = _reach_host(tmp_path)
    assert "usage:" in host._control_publicface("frobnicate")
    assert "no page matching" in host._control_publicface("approve nope")
    assert "no page matching" in host._control_publicface("unpublish nope")
    assert "no page matching" in host._control_publicface("remove nope")


# ── registry ────────────────────────────────────────────────────────


def test_publicface_registered():
    assert "publicface" in CONTROL_COMMANDS
    assert "publicface" not in _public_command_kinds()
    slots = [title for title, kinds in LIST_GROUPS if "publicface" in kinds]
    assert slots == ["memory & thinking"]
    assert "publicface" in LIST_ONELINERS
    assert "/publicface" in LIST_ONELINERS["publicface"]
    assert "/publicface" in help_text()


def _public_command_kinds():
    from nomorals.social.chat.control import PUBLIC_COMMANDS

    return PUBLIC_COMMANDS
