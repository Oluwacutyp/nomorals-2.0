"""Offline tests for nomorals/social/profiles.py (build-map #81)."""

import os
import tempfile

import pytest

from nomorals.social.profiles import (
    PROMPTS, ELEMENT_TYPES, SURFACES,
    ProfileStore, control_uprofile, format_profile, suggested_prompts,
)


@pytest.fixture
def store():
    return ProfileStore(db_path=os.path.join(tempfile.mkdtemp(), "p.db"))


# ── creation ────────────────────────────────────────────────────────────

def test_create_all_surfaces(store):
    for surface in SURFACES:
        p = store.create("Ada", surface, f"Ada {surface}")
        assert p is not None
        assert p.surface == surface
        assert p.profile_id.startswith("prof_")


def test_create_bad_surface_defaults_gig(store):
    p = store.create("Ada", "nonsense")
    assert p.surface == "gig"


def test_get_missing_returns_none(store):
    assert store.get("prof_nope") is None


def test_list_filter(store):
    store.create("A", "gig")
    store.create("B", "community")
    assert len(store.list()) == 2
    assert len(store.list("gig")) == 1


# ── prompts ─────────────────────────────────────────────────────────────

def test_answer_prompt(store):
    p = store.create("Ada", "gig")
    assert store.answer_prompt(p.profile_id, "best_work",
                               "redesigned a bank app") is True
    got = store.get(p.profile_id)
    assert got.prompts["best_work"] == "redesigned a bank app"


def test_answer_invalid_prompt_rejected(store):
    p = store.create("Ada", "gig")
    assert store.answer_prompt(p.profile_id, "not_a_prompt", "x") is False


def test_unanswered_nudges(store):
    p = store.create("Ada", "community")
    left = store.unanswered(p.profile_id)
    assert len(left) == len(PROMPTS["community"])
    store.answer_prompt(p.profile_id, "intro", "hi")
    assert len(store.unanswered(p.profile_id)) == len(left) - 1


def test_suggested_prompts_skips_answered():
    got = suggested_prompts("gig", {"best_work"})
    assert all(pid != "best_work" for pid, _ in got)


# ── elements ────────────────────────────────────────────────────────────

def test_add_element(store):
    p = store.create("Ada", "gig")
    el = store.add_element(p.profile_id, "project", "Bank redesign",
                           "3 months, Figma")
    assert el is not None
    assert el.element_id.startswith("elem_")
    got = store.get(p.profile_id)
    assert len(got.elements) == 1
    assert got.elements[0].title == "Bank redesign"


def test_add_element_bad_type_defaults(store):
    p = store.create("Ada", "gig")
    el = store.add_element(p.profile_id, "hologram", "X")
    assert el.type == "project"


def test_add_element_missing_profile(store):
    assert store.add_element("prof_nope", "project", "X") is None


def test_element_types_covered():
    assert set(ELEMENT_TYPES) >= {"project", "photo", "answer"}


# ── likes ───────────────────────────────────────────────────────────────

def test_like_element(store):
    p = store.create("Ada", "gig")
    el = store.add_element(p.profile_id, "project", "Bank redesign")
    lk = store.like(p.profile_id, el.element_id, "Emeka",
                    "love the onboarding flow!")
    assert lk is not None
    assert lk.comment == "love the onboarding flow!"


def test_like_idempotent(store):
    p = store.create("Ada", "gig")
    el = store.add_element(p.profile_id, "project", "X")
    store.like(p.profile_id, el.element_id, "Emeka", "first")
    store.like(p.profile_id, el.element_id, "Emeka", "second")
    likes = store.likes_for(el.element_id)
    assert len(likes) == 1
    assert likes[0].comment == "first"


def test_like_missing_element(store):
    p = store.create("Ada", "gig")
    assert store.like(p.profile_id, "elem_nope", "Emeka") is None


def test_like_count_on_profile(store):
    p = store.create("Ada", "gig")
    el = store.add_element(p.profile_id, "project", "X")
    store.like(p.profile_id, el.element_id, "Emeka")
    store.like(p.profile_id, el.element_id, "Mama")
    got = store.get(p.profile_id)
    assert got.elements[0].like_count == 2


def test_matches_inbox(store):
    # Ada's element liked by two people → her match inbox.
    ada = store.create("Ada", "gig")
    bob = store.create("Bob", "gig")
    el = store.add_element(ada.profile_id, "project", "X")
    store.like(ada.profile_id, el.element_id, "Emeka", "nice work!")
    store.like(ada.profile_id, el.element_id, "Mama")
    inbox = store.matches("Ada")
    assert len(inbox) == 2
    assert any(lk.comment == "nice work!" for lk in inbox)
    # Bob sees nothing — isolation.
    assert store.matches("Bob") == []


# ── display ─────────────────────────────────────────────────────────────

def test_format_profile(store):
    p = store.create("Ada", "gig", "Ada D.")
    store.answer_prompt(p.profile_id, "best_work", "bank app")
    el = store.add_element(p.profile_id, "project", "Bank redesign")
    store.like(p.profile_id, el.element_id, "Emeka")
    got = store.get(p.profile_id)
    text = format_profile(got)
    assert "Ada D." in text
    assert "bank app" in text
    assert "Bank redesign" in text
    assert "❤️×1" in text
    assert "Unanswered" in text  # still has prompts left


# ── chat control ────────────────────────────────────────────────────────

def test_control_help():
    out = control_uprofile("help")
    assert "/uprofile create" in out


def test_control_create_and_show():
    home = tempfile.mkdtemp()
    ctx = type("C", (), {"profile_store": ProfileStore(
        db_path=os.path.join(home, "p.db"))})()
    out = control_uprofile("create gig Ada Designer", context=ctx,
                           sender="Ada")
    assert "profile created" in out
    pid = out.split("(")[1].split(")")[0]
    shown = control_uprofile(f"show {pid}", context=ctx)
    assert "Ada Designer" in shown


def test_control_prompt_flow():
    home = tempfile.mkdtemp()
    ctx = type("C", (), {"profile_store": ProfileStore(
        db_path=os.path.join(home, "p.db"))})()
    out = control_uprofile("create gig Ada", context=ctx, sender="Ada")
    pid = out.split("(")[1].split(")")[0]
    ok = control_uprofile(f"prompt {pid} best_work redesigned a bank app",
                          context=ctx)
    assert "saved" in ok
    shown = control_uprofile(f"show {pid}", context=ctx)
    assert "redesigned a bank app" in shown


def test_control_like_and_matches():
    home = tempfile.mkdtemp()
    ctx = type("C", (), {"profile_store": ProfileStore(
        db_path=os.path.join(home, "p.db"))})()
    out = control_uprofile("create gig Ada", context=ctx, sender="Ada")
    pid = out.split("(")[1].split(")")[0]
    added = control_uprofile(f"add {pid} project Bank redesign | Figma",
                             context=ctx)
    eid = added.split("[")[1].split("]")[0]
    liked = control_uprofile(f"like {pid} {eid} love the flow!",
                             context=ctx, sender="Emeka")
    assert "liked" in liked
    inbox = control_uprofile("matches", context=ctx, sender="Ada")
    assert "Emeka" in inbox
    assert "love the flow!" in inbox


def test_control_never_raises():
    # Garbage in, graceful out — no tracebacks in chat.
    assert control_uprofile("") != ""
    assert control_uprofile("frobnicate !!!") != ""
    assert control_uprofile("like", ) != ""
    assert control_uprofile("prompt") != ""
