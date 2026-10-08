"""Tests for Panels: persistent, connector-backed mini-apps (Hark pattern).

All offline — data fetching goes through injectable mock fetchers.
"""

from pathlib import Path

import pytest

from nomorals.community.miniapps import (
    Panel,
    PanelStore,
    ask_panel,
    control_panel,
    create_panel,
    refresh_panel,
    render_panel,
)


@pytest.fixture
def store(tmp_path: Path) -> PanelStore:
    return PanelStore(data_dir=tmp_path / "panels")


def mock_strava(connector_id: str) -> dict:
    assert connector_id == "strava"
    return {
        "weekly_km": 42,
        "longest_run_km": 18.5,
        "resting_hr": 52,
        "training": {"vo2max": 48, "sessions": 5},
    }


def mock_broken(connector_id: str) -> dict:
    raise ConnectionError("no network")


# ── creation ───────────────────────────────────────────────────────────────


def test_create_panel(store: PanelStore):
    p = create_panel("Marathon training", "owner", "strava",
                     "track my marathon training", store=store)
    assert p.id.startswith("panel_")
    assert p.name == "Marathon training"
    assert p.data_source == "strava"
    assert p.description == "track my marathon training"
    assert p.refresh_cadence == "daily"
    assert p.data == {}


def test_create_panel_bad_cadence_falls_back(store: PanelStore):
    p = create_panel("X", "owner", "mono", refresh_cadence="never", store=store)
    assert p.refresh_cadence == "daily"


def test_create_panel_never_raises():
    p = create_panel(None, None, None, store=None)  # type: ignore[arg-type]
    assert isinstance(p, Panel)


# ── persistence ────────────────────────────────────────────────────────────


def test_panel_roundtrip(store: PanelStore):
    p = create_panel("Stocks", "owner", "binance", store=store)
    got = store.get("owner", p.id)
    assert got is not None and got.name == "Stocks"
    assert store.get("owner", p.id[:10]) is not None  # prefix match


def test_panel_find_by_name(store: PanelStore):
    create_panel("Marathon training", "owner", "strava", store=store)
    found = store.find_by_name("owner", "marathon")
    assert found is not None
    assert store.find_by_name("owner", "nope") is None


def test_panel_remove(store: PanelStore):
    p = create_panel("Temp", "owner", "mono", store=store)
    assert store.remove("owner", p.id) is True
    assert store.get("owner", p.id) is None
    assert store.remove("owner", "nope") is False


def test_panel_user_scoped(store: PanelStore):
    create_panel("Mine", "alice", "strava", store=store)
    assert store.load("bob") == []
    assert len(store.load("alice")) == 1


# ── refresh ────────────────────────────────────────────────────────────────


def test_refresh_pulls_live_data(store: PanelStore):
    p = create_panel("Marathon", "owner", "strava", store=store)
    r = refresh_panel(p.id, "owner", fetcher=mock_strava, store=store, force=True)
    assert r["ok"] is True
    assert r["panel"].data["weekly_km"] == 42
    assert r["panel"].last_refresh > 0


def test_refresh_caches_within_cadence(store: PanelStore):
    p = create_panel("Marathon", "owner", "strava", store=store)
    refresh_panel(p.id, "owner", fetcher=mock_strava, store=store, force=True)
    calls = []
    def counting(cid):
        calls.append(cid)
        return {"x": 1}
    r = refresh_panel(p.id, "owner", fetcher=counting, store=store)
    assert r.get("cached") is True
    assert calls == []


def test_refresh_fetcher_failure_is_data(store: PanelStore):
    p = create_panel("Marathon", "owner", "strava", store=store)
    r = refresh_panel(p.id, "owner", fetcher=mock_broken, store=store, force=True)
    assert r["ok"] is True
    assert "_error" in r["panel"].data


def test_refresh_no_fetcher_is_honest(store: PanelStore):
    p = create_panel("Marathon", "owner", "strava", store=store)
    r = refresh_panel(p.id, "owner", store=store, force=True)
    assert r["ok"] is True
    assert "no live connector" in r["panel"].data["_error"]


def test_refresh_unknown_panel(store: PanelStore):
    r = refresh_panel("nope", "owner", store=store)
    assert r["ok"] is False


def test_needs_refresh_manual_never(store: PanelStore):
    p = create_panel("M", "owner", "x", refresh_cadence="manual", store=store)
    assert p.needs_refresh() is False


# ── ask ────────────────────────────────────────────────────────────────────


@pytest.fixture
def trained(store: PanelStore) -> Panel:
    p = create_panel("Marathon training", "owner", "strava", store=store)
    refresh_panel(p.id, "owner", fetcher=mock_strava, store=store, force=True)
    return store.get("owner", p.id)


def test_ask_summary(store: PanelStore, trained: Panel):
    r = ask_panel(trained.id, "how is my training going?", "owner", store=store)
    assert r["ok"] is True
    assert "42" in r["answer"] and "weekly km" in r["answer"]


def test_ask_specific_metric(store: PanelStore, trained: Panel):
    r = ask_panel(trained.id, "what is my resting heart rate?", "owner", store=store)
    assert r["ok"] is True
    assert "52" in r["answer"]


def test_ask_nested_data(store: PanelStore, trained: Panel):
    r = ask_panel(trained.id, "what is my vo2max?", "owner", store=store)
    assert r["ok"] is True
    assert "48" in r["answer"]


def test_ask_no_match_suggests_keys(store: PanelStore, trained: Panel):
    r = ask_panel(trained.id, "what about my sleep?", "owner", store=store)
    assert r["ok"] is True
    assert "Nothing" in r["answer"] or "matched" in r["answer"]


def test_ask_no_data(store: PanelStore):
    p = create_panel("Empty", "owner", "strava", store=store)
    r = ask_panel(p.id, "summary", "owner", store=store)
    assert r["ok"] is True
    assert "no data" in r["answer"]


def test_ask_error_data(store: PanelStore):
    p = create_panel("Broken", "owner", "strava", store=store)
    refresh_panel(p.id, "owner", store=store, force=True)  # honest default fetcher
    r = ask_panel(p.id, "summary", "owner", store=store)
    assert r["ok"] is True
    assert "⚠️" in r["answer"]


def test_ask_unknown_panel(store: PanelStore):
    r = ask_panel("nope", "summary", "owner", store=store)
    assert r["ok"] is False


def test_ask_empty_question(store: PanelStore, trained: Panel):
    r = ask_panel(trained.id, "", "owner", store=store)
    assert r["ok"] is False


# ── render ─────────────────────────────────────────────────────────────────


def test_render_panel(store: PanelStore, trained: Panel):
    out = render_panel(trained)
    assert "Marathon training" in out
    assert "strava" in out
    assert "/panel ask" in out


# ── chat ───────────────────────────────────────────────────────────────────


def test_chat_new_list_show(store: PanelStore):
    out = control_panel('/panel new "Marathon" strava daily', "owner", store=store)
    assert "✅" in out and "Marathon" in out
    out = control_panel("/panel list", "owner", store=store)
    assert "Marathon" in out
    pid = store.load("owner")[0].id
    out = control_panel(f"/panel show {pid}", "owner", store=store)
    assert "Marathon" in out


def test_chat_new_bad_usage(store: PanelStore):
    out = control_panel("/panel new", "owner", store=store)
    assert "Usage" in out


def test_chat_ask_by_name(store: PanelStore):
    p = create_panel("Marathon training", "owner", "strava", store=store)
    refresh_panel(p.id, "owner", fetcher=mock_strava, store=store, force=True)
    out = control_panel("/panel ask marathon what was my longest run?",
                        "owner", store=store)
    assert "18.5" in out


def test_chat_refresh(store: PanelStore):
    p = create_panel("Marathon", "owner", "strava", store=store)
    out = control_panel(f"/panel refresh {p.id}", "owner",
                        store=store, fetcher=mock_strava)
    assert "🔄" in out


def test_chat_rm(store: PanelStore):
    p = create_panel("Temp", "owner", "mono", store=store)
    out = control_panel(f"/panel rm {p.id}", "owner", store=store)
    assert "deleted" in out
    assert store.get("owner", p.id) is None


def test_chat_unknown_subcommand(store: PanelStore):
    out = control_panel("/panel frobnicate", "owner", store=store)
    assert "/panel" in out  # help


def test_chat_never_raises(store: PanelStore):
    for bad in ["", "/panel", "/panel ask", None]:
        out = control_panel(bad, "owner", store=store)  # type: ignore[arg-type]
        assert isinstance(out, str)


def test_panel_never_raises_on_garbage():
    assert ask_panel(None, None, None)["ok"] is False  # type: ignore[arg-type]
    assert refresh_panel(None, None)["ok"] is False  # type: ignore[arg-type]
