"""Design follow-ups pass pins (register §3.6).

Each section pins one landed design follow-up from the held
register — the behavior the tail-batch adjudications sketched but
deliberately did not land in an error-handling batch:

1. property/value confidence (tail-6, site 272): 1 - FSD cannot
   tell "zero spread" from "no data"; a thin/no-data estimate
   rendered "confidence 100%". The displayed score now applies
   the module's own thin-market FSD floor. Stored fsd untouched.
2. documents _sniff probe (tail-11, site 215): the 4096-byte
   probe window flattened valid JSON/XML longer than the window
   to "txt". The probe now retries against the full text before
   giving up on structure; invalid content still lands on "txt".
3. games relay reconcile (tail-12, site 179): a failed accept-
   time DELETE left a stale invite row that resurrected on the
   next load. _load now reconciles: an invite whose code is an
   existing relay room_id was consumed — it is skipped and its
   row's delete retried.
4. CAPTCHA settle semantics (tail-44): "cleared" may only be
   claimed from a successful read — unreadable != absent at
   every settle point (question / click / audio / OCR solvers;
   the challenge recheck, Cloudflare and slider-drag settles
   already held this line).
5. Wallet gift-ref idempotency (gifting-148 residual): gift
   movements carry gift:<verb>:<gift id> refs; the wallet now
   treats the (verb, id) pair as an idempotency key journaled in
   game_wallet, so a re-confirm composes into a no-op instead
   of double-sending.
6. MissionWatchers truthful state (tail-92, site 869):
   subscribe/unsubscribe report whether the write persisted
   ("added"/"already"/"removed"/"absent"/"unpersisted") and the
   chat layer phrases from that state instead of claiming
   "watching" over a failed write.

Controls (marked as such) pin behavior that must NOT change.
No bare ``except: pass`` anywhere in this file.
"""
from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from unittest import mock

from nomorals.browser.audio_solver import AudioCaptchaSolver
from nomorals.browser.click_solver import ClickSolver
from nomorals.browser.ocr_solver import ImageCaptchaSolver
from nomorals.browser.question_solver import QuestionSolver
from nomorals.browser.tab_tools import docs_for_tab_checked
from nomorals.documents.parsers import _sniff
from nomorals.games.players import Player, PlayerStore
from nomorals.games.relay import GameRelay, RelayRoom
from nomorals.missions import progress as P
from nomorals.property.value import ValueEstimate, estimate_value

# ── 1 · property/value confidence ───────────────────────────────


def test_value_no_data_estimate_does_not_claim_full_confidence():
    est = estimate_value("2 bedroom flat in lekki",
                         comp_source=lambda area, br: [])
    assert est.thin_market
    assert est.fsd == 0.0  # stored value keeps its documented meaning
    assert est.confidence == 72  # 1 - _THIN_MARKET_FSD_FLOOR (0.28)
    assert "confidence 72%" in est.format()


def test_value_thin_with_real_spread_keeps_its_own_score():
    # Control: the floor only lifts a thin score that understates
    # its spread; a genuinely wide thin estimate is untouched.
    est = ValueEstimate(point_kobo=1000, fsd=0.40, thin_market=True)
    assert est.confidence == 60


def test_value_non_thin_confidence_unchanged():
    # Control: the documented HouseCanary mapping, byte-identical.
    est = ValueEstimate(point_kobo=1000, fsd=0.12)
    assert est.confidence == 88


# ── 2 · documents _sniff probe window ───────────────────────────


def test_sniff_long_valid_xml_is_xml():
    data = b"<root>" + b"a " * 3000 + b"</root>"
    assert len(data) > 4096
    assert _sniff(data, "", "") == "xml"


def test_sniff_long_valid_json_is_json():
    data = b'{"k": "' + b"x" * 5000 + b'"}'
    assert len(data) > 4096
    assert _sniff(data, "", "") == "json"


def test_sniff_long_invalid_xml_stays_txt():
    # Control: genuinely invalid content fails the full parse too —
    # no format is claimed on negative evidence.
    data = b"<root><unclosed" + b"y" * 5000
    assert _sniff(data, "", "") == "txt"


def test_sniff_short_valid_xml_still_xml():
    # Control: the windowed fast path is unchanged.
    assert _sniff(b"<root><a>1</a></root>", "", "") == "xml"


# ── shared sqlite shim (test_gifting pattern) ───────────────────


class _Q:
    """Minimal DB shim matching the Database surface the stores use."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def tx():
            yield
            self.conn.commit()

        return tx()


# ── 3 · relay reconcile-on-load ─────────────────────────────────

ADA = Player.from_sender("telegram", "1", "Ada")
BOB = Player.from_sender("telegram", "2", "Bob")


def _relay_engine(db):
    return SimpleNamespace(
        db=db, games={"duel": SimpleNamespace(max_players=2)})


def test_relay_load_reconciles_consumed_invite():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db = _Q(conn)
    first = GameRelay(_relay_engine(db))
    consumed = first.create_invite("chat:a", ADA, "duel")
    live = first.create_invite("chat:a", ADA, "duel")
    # The accept landed and saved its relay, but the invite-row
    # DELETE failed (tail-12 site 179): both rows are in the table.
    first._save_relay(RelayRoom(
        room_id=consumed.code, game_name="duel", chat_a="chat:a",
        chat_b="chat:b", player_a=ADA, player_b=BOB))
    rows = db.query("SELECT code FROM game_invites")
    assert {r["code"] for r in rows} == {consumed.code, live.code}

    second = GameRelay(_relay_engine(db))  # restart: _load reconciles
    assert consumed.code not in second.invites
    assert live.code in second.invites
    assert consumed.code in second.relays
    # the stale row's delete was retried at load
    rows = db.query("SELECT code FROM game_invites")
    assert {r["code"] for r in rows} == {live.code}


# ── 4 · CAPTCHA settle: unreadable != cleared ───────────────────


def test_docs_for_tab_checked_reports_readability():
    class DeadTab:
        url = "https://x.test/"

        def html(self):
            raise RuntimeError("tab gone")

    class LiveTab:
        url = "https://x.test/"

        def html(self):
            return "<html></html>"

    docs, read_ok = docs_for_tab_checked(DeadTab())
    assert read_ok is False
    assert docs[0]["html"] == ""  # detection contract unchanged
    _docs2, read_ok2 = docs_for_tab_checked(LiveTab())
    assert read_ok2 is True


class _DiesAfterTypeTab:
    """html() serves ``page`` until type_text lands; then the tab
    is gone — the settle read fails."""

    url = "https://x.test/form"
    title = "form"

    def __init__(self, page: str):
        self._page = page
        self.typed: list[tuple[str, str]] = []

    def html(self):
        if self.typed:
            raise RuntimeError("tab gone")
        return self._page

    def type_text(self, target, text):
        self.typed.append((target, text))


def test_question_settle_unreadable_is_not_cleared():
    tab = _DiesAfterTypeTab("<html></html>")
    solver = QuestionSolver(answer_fn=lambda q: "4",
                            sleeper=lambda s: None)
    out = solver.solve_tab(tab, {"question": "2+2?",
                                 "input_selector": "#a"})
    assert tab.typed  # the answer really was typed
    assert out["ok"] is False
    assert "could not be re-read" in out["detail"]


def test_question_settle_readable_absence_still_clears():
    # Control: a successful read with the challenge gone clears.
    tab = _DiesAfterTypeTab("<html></html>")
    tab.html = lambda: "<html><body>thanks</body></html>"
    solver = QuestionSolver(answer_fn=lambda q: "4",
                            sleeper=lambda s: None)
    out = solver.solve_tab(tab, {"question": "2+2?",
                                 "input_selector": "#a"})
    assert out["ok"] is True
    assert "cleared in-session" in out["detail"]


_GEETEST_PAGE = (
    '<html><body><div class="geetest_holder geetest_widget">'
    '<div class="geetest_item"></div></div>'
    '<div class="geetest_tip">Please click the icons</div>'
    "</body></html>")


class _ClickDiesTab:
    url = "https://x.test/login"
    title = "login"

    def __init__(self):
        self.html_calls = 0
        self.clicks: list[tuple[float, float]] = []

    def html(self):
        self.html_calls += 1
        if self.html_calls == 1:
            return _GEETEST_PAGE
        raise RuntimeError("tab gone")

    def element_image_bytes(self, selector):
        return b"\x89PNG-fake"

    def element_box(self, selector):
        return {"x": 0, "y": 0, "width": 300, "height": 300}

    def click_at_px(self, x, y):
        self.clicks.append((x, y))


def test_click_settle_unreadable_is_not_cleared():
    tab = _ClickDiesTab()
    solver = ClickSolver(
        ground_fn=lambda image, prompt: {
            "grid": {"x": 0, "y": 0, "w": 1000, "h": 1000},
            "cols": 3, "rows": 3, "cells": [(0, 0)]},
        probe=lambda: (True, "ok"), sleeper=lambda s: None)
    out = solver.solve_tab(tab)
    assert tab.clicks  # cells really were clicked
    assert out["ok"] is False
    assert "could not be re-read" in out["detail"]


_AUDIO_PAGE = ('<html><body><audio src="/audio/payload.mp3"></audio>'
               '<input type="text" id="audio-response"></body></html>')


class _AudioDiesTab(_DiesAfterTypeTab):
    def __init__(self):
        super().__init__(_AUDIO_PAGE)

    def session_fetch_bytes(self, url):
        return b"fake-audio"


def test_audio_settle_unreadable_is_not_cleared():
    tab = _AudioDiesTab()
    solver = AudioCaptchaSolver(
        transcribe_fn=lambda audio: {"ok": True, "text": "seven two",
                                     "backend": "whisper.cpp",
                                     "detail": "stub"},
        probe=lambda: (True, "ok"), sleeper=lambda s: None)
    out = solver.solve_tab(tab)
    assert tab.typed  # the transcript really was typed
    assert out["ok"] is False
    assert "could not be re-read" in out["detail"]


_OCR_HTML = ('<html><body><img src="https://x.test/cap.png">'
             '<input name="captcha_answer" type="text"></body></html>')


def test_ocr_settle_unreadable_is_not_cleared():
    tab = _DiesAfterTypeTab(_OCR_HTML)
    solver = ImageCaptchaSolver(
        read_fn=lambda data: {"text": "AB12", "mean_conf": 95.0},
        probe=lambda: (True, "ok"), sleeper=lambda s: None)
    out = solver.solve_tab(tab, {"image_bytes": b"IMG",
                                 "image_url": ""})
    assert tab.typed  # the read really was typed
    assert out["ok"] is False
    assert "could not be re-read" in out["detail"]


# ── 5 · wallet gift-ref idempotency ─────────────────────────────


def _wallet_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    q = _Q(conn)
    q.execute(
        "CREATE TABLE game_players ("
        "player_key TEXT PRIMARY KEY, platform TEXT NOT NULL DEFAULT '', "
        "display TEXT NOT NULL DEFAULT '', username TEXT NOT NULL "
        "DEFAULT '', deleted_at REAL NOT NULL DEFAULT 0, "
        "coins INTEGER NOT NULL DEFAULT 0, "
        "points INTEGER NOT NULL DEFAULT 0, wins INTEGER NOT NULL "
        "DEFAULT 0, losses INTEGER NOT NULL DEFAULT 0, "
        "draws INTEGER NOT NULL DEFAULT 0, streak INTEGER NOT NULL "
        "DEFAULT 0, best_streak INTEGER NOT NULL DEFAULT 0, "
        "games_played INTEGER NOT NULL DEFAULT 0, xp INTEGER NOT NULL "
        "DEFAULT 0, per_game TEXT NOT NULL DEFAULT '{}', "
        "items TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL "
        "DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)")
    q.execute(
        "CREATE TABLE game_wallet ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "player_key TEXT NOT NULL, amount INTEGER NOT NULL DEFAULT 0, "
        "reason TEXT NOT NULL DEFAULT '', at REAL NOT NULL DEFAULT 0)")
    return q


def _wallet_rows(db, player_key, reason_prefix):
    rows = db.query(
        "SELECT reason FROM game_wallet WHERE player_key = ?",
        (player_key,))
    return [r["reason"] for r in rows
            if r["reason"].startswith(reason_prefix)]


def test_gift_ref_spend_and_credit_are_idempotent():
    db = _wallet_db()
    store = PlayerStore(db)
    ada = Player(key="telegram:111", platform="telegram", name="Ada")
    bob = Player(key="telegram:222", platform="telegram", name="Bob")
    store.add_coins(ada, 500, "seed funding")

    gid = "gift_01TESTULID0000000000000"
    assert store.spend_coins(ada, 100, f"gift:out:{gid}") == 400
    # a re-confirm replays the whole transfer: no second movement
    assert store.spend_coins(ada, 100, f"gift:out:{gid}") == 400
    assert _wallet_rows(db, ada.key, "gift:out:") == [f"gift:out:{gid}"]

    assert store.add_coins(bob, 100, f"gift:in:{gid}:Ada") == 100
    assert store.add_coins(bob, 100, f"gift:in:{gid}:Ada") == 100
    # the key is (verb, gift id): a drifted display suffix must not
    # re-credit either
    assert store.add_coins(bob, 100, f"gift:in:{gid}:Ada L.") == 100
    assert len(_wallet_rows(db, bob.key, "gift:in:")) == 1

    # refund refs are idempotent too
    assert store.add_coins(ada, 100, f"gift:refund:{gid}") == 500
    assert store.add_coins(ada, 100, f"gift:refund:{gid}") == 500
    assert len(_wallet_rows(db, ada.key, "gift:refund:")) == 1


def test_gift_ref_replay_never_reads_as_unaffordable():
    db = _wallet_db()
    store = PlayerStore(db)
    ada = Player(key="telegram:111", platform="telegram", name="Ada")
    store.add_coins(ada, 150, "seed funding")
    gid = "gift_01TESTULID0000000000001"
    assert store.spend_coins(ada, 100, f"gift:out:{gid}") == 50
    # balance (50) is now below the amount, but the debit already
    # landed: replay reports the standing balance, never None.
    assert store.spend_coins(ada, 100, f"gift:out:{gid}") == 50


def test_gift_ref_scoping_controls():
    # Controls: a different gift id is a different movement, and
    # non-gift reasons stay repeatable (idempotency is ref-scoped).
    db = _wallet_db()
    store = PlayerStore(db)
    ada = Player(key="telegram:111", platform="telegram", name="Ada")
    store.add_coins(ada, 500, "seed funding")
    assert store.spend_coins(ada, 100, "gift:out:gift_AAA") == 400
    assert store.spend_coins(ada, 100, "gift:out:gift_BBB") == 300
    assert store.add_coins(ada, 10, "daily reward") == 310
    assert store.add_coins(ada, 10, "daily reward") == 320


# ── 6 · MissionWatchers truthful persistence state ──────────────


class _DictKV:
    """In-memory KVStore double with the real get/set surface."""

    def __init__(self, db, box):
        self._box = box

    def get(self, key):
        return self._box.get(key)

    def set(self, key, value):
        self._box[key] = value


def test_watchers_report_persistence_state():
    box: dict = {}
    watchers = P.MissionWatchers(object())
    with mock.patch.object(
            P, "KVStore", lambda db: _DictKV(db, box)):
        assert watchers.subscribe("m1", "telegram:1") == "added"
        assert watchers.subscribe("m1", "telegram:1") == "already"
        assert watchers.watchers("m1") == ["telegram:1"]
        assert watchers.unsubscribe("m1", "telegram:1") == "removed"
        assert watchers.unsubscribe("m1", "telegram:1") == "absent"
        assert watchers.watchers("m1") == []


def test_watchers_no_db_reports_unpersisted():
    watchers = P.MissionWatchers(None)
    assert watchers.subscribe("m1", "telegram:1") == "unpersisted"
