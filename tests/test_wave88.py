"""Wave 88 — patient local models, honest status, alive-while-thinking,
and a smarter hangman with a Word of the Day.

The incident that drove this wave: a 4.4 GB Q4 7B on a phone took minutes
for its first answer; a frozen llama-server then burned the full timeout
on EVERY message, the raw error line read like a crash, the typing
indicator went dead after 90 s, and the owner (rightly) reached for
Ctrl+C.  Each piece below fixes one link of that chain.
"""
from __future__ import annotations

import threading
import time
import types
import unittest
from datetime import date
from types import SimpleNamespace

from nomorals.core.config import _ENV_MAP, Settings
from nomorals.core.events import EventBus
from nomorals.core.errors import TimeoutError_
from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.router import LLMRouter
from nomorals.storage.db import Database

from nomorals.games import GameEngine, Player
from nomorals.games.ai import GameMind
from nomorals.games.games.easy import (
    _GLOBAL_FREQ,
    HANGMAN_LETTER_PRIORITY,
    HANGMAN_WORDS,
    hangman_daily_word,
)


# ── fakes ────────────────────────────────────────────────────────────────────

class Ctx:
    def __init__(self, db: Database) -> None:
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, sent


class _DeadLocal:
    name = "dead_local"
    model_id = "local"
    preflight_health = True
    capabilities = {"chat"}

    def health(self) -> bool:
        return False

    def chat(self, *a, **k):
        raise AssertionError("a wedged server must never be called")


class _Cloud:
    name = "cloud"
    model_id = "cloud"
    capabilities = {"chat"}

    def health(self) -> bool:
        return True

    def chat(self, messages, params=None, **k):
        return LLMResponse(text="hello from cloud")


class _SlowLocal:
    calls = 0

    name = "slow_local"
    model_id = "local"
    preflight_health = True
    capabilities = {"chat"}

    def health(self) -> bool:
        return True

    def chat(self, messages, params=None, **k):
        _SlowLocal.calls += 1
        raise TimeoutError_("request to http://127.0.0.1:8080 timed out")


# ── the model plane ──────────────────────────────────────────────────────────

class TestRouterPreflight(unittest.TestCase):
    """A local server can wedge silently (port held, never answers). The
    router must probe it cheaply and skip it — not pay the patient full
    timeout (600 s) on every single message."""

    def test_wedged_local_is_skipped_fast_and_fallback_answers(self) -> None:
        bus = EventBus()
        router = LLMRouter(bus=bus)
        router.add(_DeadLocal(), primary=True, name="dead_local")
        router.add(_Cloud(), primary=False, name="cloud")
        t0 = time.perf_counter()
        resp = router.chat([Message.user("hi")])
        elapsed = time.perf_counter() - t0
        self.assertTrue(resp.ok)
        self.assertIn("cloud", resp.text)
        self.assertLess(elapsed, 2.0,
                        "preflight must skip a dead server in ~5 s max")
        snap = router.stats_snapshot()
        self.assertIn("not responding",
                      snap["health"]["dead_local"]["last_error"])

    def test_timeout_cools_the_provider_down_immediately(self) -> None:
        # A request that burned its full timeout means the backend is
        # stuck — the next message must NOT pay the same timeout again.
        bus = EventBus()
        router = LLMRouter(bus=bus)
        router.add(_SlowLocal(), primary=True, name="slow_local")
        router.add(_Cloud(), primary=False, name="cloud")
        first = router.chat([Message.user("hi")])
        self.assertTrue(first.ok)
        self.assertEqual(_SlowLocal.calls, 1)
        # second message: the wedged provider is cooling — skipped, not
        # retried, no second six-minute wait
        _SlowLocal.calls = 0
        second = router.chat([Message.user("hi again")])
        self.assertTrue(second.ok)
        self.assertIn("cloud", second.text)
        self.assertEqual(_SlowLocal.calls, 0,
                         "a timed-out provider must cool down immediately")


class TestLocalTimeoutWiring(unittest.TestCase):
    def test_env_mapping_exists(self) -> None:
        self.assertEqual(_ENV_MAP["NM_LLM_LOCAL_TIMEOUT"],
                         "llm.local_request_timeout")

    def test_default_is_patient(self) -> None:
        self.assertGreaterEqual(Settings().llm.local_request_timeout, 600.0)

    def test_provider_kwargs_carry_the_timeout(self) -> None:
        from nomorals.agents.context import provider_kwargs

        settings = Settings()
        settings.llm.local_request_timeout = 777.0
        kwargs = provider_kwargs(settings, "llama_cpp")
        self.assertEqual(kwargs["timeout"], 777.0)


# ── honest, plain status lines ──────────────────────────────────────────────

class TestPlainModelReason(unittest.TestCase):
    def _reason(self, error: str) -> str:
        from nomorals.agents.partner_runtime import PartnerRuntime
        return PartnerRuntime._plain_model_reason(error)

    def test_timeout_reads_human(self) -> None:
        self.assertIn("took too long",
                      self._reason("tool.timeout: request to http://localhost:8080"
                                   "/v1/chat/completions timed out"))

    def test_connection_refused_says_not_running(self) -> None:
        r = self._reason("http.request: [Errno 111] Connection refused")
        self.assertIn("not running", r)

    def test_wedge_says_frozen(self) -> None:
        self.assertIn("frozen", self._reason("server not responding to health probe"))

    def test_rate_limit_is_reassuring(self) -> None:
        self.assertIn("rate limited", self._reason("model.provider.rate_limited: 429"))

    def test_bad_credentials_named(self) -> None:
        self.assertIn("credentials", self._reason("http: 401 unauthorized"))

    def test_unknown_error_stays_short(self) -> None:
        r = self._reason("something " + "x" * 100)
        self.assertLessEqual(len(r), 60)


# ── alive while thinking ─────────────────────────────────────────────────────

class _FakeGateway:
    def __init__(self) -> None:
        self.typings = 0
        self.sent: list[str] = []

    def typing(self, platform, chat, seconds=3.0) -> bool:
        self.typings += 1
        return True

    def send(self, platform, chat, text) -> SimpleNamespace:
        self.sent.append(text)
        return SimpleNamespace(ok=True, error="")


class TestSlowReplyNotice(unittest.TestCase):
    """On a phone a cold 7B needs minutes.  Silence reads as death — which
    is exactly when owners reach for Ctrl+C.  One short 'still thinking'
    line in a DM keeps her alive in the owner's head."""

    def _run(self, *, kind, budget: float, notice: float,
             in_groups: bool = True):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.social.chat.base import ChatKind

        partner = SimpleNamespace(
            typing_keepalive_seconds=0.02,
            typing_keepalive_budget=budget,
            slow_reply_notice_seconds=notice,
            typing_in_groups=in_groups,
        )
        rt = SimpleNamespace(
            settings=SimpleNamespace(partner=partner),
            gateway=_FakeGateway(),
        )
        chat = SimpleNamespace(platform="telegram", key="t:1", kind=kind)
        stop = threading.Event()
        PartnerRuntime._typing_keepalive(rt, chat, stop)
        return rt.gateway

    def test_dm_gets_one_notice_when_slow(self) -> None:
        from nomorals.social.chat.base import ChatKind
        gw = self._run(kind=ChatKind.DM, budget=0.25, notice=0.05)
        notices = [t for t in gw.sent if "still thinking" in t]
        self.assertEqual(len(notices), 1, "exactly one notice, not a spam")
        self.assertGreaterEqual(gw.typings, 2, "typing kept refreshing")

    def test_groups_never_get_the_notice(self) -> None:
        from nomorals.social.chat.base import ChatKind
        gw = self._run(kind=ChatKind.GROUP, budget=0.25, notice=0.05)
        self.assertEqual(gw.sent, [])

    def test_notice_can_be_disabled(self) -> None:
        from nomorals.social.chat.base import ChatKind
        gw = self._run(kind=ChatKind.DM, budget=0.2, notice=0.0)
        self.assertEqual(gw.sent, [])

    def test_budget_replaces_the_old_90s_hardcode(self) -> None:
        # the budget must be respected: a 1 s budget ends in ~1 s, not 90
        from nomorals.social.chat.base import ChatKind
        t0 = time.perf_counter()
        self._run(kind=ChatKind.DM, budget=1.0, notice=0.0)
        self.assertLess(time.perf_counter() - t0, 3.0)


# ── hangman: Word of the Day + deck-aware house ──────────────────────────────

class TestDailyWord(unittest.TestCase):
    def test_deterministic_per_date(self) -> None:
        for day in ("2026-09-17", "2026-09-18", "2026-10-01"):
            self.assertEqual(hangman_daily_word(day), hangman_daily_word(day))

    def test_always_a_real_word_from_the_deck(self) -> None:
        deck = {w for w, _ in HANGMAN_WORDS}
        for i in range(60):
            word = hangman_daily_word(date(2026, 1, 1).fromordinal(
                date(2026, 1, 1).toordinal() + i).isoformat())
            self.assertIn(word, deck)

    def test_dates_usually_differ(self) -> None:
        words = {hangman_daily_word(f"2026-09-{d:02d}") for d in range(1, 30)}
        self.assertGreater(len(words), 1)

    def test_daily_room_uses_the_daily_word(self) -> None:
        engine, sent = make_engine()
        ada = Player.from_sender("telegram", "456", "Ada")
        today = hangman_daily_word()
        room, msgs = engine.start("chat-daily", "hangman", ada, daily=True)
        self.assertEqual(room.state["word"], today)
        self.assertTrue(room.state["daily"])
        self.assertTrue(any("today's word" in m for m in msgs))
        engine, _ = make_engine()
        room2, _ = engine.start("chat-normal", "hangman", ada)
        self.assertFalse(room2.state["daily"])


class TestDeckAwareHouse(unittest.TestCase):
    def test_priority_table_covers_every_word(self) -> None:
        for word, category in HANGMAN_WORDS:
            self.assertIn((category, len(word)), HANGMAN_LETTER_PRIORITY)

    def test_priority_orders_deck_letters_first(self) -> None:
        # The 8-letter animal deck: every letter that appears in any deck
        # word must rank above every letter no deck word contains, and the
        # letters appearing in the most deck words lead the table.
        deck_words = [w for w, c in HANGMAN_WORDS
                      if c == "animal" and len(w) == 8]
        self.assertTrue(deck_words)
        deck = set("".join(deck_words).lower())
        prio = HANGMAN_LETTER_PRIORITY[("animal", 8)]
        last_deck = max(prio.index(ch) for ch in deck)
        first_non_deck = min(prio.index(ch) for ch in prio
                             if ch not in deck)
        self.assertLess(last_deck, first_non_deck)
        # count 'o' and 'l' across the deck: the top of the table must be
        # exactly the letters with the highest counts, in frequency order
        from collections import Counter
        counts = Counter()
        for w in deck_words:
            counts.update(set(w.lower()))
        top_count = max(counts.values())
        leaders = sorted((ch for ch, n in counts.items()
                          if n == top_count),
                         key=lambda ch: _GLOBAL_FREQ.index(ch))
        self.assertTrue(all(ch in deck for ch in prio[:len(leaders)]))
        self.assertEqual(prio[0], leaders[0])

    def test_letter_guess_uses_priority_and_skips_revealed(self) -> None:
        mind = GameMind()
        prio = HANGMAN_LETTER_PRIORITY[("animal", 8)]
        self.assertEqual(mind.letter_guess(set(), 8, "animal", priority=prio),
                         prio[0])
        self.assertEqual(mind.letter_guess({prio[0]}, 8, "animal", priority=prio),
                         prio[1])

    def test_letter_guess_defaults_to_frequency(self) -> None:
        mind = GameMind()
        self.assertEqual(mind.letter_guess(set(), 8, "animal"), "e")

    def test_letter_guess_never_dies_on_full_board(self) -> None:
        mind = GameMind()
        all_but_z = set("abcdefghijklmnopqrstuvwxy")
        self.assertEqual(mind.letter_guess(all_but_z, 8, "animal"), "z")
        self.assertEqual(mind.letter_guess(set("abcdefghijklmnopqrstuvwxyz"),
                                           8, "animal",
                                           priority="z"), "z")


class TestDailyCommandParsing(unittest.TestCase):
    def test_hangman_accepts_the_daily_flag(self) -> None:
        from nomorals.social.chat.control import parse_control
        c = parse_control("/hangman daily")
        self.assertEqual(c.kind, "hangman")
        self.assertEqual(c.tail, "daily")

    def test_plain_hangman_still_works(self) -> None:
        from nomorals.social.chat.control import parse_control
        c = parse_control("/hangman")
        self.assertEqual(c.kind, "hangman")
        self.assertNotEqual(c, None)


if __name__ == "__main__":
    unittest.main()
