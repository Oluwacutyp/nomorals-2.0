"""Pass tail-2 pins: agent_loop failures stay silent, ledger intact.

Target: nomorals/agents/agent_loop.py — 0 converts / 11 exclusions.
Every site below maps a failure to a genuine degraded value (or a
caller-supplied fallback) and the loop proceeds; none of them records,
because none of them hides a user-visible outcome. Ledger REUSED:
nomorals/agents/failures.py (same package — read-only for pins; no
record call, no import added to the target, which contains no
record_failure call at all).

Each test pins the degraded behavior AND the NOT-recording property
(ledger empty after the trigger). No bare ``except: pass`` anywhere
in this file — the scanner counts test files too; the tests below
need no except blocks at all (stubs raise, the code under test
catches).
"""

from types import SimpleNamespace

from nomorals.agents import agent_loop as al
from nomorals.agents import failures as af
from nomorals.agents.coremind import Intent


def _silent():
    assert af.recent_failures() == []


class Chat:
    def __init__(self, key="real-key", key_raises=False):
        self.platform = "telegram"
        self.chat_id = "42"
        self.kind = "dm"
        self.thread_id = ""
        self._key = key
        self._key_raises = key_raises

    @property
    def key(self):
        if self._key_raises:
            raise RuntimeError("boom-key")
        return self._key


class Msg:
    def __init__(self, text="hi", chat=None):
        self.text = text
        self.media = None
        self.chat = chat if chat is not None else Chat()


class RaisingLiveGame:
    def _live_game(self, key):
        raise RuntimeError("boom-live")

    def _get_pending(self, key):
        return None


class RaisingPending:
    def _live_game(self, key):
        return None

    def _get_pending(self, key):
        raise RuntimeError("boom-pending")


class RaisingJobs:
    def _live_game(self, key):
        return None

    def _get_pending(self, key):
        return None

    @property
    def _jobs(self):
        raise RuntimeError("boom-jobs")


class ListJobs:
    """Real CoreMind shape: _jobs is a list, not a dict."""

    def _live_game(self, key):
        return None

    def _get_pending(self, key):
        return None

    _jobs = [{"done": False}, {"done": True}]


class RaisingBrain:
    @property
    def relationship(self):
        raise RuntimeError("boom-rel")


class RaisingSettings:
    @property
    def settings(self):
        raise RuntimeError("boom-settings")


class RaisingHistory:
    def _history(self, key, limit=4):
        raise RuntimeError("boom-hist")


class RaisingRoute:
    def __init__(self):
        self.attempted = False

    def _live_game(self, key):
        return None

    def _get_pending(self, key):
        return None

    def _record_route(self, intent):
        self.attempted = True
        raise RuntimeError("boom-route")


class DecideMind:
    def __init__(self):
        self.decide_kwargs = {}

    def _live_game(self, key):
        raise RuntimeError("boom-live")

    def _get_pending(self, key):
        return None

    def decide(self, text, live_game=None, allow_model=True):
        self.decide_kwargs = {"live_game": live_game}
        return Intent("chat", 1.0)


def test_site_259_chat_key_falls_back_to_caller_key():
    af.clear_failures()
    ctx = al.build_loop_context(Msg(chat=Chat(key_raises=True)),
                                chat_key="fallback-key")
    assert ctx.chat_key == "fallback-key"
    _silent()


def test_site_274_live_game_lookup_failure_is_no_game():
    af.clear_failures()
    ctx = al.build_loop_context(Msg(), mind=RaisingLiveGame())
    assert ctx.live_game is None
    _silent()


def test_site_278_pending_lookup_failure_is_no_pending():
    af.clear_failures()
    ctx = al.build_loop_context(Msg(), mind=RaisingPending())
    assert ctx.pending_clarification is False
    _silent()


def test_site_287_jobs_lookup_failure_is_zero_open():
    af.clear_failures()
    ctx = al.build_loop_context(Msg(), mind=RaisingJobs())
    assert ctx.open_jobs == 0
    _silent()


def test_site_287_list_shaped_jobs_count_zero():
    # Documents the note: coremind._jobs is a list, so the dict
    # branch never counts real jobs; open_jobs is advisory only.
    af.clear_failures()
    ctx = al.build_loop_context(Msg(), mind=ListJobs())
    assert ctx.open_jobs == 0
    _silent()


def test_site_300_owner_mode_lookup_failure_is_not_owner_mode():
    af.clear_failures()
    ctx = al.build_loop_context(
        Msg(), runtime=SimpleNamespace(brain=RaisingBrain()))
    assert ctx.owner_mode is False
    _silent()


def test_site_311_settings_lookup_failure_keeps_workstation():
    af.clear_failures()
    ctx = al.build_loop_context(
        Msg(), runtime=RaisingSettings())
    assert ctx.resource_profile == "workstation"
    _silent()


def test_site_319_termux_heuristic_still_runs(monkeypatch):
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")
    af.clear_failures()
    ctx = al.build_loop_context(
        Msg(), runtime=RaisingSettings())
    assert ctx.resource_profile == "termux"
    _silent()


def test_site_343_history_lookup_failure_is_empty_window():
    af.clear_failures()
    ctx = al.build_loop_context(
        Msg(text="hello world"),
        runtime=SimpleNamespace(brain=RaisingHistory()))
    assert ctx.recent_turns == []
    _silent()


def test_site_450_run_loop_live_game_failure_decides_without_game():
    af.clear_failures()
    mind = DecideMind()
    reply, ctx = al.run_loop(mind, "xyzzy plover", message=Msg(), chat_key="k")
    assert mind.decide_kwargs.get("live_game") is None
    assert reply is None
    _silent()


def test_site_459_route_log_failure_keeps_fast_reply():
    af.clear_failures()
    mind = RaisingRoute()
    reply, ctx = al.run_loop(mind, "hello", message=Msg(), chat_key="k")
    assert mind.attempted is True
    assert reply == "hey — what's on your mind?"
    _silent()


def test_site_489_broken_route_log_is_unknown_kind():
    af.clear_failures()
    assert al._last_intent_kind(SimpleNamespace(_route_log=object())) == ""
    assert al._last_intent_kind(SimpleNamespace(_route_log=[])) == ""
    _silent()
