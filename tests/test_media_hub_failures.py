"""Pass tail-115 pins for nomorals/media/__init__.py — 0 converts / 3 exclusions.

315 (MediaHub.auto_send_target): a crashing gateway status() returns
("", "") — the function's documented "no target" value, identical to
having no gateway at all. Callers treat it as "do not send": the
failure direction is abstention, never a fabricated target.

333 (auto_send_target): a crashing chats-registry query falls through
to the function's own designed next rung — the first online platform
with an empty chat_id, the same value the no-db path returns. A
healthy query returns the real (platform, chat_id).

466 (_summarize): a crashing Brain/router falls through to the
extractive summarizer (in-code note: "fall through to extractive"),
producing exactly the no-router result from the real text.

No except handlers appear in this file (AST-checked).
"""
import nomorals.media as media_mod
from nomorals.media import MediaHub


class _GW:
    def __init__(self, fn):
        self._fn = fn

    def status(self):
        return self._fn()


class _Ctx:
    pass


def _hub(ctx):
    h = MediaHub.__new__(MediaHub)
    h.context = ctx
    return h


def _boom():
    raise RuntimeError("gateway down")


def test_315_gateway_status_crash_means_no_target():
    ctx = _Ctx()
    ctx.gateway = _GW(_boom)
    ctx.db = None
    assert _hub(ctx).auto_send_target() == ("", "")


def test_315_no_gateway_control():
    ctx = _Ctx()
    ctx.gateway = None
    ctx.db = None
    assert _hub(ctx).auto_send_target() == ("", "")


def test_333_chats_query_crash_falls_to_first_online_platform():
    ctx = _Ctx()
    ctx.gateway = _GW(lambda: {"tg": {"connected": True}})

    class DB:
        def query_one(self, *a):
            raise RuntimeError("db down")

    ctx.db = DB()
    assert _hub(ctx).auto_send_target() == ("tg", "")


def test_333_healthy_query_control():
    ctx = _Ctx()
    ctx.gateway = _GW(lambda: {"tg": {"connected": True}})

    class DB:
        def query_one(self, *a):
            return {"platform": "tg", "chat_id": "42"}

    ctx.db = DB()
    assert _hub(ctx).auto_send_target() == ("tg", "42")


def test_333_no_db_control():
    ctx = _Ctx()
    ctx.gateway = _GW(lambda: {"tg": {"connected": True}})
    ctx.db = None
    assert _hub(ctx).auto_send_target() == ("tg", "")


_TEXT = ("Alpha bravo charlie delta echo foxtrot golf hotel india juliet. "
         "Kilo lima mike november oscar papa quebec romeo sierra tango. "
         "Uniform victor whiskey xray yankee zulu alpha bravo charlie delta.")


def test_466_brain_crash_falls_through_to_extractive():
    class BoomRouter:
        pass

    trig = media_mod._summarize(_TEXT, router=BoomRouter())
    ctrl = media_mod._summarize(_TEXT, router=None)
    assert trig == ctrl
    assert len(trig) > 0
