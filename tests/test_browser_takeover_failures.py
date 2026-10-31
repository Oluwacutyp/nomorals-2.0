"""Failure pins for nomorals/browser/takeover.py (tail-69).

Every site is an adjudicated EXCLUSION: profile tuning falls back to
the documented default interval, a dead peer ends the send, handoff
notes are bookkeeping behind an already-recorded grant state, a
crashing cleared-check reads as NOT cleared (fail-closed — the wall
is verified before any resume and nothing is saved on a non-clear),
a failed screenshot surfaces as the handler's explicit 502, and
shutdown is best-effort teardown. The module's decision paths
(token validation, limiter-style gates, audit) all bind and report.
Each pin proves the guarded operation was attempted (spy), the
degraded value that follows, an honest control, and that nothing was
recorded on the browser failure ledger.
"""
from unittest import mock

from nomorals.browser import failures as browser_failures
from nomorals.browser import takeover as tk


def _grant(session_id, **kw):
    return tk.TakeoverGrant(token="t", token_hash="h" * 64, url="",
                            session_id=session_id, owner_chat="",
                            created_at=0.0, expires_at=1e18, **kw)


# 146 — frame_interval: the profile lookup raises -> "" -> the
# documented 1.0s default, the same value an unknown profile gets.
# Sole consumer is the frame throttle.
def test_146_profile_lookup_failure_uses_default_interval():
    before = browser_failures.failure_count()
    assert tk.TakeoverConfig(profile="termux").frame_interval() == 3.0
    cfg = tk.TakeoverConfig()
    with mock.patch("nomorals.core.profiles.get_profile_kind",
                    side_effect=RuntimeError("profile boom")):
        assert cfg.frame_interval() == 1.0
    assert browser_failures.failure_count() == before


# 283 — Handler._send: the peer disconnects mid-response -> the send
# ends silently; the response channel is the failed thing, so there
# is no surface left to report to and no state depends on it.
def test_283_send_survives_dead_peer():
    before = browser_failures.failure_count()
    mgr = tk.TakeoverManager(tk.TakeoverConfig())
    handler_cls = tk._make_handler(mgr)
    handler = handler_cls.__new__(handler_cls)
    handler.send_response = lambda *a: None
    handler.send_header = lambda *a: None
    handler.end_headers = lambda *a: None

    class DeadWfile:
        def write(self, body):
            raise BrokenPipeError("peer gone")

    handler.wfile = DeadWfile()
    handler._send(200, b"{}")  # must not raise

    class GoodWfile:
        def __init__(self):
            self.data = b""

        def write(self, body):
            self.data += body

    good = GoodWfile()
    handler.wfile = good
    handler._send(200, b"hello")
    assert good.data == b"hello"  # honest control
    assert browser_failures.failure_count() == before


# 503 — mark_done: handoff.owner_note raises -> swallowed; the grant
# state (done, done_note) was already set under the lock and the
# audit entry still runs — the note is bookkeeping only.
def test_503_owner_note_failure_keeps_done_state():
    before = browser_failures.failure_count()
    mgr = tk.TakeoverManager(tk.TakeoverConfig())

    class NoteBoomHandoff:
        def __init__(self):
            self.notes = []

        def owner_note(self, note):
            self.notes.append(note)
            raise RuntimeError("note boom")

    grant = _grant("s1")
    grant.handoff = NoteBoomHandoff()
    mgr.mark_done(grant, note="done")
    assert grant.handoff.notes == ["done"]  # note was really attempted
    assert grant.done is True

    class NoteOkHandoff:
        def __init__(self):
            self.notes = []

        def owner_note(self, note):
            self.notes.append(note)

    grant2 = _grant("s2")
    grant2.handoff = NoteOkHandoff()
    mgr.mark_done(grant2, note="finished")
    assert grant2.handoff.notes == ["finished"]  # honest control
    assert grant2.done is True
    assert browser_failures.failure_count() == before


# 548 — _safe_check: a crashing cleared-check reads as NOT cleared.
# Fail-closed: wait_for_resolution returns still_blocked, and its
# caller saves/continues only on an actual clear.
def test_548_crashing_check_reads_not_cleared():
    before = browser_failures.failure_count()
    mgr = tk.TakeoverManager(tk.TakeoverConfig())

    def boom():
        raise RuntimeError("check boom")

    assert mgr._safe_check(boom) is False
    assert mgr._safe_check(lambda: True) is True  # honest controls
    assert mgr._safe_check(lambda: False) is False
    assert browser_failures.failure_count() == before


# 602 — frame_for: the screenshot raises -> None, which the handler
# serves as an explicit 502 "no frame available"; no frame is faked.
def test_602_screenshot_failure_is_no_frame():
    before = browser_failures.failure_count()
    mgr = tk.TakeoverManager(tk.TakeoverConfig())

    class ShotBoomTab:
        url = "https://x.test/"

        def screenshot_png(self):
            raise RuntimeError("shot boom")

    class ShotOkTab:
        url = "https://x.test/"

        def screenshot_png(self):
            return b"\x89PNG\r\n\x1a\nFAKE"

    assert mgr.frame_for(_grant("s3", tab=ShotBoomTab())) is None
    assert mgr.frame_for(_grant("s4", tab=ShotOkTab())) == \
        b"\x89PNG\r\n\x1a\nFAKE"  # honest control
    assert browser_failures.failure_count() == before


# 617 — shutdown: server.shutdown raises -> swallowed ("shutdown is
# best effort"); the manager still drops its server reference.
def test_617_shutdown_failure_is_best_effort():
    before = browser_failures.failure_count()
    mgr = tk.TakeoverManager(tk.TakeoverConfig())

    class ServerBoom:
        def __init__(self):
            self.calls = []

        def shutdown(self):
            self.calls.append("shutdown")
            raise RuntimeError("shutdown boom")

        def server_close(self):
            self.calls.append("close")

    boom = ServerBoom()
    mgr._server = boom
    mgr.shutdown()  # must not raise
    assert boom.calls == ["shutdown"]  # shutdown was really attempted
    assert mgr._server is None

    class ServerOk:
        def __init__(self):
            self.calls = []

        def shutdown(self):
            self.calls.append("shutdown")

        def server_close(self):
            self.calls.append("close")

    ok = ServerOk()
    mgr._server = ok
    mgr.shutdown()
    assert ok.calls == ["shutdown", "close"]  # honest control
    assert mgr._server is None
    assert browser_failures.failure_count() == before
