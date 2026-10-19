"""Pass #4: browser + takeover fixes (failing-first).

1. The job's ``wall_escalated`` ping must carry the takeover link — the
   solver's escalation message has it, but the job-scoped ping (the one
   the owner sees in the job's context) said only "escalation sent"
   with no actionable link.
2. ``SolvedSessionStore.save`` used a pid-only tmp name: two threads in
   the same process (two concurrent ``/browser run`` jobs on one site)
   shared the tmp path and could interleave writes into torn JSON.
   Tmp names must be unique per call.
3. ``ChallengeSolver.solve`` let ``takeover_manager.request_takeover``
   raise (e.g. stream-server bind failure): the encounter was never
   recorded in memory and the outcome was a raw exception instead of
   an honest ``takeover_failed`` outcome.
"""
from __future__ import annotations

import threading
from pathlib import Path

from nomorals.browser.challenge import ChallengeSolver
from nomorals.browser.challenge_memory import ChallengeMemory
from nomorals.browser.jobs import BrowserRunJob
from nomorals.browser.solved_sessions import SolvedSessionStore

CF_HTML = (
    "<html><head><title>Just a moment...</title></head><body>"
    "Verify you are human — checking your browser</body></html>"
)
URL = "https://example.test/"
LINK = "http://127.0.0.1:9/takeover/TOKEN123"


class FakeTab:
    def __init__(self, html: str = CF_HTML) -> None:
        self._html = html
        self._url = URL
        self.reloads = 0

    @property
    def url(self) -> str:
        return self._url

    @property
    def title(self) -> str:
        return "Just a moment..."

    def html(self) -> str:
        return self._html

    def reload(self) -> None:
        self.reloads += 1


# -- 1: escalation ping carries the link -------------------------------------

def test_wall_escalated_ping_carries_takeover_link():
    sent: list[str] = []
    job = BrowserRunJob(job_id="br-link1", url=URL, goal="browse",
                        _notify=lambda m, u: sent.append(m))
    job.wall_event("wall_escalated",
                   {"detection": "cloudflare", "via": "takeover_pending",
                    "takeover_url": LINK})
    assert sent, "escalation ping was not sent"
    assert LINK in sent[0], (
        "job's escalation ping has no takeover link; the owner cannot "
        "act on it")


def test_wall_escalated_ping_without_link_still_sends():
    sent: list[str] = []
    job = BrowserRunJob(job_id="br-link2", url=URL, goal="browse",
                        _notify=lambda m, u: sent.append(m))
    job.wall_event("wall_escalated",
                   {"detection": "cloudflare", "via": "takeover_pending"})
    assert sent, "escalation ping was not sent"
    assert "Takeover link" not in sent[0]


# -- 2: concurrent same-site saves use distinct tmp files --------------------

def test_concurrent_same_site_saves_use_distinct_tmp_files(tmp_path,
                                                           monkeypatch):
    store = SolvedSessionStore(tmp_path)
    seen: list[str] = []
    lock = threading.Lock()
    orig_write_text = Path.write_text

    def recording_write_text(self, data, *args, **kwargs):
        if self.name.startswith("example.test."):
            with lock:
                seen.append(str(self))
        return orig_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    def worker(n: int) -> None:
        for i in range(25):
            store.save(URL, URL, [{
                "name": f"c{n}", "value": f"v{n}-{i}-{'x' * 300}",
                "domain": "example.test", "path": "/",
            }])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen, "no tmp writes were recorded"
    assert len(set(seen)) == len(seen), (
        f"{len(seen) - len(set(seen))} tmp-path collisions across threads: "
        "concurrent saves share one tmp file and can tear each other's JSON")


def test_concurrent_same_site_saves_leave_valid_record(tmp_path):
    store = SolvedSessionStore(tmp_path)

    def worker(n: int) -> None:
        for i in range(25):
            store.save(URL, URL, [{
                "name": f"c{n}", "value": f"v{n}-{i}-{'x' * 300}",
                "domain": "example.test", "path": "/",
            }])

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    record = store.load(URL)
    assert record is not None and record.get("cookies"), (
        "final solved-session record is missing or torn after "
        "concurrent same-site saves")


# -- 3: takeover request failure is an honest outcome -------------------------

class ExplodingTakeoverManager:
    def request_takeover(self, **kwargs):
        raise OSError("bind failed: address already in use")


def test_takeover_request_failure_returns_honest_outcome(tmp_path):
    memory = ChallengeMemory(tmp_path / "challenge-memory.sqlite3")
    solver = ChallengeSolver(
        takeover_manager=ExplodingTakeoverManager(), memory=memory,
        sleeper=lambda s: None, wait_s=0.0)
    outcome = solver.solve(tab=FakeTab())
    assert outcome.cleared is False
    assert outcome.via == "takeover_failed", (
        f"expected an honest takeover_failed outcome, got {outcome.via!r}")
    assert "bind failed" in outcome.message
    assert outcome.takeover_grant is None
    assert memory.stats()["encounters"] == 1, (
        "the failed-takeover encounter was never recorded in memory")
