"""Browser area, second wave: playbooks, profiles, flows, recorder,
downloads, concurrency, handoff. No network, no real browser — every
test drives the new modules with fakes."""

import hashlib

import pytest

from nomorals.browser.playbooks import (
    PlaybookError,
    PlaybookRegistry,
    PlaybookStore,
    SitePlaybook,
    builtin_registry,
)
from nomorals.browser.profiles import BrowserProfile, ProfileStore
from nomorals.browser.flows import BrowserFlow, FlowError, FlowStep
from nomorals.browser.recorder import (
    Recording,
    RecordingDiffError,
    replay_or_raise,
)
from nomorals.browser.downloads import DownloadError, DownloadQueue
from nomorals.browser.concurrency import ConcurrencyError, DomainConcurrency
from nomorals.browser.handoff import HandoffError, HandoffSession


# ── playbooks ──────────────────────────────────────────────────────────────


def test_playbook_validates_and_matches():
    reg = builtin_registry()
    assert reg.names() == ["generic-article", "generic-listing", "generic-search"]
    pb = reg.get("generic-search")
    assert pb is not None and pb.is_valid
    assert pb.matches_host("news.example.com")
    assert not pb.matches_host("example.org")
    # Longest host-pattern match wins.
    assert reg.match_for_url("https://news.example.com/search?q=x") is pb
    assert reg.match_for_url("https://unrelated.test/x") is None


def test_playbook_validation_reports_every_problem():
    pb = SitePlaybook(name="", host="*.*")
    errs = pb.validate()
    assert any("name" in e for e in errs)
    assert any("host" in e for e in errs)
    assert any("search" in e for e in errs)
    assert any("success" in e for e in errs)
    assert not pb.is_valid
    with pytest.raises(PlaybookError):
        PlaybookRegistry([pb])


def test_playbook_bad_rate_and_fallback_rejected():
    pb = SitePlaybook.from_dict({
        "name": "bad", "host": "bad.example",
        "search": {"url_template": "https://bad.example/s?q={query}",
                   "results_selector": ".r"},
        "content": {"body_selector": "main"},
        "rate": {"min_delay_ms": -1, "max_requests_per_minute": 0},
        "success": {"required_selectors": ["main"]},
        "fallback": {"chain": ["explode", "abort", "retry"]},
    })
    errs = pb.validate()
    assert any("min_delay_ms" in e for e in errs)
    assert any("max_requests_per_minute" in e for e in errs)
    assert any("explode" in e for e in errs)
    assert any("abort" in e for e in errs)


def test_playbook_success_check_catches_block_pages():
    pb = builtin_registry().get("generic-search")
    ok, missing = pb.success.check(selectors_present=["main"],
                                   text="results here")
    assert ok and missing == []
    ok, missing = pb.success.check(selectors_present=["main"],
                                   text="Just a moment... verifying")
    assert not ok and any("forbidden" in m for m in missing)


def test_playbook_store_round_trip(tmp_path):
    store = PlaybookStore(tmp_path / "playbooks.json")
    reg = builtin_registry()
    store.save(reg)
    loaded = store.load()
    assert loaded.names() == reg.names()
    assert loaded.get("generic-listing").login.url == "https://example.net/login"
    # Missing file -> empty registry, not an error.
    assert PlaybookStore(tmp_path / "nope.json").load().names() == []


# ── profiles ───────────────────────────────────────────────────────────────


def test_profile_save_load_and_header_rotation(tmp_path):
    store = ProfileStore(tmp_path / "profiles")
    profile = BrowserProfile(
        name="owner-main",
        cookies=[{"name": "session", "value": "SECRET-VALUE",
                  "domain": "example.com", "path": "/"}],
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        proxy="http://user:hunter2@proxy.example:8080",
    )
    assert profile.is_valid
    store.save(profile)
    assert store.list_profiles() == ["owner-main"]
    loaded = store.load("owner-main")
    assert loaded.user_agent == profile.user_agent
    # Cookie values never appear in the masked summary.
    assert "SECRET-VALUE" not in str(loaded.masked_summary())
    assert "hunter2" not in loaded.masked_proxy()
    # Rotation cycles real-browser UAs only, and persists.
    before = loaded.user_agent
    loaded.rotate_headers()
    assert loaded.user_agent != before
    store.save(loaded)
    assert store.load("owner-main").rotation_index == loaded.rotation_index
    # Cookie header matches domain/path, skips other domains.
    assert "session=SECRET-VALUE" in loaded.cookie_header_for("https://example.com/x")
    assert loaded.cookie_header_for("https://other.test/") == ""


def test_profile_rejects_spoofing_headers():
    profile = BrowserProfile(name="bad", headers={"X-Fake-Device": "Pixel"})
    assert any("X-Fake-Device" in e for e in profile.validate())


# ── flows ──────────────────────────────────────────────────────────────────


def _recording_executor(seen):
    def run(step):
        seen.append(step)
        if step["kind"] == "navigate":
            return {"ok": True, "url": step["target"]}
        return {"ok": True}
    return run


def test_flow_survives_restart_and_resumes(tmp_path):
    checkpoint = tmp_path / "flow.json"
    seen_first: list = []
    flow = BrowserFlow(
        [FlowStep(kind="navigate", target="https://example.com"),
         FlowStep(kind="click", target="a.next"),
         FlowStep(kind="extract", target="main")],
        checkpoint_path=checkpoint, executor=_recording_executor(seen_first))
    flow.run_step()
    flow.run_step()
    assert flow.index == 2 and flow.status == "running"
    flow.pause()
    assert flow.status == "paused"
    # Fresh object, fresh executor: resumes at step 3, never replays 1-2.
    seen_second: list = []
    revived = BrowserFlow.load(checkpoint, executor=_recording_executor(seen_second))
    assert revived.index == 2 and revived.status == "paused"
    revived.resume()
    assert revived.done and revived.index == 3
    assert [s["kind"] for s in seen_second] == ["extract"]
    assert len(revived.results) == 3


def test_flow_failure_marks_failed_and_can_retry(tmp_path):
    calls = {"n": 0}

    def flaky(step):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return {"ok": True}

    flow = BrowserFlow([FlowStep(kind="click", target="button")],
                       checkpoint_path=tmp_path / "f.json", executor=flaky)
    with pytest.raises(RuntimeError):
        flow.run_step()
    assert flow.status == "failed"
    flow.reset_failed()
    flow.resume()
    assert flow.done


def test_flow_rejects_unknown_step_kind():
    with pytest.raises(FlowError):
        BrowserFlow([FlowStep(kind="teleport", target="x")])


# ── recorder ───────────────────────────────────────────────────────────────


def test_recorder_diff_is_never_silently_ignored():
    rec = Recording("checkout")
    rec.record_action("click", "button.buy")
    rec.snapshot("h1", "Product page")
    rec.snapshot("p.price", "$10")
    # Identical DOM -> clean.
    report = rec.diff({"h1": "Product page", "p.price": "$10"})
    assert not report.has_diffs
    # Changed text -> a changed entry, with both hashes.
    report = rec.diff({"h1": "Product page", "p.price": "$12"})
    assert report.has_diffs and len(report.changed) == 1
    assert report.changed[0].selector == "p.price"
    assert report.changed[0].expected_hash != report.changed[0].actual_hash
    # Missing selector -> a missing entry.
    report = rec.diff({"h1": "Product page"})
    assert len(report.missing) == 1 and report.missing[0].selector == "p.price"
    # New selector -> an added entry (current text supplied by the caller).
    report = rec.diff({"h1": "Product page", "p.price": "$10",
                       "div.banner": "promo!"},
                      extra_selectors=["div.banner"])
    assert len(report.added) == 1
    # assert_no_diff raises carrying the full report.
    with pytest.raises(RecordingDiffError) as excinfo:
        rec.assert_no_diff({"h1": "Product page", "p.price": "$12"})
    assert excinfo.value.report.has_diffs


def test_recorder_replay_runs_actions_then_diffs():
    rec = Recording("r")
    rec.record_action("navigate", "https://example.com")
    rec.snapshot("h1", "Hello")
    dom = {"h1": "Hello"}
    ran = []

    def executor(action):
        ran.append(action)
        return {"ok": True}

    report = rec.replay(executor, lambda sel: dom.get(sel))
    assert not report.has_diffs and ran[0]["kind"] == "navigate"
    dom["h1"] = "Goodbye"
    with pytest.raises(RecordingDiffError):
        replay_or_raise(rec, executor, lambda sel: dom.get(sel))


# ── downloads ──────────────────────────────────────────────────────────────


def _fake_fetcher_factory(payload: bytes, *, fail=False):
    def fetcher(job):
        dest = job["dest"]
        if fail:
            return {"ok": False, "error": "network down", "backend": "test-fetcher"}
        import pathlib
        pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(dest).write_bytes(payload)
        return {"ok": True, "path": dest, "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "resumed": False, "backend": "test-fetcher"}
    return fetcher


def test_download_queue_runs_and_verifies(tmp_path):
    payload = b"hello-download"
    queue = DownloadQueue(tmp_path / "q.json",
                          fetcher=_fake_fetcher_factory(payload))
    job = queue.enqueue("https://example.com/file.bin",
                        tmp_path / "file.bin",
                        expected_sha256=hashlib.sha256(payload).hexdigest())
    done = queue.run_job(job.job_id)
    assert done.status == "done"
    assert done.sha256 == hashlib.sha256(payload).hexdigest()
    assert queue.verify(job.job_id) == (True, "hash matches recorded sha256")


def test_download_integrity_mismatch_fails_job(tmp_path):
    payload = b"real-bytes"
    queue = DownloadQueue(tmp_path / "q.json",
                          fetcher=_fake_fetcher_factory(payload))
    job = queue.enqueue("https://example.com/file.bin", tmp_path / "file.bin",
                        expected_sha256="0" * 64)
    done = queue.run_job(job.job_id)
    assert done.status == "failed"
    assert "integrity_mismatch" in done.error


def test_download_queue_persists_and_resumes(tmp_path):
    payload = b"persisted"
    queue = DownloadQueue(tmp_path / "q.json",
                          fetcher=_fake_fetcher_factory(payload))
    job = queue.enqueue("https://example.com/f.bin", tmp_path / "f.bin")
    queue.pause(job.job_id)
    # A fresh queue loaded from disk sees the paused job and resumes it.
    revived = DownloadQueue(tmp_path / "q.json",
                            fetcher=_fake_fetcher_factory(payload))
    assert revived.get(job.job_id).status == "paused"
    done = revived.resume(job.job_id)
    assert done.status == "done"
    with pytest.raises(DownloadError):
        queue.enqueue("ftp://example.com/f", tmp_path / "x")


# ── concurrency ────────────────────────────────────────────────────────────


def test_per_domain_cap_enforced_and_other_domains_unaffected():
    limiter = DomainConcurrency(max_per_domain=2, global_cap=10)
    assert limiter.try_acquire("https://a.example/x")
    assert limiter.try_acquire("https://a.example/y")
    # Third concurrent slot on the same domain is refused.
    assert not limiter.try_acquire("https://a.example/z", timeout=0.05)
    # A different domain is unaffected.
    assert limiter.try_acquire("https://b.example/x")
    limiter.release("https://a.example/x")
    assert limiter.try_acquire("a.example")
    stats = limiter.stats()
    assert stats["global"]["in_use"] == 3
    assert stats["per_domain"]["a.example"]["in_use"] == 2
    with pytest.raises(ConcurrencyError):
        limiter.release("https://never-acquired.example/")


def test_global_cap_enforced():
    limiter = DomainConcurrency(max_per_domain=5, global_cap=2)
    assert limiter.try_acquire("https://a.example/1")
    assert limiter.try_acquire("https://b.example/1")
    assert not limiter.try_acquire("https://c.example/1", timeout=0.05)
    limiter.release("https://b.example/1")
    with limiter.slot("https://d.example/1"):
        assert limiter.stats()["global"]["in_use"] == 2
    # Slot released on context exit: a new acquire succeeds.
    assert limiter.try_acquire("https://c.example/1")


def test_concurrency_rejects_bad_config():
    with pytest.raises(ConcurrencyError):
        DomainConcurrency(max_per_domain=0)
    with pytest.raises(ConcurrencyError):
        DomainConcurrency(global_cap=0)


# ── handoff ────────────────────────────────────────────────────────────────


def test_handoff_state_machine_full_cycle():
    h = HandoffSession("h1", task_label="checkout")
    assert h.state == "idle"
    # Owner cannot take the mouse before the agent pauses.
    with pytest.raises(HandoffError):
        h.hand_mouse()
    h.start_live()
    assert h.state == "live"
    shown = h.show_live()
    assert shown["state"] == "live"  # read-only: no state change
    h.pause()
    h.hand_mouse(note="solving captcha")
    assert h.state == "handed"
    h.owner_note("solved the captcha, clicked continue")
    h.resume()
    assert h.state == "resumed"
    h.finish()
    assert h.state == "done"
    with pytest.raises(HandoffError):
        h.resume()


def test_handoff_paused_resume_without_owner_action():
    h = HandoffSession("h2")
    h.start_live()
    h.pause()
    h.resume()
    assert h.state == "resumed"


def test_handoff_survives_restart_still_handed(tmp_path):
    path = tmp_path / "handoff.json"
    h = HandoffSession("h3", state_path=path)
    h.start_live()
    h.pause()
    h.hand_mouse()
    revived = HandoffSession.load(path)
    assert revived.state == "handed"
    # Still handed: owner notes work, resume works, auto-resume never happened.
    revived.owner_note("did the login")
    revived.resume()
    assert revived.state == "resumed"
    # Notes are rejected outside HANDED.
    with pytest.raises(HandoffError):
        revived.owner_note("too late")
