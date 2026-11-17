"""Antivirus wiring bundle: scan on intake + scheduled sweep + ``nm av``.

Failing-first: on the pristine base the scanner itself exists
(``core/antivirus.py``, register §11(1)) and the ``av_scan`` tool
exists — but nothing ever calls it on a download, no sweep job
exists, and there is no ``nm av`` verb. Every test here fails there
(``ImportError`` / ``AttributeError`` / missing command).

1. Scan on intake — ``MediaDownloader._succeed`` is the single funnel
   every landed download passes through (two call sites, both into
   ``_succeed``). The intake scan runs there, after the extension fix
   and before the ledger / artifact registration / archive unpack:
   a clean file flows on, an unknown verdict fails open (the reason
   recorded on the report), an infected file is quarantined and the
   download is reported blocked with the threat named — never handed
   to the owner.
2. Scheduled sweep — ``security/av_sweep.py`` registers a durable
   daily job (03:00 local) whose tick scans the media dir, quarantines
   hits, publishes through the Notifier only when there is something
   to report, and writes durable last-sweep state.
3. ``nm av`` — status (engine + last sweep + quarantine), scan a path
   on demand, run a sweep now, list the quarantine.

One correction to the scout's note, recorded here because the brief
carried it: ``agents/apkforge/dissect.py:276`` is APKiD's own
``scanner.scan_file`` — a different scanner, not a consumer of
``core.antivirus``. Before this lane the only live consumer of the
antivirus scanner was the ``av_scan`` tool.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

# The EICAR standard test string, assembled from parts so this file
# never contains the contiguous signature.
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def eicar_bytes() -> bytes:
    return (_EICAR_A + _EICAR_B).encode("ascii")


def _ctx(home: Path):
    """Fake Devon context: settings resolve home + quarantine + media."""
    home = Path(home)
    settings = SimpleNamespace(
        home=str(home), home_path=home,
        workspace_dir=home / "workspace")

    def _resolve(relative: str) -> Path:
        p = Path(os.path.expanduser(relative))
        return p if p.is_absolute() else (home / p).resolve()

    settings.resolve = _resolve
    return SimpleNamespace(db=None, settings=settings)


def _ns(**kwargs):
    args = argparse.Namespace(
        av_action="status", av_rest=[], json=False,
        no_quarantine=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


class _FakeNotifier:
    def __init__(self):
        self.calls: list[dict] = []

    def publish(self, kind, title="", body="", **kwargs):
        self.calls.append({"kind": kind, "title": title, "body": body})
        return {"delivered": True}


class _FakeScheduler:
    """Minimal stand-in for the real Scheduler's job surface."""

    def __init__(self):
        self.jobs: list[dict] = []
        self._seq = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, payload_kind, payload, **kwargs):
        self._seq += 1
        job = {"id": f"job-{self._seq}", "name": name,
               "spec": spec, "payload_kind": payload_kind,
               "payload": payload, **kwargs}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        before = len(self.jobs)
        self.jobs = [j for j in self.jobs if j["id"] != job_id]
        return len(self.jobs) < before


class _AvEnvCase(unittest.TestCase):
    """Base: the process-wide antivirus cache is restored after each
    test so a faked engine never leaks into a later test file."""

    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("NM_AV", None)
        os.environ.pop("NM_AV_QUARANTINE", None)
        from nomorals.core.antivirus import detect_antivirus

        detect_antivirus(refresh=True)

    def tearDown(self):
        os.environ.pop("NM_AV", None)
        os.environ.pop("NM_AV_QUARANTINE", None)
        from nomorals.core.antivirus import detect_antivirus

        detect_antivirus(refresh=True)
        self._env.stop()


# ── quarantine-dir resolution ──────────────────────────────────────

class QuarantineDirTests(_AvEnvCase):
    def test_env_override_wins(self):
        from nomorals.core.antivirus import quarantine_dir_for

        with mock.patch.dict(os.environ,
                             {"NM_AV_QUARANTINE": "~/qdir"}):
            got = quarantine_dir_for(_ctx(Path(tempfile.mkdtemp())))
        self.assertEqual(got, Path(os.path.expanduser("~/qdir")))

    def test_settings_resolve_next(self):
        from nomorals.core.antivirus import quarantine_dir_for

        home = Path(tempfile.mkdtemp())
        self.assertEqual(quarantine_dir_for(_ctx(home)),
                         (home / "quarantine").resolve())

    def test_no_settings_no_workspace_is_report_only(self):
        from nomorals.core.antivirus import quarantine_dir_for

        self.assertIsNone(quarantine_dir_for(SimpleNamespace()))


# ── intake scan ────────────────────────────────────────────────────

class IntakeScanTests(_AvEnvCase):
    def _report(self, path: Path):
        from nomorals.media.downloader import DownloadReport

        return DownloadReport(url="https://example.invalid/f.bin",
                              ok=True, path=str(path),
                              size_bytes=path.stat().st_size,
                              extractor="test")

    def test_clean_file_passes_through(self):
        from nomorals.media.downloader import _scan_intake

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "hello.txt"
            target.write_text("hello, world\n", encoding="utf-8")
            report = self._report(target)
            _scan_intake(report, _ctx(Path(td)))
            self.assertTrue(report.ok)
            self.assertEqual(report.path, str(target))
            self.assertEqual(report.av["verdict"], "clean")
            self.assertEqual(report.av["engine"], "builtin")

    def test_eicar_is_blocked_and_quarantined(self):
        from nomorals.media.downloader import _scan_intake

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            target = home / "eicar.com"
            target.write_bytes(eicar_bytes())
            report = self._report(target)
            _scan_intake(report, _ctx(home))
            self.assertFalse(report.ok)
            self.assertEqual(report.failure.value, "site_error")
            self.assertIn("EICAR-Test-File", report.likely_cause)
            self.assertIn("quarantined", report.likely_cause)
            self.assertEqual(report.path, "")
            self.assertFalse(target.exists())
            self.assertEqual(report.av["verdict"], "infected")
            qdir = (home / "quarantine").resolve()
            self.assertTrue(report.av["quarantined_to"].startswith(
                str(qdir)))
            self.assertTrue(Path(report.av["quarantined_to"]).exists())
            # the sidecar names the original path
            sidecar = Path(report.av["quarantined_to"] + ".json")
            self.assertIn("eicar.com", sidecar.read_text(
                encoding="utf-8"))

    def test_unknown_verdict_fails_open(self):
        from nomorals.media.downloader import _scan_intake

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "hello.txt"
            target.write_text("hello, world\n", encoding="utf-8")
            report = self._report(target)
            with mock.patch.dict(os.environ, {"NM_AV": "none"}):
                from nomorals.core.antivirus import detect_antivirus

                detect_antivirus(refresh=True)
                _scan_intake(report, _ctx(Path(td)))
            self.assertTrue(report.ok)
            self.assertEqual(report.path, str(target))
            self.assertEqual(report.av["verdict"], "unknown")
            self.assertIn("no antivirus engine",
                          report.av["detail"])

    def test_scan_failure_fails_open(self):
        from nomorals.media.downloader import _scan_intake

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "hello.txt"
            target.write_text("hello, world\n", encoding="utf-8")
            report = self._report(target)
            with mock.patch(
                    "nomorals.core.antivirus.scan_file",
                    side_effect=RuntimeError("boom")):
                _scan_intake(report, _ctx(Path(td)))
            self.assertTrue(report.ok)
            self.assertEqual(report.av["verdict"], "unknown")
            self.assertIn("boom", report.av["detail"])

    def test_quarantine_failure_still_blocks(self):
        from nomorals.media.downloader import _scan_intake

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            # the quarantine dir is a plain file: the move must fail
            (home / "quarantine").write_text("not a dir",
                                             encoding="utf-8")
            target = home / "eicar.com"
            target.write_bytes(eicar_bytes())
            report = self._report(target)
            _scan_intake(report, _ctx(home))
            self.assertFalse(report.ok)
            self.assertIn("EICAR-Test-File", report.likely_cause)
            self.assertIn("quarantine failed", report.likely_cause)
            # the file stays where it was, but the report no longer
            # points at it — nothing hands it to the owner
            self.assertTrue(target.exists())
            self.assertEqual(report.path, "")

    def test_missing_path_is_not_scanned(self):
        from nomorals.media.downloader import (
            DownloadReport,
            _scan_intake,
        )

        report = DownloadReport(url="https://example.invalid/f.bin",
                                ok=True, path="/definitely/not/here.bin",
                                size_bytes=0, extractor="test")
        _scan_intake(report, _ctx(Path(tempfile.mkdtemp())))
        self.assertTrue(report.ok)
        self.assertIsNone(report.av)


class SucceedWiringTests(_AvEnvCase):
    """The intake scan runs inside ``_succeed`` — the funnel — and an
    infected file never reaches the artifact registration, the ledger,
    or the archive unpack."""

    def _downloader(self, home: Path):
        from nomorals.media.downloader import MediaDownloader

        dl = MediaDownloader(_ctx(home), direct_retries=1,
                             max_proxies=1)
        self._calls: list[str] = []
        calls = self._calls
        dl._record_download_outcome = lambda *a, **k: calls.append(
            "outcome")
        dl._learn_from_success = lambda *a, **k: calls.append("learn")
        dl._intake_exploration = lambda *a, **k: calls.append(
            "explore")
        dl._register_download_artifact = lambda *a, **k: calls.append(
            "artifact")
        return dl

    def _report(self, path: Path):
        from nomorals.media.downloader import DownloadReport

        return DownloadReport(url="https://example.invalid/f.bin",
                              ok=True, path=str(path),
                              size_bytes=path.stat().st_size,
                              extractor="test")

    def test_infected_never_reaches_artifact_or_unpack(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = home / "eicar.com"
            target.write_bytes(eicar_bytes())
            out = dl._succeed(self._report(target), 0.0)
            self.assertFalse(out.ok)
            self.assertEqual(out.av["verdict"], "infected")
            self.assertEqual(self._calls, [])
            self.assertIsNone(out.archive)

    def test_clean_flows_through_bookkeeping(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            dl = self._downloader(home)
            target = home / "hello.txt"
            target.write_text("hello, world\n", encoding="utf-8")
            out = dl._succeed(self._report(target), 0.0)
            self.assertTrue(out.ok)
            self.assertEqual(out.av["verdict"], "clean")
            self.assertEqual(self._calls,
                             ["outcome", "learn", "explore", "artifact"])


# ── scheduled sweep ────────────────────────────────────────────────

class SweepJobTests(_AvEnvCase):
    def test_spec_parses_as_daily_cron(self):
        from nomorals.agents.scheduler import parse_schedule_spec
        from nomorals.security.av_sweep import AV_SWEEP_SPEC

        kind, _parsed = parse_schedule_spec(AV_SWEEP_SPEC)
        self.assertEqual(kind, "cron")

    def test_ensure_registers_and_is_idempotent(self):
        from nomorals.security.av_sweep import (
            AV_SWEEP_JOB,
            AV_SWEEP_SPEC,
            AV_SWEEP_TOOL,
            ensure_av_sweep_job,
        )

        sched = _FakeScheduler()
        first = ensure_av_sweep_job(sched)
        self.assertEqual(first["name"], AV_SWEEP_JOB)
        self.assertEqual(first["spec"], AV_SWEEP_SPEC)
        self.assertEqual(first["payload"]["tool"], AV_SWEEP_TOOL)
        second = ensure_av_sweep_job(sched)
        self.assertEqual(second["id"], first["id"])
        self.assertEqual(len(sched.jobs), 1)

    def test_ensure_replaces_stale_spec(self):
        from nomorals.security.av_sweep import (
            AV_SWEEP_JOB,
            ensure_av_sweep_job,
        )

        sched = _FakeScheduler()
        sched.add(AV_SWEEP_JOB, "cron 0 4 * * *", "tool",
                  {"tool": "__av_sweep_tick__", "args": {}})
        ensure_av_sweep_job(sched)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(sched.jobs[0]["spec"],
                         "cron 0 3 * * *")

    def test_sweep_quarantines_and_reports(self):
        from nomorals.security.av_sweep import run_av_sweep_tick_once

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            media = home / "media"
            media.mkdir()
            (media / "clean.txt").write_text("fine\n", encoding="utf-8")
            bad = media / "eicar.com"
            bad.write_bytes(eicar_bytes())
            qdir = home / "quarantine"
            state = home / "data" / "av_sweep.json"
            notifier = _FakeNotifier()
            out = run_av_sweep_tick_once(
                notifier=notifier, roots=[media], quarantine_dir=qdir,
                state_path=state)
            self.assertTrue(out["ok"])
            self.assertEqual(out["scanned"], 2)
            self.assertEqual(out["infected"], 1)
            self.assertEqual(out["clean"], 1)
            self.assertFalse(bad.exists())
            self.assertEqual(len(out["quarantined"]), 1)
            self.assertEqual(len(notifier.calls), 1)
            self.assertEqual(notifier.calls[0]["kind"], "av_sweep")
            self.assertIn("EICAR-Test-File", notifier.calls[0]["body"])
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["infected"], 1)
            self.assertEqual(saved["scanned"], 2)

    def test_clean_sweep_is_silent_but_recorded(self):
        from nomorals.security.av_sweep import run_av_sweep_tick_once

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            media = home / "media"
            media.mkdir()
            (media / "clean.txt").write_text("fine\n", encoding="utf-8")
            state = home / "data" / "av_sweep.json"
            notifier = _FakeNotifier()
            out = run_av_sweep_tick_once(
                notifier=notifier, roots=[media],
                quarantine_dir=home / "quarantine", state_path=state)
            self.assertTrue(out["ok"])
            self.assertEqual(out["infected"], 0)
            self.assertEqual(notifier.calls, [])
            saved = json.loads(state.read_text(encoding="utf-8"))
            self.assertEqual(saved["clean"], 1)

    def test_sweep_never_raises(self):
        from nomorals.security.av_sweep import run_av_sweep_tick_once

        out = run_av_sweep_tick_once(roots=None, context=None)
        self.assertFalse(out["ok"])
        self.assertIn("error", out)
        self.assertTrue(out["error"])


# ── nm av ──────────────────────────────────────────────────────────

class AvCliTests(_AvEnvCase):
    def test_status_names_engine_and_last_sweep(self):
        from nomorals.cmdline.commands.av import _cmd_av
        from nomorals.security.av_sweep import run_av_sweep_tick_once

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            media = home / "media"
            media.mkdir()
            (media / "clean.txt").write_text("fine\n", encoding="utf-8")
            run_av_sweep_tick_once(
                roots=[media], quarantine_dir=home / "quarantine",
                state_path=home / "data" / "av_sweep.json")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="status"), _ctx(home))
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("builtin", text)
            self.assertIn("last sweep", text)
            self.assertIn("scanned 1", text)

    def test_status_without_state_is_honest(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="status"), _ctx(Path(td)))
            self.assertEqual(rc, 0)
            self.assertIn("never run", buf.getvalue())

    def test_scan_reports_verdict(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            target = home / "eicar.com"
            target.write_bytes(eicar_bytes())
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="scan", av_rest=[str(target)]),
                             _ctx(home))
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("infected", text)
            self.assertIn("EICAR-Test-File", text)
            self.assertFalse(target.exists())

    def test_scan_needs_a_path(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _cmd_av(_ns(av_action="scan", av_rest=[]),
                             _ctx(Path(td)))
            self.assertEqual(rc, 2)

    def test_scan_json(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            target = home / "clean.txt"
            target.write_text("fine\n", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="scan", av_rest=[str(target)],
                                 json=True), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertEqual(json.loads(buf.getvalue())["verdict"],
                             "clean")

    def test_sweep_runs_now(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            media = home / "workspace" / "media"
            media.mkdir(parents=True)
            (media / "clean.txt").write_text("fine\n", encoding="utf-8")
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="sweep"), _ctx(home))
            self.assertEqual(rc, 0)
            self.assertIn("swept 1 file(s)", buf.getvalue())

    def test_quarantine_lists_hits(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            target = home / "eicar.com"
            target.write_bytes(eicar_bytes())
            _cmd_av(_ns(av_action="scan", av_rest=[str(target)]),
                    _ctx(home))
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="quarantine"), _ctx(home))
            self.assertEqual(rc, 0)
            text = buf.getvalue()
            self.assertIn("EICAR-Test-File", text)
            self.assertIn("eicar.com", text)

    def test_unknown_action_is_usage_error(self):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stderr(buf):
                rc = _cmd_av(_ns(av_action="explode"), _ctx(Path(td)))
            self.assertEqual(rc, 2)

    def test_parser_and_dispatch_cover_av(self):
        from nomorals.cmdline import _COMMANDS

        self.assertIn("av", _COMMANDS)
        self.assertTrue(callable(_COMMANDS["av"].handler))
        from nomorals.cmdline.commands.meta import _cli_subparsers

        sub = _cli_subparsers()
        self.assertIn("av", sub.choices)


if __name__ == "__main__":
    unittest.main()
