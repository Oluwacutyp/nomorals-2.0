"""One general command that installs every missing model.

The owner asked (2026-11-14): "I meant a command that generally
install every missing model." The per-area commands already exist
(``/character provision`` for the three character backends,
``/dj desk provision`` for madmom) — this suite pins the ONE command
that covers all of Devon's models: ``/models``.

Inventory (from the code, not a wishlist):

* characters — xtts / rvc / faceswap, delegated to
  ``characters.provision.provision_backends`` (never duplicated)
* DJ — madmom, delegated to ``media.dj_desk.provision_madmom``
* voice STT — faster-whisper / parakeet / whisper-cpp (profile order
  from ``audio.pipeline._STT_ORDER``)
* voice TTS — chatterbox / kokoro / piper / f5tts (profile order from
  ``audio.pipeline._TTS_ORDER``; xtts is the character backend)
* image — sd15 (Stable Diffusion 1.5 weights) / ace-step (ACE-Step
  1.5 engine + weights)
* small ML — sklearn / lightgbm / xgboost / catboost
  (``llm.specialists.backend_available``)

Journal-trained learners (ladder jobs) are STATUS ONLY — they train
from journal rows, not downloads: rows held vs floor, trained or not
yet, never "installed". Codebeast and the big-merge specialists are
OUT OF SCOPE (separate programmes) and the report says so.

Every install/download is mocked here — no real pip installs, no real
weight downloads. The seams are module-level functions on
``nomorals.tools.models_provision`` (``_run``, ``_measure_box``,
``_detect_*``, the two delegates) patched per test.
"""

from __future__ import annotations

import json
import types

BIG_BOX = {"ok": True, "total_ram_gb": 16.0, "free_ram_gb": 10.0,
           "free_disk_gb": 120.0, "cpus": 8, "platform": "linux",
           "profile": "workstation", "detail": "big box"}

TINY_BOX = {"ok": True, "total_ram_gb": 0.5, "free_ram_gb": 0.4,
            "free_disk_gb": 0.5, "cpus": 1, "platform": "linux",
            "profile": "laptop", "detail": "tiny box"}

EXPECTED_BACKENDS = (
    "xtts", "rvc", "faceswap", "madmom",
    "faster-whisper", "parakeet", "whisper-cpp",
    "chatterbox", "kokoro", "piper", "f5tts",
    "sd15", "ace-step",
    "sklearn", "lightgbm", "xgboost", "catboost",
)

EXPECTED_LEARNERS = (
    "character_dialogue", "character_tactics",
    "dj_sound_selection", "dj_cue_suggest",
    "npc_line_scorer", "npc_tactics",
    "pumpdesk_win", "error_triage", "morning_rank",
)


def _ok_run(args, timeout=None):
    return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")


def _fail_run(args, timeout=None):
    return types.SimpleNamespace(returncode=1, stdout="",
                                 stderr="boom")


def _modelsprov():
    import nomorals.tools.models_provision as mp

    return mp


def _control_models():
    from nomorals.audio.models import control_models

    return control_models


def _char_report(name, status="installed"):
    return {"ok": status in ("installed", "already-installed"),
            "status": status, "detail": f"{name} {status}",
            "steps": [f"{name}: ok"]}


# ── import is inert ────────────────────────────────────────────────────

def test_import_performs_no_work(monkeypatch):
    import subprocess

    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.append(a) or _ok_run(a))
    mp = _modelsprov()
    assert mp.BACKENDS == EXPECTED_BACKENDS
    assert calls == []


# ── inventory ──────────────────────────────────────────────────────────

def test_inventory_lists_every_backend(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_BOX))
    got = [row["id"] for row in mp.inventory()]
    assert got == list(EXPECTED_BACKENDS)


def test_inventory_marks_installed_honestly(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_BOX))
    for name in EXPECTED_BACKENDS:
        monkeypatch.setattr(mp, f"_detect_{name.replace('-', '_')}",
                            lambda: False)
    rows = {row["id"]: row for row in mp.inventory()}
    assert all(not row["installed"] for row in rows.values())
    monkeypatch.setattr(mp, "_detect_piper", lambda: True)
    rows = {row["id"]: row for row in mp.inventory()}
    assert rows["piper"]["installed"] is True
    assert rows["kokoro"]["installed"] is False


def test_inventory_follows_profile_orders():
    mp = _modelsprov()
    stt_termux = [row["id"] for row in mp.inventory(profile="termux")
                  if row["family"] == "voice-stt"]
    assert stt_termux == ["parakeet", "whisper-cpp", "faster-whisper"]
    stt_laptop = [row["id"] for row in mp.inventory(profile="laptop")
                  if row["family"] == "voice-stt"]
    assert stt_laptop == ["faster-whisper", "parakeet", "whisper-cpp"]
    tts_termux = [row["id"] for row in mp.inventory(profile="termux")
                  if row["family"] == "voice-tts" and row["in_profile"]]
    assert tts_termux == ["chatterbox", "piper", "kokoro"]
    # f5tts is listed (the inventory is complete) but not for termux.
    f5 = [row for row in mp.inventory(profile="termux")
          if row["id"] == "f5tts"][0]
    assert f5["in_profile"] is False


# ── bare command provisions all missing ────────────────────────────────

def _mock_all_missing(monkeypatch, mp):
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(mp, "_run", _ok_run)
    for name in EXPECTED_BACKENDS:
        monkeypatch.setattr(mp, f"_detect_{name.replace('-', '_')}",
                            lambda: False)
    monkeypatch.setattr(
        mp, "_char_provision_backends",
        lambda backends=None: {
            "ok": True, "box": dict(BIG_BOX),
            "backends": {b: _char_report(b) for b in
                         (backends or ("xtts", "rvc", "faceswap"))}})
    monkeypatch.setattr(
        mp, "_madmom_provision",
        lambda dest=None: {"ok": True, "path": "/tmp/madmom",
                           "steps": ["madmom: ok"]})


def test_bare_provisions_all_missing(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    result = mp.provision_models()
    assert result["ok"] is True
    got = result["backends"]
    assert sorted(got) == sorted(EXPECTED_BACKENDS)
    assert all(rep["status"] in ("installed", "already-installed")
               for rep in got.values())


def test_single_backend_install_only(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    result = mp.provision_models(["piper"])
    assert sorted(result["backends"]) == ["piper"]
    assert result["backends"]["piper"]["status"] == "installed"


def test_unknown_backend_reported(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    result = mp.provision_models(["nope"])
    assert result["ok"] is False
    assert "nope" in (result.get("error") or "")


# ── honest box check ───────────────────────────────────────────────────

def test_unmet_floor_refuses_with_numbers(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(TINY_BOX))
    monkeypatch.setattr(mp, "_run", _ok_run)
    for name in EXPECTED_BACKENDS:
        monkeypatch.setattr(mp, f"_detect_{name.replace('-', '_')}",
                            lambda: False)
    calls = []
    monkeypatch.setattr(
        mp, "_char_provision_backends",
        lambda backends=None: calls.append("char") or {"ok": True,
                                                       "box": dict(TINY_BOX),
                                                       "backends": {}})
    monkeypatch.setattr(
        mp, "_madmom_provision",
        lambda dest=None: calls.append("madmom") or {"ok": True,
                                                     "path": "", "steps": []})
    result = mp.provision_models()
    assert result["ok"] is False
    # Nothing was installed: the delegates were never called, no pip ran.
    assert calls == []
    for name, rep in result["backends"].items():
        if name == "f5tts":
            # The tiny box is a laptop profile — f5tts is not in the
            # laptop order, so the profile skip fires before the floor.
            assert rep["status"] == "skipped", name
            continue
        assert rep["status"] == "unavailable", name
        # The measured numbers are in the detail, not a bare refusal.
        assert "0.5" in rep["detail"], name


def test_failed_pip_reports_failed_not_installed(monkeypatch):
    mp = _modelsprov()
    monkeypatch.setattr(mp, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(mp, "_run", _fail_run)
    monkeypatch.setattr(mp, "_detect_kokoro", lambda: False)
    result = mp.provision_models(["kokoro"])
    rep = result["backends"]["kokoro"]
    assert rep["status"] == "failed"
    assert rep["ok"] is False


# ── learners are status only ───────────────────────────────────────────

def test_learners_report_status_never_installed(monkeypatch, tmp_path):
    mp = _modelsprov()
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("NM_FAILURE_GRADES", str(tmp_path / "grades.jsonl"))
    monkeypatch.setenv("NM_NPC_LEARN_DIR", str(tmp_path / "npc"))
    monkeypatch.setenv("NOMORALS_PUMPDESK_DIR", str(tmp_path / "pump"))
    rows = mp.learner_status()
    assert [row["job"] for row in rows] == list(EXPECTED_LEARNERS)
    for row in rows:
        assert row["status"] != "installed", row["job"]
        assert row["status"] in ("not trained yet", "trained",
                                 "unknown"), row["job"]
        assert row["floor"] in (30, 40), row["job"]
        assert row["rows"] is None or row["rows"] >= 0, row["job"]
    # Empty journals: not trained yet, 0 rows held.
    by_job = {row["job"]: row for row in rows}
    assert by_job["error_triage"]["rows"] == 0
    assert by_job["error_triage"]["status"] == "not trained yet"


def test_learner_rows_counted_from_journals(monkeypatch, tmp_path):
    mp = _modelsprov()
    grades = tmp_path / "grades.jsonl"
    grades.write_text("\n".join(
        json.dumps({"site": "x", "real": True}) for _ in range(5)) + "\n")
    monkeypatch.setenv("NM_FAILURE_GRADES", str(grades))
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    rows = {row["job"]: row for row in mp.learner_status()}
    assert rows["error_triage"]["rows"] == 5
    assert rows["error_triage"]["status"] == "not trained yet"
    assert "5/30" in rows["error_triage"]["detail"]


def test_learner_trained_model_reported(monkeypatch, tmp_path):
    mp = _modelsprov()
    models_dir = tmp_path / "models" / "error-triage"
    models_dir.mkdir(parents=True)
    (models_dir / "manifest.json").write_text(json.dumps({
        "job": "error-triage", "latest": 2, "backend": "gradboost",
        "trained_at": 1700000000.0, "n_rows": 45,
        "report": {"score": 0.81, "baseline_score": 0.5}}))
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    rows = {row["job"]: row for row in mp.learner_status()}
    rep = rows["error_triage"]
    assert rep["status"] == "trained"
    assert rep["version"] == 2
    assert rep["score"] == 0.81


# ── journal paths pinned to the learners' own paths ────────────────────

def test_journal_paths_match_learner_modules():
    mp = _modelsprov()
    from nomorals.agents.failure_models import DEFAULT_GRADES_PATH
    from nomorals.agents.pumpdesk.store import pumpdesk_dir
    from nomorals.games.npc_learn import _default_npc_dir

    assert str(mp._failure_grades_path()) == DEFAULT_GRADES_PATH
    assert str(mp._pumpdesk_dir()) == pumpdesk_dir()
    assert str(mp._npc_learn_dir()) == str(_default_npc_dir())


# ── control surface ────────────────────────────────────────────────────

def test_control_status_reports_installed_vs_missing(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    monkeypatch.setattr(mp, "_detect_piper", lambda: True)
    monkeypatch.setattr(mp, "_detect_kokoro", lambda: True)
    text = _control_models()("status")
    assert "installed" in text
    assert "missing" in text
    assert "piper" in text and "kokoro" in text


def test_control_bare_means_all(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    text = _control_models()("")
    assert "models" in text.lower()
    for name in ("xtts", "madmom", "faster-whisper", "sd15", "sklearn"):
        assert name in text, name


def test_control_names_out_of_scope(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    text = _control_models()("")
    assert "codebeast" in text.lower()
    assert "big-merge" in text.lower() or "big merge" in text.lower()


def test_control_unknown_arg_is_usage_not_crash(monkeypatch):
    mp = _modelsprov()
    _mock_all_missing(monkeypatch, mp)
    text = _control_models()("nope")
    assert "nope" in text
    assert "/models" in text


def test_control_learners_lists_jobs(monkeypatch, tmp_path):
    monkeypatch.setenv("NM_MODELS_DIR", str(tmp_path / "models"))
    text = _control_models()("learners")
    for job in EXPECTED_LEARNERS:
        assert job in text, job


# ── registry wiring ────────────────────────────────────────────────────

def test_registry_wiring():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        help_text,
        list_catalog,
    )

    assert CONTROL_COMMANDS["models"] == (0, None)
    page = COMMAND_DETAILS["models"]
    assert page["what"] and page["usage"] and page["example"]
    assert LIST_ONELINERS["models"]
    slots = [g for g, kinds in LIST_GROUPS if "models" in kinds]
    assert len(slots) == 1
    assert "/models" in help_text()
    assert "/models" in list_catalog()


def test_dispatch_branch_and_tour_denylist():
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    runtime = (repo / "nomorals" / "agents" / "partner"
               / "runtime.py").read_text()
    assert 'kind == "models"' in runtime
    meta = (repo / "nomorals" / "agents" / "partner"
            / "runtime_meta.py").read_text()
    assert '"models"' in meta
