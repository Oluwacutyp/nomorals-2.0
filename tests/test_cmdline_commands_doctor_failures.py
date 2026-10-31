"""Tail-70 pins: nomorals/cmdline/commands/doctor.py silent-swallow sites.

Adjudications (see ~/workspace/nm-tail70-verify/SURVEY.md):
- 32 EXCLUSION — the native section degrades to a visibly honest
  {"overall": "unknown"}; "doctor must always report" is the contract.
- 77 CONVERTED by parent amendment — the registry probe IS the
  operation; its failure is now recorded as a failed-module entry
  ({"module": "tool registry probe", "error": ...}) instead of
  landing on _failed == [], byte-identical to "every module
  registered cleanly", on the surface whose exit code is a deploy
  contract.  Doctor still always reports; the entry trips rc 1.
- 83 EXCLUSION — _status None is visibly absent (payload null, text
  section omitted), never a success-shaped value.
- 200 EXCLUSION — the durable .env write precedes the guard and fails
  loud; the guarded in-memory settings mirror is best-effort by its
  own contract ("the file is the durable truth").
- 345 EXCLUSION — EOF/Ctrl-C becomes _SetupAborted: printed sentence,
  rc 1, nothing written.  Not a swallow.
- 529 EXCLUSION — post-write EOF at the optional test prompt defaults
  to "don't test" and says so; the written config is unaffected.

Each pin proves the guarded operation was ATTEMPTED (spy).
"""

from __future__ import annotations

import argparse
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.cmdline.commands import doctor as doc


def _settings(tmp_path):
    return SimpleNamespace(
        db_path=str(tmp_path / "doc.db"), profile="test",
        concurrency=SimpleNamespace(processes=None, use_processes=False))


def _doctor_json(tmp_path, capsys):
    rc = doc._cmd_doctor(argparse.Namespace(json=True, build_native=False),
                         _settings(tmp_path))
    return rc, json.loads(capsys.readouterr().out)


# ── _cmd_doctor section guards (32 / 77 / 83) ───────────────────────────────

def test_32_native_info_failure_reports_unknown(tmp_path, capsys,
                                                monkeypatch):
    import nomorals.native as native
    calls = []

    def raiser():
        calls.append(1)
        raise RuntimeError("native broken")

    monkeypatch.setattr(native, "info", raiser)
    rc, payload = _doctor_json(tmp_path, capsys)
    assert calls == [1]
    assert payload["native"]["overall"] == "unknown"
    assert payload["native"]["fallback"] == "pure-python"
    assert payload["native"]["kernels"] == {}


def test_77_registry_probe_failure_reports_failed_entry(tmp_path, capsys,
                                                        monkeypatch):
    # CONVERTED by parent amendment (tail-70): a wholesale probe failure
    # is recorded as a failed-module entry — never byte-identical to a
    # clean registration — and trips the hard-fail exit code, while
    # doctor still produces its full report.
    from nomorals.tools.registry import ToolRegistry
    calls = []

    def raiser(self):
        calls.append(1)
        raise RuntimeError("registry exploded")

    monkeypatch.setattr(ToolRegistry, "register_builtins", raiser)
    rc, payload = _doctor_json(tmp_path, capsys)
    assert calls == [1]  # the probe was attempted
    assert payload["tool_modules"]["failed"] == [
        {"module": "tool registry probe",
         "error": "RuntimeError: registry exploded"}]
    assert rc == 1  # _hard_fail: the deploy VERIFY FAILED contract
    assert "database" in payload and "config_keys" in payload

    # Healthy-probe control: the real register_builtins reports no
    # failed modules, exactly as before the convert.
    monkeypatch.undo()
    rc2, payload2 = _doctor_json(tmp_path, capsys)
    assert payload2["tool_modules"]["failed"] == []


def test_83_config_status_failure_reports_none(tmp_path, capsys,
                                               monkeypatch):
    import nomorals.core.config as cfg
    calls = []

    def raiser():
        calls.append(1)
        raise RuntimeError("config broken")

    monkeypatch.setattr(cfg, "config_env_status", raiser)
    rc, payload = _doctor_json(tmp_path, capsys)
    assert calls == [1]
    assert payload["config_keys"] is None


# ── 200: _cmd_models promote-local settings mirror ──────────────────────────

def test_200_settings_mirror_failure_file_still_durable(tmp_path, capsys,
                                                        monkeypatch):
    from nomorals.storage.db import Database

    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    class FrozenLLM:
        def __setattr__(self, key, value):
            raise RuntimeError("settings frozen")

    db = Database(str(tmp_path / "m.db"))
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(home=str(home), llm=FrozenLLM()))
    args = argparse.Namespace(promote_local=str(gguf), lora="",
                              activate="", catalog=False, search="",
                              kind="", max_params=0, json=False)
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.chdir(work)
    try:
        rc = doc._cmd_models(args, ctx)
    finally:
        db.close()
    assert rc == 0
    env_text = (home / ".env").read_text()
    assert "NM_LLM_PROVIDER=llama_cpp" in env_text
    assert f"NM_LLM_LOCAL_MODEL={gguf}" in env_text
    assert "local model promoted" in capsys.readouterr().out


# ── 345 / 529: setup wizard KeyboardInterrupt/EOF handling ──────────────────

def _wizard_ctx():
    return SimpleNamespace(router=SimpleNamespace(active_model="mock"))


def test_345_eof_at_wizard_prompt_aborts_cleanly(tmp_path, capsys,
                                                 monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("NM_HOME", str(home))

    def eof_input(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof_input)
    rc = doc._cmd_setup(argparse.Namespace(), _wizard_ctx())
    assert rc == 1
    assert "Setup aborted" in capsys.readouterr().out
    assert not (home / ".env").exists()  # abort precedes any write


def test_529_eof_at_test_prompt_skips_test_after_write(tmp_path, capsys,
                                                       monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.chdir(work)
    answers = iter(["1", "sk-or-probe", "1"])

    def fake_input(prompt=""):
        try:
            return next(answers)
        except StopIteration as exc:
            assert isinstance(exc, Exception)
            raise EOFError from None

    monkeypatch.setattr(builtins, "input", fake_input)
    rc = doc._cmd_setup(argparse.Namespace(), _wizard_ctx())
    assert rc == 0
    out = capsys.readouterr().out
    assert "(no input available — skipping the test)" in out
    env_text = (home / ".env").read_text()
    assert "NM_OPENROUTER_API_KEY=sk-or-probe" in env_text
    assert "NM_LLM_PROVIDER=openrouter" in env_text
