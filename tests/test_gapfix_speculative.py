"""Gapfix R2-34 — speculative decoding wired into the local server.

Failing-first: at base 0279bee9 the speculative module exists but is never
imported by production code, there is no config surface for a draft model,
no caller passes one, and the live boot path (ensure_local_gguf) raises
TypeError on every auto-start attempt (bogus ``model_path=`` kwarg).
"""

from __future__ import annotations

import argparse
import os
import socket
import stat
from types import SimpleNamespace

from nomorals.llm.local_server import GGUFServerManager


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _write_gguf(path, size_mb: int = 10) -> None:
    with open(path, "wb") as fh:
        fh.write(b"GGUF")
        fh.write(b"\x00" * (size_mb * 1024 * 1024 - 4))


# ── config surface ────────────────────────────────────────────────────────


def test_config_keys_exist():
    from nomorals.core.config import LLMSettings

    llm = LLMSettings()
    assert llm.local_draft_model == ""
    assert llm.local_draft_max == 4
    assert llm.local_draft_min == 1
    assert llm.local_draft_p_min == 0.9


def test_env_map_has_draft_keys():
    from nomorals.core.config import _ENV_MAP

    assert _ENV_MAP["NM_LLM_LOCAL_DRAFT_MODEL"] == "llm.local_draft_model"
    assert _ENV_MAP["NM_LLM_LOCAL_DRAFT_MAX"] == "llm.local_draft_max"
    assert _ENV_MAP["NM_LLM_LOCAL_DRAFT_MIN"] == "llm.local_draft_min"
    assert _ENV_MAP["NM_LLM_LOCAL_DRAFT_P_MIN"] == "llm.local_draft_p_min"


def test_env_loads_draft_keys(monkeypatch):
    from nomorals.core.config import load_settings

    monkeypatch.setenv("NM_LLM_LOCAL_DRAFT_MODEL", "/models/draft.gguf")
    monkeypatch.setenv("NM_LLM_LOCAL_DRAFT_MAX", "8")
    monkeypatch.setenv("NM_LLM_LOCAL_DRAFT_P_MIN", "0.85")
    settings = load_settings()
    assert settings.llm.local_draft_model == "/models/draft.gguf"
    assert settings.llm.local_draft_max == 8
    assert settings.llm.local_draft_p_min == 0.85


def test_template_documents_draft_keys():
    from pathlib import Path

    template = Path(__file__).parents[1] / "scripts" / "devon_ec2_env.template"
    text = template.read_text()
    assert "NM_LLM_LOCAL_DRAFT_MODEL" in text


# ── argv construction routes through the speculative module ─────────────


def test_invalid_draft_p_min_emits_no_draft_flags():
    # SpeculativeConfig.validate() rejects draft_p_min outside (0, 1];
    # the manager must refuse it, not emit it raw to llama-server.
    mgr = GGUFServerManager(port=18081, draft_model="draft.gguf",
                            draft_p_min=2.0)
    args = mgr.build_args("target.gguf", "llama-server")
    assert "--draft-model" not in args


def test_draft_flags_present_and_max_honored():
    mgr = GGUFServerManager(port=18082, draft_model="draft.gguf",
                            draft_max=7, draft_min=2, draft_p_min=0.8)
    args = mgr.build_args("target.gguf", "llama-server")
    assert args[args.index("--draft-model") + 1] == "draft.gguf"
    assert args[args.index("--draft-max") + 1] == "7"
    assert args[args.index("--draft-min") + 1] == "2"
    assert args[args.index("--draft-p-min") + 1] == "0.8"


def test_no_draft_argv_byte_identical():
    # The no-draft default path must not change at all.
    mgr = GGUFServerManager(port=18083, ctx_size=2048, threads=4,
                            kv_cache_quant="q8_0", flash_attn=False,
                            extra_args="")
    args = mgr.build_args("target.gguf", "llama-server")
    assert args == [
        "llama-server", "-m", "target.gguf",
        "--host", "127.0.0.1", "--port", "18083",
        "--ctx-size", "2048", "-t", "4", "--parallel", "1",
        "--ctk", "q8_0", "--ctv", "q8_0",
        "--flash-attn", "off",
    ]


def test_kv_flags_emitted_exactly_once_with_draft():
    # The speculative fragment and the KV fragment must not duplicate
    # --ctk/--ctv when both are built from the same config.
    mgr = GGUFServerManager(port=18084, draft_model="draft.gguf",
                            kv_cache_quant="q8_0")
    args = mgr.build_args("target.gguf", "llama-server")
    assert args.count("--ctk") == 1
    assert args.count("--ctv") == 1
    assert args.count("--draft-model") == 1


def test_draft_equal_target_emits_nothing():
    mgr = GGUFServerManager(port=18085, draft_model="target.gguf")
    args = mgr.build_args("target.gguf", "llama-server")
    assert "--draft-model" not in args


# ── draft file validation: refuse honestly, never spawn a broken server ──


def test_doctor_flags_missing_draft_file(tmp_path):
    mgr = GGUFServerManager(port=18086,
                            draft_model=str(tmp_path / "missing-draft.gguf"))
    diagnosis = mgr.doctor("no-such-model")
    joined = " ".join(diagnosis.problems).lower()
    assert "draft" in joined


def test_doctor_flags_non_gguf_draft(tmp_path):
    bad = tmp_path / "draft.gguf"
    bad.write_bytes(b"NOTA" + b"\x00" * (11 * 1024 * 1024))
    mgr = GGUFServerManager(port=18087, draft_model=str(bad))
    diagnosis = mgr.doctor("no-such-model")
    joined = " ".join(diagnosis.problems).lower()
    assert "draft" in joined


def test_start_refuses_missing_draft_before_spawn(tmp_path, monkeypatch):
    import subprocess

    import nomorals.llm.local_server as ls

    model = tmp_path / "model.gguf"
    _write_gguf(model)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "llama-server"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    # Popen must never be reached: a missing draft refuses first.
    calls: list[list[str]] = []
    real_popen = subprocess.Popen

    def recording_popen(*a, **k):
        calls.append(list(a[0]) if a else [])
        return real_popen(*a, **k)

    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    monkeypatch.setattr(ls.subprocess, "Popen", recording_popen)
    mgr = GGUFServerManager(port=_free_port(), cache_dir=str(tmp_path),
                            boot_timeout=30.0,
                            draft_model=str(tmp_path / "missing-draft.gguf"))
    diagnosis = mgr.start(str(model))
    assert diagnosis.ok is False
    assert "draft" in " ".join(diagnosis.problems).lower()
    assert calls == []


# ── live boot path: ensure_local_gguf ─────────────────────────────────────


def test_ensure_local_gguf_passes_draft_without_bogus_kwarg(tmp_path,
                                                            monkeypatch):
    import nomorals.llm.local_server as ls

    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    draft = tmp_path / "draft.gguf"
    draft.write_bytes(b"GGUF")
    seen: dict[str, object] = {}

    class FakeManager:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def _healthy(self):
            return False

        def heal(self):
            return SimpleNamespace(ok=False, model_path="")

    monkeypatch.setattr(ls, "GGUFServerManager", FakeManager)
    from nomorals.agents.context import ensure_local_gguf

    settings = SimpleNamespace(
        llm=SimpleNamespace(
            local_model=str(model), local_auto_start=True,
            local_port=_free_port(), local_host="127.0.0.1",
            local_draft_model=str(draft), local_draft_max=7,
            local_draft_min=2, local_draft_p_min=0.8,
        )
    )
    result = ensure_local_gguf(settings)
    # No bogus model_path kwarg (the base TypeError), draft settings passed.
    assert "model_path" not in seen
    assert seen.get("draft_model") == str(draft)
    assert seen.get("draft_max") == 7
    assert seen.get("draft_min") == 2
    assert seen.get("draft_p_min") == 0.8
    # Heal declined; the path itself is still returned, not None.
    assert result == str(model)


# ── nm models --promote-local --draft ─────────────────────────────────────


def _models_ctx(tmp_path):
    from nomorals.storage.db import Database

    home = tmp_path / "home"
    home.mkdir()
    db = Database(str(tmp_path / "m.db"))
    ctx = SimpleNamespace(
        db=db, settings=SimpleNamespace(home=str(home),
                                        llm=SimpleNamespace()))
    return ctx, home, db


def test_promote_local_draft_writes_env(tmp_path, capsys, monkeypatch):
    from nomorals.cmdline.commands import doctor as doc

    ctx, home, db = _models_ctx(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    draft = tmp_path / "draft.gguf"
    _write_gguf(draft)  # promote validates the draft with gguf_check (≥10MB)
    args = argparse.Namespace(promote_local=str(model), draft=str(draft),
                              lora="", activate="", catalog=False, search="",
                              kind="", max_params=0, json=False)
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.chdir(work)
    try:
        rc = doc._cmd_models(args, ctx)
    finally:
        db.close()
    assert rc == 0
    env_text = (home / ".env").read_text()
    assert f"NM_LLM_LOCAL_DRAFT_MODEL={draft}" in env_text
    assert ctx.settings.llm.local_draft_model == str(draft)
    assert "draft" in capsys.readouterr().out.lower()


def test_promote_local_missing_draft_refuses_and_writes_nothing(
        tmp_path, capsys, monkeypatch):
    from nomorals.cmdline.commands import doctor as doc

    ctx, home, db = _models_ctx(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    args = argparse.Namespace(
        promote_local=str(model), draft=str(tmp_path / "no-draft.gguf"),
        lora="", activate="", catalog=False, search="", kind="",
        max_params=0, json=False)
    monkeypatch.setenv("NM_HOME", str(home))
    monkeypatch.chdir(work)
    try:
        rc = doc._cmd_models(args, ctx)
    finally:
        db.close()
    assert rc == 1
    assert not (home / ".env").exists()
    assert "draft" in capsys.readouterr().out.lower()
