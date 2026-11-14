"""Character-model provision: the missing install command.

The owner asked (2026-11-13): "Weren't you supposed to give me a
certain command to install the models?" Code truth: the character
generative specialists (XTTS voice cloning, RVC voice swap, face swap)
existed as slots only — honestly unavailable, with NO provision path
anywhere in the repo. This suite pins the provision that closes that
gap: ``/character provision`` installs the three backends on Devon's
own box (same shape as ``provision_madmom`` / ``/dj desk provision``),
reports per backend, and never fakes a working install.

Every install/download is mocked here — no real pip installs, no real
weight downloads. The seams are module-level functions on
``nomorals.characters.provision`` (``_run``, ``_measure_box``,
``_detect_*``, ``_warm_faceswap_models``) patched per test.
"""

from __future__ import annotations

import subprocess
import types

BIG_BOX = {"ok": True, "total_ram_gb": 16.0, "free_ram_gb": 10.0,
           "free_disk_gb": 120.0, "cpus": 8, "platform": "linux",
           "profile": "workstation", "detail": "big box"}


def _ok_run(args, timeout=None):
    return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")


def _fail_run(args, timeout=None):
    return types.SimpleNamespace(returncode=1, stdout="",
                                 stderr="boom")


def _provision():
    import nomorals.characters.provision as prov

    return prov


# ── import is inert ────────────────────────────────────────────────────

def test_import_performs_no_work(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: calls.append(a) or _ok_run(a))
    prov = _provision()
    assert prov.BACKENDS == ("xtts", "rvc", "faceswap")
    assert calls == []


# ── report shape ───────────────────────────────────────────────────────

def test_provision_all_success_shape(monkeypatch):
    prov = _provision()
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(prov, "_run", _ok_run)
    monkeypatch.setattr(prov, "_detect_xtts", lambda: False)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: False)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: False)
    monkeypatch.setattr(prov, "_warm_faceswap_models",
                        lambda: {"ok": True, "detail": "warmed"})
    result = prov.provision_backends()
    assert result["ok"] is True
    assert set(result["backends"]) == {"xtts", "rvc", "faceswap"}
    for name, rep in result["backends"].items():
        assert rep["status"] == "installed", name
        assert rep["steps"], name
    text = prov.provision_report_text(result)
    assert "xtts" in text.lower()
    assert "rvc" in text.lower()
    assert "face" in text.lower()


def test_provision_never_raises_on_runner_error(monkeypatch):
    prov = _provision()
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(BIG_BOX))

    def _boom(args, timeout=None):
        raise OSError("no such python")

    monkeypatch.setattr(prov, "_run", _boom)
    monkeypatch.setattr(prov, "_detect_xtts", lambda: False)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: False)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: False)
    result = prov.provision_backends()
    assert result["ok"] is False
    assert all(rep["status"] == "failed"
               for rep in result["backends"].values())


def test_failed_pip_is_failed_report_not_exception(monkeypatch):
    prov = _provision()
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(prov, "_run", _fail_run)
    monkeypatch.setattr(prov, "_detect_xtts", lambda: False)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: False)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: False)
    result = prov.provision_backends(["xtts"])
    assert result["ok"] is False
    assert result["backends"]["xtts"]["status"] == "failed"
    assert "boom" in result["backends"]["xtts"]["detail"]


def test_already_installed_short_circuits(monkeypatch):
    prov = _provision()
    calls = []
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(prov, "_run",
                        lambda a, timeout=None: calls.append(a) or _ok_run(a))
    monkeypatch.setattr(prov, "_detect_xtts", lambda: True)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: True)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: True)
    result = prov.provision_backends()
    assert result["ok"] is True
    assert all(rep["status"] == "already-installed"
               for rep in result["backends"].values())
    assert calls == []


# ── honest box check ───────────────────────────────────────────────────

def test_small_box_xtts_unavailable_with_numbers(monkeypatch):
    prov = _provision()
    small = dict(BIG_BOX, total_ram_gb=2.0, free_ram_gb=1.2,
                 free_disk_gb=20.0)
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(small))
    calls = []
    monkeypatch.setattr(prov, "_run",
                        lambda a, timeout=None: calls.append(a) or _ok_run(a))
    monkeypatch.setattr(prov, "_detect_xtts", lambda: False)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: False)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: False)
    monkeypatch.setattr(prov, "_warm_faceswap_models",
                        lambda: {"ok": True, "detail": "warmed"})
    result = prov.provision_backends()
    xtts = result["backends"]["xtts"]
    assert xtts["status"] == "unavailable"
    # the measured numbers are in the reason, not a bare refusal
    assert "2.0" in xtts["detail"] or "1.2" in xtts["detail"]
    assert "ram" in xtts["detail"].lower()
    # XTTS was never attempted on the small box…
    assert not any("TTS" in " ".join(map(str, c)) for c in calls)
    # …but the lighter backends were
    assert result["backends"]["rvc"]["status"] == "installed"
    assert result["backends"]["faceswap"]["status"] == "installed"


def test_status_reflects_detectors(monkeypatch):
    prov = _provision()
    monkeypatch.setattr(prov, "_detect_xtts", lambda: True)
    monkeypatch.setattr(prov, "_detect_rvc", lambda: False)
    monkeypatch.setattr(prov, "_detect_faceswap", lambda: True)
    status = prov.backend_status()
    assert status["xtts"]["installed"] is True
    assert status["rvc"]["installed"] is False
    assert status["faceswap"]["installed"] is True


def test_xtts_report_names_private_use(monkeypatch):
    prov = _provision()
    monkeypatch.setattr(prov, "_measure_box", lambda: dict(BIG_BOX))
    monkeypatch.setattr(prov, "_run", _ok_run)
    monkeypatch.setattr(prov, "_detect_xtts", lambda: False)
    result = prov.provision_backends(["xtts"])
    assert "private" in result["backends"]["xtts"]["detail"].lower()


# ── specialists light up once provisioned ──────────────────────────────

def test_character_voice_uses_provisioned_backend(monkeypatch):
    import nomorals.characters.specialists as cs
    import nomorals.voice.tts as tts_mod
    from nomorals.characters.character import Character
    from nomorals.voice.catalogue import VoiceCatalogue

    monkeypatch.setattr(tts_mod, "available_backends", lambda: ["xtts"])
    entry = types.SimpleNamespace(backend="xtts")
    monkeypatch.setattr(VoiceCatalogue, "get", lambda self, n: entry)

    class _FakeTTS:
        def __init__(self, *a, **k):
            pass

        def speak(self, line, voice_name="", out_path="", audience=""):
            return {"ok": True, "path": "/tmp/out.wav"}

    monkeypatch.setattr(tts_mod, "UniversalTTS", _FakeTTS)
    char = Character(name="Mara", voice_name="mara-dry")
    result = cs.character_voice(char, "Hello there.")
    assert result["ok"] is True
    assert result["backend"] == "xtts"


def test_character_voice_unprovisioned_names_command(monkeypatch):
    import nomorals.characters.specialists as cs
    import nomorals.voice.tts as tts_mod
    from nomorals.characters.character import Character

    monkeypatch.setattr(tts_mod, "available_backends", lambda: [])
    char = Character(name="Mara", voice_name="mara-dry")
    result = cs.character_voice(char, "Hello there.")
    assert result["ok"] is False
    assert "/character provision" in result["reason"]


def test_swap_voice_uses_provisioned_rvc(monkeypatch):
    import nomorals.characters.specialists as cs
    import nomorals.voice.rvc_bridge as rvc
    from nomorals.characters.character import Character

    monkeypatch.setattr(rvc, "detect_rvc",
                        lambda: {"ok": True, "method": "python",
                                 "detail": "rvc_python"})
    monkeypatch.setattr(rvc, "convert",
                        lambda *a, **k: {"path": "/tmp/swapped.wav"})
    char = Character(name="Mara")
    char.rvc_model = "mara-v1"
    result = cs.swap_voice(char, "/tmp/line.wav")
    assert result["ok"] is True
    assert result["path"] == "/tmp/swapped.wav"


def test_swap_voice_unprovisioned_names_command(monkeypatch):
    import nomorals.characters.specialists as cs
    import nomorals.voice.rvc_bridge as rvc
    from nomorals.characters.character import Character

    monkeypatch.setattr(rvc, "detect_rvc",
                        lambda: {"ok": False, "method": "",
                                 "detail": "RVC not installed"})
    char = Character(name="Mara")
    char.rvc_model = "mara-v1"
    result = cs.swap_voice(char, "/tmp/line.wav")
    assert result["ok"] is False
    assert "/character provision" in result["reason"]


def test_face_swap_guard_still_holds(tmp_path):
    import nomorals.characters.specialists as cs
    from nomorals.characters.character import Character

    char = Character(name="Mara")
    result = cs.swap_face(char, str(tmp_path / "target.png"))
    assert result["ok"] is False
    assert "face" in result["reason"].lower()


# ── the chat command ───────────────────────────────────────────────────

class _StubStore:
    def list(self):
        return []


def test_provision_verb_status(monkeypatch):
    import nomorals.audio.characters as ac
    import nomorals.characters.provision as prov

    monkeypatch.setattr(prov, "backend_status", lambda: {
        "xtts": {"installed": False, "detail": "no TTS backend"},
        "rvc": {"installed": False, "detail": "RVC not installed"},
        "faceswap": {"installed": False, "detail": "no insightface"},
    })
    text = ac.control_character("provision status", store=_StubStore())
    assert "xtts" in text.lower()
    assert "rvc" in text.lower()
    assert "face" in text.lower()


def test_provision_verb_runs_and_reports(monkeypatch):
    import nomorals.audio.characters as ac
    import nomorals.characters.provision as prov

    seen = {}

    def _fake(backends=None):
        seen["backends"] = backends
        return {"ok": True, "box": dict(BIG_BOX),
                "backends": {
                    "rvc": {"ok": True, "status": "installed",
                            "detail": "rvc-python installed",
                            "steps": ["pip install rvc-python: rc=0"]}}}

    monkeypatch.setattr(prov, "provision_backends", _fake)
    text = ac.control_character("provision rvc", store=_StubStore())
    assert seen["backends"] == ["rvc"]
    assert "installed" in text.lower()


def test_provision_verb_unknown_backend(monkeypatch):
    import nomorals.audio.characters as ac
    import nomorals.characters.provision as prov

    called = []
    monkeypatch.setattr(prov, "provision_backends",
                        lambda b=None: called.append(b) or {"ok": True,
                                                            "backends": {}})
    text = ac.control_character("provision wat", store=_StubStore())
    assert called == []
    assert "provision" in text.lower()
