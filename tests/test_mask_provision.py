"""§1.1d tail: the SAM consent-to-download flow, every state.

Probe / installer / re-probe are fakes throughout — no network,
no real download, no real install ever runs here. The flow's
states are the contract: not_needed / available / offered /
installed / refused / failed, each reached by its honest path.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.media.imggen import mask_provision as mp
from nomorals.media.imggen.mask_provision import (
    MaskProvisionError,
    MaskProvisionFlow,
    _SamInstaller,
)

MISSING_BOTH = {
    "sam": {"usable": False,
            "reason": "no SAM checkpoint on disk; place one at "
                      "~/.nomorals/models/sam_vit_h.pth",
            "package": None, "checkpoint": None},
}
MISSING_CKPT = {
    "sam": {"usable": False,
            "reason": "no SAM checkpoint on disk; place one at "
                      "~/.nomorals/models/sam_vit_h.pth",
            "package": "segment-anything",
            "checkpoint": None},
}
USABLE = {
    "sam": {"usable": True, "reason": "ok",
            "package": "segment-anything",
            "checkpoint": "/models/sam_vit_b.pth"},
}
FACTS = {"path": "/models/sam_vit_b.pth", "bytes": 375_000_000,
         "sha256": "ab" * 32}


class FakeInstaller:
    def __init__(self, *, facts=None, error=None):
        self.calls: list[dict] = []
        self._facts = facts
        self._error = error

    def install(self, sam):
        self.calls.append(sam)
        if self._error is not None:
            raise self._error
        return self._facts


def flow(tmp_path, probe_status, *, installer=None, reprobe=None,
         routes=None):
    return MaskProvisionFlow(
        probe=lambda: probe_status,
        installer=installer or FakeInstaller(facts=FACTS),
        reprobe=reprobe or (lambda: USABLE),
        routes=routes or (lambda sam: ["package", "checkpoint"]
                          if not sam.get("package")
                          else ["checkpoint"]),
        state_path=tmp_path / "mask_provision.json",
        clock=lambda: 1234.5)


def test_assess_available(tmp_path):
    rec = flow(tmp_path, USABLE).assess()
    assert rec["state"] == "available"
    assert rec["offer"] == "" and rec["attempts"] == []


def test_assess_not_needed_makes_no_offer(tmp_path):
    inst = FakeInstaller()
    rec = flow(tmp_path, MISSING_BOTH, installer=inst).assess(
        need=False)
    assert rec["state"] == "not_needed"
    assert rec["offer"] == ""
    assert inst.calls == []


def test_assess_offered_with_plain_offer(tmp_path):
    rec = flow(tmp_path, MISSING_BOTH).assess()
    assert rec["state"] == "offered"
    assert "no SAM checkpoint on disk" in rec["offer"]
    assert "pip install segment-anything" in rec["offer"]
    assert "375" in rec["offer"]
    assert "won't ask again" in rec["offer"]


def test_assess_checkpoint_only_offer_skips_pip(tmp_path):
    rec = flow(tmp_path, MISSING_CKPT).assess()
    assert rec["state"] == "offered"
    assert "pip install" not in rec["offer"]
    assert "download the SAM ViT-B checkpoint" in rec["offer"]


def test_assess_failed_when_no_route_applies(tmp_path):
    rec = flow(tmp_path, MISSING_BOTH, routes=lambda sam: []).assess()
    assert rec["state"] == "failed"
    assert "no install route" in rec["detail"]


def test_assess_failed_when_probe_raises(tmp_path):
    def boom():
        raise RuntimeError("probe exploded")

    f = MaskProvisionFlow(probe=boom,
                          state_path=tmp_path / "s.json")
    rec = f.assess()
    assert rec["state"] == "failed"
    assert "probe exploded" in rec["reason"]


def test_refusal_is_remembered_across_instances(tmp_path):
    flow(tmp_path, MISSING_BOTH).refuse()
    rec = flow(tmp_path, MISSING_BOTH).assess()
    assert rec["state"] == "refused"
    assert rec["offer"] == ""


def test_refusal_reopens_when_the_reason_changes(tmp_path):
    flow(tmp_path, MISSING_BOTH).refuse()
    changed = {"sam": {**MISSING_BOTH["sam"],
                       "reason": "no SAM package installed "
                                 "(pip install segment-anything)",
                       "package": None,
                       "checkpoint": "/models/sam_vit_h.pth"}}
    f = MaskProvisionFlow(
        probe=lambda: changed,
        installer=FakeInstaller(),
        reprobe=lambda: USABLE,
        routes=lambda sam: ["package"],
        state_path=tmp_path / "mask_provision.json")
    assert f.assess()["state"] == "offered"


def test_reopen_clears_the_refusal(tmp_path):
    f = flow(tmp_path, MISSING_BOTH)
    f.refuse()
    assert f.assess()["state"] == "refused"
    assert f.reopen()["state"] == "offered"


def test_accept_installs_and_verifies(tmp_path):
    inst = FakeInstaller(facts=FACTS)
    f = flow(tmp_path, MISSING_BOTH, installer=inst)
    rec = f.accept()
    assert rec["state"] == "installed"
    assert inst.calls == [MISSING_BOTH["sam"]]
    assert rec["checkpoint"] == FACTS
    saved = json.loads(
        (tmp_path / "mask_provision.json").read_text())
    assert saved["state"] == "installed"
    assert saved["updated_at"] == 1234.5


def test_accept_already_available_installs_nothing(tmp_path):
    inst = FakeInstaller()
    rec = flow(tmp_path, USABLE, installer=inst).accept()
    assert rec["state"] == "available"
    assert inst.calls == []


def test_accept_failed_install_keeps_receipts(tmp_path):
    err = MaskProvisionError(
        ["pip install of segment-anything failed (exit 1): boom"])
    rec = flow(tmp_path, MISSING_BOTH,
               installer=FakeInstaller(error=err)).accept()
    assert rec["state"] == "failed"
    assert rec["attempts"] == [
        "pip install of segment-anything failed (exit 1): boom"]


def test_accept_failed_when_reprobe_still_missing(tmp_path):
    rec = flow(tmp_path, MISSING_BOTH,
               reprobe=lambda: MISSING_CKPT).accept()
    assert rec["state"] == "failed"
    assert "still reports SAM unusable" in rec["detail"]
    assert rec["checkpoint"] == FACTS  # what landed is recorded


def test_accept_failed_when_reprobe_cannot_run(tmp_path):
    def bad_reprobe():
        raise MaskProvisionError(["fresh-process re-probe failed: x"])

    rec = flow(tmp_path, MISSING_BOTH,
               reprobe=bad_reprobe).accept()
    assert rec["state"] == "failed"
    assert rec["attempts"] == ["fresh-process re-probe failed: x"]


# ── the real installer, staged ─────────────────────────────────────


def test_installer_pip_failure_is_a_named_attempt(tmp_path):
    inst = _SamInstaller(
        runner=lambda argv: (1, "ERROR: no matching distribution"),
        downloader=lambda url, dest: FACTS,
        dest_dir=tmp_path)
    with pytest.raises(MaskProvisionError) as ei:
        inst.install(MISSING_BOTH["sam"])
    assert "pip install of segment-anything failed (exit 1)" \
        in ei.value.attempts[0]


def test_installer_download_failure_is_a_named_attempt(tmp_path):
    def bad_download(url, dest):
        raise OSError("network unreachable")

    inst = _SamInstaller(runner=lambda argv: (0, ""),
                         downloader=bad_download, dest_dir=tmp_path)
    with pytest.raises(MaskProvisionError) as ei:
        inst.install(MISSING_CKPT["sam"])
    assert "download of the SAM checkpoint" in ei.value.attempts[0]
    assert "network unreachable" in ei.value.attempts[0]


def test_installer_success_returns_landed_facts(tmp_path):
    seen = {}

    def download(url, dest):
        seen["url"], seen["dest"] = url, dest
        return FACTS

    inst = _SamInstaller(runner=lambda argv: (0, ""),
                         downloader=download, dest_dir=tmp_path)
    assert inst.install(MISSING_BOTH["sam"]) == FACTS
    assert seen["url"] == mp.SAM_CHECKPOINT_URL
    assert seen["dest"] == tmp_path / "sam_vit_b.pth"


# ── the downloader, staged at the urlopen seam ─────────────────────


class _FakeResp:
    def __init__(self, body: bytes, length: int | None = None):
        self._buf = io.BytesIO(body)
        self.headers = {} if length is None else {
            "Content-Length": str(length)}

    def read(self, n=-1):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_lands_verified_bytes(tmp_path, monkeypatch):
    body = b"weights" * 1000
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResp(body, len(body)))
    dest = tmp_path / "sam_vit_b.pth"
    facts = mp._download_file("https://example.invalid/x", dest)
    assert dest.read_bytes() == body
    assert facts["bytes"] == len(body)
    assert facts["sha256"] == hashlib.sha256(body).hexdigest()
    assert not (tmp_path / "sam_vit_b.pth.part").exists()


def test_download_truncation_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda req, timeout=None: _FakeResp(b"short", 10_000))
    dest = tmp_path / "sam_vit_b.pth"
    with pytest.raises(MaskProvisionError, match="truncated"):
        mp._download_file("https://example.invalid/x", dest)
    assert not dest.exists()
    assert not (tmp_path / "sam_vit_b.pth.part").exists()


# ── surfaces: tools + CLI carry the same flow ──────────────────────


def _registry_with_fake_flow(monkeypatch, tmp_path, status):
    from nomorals.tools import media_gen
    from nomorals.tools.registry import ToolRegistry

    fake = flow(tmp_path, status)
    monkeypatch.setattr(
        "nomorals.media.imggen.mask_provision.default_flow",
        lambda: fake)
    reg = ToolRegistry()
    reg.context = SimpleNamespace(workspace=str(tmp_path))
    media_gen.register(reg)
    return reg


def test_tools_serve_status_and_carry_consent(monkeypatch, tmp_path):
    reg = _registry_with_fake_flow(monkeypatch, tmp_path,
                                   MISSING_BOTH)
    out = reg.call("mask_rung_status")
    assert out.ok, getattr(out.error, "message", out.error)
    assert set(out.value["rungs"]) == {
        "sam", "grabcut", "flood", "box", "file"}
    assert out.value["sam_provision"]["state"] == "offered"

    out = reg.call("mask_rung_provision", decision="accept")
    assert out.ok, getattr(out.error, "message", out.error)
    assert out.value["sam_provision"]["state"] == "installed"

    out = reg.call("mask_rung_provision", decision="refuse")
    assert out.ok
    assert out.value["sam_provision"]["state"] == "refused"

    out = reg.call("mask_rung_provision", decision="maybe")
    assert not out.ok


def test_cli_masks_prints_the_offer(monkeypatch, tmp_path, capsys):
    from nomorals.cmdline.commands import imggen as imggen_cmd

    fake = flow(tmp_path, MISSING_BOTH)
    monkeypatch.setattr(
        "nomorals.media.imggen.mask_provision.default_flow",
        lambda: fake)
    args = SimpleNamespace(accept=False, refuse=False,
                           reopen=False, json=False)
    rc = imggen_cmd._cmd_imggen_masks(args, None)
    text = capsys.readouterr().out
    assert rc == 0
    assert "sam provision: offered" in text
    assert "375" in text

    args = SimpleNamespace(accept=True, refuse=False,
                           reopen=False, json=True)
    rc = imggen_cmd._cmd_imggen_masks(args, None)
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["sam_provision"]["state"] == "installed"
