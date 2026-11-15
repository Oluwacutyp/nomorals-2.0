"""Digital will / legacy, wired into production (E-6).

MORE_TO_ADD gap 8: the domain (nomorals/legacy/will.py) existed and
was unit-pinned (tests/test_more_gaps710.py) but nothing in production
reached it — no durable store, no chat command, no CLI. The sealed
body can only be written through the owner's real vault seal; this
module never sees a key and never invents a cipher.

Every test here drives the PRODUCTION wiring: the durable WillStore
and the /legacy control — never the dormant organ in isolation.

Teeth: on the pristine base each test fails for the missing wiring
(Will has no to_dict, no WillStore, no _control_legacy, no "legacy"
command kind).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from nomorals.accounts.vault import VaultLockedError
from nomorals.legacy.will import (
    SEALED_MARKER,
    Beneficiary,
    DigitalAsset,
    Executor,
    Will,
    WillStore,
)
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    help_text,
)


class _FakeVault:
    """Test double for the accounts vault: toy XOR cipher, TEST ONLY.

    Production seals through the real vault (encrypt_blob, purpose
    "legacy-will"). This double exists only so the wiring tests can
    exercise seal/reveal without owner credentials.
    """

    _PURPOSE = "legacy-will"

    def encrypt_blob(self, plaintext, *, purpose):
        assert purpose == self._PURPOSE, purpose
        body = "".join(chr(ord(c) ^ 0x5A) for c in plaintext)
        return f"fake:{purpose}:{body}"

    def decrypt_blob(self, blob, *, purpose):
        assert purpose == self._PURPOSE, purpose
        prefix = f"fake:{purpose}:"
        assert blob.startswith(prefix), blob
        return "".join(chr(ord(c) ^ 0x5A) for c in blob[len(prefix):])


class _TamperVault(_FakeVault):
    """Decrypts to the WRONG text — proves reveal fails loudly, never garbage."""

    def decrypt_blob(self, blob, *, purpose):
        return "tampered nonsense"


def _reach_host(tmp_path):
    """Production control host on a throwaway home (decisions/focus harness)."""
    from nomorals.agents.partner.runtime_reach import RuntimeReachMixin

    host = RuntimeReachMixin()
    host.context = SimpleNamespace(
        settings=SimpleNamespace(home_path=Path(tmp_path)))
    return host


def _sealable_will():
    return Will(
        owner_name="Owner",
        executor=Executor(name="Exec", contact="@exec", backup_name="Backup"),
        beneficiaries=[Beneficiary(name="Kid", relationship="child",
                                   share_note="half")],
        digital_assets=[DigitalAsset(label="wallet", kind="wallet",
                                     location_hint="desk drawer",
                                     instruction="transfer to Kid")],
        instructions=["look after the dog"],
        witnesses=["Wit One"],
        created=date(2026, 11, 15),
    )


def _create_will(host, owner="Owner", executor="Exec"):
    out = host._control_legacy(f"create {owner} | {executor}")
    assert out.startswith("recorded: "), out
    return out


# ── durable store ───────────────────────────────────────────────────


def test_will_dict_round_trip():
    will = _sealable_will()
    will.seal_content("my final wishes", lambda pt: f"blob:{pt}",
                      is_owner=True)
    assert Will.from_dict(will.to_dict()) == will
    assert Beneficiary.from_dict(
        Beneficiary(name="Kid").to_dict()) == Beneficiary(name="Kid")
    assert DigitalAsset.from_dict(
        DigitalAsset(label="wallet", kind="wallet").to_dict()) == (
        DigitalAsset(label="wallet", kind="wallet"))
    assert Executor.from_dict(
        Executor(name="Exec").to_dict()) == Executor(name="Exec")


def test_store_persists_across_instances(tmp_path):
    path = str(tmp_path / "legacy.json")
    secret = "my final wishes — the house goes to Kid"
    store = WillStore(path)
    assert store.get() is None
    will = _sealable_will()
    will.seal_content(secret, lambda pt: f"blob:{pt}", is_owner=True)
    store.put(will)

    raw = Path(path).read_text(encoding="utf-8")
    assert secret not in raw
    assert "blob:" in raw

    fresh = WillStore(path)
    got = fresh.get()
    assert got is not None and got.owner_name == "Owner"
    assert got.executor.name == "Exec"
    assert [b.name for b in got.beneficiaries] == ["Kid"]
    assert [a.label for a in got.digital_assets] == ["wallet"]
    assert got.instructions == ["look after the dog"]
    assert got.witnesses == ["Wit One"]
    assert got.created == date(2026, 11, 15)
    assert got.summary(is_owner=True)["sealed_content"] == SEALED_MARKER
    assert got.reveal(lambda blob: blob[len("blob:"):], is_owner=True) == secret


def test_store_refuses_corrupt_file(tmp_path):
    missing = WillStore(str(tmp_path / "nope.json"))
    assert missing.get() is None

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    try:
        WillStore(str(corrupt))
    except ValueError as exc:
        assert "unreadable" in str(exc)
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("corrupt will file must refuse to load")

    smuggled = tmp_path / "smuggled.json"
    smuggled.write_text(json.dumps({
        "will": {"owner_name": "Owner",
                 "executor": {"name": "Exec"},
                 "plaintext": "never store this"},
    }), encoding="utf-8")
    try:
        WillStore(str(smuggled))
    except ValueError as exc:
        assert "plaintext" in str(exc)
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("a row smuggling plaintext must refuse to load")

    bad_row = tmp_path / "bad.json"
    bad_row.write_text(json.dumps({
        "will": {"owner_name": 123, "executor": {"name": "Exec"}},
    }), encoding="utf-8")
    try:
        WillStore(str(bad_row))
    except ValueError:
        pass
    else:  # pragma: no cover - the refusal is the point
        raise AssertionError("a bad will row must refuse to load")


# ── /legacy control ─────────────────────────────────────────────────


def test_legacy_create_and_show(tmp_path):
    host = _reach_host(tmp_path)
    assert "no will yet" in host._control_legacy("show")
    _create_will(host)
    shown = host._control_legacy("show")
    assert "Owner" in shown and "Exec" in shown
    assert "no sealed body" in shown
    assert "a will already exists" in host._control_legacy("create Other | X")


def test_legacy_adds(tmp_path):
    host = _reach_host(tmp_path)
    _create_will(host)
    assert "recorded beneficiary" in host._control_legacy(
        "beneficiary Kid | child | half")
    assert "recorded asset" in host._control_legacy(
        "asset wallet | wallet | desk drawer | transfer to Kid")
    assert "recorded instruction" in host._control_legacy(
        "instruction look after the dog")
    assert "recorded witness" in host._control_legacy("witness Wit One")
    assert "executor set" in host._control_legacy(
        "executor NewExec | @new | Backup")
    shown = host._control_legacy("show")
    assert "Kid" in shown and "wallet" in shown
    assert "look after the dog" in shown and "Wit One" in shown
    assert "NewExec" in shown
    assert "usage:" in host._control_legacy("beneficiary")
    assert "usage:" in host._control_legacy("asset")


def test_legacy_seal_reveal_via_vault(tmp_path):
    host = _reach_host(tmp_path)
    host._legacy_vault = lambda: _FakeVault()
    _create_will(host)
    secret = "my final wishes — the house goes to Kid"
    out = host._control_legacy(f"seal {secret}")
    assert "sealed" in out
    shown = host._control_legacy("show")
    assert SEALED_MARKER in shown
    assert secret not in shown
    raw = Path(tmp_path / "data" / "legacy.json").read_text(encoding="utf-8")
    assert secret not in raw
    assert host._control_legacy("reveal") == secret


def test_legacy_seal_refuses_locked_vault(tmp_path):
    host = _reach_host(tmp_path)

    def _locked():
        raise VaultLockedError(["power key: not configured"])

    host._legacy_vault = _locked
    _create_will(host)
    out = host._control_legacy("seal some wishes")
    assert "vault locked" in out
    shown = host._control_legacy("show")
    assert "no sealed body" in shown


def test_legacy_reveal_fails_on_tamper(tmp_path):
    host = _reach_host(tmp_path)
    host._legacy_vault = lambda: _FakeVault()
    _create_will(host)
    host._control_legacy("seal my final wishes")
    host._legacy_vault = lambda: _TamperVault()
    out = host._control_legacy("reveal")
    assert "reveal failed" in out
    assert "tampered nonsense" not in out


def test_legacy_reset_needs_confirmation(tmp_path):
    host = _reach_host(tmp_path)
    assert "no will" in host._control_legacy("reset yes")
    _create_will(host)
    assert "confirmation" in host._control_legacy("reset")
    assert "Owner" in host._control_legacy("show")
    assert "deleted" in host._control_legacy("reset yes")
    assert "no will yet" in host._control_legacy("show")


def test_legacy_unknown_verb_and_empty_states(tmp_path):
    host = _reach_host(tmp_path)
    assert "no will yet" in host._control_legacy("show")
    assert "no will yet" in host._control_legacy("seal wishes")
    assert "no will yet" in host._control_legacy("reveal")
    assert "no will yet" in host._control_legacy("executor X")
    assert "no will yet" in host._control_legacy("beneficiary Kid")
    _create_will(host)
    assert "usage:" in host._control_legacy("frobnicate")


# ── registry ────────────────────────────────────────────────────────


def test_legacy_registered():
    assert "legacy" in CONTROL_COMMANDS
    assert "legacy" not in _public_command_kinds()
    slots = [title for title, kinds in LIST_GROUPS if "legacy" in kinds]
    assert slots == ["memory & thinking"]
    assert "legacy" in LIST_ONELINERS and "/legacy" in LIST_ONELINERS["legacy"]
    assert "/legacy" in help_text()


def _public_command_kinds():
    from nomorals.social.chat.control import PUBLIC_COMMANDS

    return PUBLIC_COMMANDS
