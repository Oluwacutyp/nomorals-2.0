"""Tail-63 pins for nomorals/accounts/identity_bank.py —
0 converts / 6 exclusions. Each test pins one site with a
control half proving the guarded operation was actually
ATTEMPTED via the double's/spy's own call log. The file
contains no exception handlers, so it adds zero census sites.

identity_bank.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the accounts package ledger (nomorals/accounts/failures.py):
recent_failures() snapshot before/after every trigger.

Verdict notes (full grounds in the tail-63 SURVEY.md):
- 356 / 369 / 469 / 480: Persona.from_dict failure on ONE
  stored row -> the row is skipped, the rest load. The
  fallback IS the documented design ("skip corrupt rows");
  from_dict is deliberately lenient (coerce-with-defaults),
  so a raise means a genuinely unparseable row that no
  consumer could honor as a Persona.
- 442 / 455: get() on a corrupt row -> None, byte-identical
  to a missing id. Both production consumers
  (trial/flow.py) treat None as "mint fresh"
  (get(...) or get_or_mint(...)), which is get()'s
  documented contract (Persona | None).
"""

import json

import pytest

from nomorals.accounts import identity_bank as IB
from nomorals.accounts import failures as acct_fail
from nomorals.storage.db import Database


def _snap():
    return list(acct_fail.recent_failures(200))


GOOD = {"id": "good1", "service": "svc", "first_name": "Ada",
        "last_name": "Trial", "created_at": 1000.0}
BAD = {"id": "bad1", "service": "svc", "first_name": "Bad",
       "last_name": "Trial", "created_at": "not-a-float"}


class Cred:
    def __init__(self, meta):
        self.metadata = meta
        self.username = meta.get("id", "")


class FakeVault:
    def __init__(self, creds):
        self._creds = creds
        self.calls = 0

    def list_all(self, service=None, active_only=True):
        self.calls += 1
        return list(self._creds)


@pytest.fixture()
def from_dict_spy(monkeypatch):
    seen = []
    real = IB.Persona.from_dict.__func__

    def spy(cls, data):
        seen.append(data.get("id"))
        return real(cls, data)

    monkeypatch.setattr(IB.Persona, "from_dict", classmethod(spy))
    return seen


def _db_bank_with_rows():
    db = Database(":memory:")
    bank = IB.IdentityBank(db=db)
    with db.transaction():
        db.execute(
            "INSERT INTO identity_bank_personas "
            "(id, service, data, updated_at) VALUES (?, ?, ?, ?)",
            ("bad1", "svc", json.dumps(BAD), 1.0))
        db.execute(
            "INSERT INTO identity_bank_personas "
            "(id, service, data, updated_at) VALUES (?, ?, ?, ?)",
            ("good1", "svc", json.dumps(GOOD), 2.0))
    return bank, db


# 356 — vault _load_all skips the corrupt row, loads the rest
def test_356_vault_load_all_skips_corrupt(from_dict_spy):
    before = _snap()
    vault = FakeVault([Cred(dict(BAD)), Cred(dict(GOOD))])
    bank = IB.IdentityBank(vault=vault)
    out = bank._load_all("svc")
    assert [p.id for p in out] == ["good1"]
    assert from_dict_spy == ["bad1", "good1"]  # corrupt row attempted
    assert vault.calls == 1
    assert _snap() == before  # NOT recorded


# 369 — db _load_all skips the corrupt row, loads the rest
def test_369_db_load_all_skips_corrupt(from_dict_spy):
    before = _snap()
    bank, db = _db_bank_with_rows()
    out = bank._load_all("svc")
    assert [p.id for p in out] == ["good1"]
    assert from_dict_spy == ["bad1", "good1"]
    rows = db.query("SELECT id FROM identity_bank_personas")
    assert len(rows) == 2  # the corrupt row really is stored
    assert _snap() == before


# 442 — vault get() on a corrupt row returns None (== missing)
def test_442_vault_get_corrupt_returns_none(from_dict_spy):
    before = _snap()
    vault = FakeVault([Cred(dict(BAD))])
    bank = IB.IdentityBank(vault=vault)
    assert bank.get("bad1") is None
    assert from_dict_spy == ["bad1"]  # parse was attempted
    assert _snap() == before


# 455 — db get() on a corrupt row returns None (== missing)
def test_455_db_get_corrupt_returns_none(from_dict_spy):
    before = _snap()
    bank, db = _db_bank_with_rows()
    assert bank.get("bad1") is None
    assert from_dict_spy == ["bad1"]
    row = db.query_one(
        "SELECT id FROM identity_bank_personas WHERE id = ?",
        ("bad1",))
    assert row is not None  # the row exists; only its parse failed
    assert _snap() == before


# 469 — vault list() skips the corrupt row
def test_469_vault_list_skips_corrupt(from_dict_spy):
    before = _snap()
    vault = FakeVault([Cred(dict(BAD)), Cred(dict(GOOD))])
    bank = IB.IdentityBank(vault=vault)
    out = bank.list()
    assert [p.id for p in out] == ["good1"]
    assert from_dict_spy == ["bad1", "good1"]
    assert _snap() == before


# 480 — db list() skips the corrupt row
def test_480_db_list_skips_corrupt(from_dict_spy):
    before = _snap()
    bank, db = _db_bank_with_rows()
    out = bank.list()
    assert [p.id for p in out] == ["good1"]
    assert from_dict_spy == ["bad1", "good1"]
    assert _snap() == before
