"""Freeze enforcement for nomorals/storage/migrations.py.

INCIDENT 2026-10-24: EC2 Devon refused to start with
``MigrationError: refusing to migrate: migration 0043... was modified
after being applied`` (12 migrations: 0043, 0044, 0045, 0047, 0048, 0049,
0050, 0052, 0076, 0079, 0080, 0088). Root cause: Pass #105 inserted
``record_failure(...)`` calls plus ``except Exception:`` -> ``except
Exception as e:`` rebinds inside 12 already-applied migration functions.
The tamper guard (``nomorals/storage/schema.py``, ``Migration.source_checksum``
-> ``_fn_fingerprint``) hashes each function body's AST, so those insertions
changed all 12 fingerprints. The file's own header already said it: add a
migration at the end of MIGRATIONS, NEVER edit an earlier one. #105 violated
the file's own rule. The DB was not corrupt and no data was at risk — the
guard was doing its job.

The fix reverted migrations.py to its pre-#105 content byte-for-byte. This
test pins the freeze structurally: no record call and no failures-ledger
import anywhere in the file, so no future ledger pass can repeat #105.

Deliberately NOT fingerprint values: ``ast.dump`` output differs between
Python 3.11 (EC2) and 3.12 (this VM) because 3.12 adds ``type_params``
fields, so value pins would be interpreter-fragile. Structural assertions
only.
"""

from __future__ import annotations

import ast
from pathlib import Path

_MIGRATIONS = Path(__file__).resolve().parents[1] / "nomorals" / "storage" / "migrations.py"


def _tree() -> ast.Module:
    return ast.parse(_MIGRATIONS.read_text(encoding="utf-8"))


def test_migrations_contain_no_record_failure_calls():
    calls = []
    for node in ast.walk(_tree()):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "record_failure":
            calls.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr == "record_failure":
            calls.append(node.lineno)
    assert calls == [], f"record_failure calls in frozen migrations.py: {calls}"


def test_migrations_import_no_failures_ledger():
    imports = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom):
            if node.module in ("failures", "nomorals.storage.failures"):
                imports.append(node.lineno)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in ("failures", "nomorals.storage.failures"):
                    imports.append(node.lineno)
    assert imports == [], f"failures-ledger imports in frozen migrations.py: {imports}"
