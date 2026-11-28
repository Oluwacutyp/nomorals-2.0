"""Telegram user database (osint_telegramdb): the leaked-corpus
lookup behind /lookup's fourth phone route.

The database is the Telegram user corpus mirrored publicly on
Hugging Face (rehuuuu/Telegram-DB, 187.7M rows): account_id,
phone, username.  A local indexed SQLite copy is built once by
the provision and queried read-only afterwards — no network at
lookup time, no new runtime dependency (stdlib sqlite3).

These tests use only the standard library: fixtures are built
with the module's own ``_write_sqlite`` helper, so a fixture is
a real database file, not a mock of one.
"""

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROWS = [
    # Synthetic fixture rows (fake IDs/handles/numbers — test
    # data only, not real accounts).
    ("100000001", "fixture_alpha", "15551234567"),
    ("100000002", "fixture_beta", "15551234568"),
    ("100000003", "fixture_gamma", "15551234569"),
]


def _fixture_db(path: Path) -> Path:
    from nomorals.agents.osint_telegramdb import _write_sqlite

    path.parent.mkdir(parents=True, exist_ok=True)
    _write_sqlite(iter(ROWS), path, dataset="test-fixture")
    return path


class TelegramDbLookupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-tdb-")
        self.home = Path(self._tmp.name)
        self.db = _fixture_db(self.home / "osint" / "telegramdb.sqlite")

    def tearDown(self):
        self._tmp.cleanup()

    def test_lookup_by_id(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb("100000001", "",
                                 db_path=self.db) == "15551234567"

    def test_lookup_by_username(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb("", "fixture_beta",
                                 db_path=self.db) == "15551234568"

    def test_lookup_by_username_is_case_insensitive(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb("", "FIXTURE_BETA",
                                 db_path=self.db) == "15551234568"
        assert lookup_telegramdb("", "Fixture_Gamma",
                                 db_path=self.db) == "15551234569"

    def test_lookup_prefers_id_over_username(self):
        # An ID row and a username row naming different accounts:
        # the ID wins — the ID is the identity, the handle can move.
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb("100000002", "fixture_alpha",
                                 db_path=self.db) == "15551234568"

    def test_lookup_most_frequent_phone_wins_for_duplicates(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        dup = _fixture_db(self.home / "osint2" / "telegramdb.sqlite")
        with sqlite3.connect(dup) as con:
            con.executemany(
                "INSERT INTO telegram_users VALUES (?, ?, ?)",
                [("777", "dup_user", "1111111111"),
                 ("777", "dup_user", "1111111111"),
                 ("777", "dup_user", "2222222222")])
            con.commit()
        assert lookup_telegramdb("777", "", db_path=dup) == "1111111111"

    def test_lookup_miss_returns_none(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb("123456789", "",
                                 db_path=self.db) is None
        assert lookup_telegramdb("", "nobody_here_ever",
                                 db_path=self.db) is None
        assert lookup_telegramdb("", "", db_path=self.db) is None

    def test_lookup_missing_file_returns_none(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        assert lookup_telegramdb(
            "100000001", "",
            db_path=self.home / "osint" / "no-such.sqlite") is None

    def test_lookup_corrupt_file_returns_none(self):
        from nomorals.agents.osint_telegramdb import lookup_telegramdb

        bad = self.home / "bad.sqlite"
        bad.write_text("this is not a database", encoding="utf-8")
        assert lookup_telegramdb("100000001", "",
                                 db_path=bad) is None

    def test_db_status_missing(self):
        from nomorals.agents.osint_telegramdb import db_status

        st = db_status(home=self.home / "empty-home")
        assert st["present"] is False
        assert st["rows"] is None

    def test_db_status_present(self):
        from nomorals.agents.osint_telegramdb import db_status

        st = db_status(home=self.home)
        assert st["present"] is True
        assert st["rows"] == len(ROWS)
        assert st["bytes"] and st["bytes"] > 0
        assert st["dataset"] == "test-fixture"

    def test_telegramdb_path(self):
        from nomorals.agents.osint_telegramdb import telegramdb_path

        assert str(telegramdb_path(home=self.home)).endswith(
            os.path.join("osint", "telegramdb.sqlite"))

    def test_write_and_verify_roundtrip(self):
        from nomorals.agents.osint_telegramdb import _write_sqlite, lookup_telegramdb, verify_db

        path = self.home / "roundtrip.sqlite"
        _write_sqlite(iter(ROWS), path, dataset="test-fixture")
        assert verify_db(path)["ok"] is True
        assert lookup_telegramdb("100000003", "",
                                 db_path=path) == "15551234569"

    def test_build_refuses_when_disk_short(self):
        from nomorals.agents import osint_telegramdb as tdb

        # A fresh home — the fixture DB from setUp would make the
        # build return "already stocked" before the disk guard.
        home = Path(tempfile.mkdtemp(prefix="nm-test-tdb-"))
        res = tdb.build_db(home=home, need_bytes=10 ** 18)
        assert res["status"] == "blocked"
        assert "free" in str(res.get("detail") or "").lower()
        assert not (home / "osint" / "telegramdb.sqlite").exists()

    def test_build_from_local_parquet_files(self):
        # The real build path, minus the network: two parquet
        # shards already on disk stand in for the HF file list.
        # (Needs duckdb; the test env has it only when the lane's
        # verify venv is on the path — skip honestly otherwise.)
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb not installed in this venv")
        from nomorals.agents import osint_telegramdb as tdb

        shards = [
            "/home/hatch/workspace/gapfix165-lookupdb-verify/tmp/"
            "data_99.parquet",
            "/home/hatch/workspace/gapfix165-lookupdb-verify/tmp/"
            "data_10.parquet",
        ]
        if not all(os.path.exists(s) for s in shards):
            self.skipTest("local parquet shards not present")
        # A fresh home — the fixture DB from setUp would make the
        # build return "already stocked" before it ever reads.
        home = Path(tempfile.mkdtemp(prefix="nm-test-tdb-"))
        res = tdb.build_db(home=home, parquet_files=shards,
                           need_bytes=0)
        assert res["status"] == "ready", res
        assert res["rows"] and res["rows"] > 50000
        # A real row from the shards (ID seen in the shard probe):
        # the build returns its real number — asserted by shape
        # only, never written out in full.
        phone = tdb.lookup_telegramdb("990055504", "", home=home)
        assert phone and len(phone) == 12
        assert phone.startswith("99") and phone.endswith("97")


if __name__ == "__main__":
    unittest.main()
