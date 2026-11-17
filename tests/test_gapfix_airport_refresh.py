"""Airport dataset refresh (R3-67 named remainder).

The airport→country map (shipped 1812774a) had no refresh anywhere:
``travel/airports.py`` loaded the CSV with no fetch, validation, or
job, so the table silently went stale forever and the only refresh was
a human re-running a generator script that lived outside the repo.
This suite pins the in-repo refresh: the generator rules, the
validation, the atomic replace, the cache invalidation, the monthly
job, and the ``nm airports`` console surface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

_FIXTURE_DUMP = """id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,continent,iso_country,iso_region,municipality,scheduled_service,gps_code,iata_code,local_code,home_link,wikipedia_link,keywords
1,AAAA,small_airport,Alpha Field,0.0,0.0,0,AF,US,US-CA,Alpha,0,,AAA,AAA,,
2,BBBB,small_airport,Beta Field,0.0,0.0,0,AF,NG,NG-LA,Beta,0,,BBB,BBB,,
3,CCCC,small_airport,Gamma Field,0.0,0.0,0,EU,GB,GB-ENG,Gamma,0,,CCC,CCC,,
4,DDDD,small_airport,Bad Code Field,0.0,0.0,0,EU,FR,FR-ID,Delta,0,,12X,12X,,
5,EEEE,small_airport,Bad Country Field,0.0,0.0,0,EU,ZZZ,ZZ-ID,Epsilon,0,,EEE,EEE,,
6,FFFF,small_airport,No Code Field,0.0,0.0,0,EU,DE,DE-BE,Zeta,0,,,,,
7,GGGG,small_airport,Conflict Field,0.0,0.0,0,AF,US,US-NY,Eta,0,,AAA,AAA,,
8,HHHH,small_airport,Same Code Same Country,0.0,0.0,0,AF,US,US-TX,Theta,0,,AAA,AAA,,
"""


def _big_dump(n_codes: int, country: str | None = None) -> str:
    """A well-formed dump with ``n_codes`` distinct IATA codes.

    With no ``country`` the codes cycle through distinct 2-letter
    countries, like the real dump (239 countries); a single explicit
    country makes a degenerate dump validation must refuse.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "ident", "type", "name", "latitude_deg",
                "longitude_deg", "elevation_ft", "continent",
                "iso_country", "iso_region", "municipality",
                "scheduled_service", "gps_code", "iata_code",
                "local_code", "home_link", "wikipedia_link",
                "keywords"])
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(n_codes):
        code = (letters[(i // 676) % 26]
                + letters[(i // 26) % 26]
                + letters[i % 26])
        place = country or (letters[(i // 26) % 26] + letters[i % 26])
        w.writerow([i, code, "small_airport", f"Field {i}", 0.0, 0.0, 0,
                    "AF", place, f"{place}-X", f"Town {i}", 0, "",
                    code, code, "", "", ""])
    return buf.getvalue()


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
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class BuildRulesTests(unittest.TestCase):
    def test_fixture_dump_builds_expected_mapping(self):
        from nomorals.travel.refresh import build_mapping

        mapping, stats = build_mapping(_FIXTURE_DUMP)
        self.assertEqual(mapping, {"AAA": "US", "BBB": "NG",
                                   "CCC": "GB"})
        self.assertEqual(stats["total_rows"], 8)
        # bad code (12X) + bad country (EEE) + conflict (AAA→US vs
        # the later AAA row is the SAME country, so no conflict) —
        # the dropped rows are the bad code and the bad country.
        self.assertEqual(stats["dropped"], 2)
        self.assertEqual(stats["codes"], 3)
        self.assertEqual(stats["countries"], 3)

    def test_conflicting_code_is_dropped_first_wins(self):
        from nomorals.travel.refresh import build_mapping

        dump = _FIXTURE_DUMP.replace(
            "8,HHHH,small_airport,Same Code Same Country,0.0,0.0,0,AF,"
            "US,US-TX,Theta,0,,AAA,AAA,,",
            "8,HHHH,small_airport,Conflict Field,0.0,0.0,0,AF,"
            "CA,CA-ON,Theta,0,,AAA,AAA,,")
        mapping, stats = build_mapping(dump)
        self.assertEqual(mapping["AAA"], "US")
        self.assertEqual(stats["dropped"], 3)

    def test_render_is_sorted_with_header(self):
        from nomorals.travel.refresh import build_mapping, render_mapping

        mapping, _stats = build_mapping(_FIXTURE_DUMP)
        text = render_mapping(mapping)
        rows = list(csv.DictReader(io.StringIO(text)))
        self.assertEqual([r["iata_code"] for r in rows],
                         ["AAA", "BBB", "CCC"])
        self.assertEqual(rows[0]["iso_country"], "US")
        self.assertTrue(text.startswith("iata_code,iso_country\r\n")
                        or text.startswith("iata_code,iso_country\n"))

    def test_rules_reproduce_seed_shape(self):
        """The ported rules on a well-formed dump keep every row."""
        from nomorals.travel.refresh import build_mapping

        mapping, stats = build_mapping(_big_dump(6000))
        self.assertEqual(stats["total_rows"], 6000)
        self.assertEqual(stats["dropped"], 0)
        self.assertEqual(stats["codes"], 6000)


class ValidationTests(unittest.TestCase):
    def test_good_mapping_passes(self):
        from nomorals.travel.refresh import build_mapping, validate_mapping

        mapping, stats = build_mapping(_big_dump(6000))
        self.assertEqual(validate_mapping(mapping, stats), "")

    def test_too_few_rows_fails(self):
        from nomorals.travel.refresh import build_mapping, validate_mapping

        mapping, stats = build_mapping(_FIXTURE_DUMP)
        self.assertIn("too few", validate_mapping(mapping, stats))

    def test_high_dropped_fraction_fails(self):
        from nomorals.travel.refresh import build_mapping, validate_mapping

        bad = []
        for i in range(6000):
            bad.append(f"{i},X{i},small_airport,F{i},0,0,0,AF,US,"
                       f"US-X,T{i},0,,12X,12X,,")
        mapping, stats = build_mapping(_big_dump(6000)
                                       + "\n".join(bad))
        self.assertEqual(stats["total_rows"], 12000)
        self.assertEqual(stats["dropped"], 6000)
        self.assertEqual(stats["codes"], 6000)
        self.assertIn("dropped", validate_mapping(mapping, stats))

    def test_non_csv_text_fails(self):
        from nomorals.travel.refresh import build_mapping, validate_mapping

        mapping, stats = build_mapping("<html>not a csv</html>\n")
        self.assertEqual(stats["codes"], 0)
        self.assertNotEqual(validate_mapping(mapping, stats), "")


class RefreshTests(unittest.TestCase):
    def _seed_csv(self, path: Path, mapping: dict[str, str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["iata_code", "iso_country"])
            for code in sorted(mapping):
                w.writerow([code, mapping[code]])

    def test_refresh_replaces_and_invalidates_cache(self):
        from nomorals.travel import airports
        from nomorals.travel.refresh import refresh_airports

        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "airports_iata_country.csv"
            self._seed_csv(data, {"AAA": "US", "BBB": "NG"})
            old_airports_path = airports.AIRPORTS_DATA_PATH
            airports.AIRPORTS_DATA_PATH = data
            try:
                airports.invalidate_cache()
                self.assertEqual(airports.country_for_iata("AAA"), "US")
                out = refresh_airports(
                    fetcher=lambda: _big_dump(6000),
                    data_path=data, state_path=Path(td) / "state.json")
                self.assertTrue(out["ok"])
                self.assertEqual(out["rows"], 6000)
                # Same process, same call — the refresh must have
                # invalidated the module cache, or AAA still reads US.
                self.assertEqual(airports.country_for_iata("AAA"), "AA")
                self.assertEqual(airports.airport_country_count(), 6000)
            finally:
                airports.AIRPORTS_DATA_PATH = old_airports_path
                airports.invalidate_cache()

    def test_failed_fetch_leaves_old_table_byte_identical(self):
        from nomorals.travel.refresh import refresh_airports

        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "airports_iata_country.csv"
            self._seed_csv(data, {"AAA": "US"})
            before = data.read_bytes()
            sha_before = hashlib.sha256(before).hexdigest()

            def _boom():
                raise ConnectionError("network down")

            out = refresh_airports(fetcher=_boom, data_path=data,
                                   state_path=Path(td) / "state.json")
            self.assertFalse(out["ok"])
            self.assertIn("fetch", out["error"])
            self.assertEqual(data.read_bytes(), before)
            self.assertEqual(hashlib.sha256(data.read_bytes())
                             .hexdigest(), sha_before)
            state = json.loads((Path(td) / "state.json")
                               .read_text(encoding="utf-8"))
            self.assertFalse(state["ok"])
            self.assertIn("fetch", state["error"])

    def test_bad_dump_leaves_old_table_untouched(self):
        from nomorals.travel.refresh import refresh_airports

        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "airports_iata_country.csv"
            self._seed_csv(data, {"AAA": "US"})
            before = data.read_bytes()
            out = refresh_airports(
                fetcher=lambda: "<html>not a csv</html>\n",
                data_path=data, state_path=Path(td) / "state.json")
            self.assertFalse(out["ok"])
            self.assertEqual(data.read_bytes(), before)

    def test_tick_never_raises_and_reports_failure(self):
        from nomorals.travel.refresh import run_airport_refresh_tick_once

        class _Notifier:
            def __init__(self):
                self.calls = []

            def publish(self, kind, title="", body="", **_kw):
                self.calls.append((kind, title, body))

        out = run_airport_refresh_tick_once(
            fetcher=lambda: (_ for _ in ()).throw(
                ConnectionError("network down")),
            notifier=_Notifier())
        self.assertFalse(out["ok"])
        self.assertIn("fetch", out["error"])

    def test_tick_publishes_only_on_failure(self):
        from nomorals.travel.refresh import run_airport_refresh_tick_once

        class _Notifier:
            def __init__(self):
                self.calls = []

            def publish(self, kind, title="", body="", **_kw):
                self.calls.append((kind, title, body))

        with tempfile.TemporaryDirectory() as td:
            data = Path(td) / "airports_iata_country.csv"
            self._seed_csv(data, {"AAA": "US"})
            notifier = _Notifier()
            out = run_airport_refresh_tick_once(
                fetcher=lambda: _big_dump(6000),
                data_path=data, state_path=Path(td) / "state.json",
                notifier=notifier)
            self.assertTrue(out["ok"])
            self.assertEqual(notifier.calls, [])


class JobTests(unittest.TestCase):
    def test_job_registered_monthly_first_0400(self):
        from nomorals.travel.refresh import (
            AIRPORTS_REFRESH_JOB,
            AIRPORTS_REFRESH_SPEC,
            AIRPORTS_REFRESH_TOOL,
            ensure_airport_refresh_job,
        )

        sched = _FakeScheduler()
        self.assertIsNone(ensure_airport_refresh_job(None))
        out = ensure_airport_refresh_job(sched)
        self.assertEqual(out["name"], AIRPORTS_REFRESH_JOB)
        self.assertEqual(len(sched.jobs), 1)
        job = sched.jobs[0]
        self.assertEqual(job["spec"], "cron 0 4 1 * *")
        self.assertEqual(job["spec"], AIRPORTS_REFRESH_SPEC)
        self.assertEqual(job["payload"]["tool"], AIRPORTS_REFRESH_TOOL)
        self.assertTrue(job.get("routine"))

    def test_job_idempotent_and_replaces_stale_spec(self):
        from nomorals.travel.refresh import ensure_airport_refresh_job

        sched = _FakeScheduler()
        ensure_airport_refresh_job(sched)
        ensure_airport_refresh_job(sched)
        self.assertEqual(len(sched.jobs), 1)
        sched.jobs[0]["spec"] = "cron 0 3 1 * *"
        ensure_airport_refresh_job(sched)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(sched.jobs[0]["spec"], "cron 0 4 1 * *")


class CliTests(unittest.TestCase):
    def _ctx(self, home: Path):
        from types import SimpleNamespace

        settings = SimpleNamespace(home=str(home), home_path=home)
        return SimpleNamespace(db=None, settings=settings)

    def _ns(self, **kwargs):
        args = argparse.Namespace(airports_action="status",
                                  airports_rest=[], json=False)
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_status_names_rows_and_last_refresh(self):
        from nomorals.cmdline.commands.airports import _cmd_airports
        from nomorals.travel import airports

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            data = home / "airports_iata_country.csv"
            with open(data, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["iata_code", "iso_country"])
                w.writerow(["AAA", "US"])
                w.writerow(["BBB", "NG"])
            old = airports.AIRPORTS_DATA_PATH
            airports.AIRPORTS_DATA_PATH = data
            try:
                airports.invalidate_cache()
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    rc = _cmd_airports(self._ns(), self._ctx(home))
                finally:
                    sys.stdout = old_stdout
                self.assertEqual(rc, 0)
                text = buf.getvalue()
                self.assertIn("2", text)
                self.assertIn("never", text)
            finally:
                airports.AIRPORTS_DATA_PATH = old
                airports.invalidate_cache()

    def test_refresh_runs_now(self):
        from nomorals.cmdline.commands.airports import _cmd_airports
        from nomorals.travel import airports
        from nomorals.travel import refresh as refresh_mod

        with tempfile.TemporaryDirectory() as td:
            home = Path(td)
            data = home / "airports_iata_country.csv"
            with open(data, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["iata_code", "iso_country"])
                w.writerow(["AAA", "US"])
            old_path = airports.AIRPORTS_DATA_PATH
            old_data = refresh_mod.DATA_PATH
            old_fetch = refresh_mod._fetch_dump
            airports.AIRPORTS_DATA_PATH = data
            refresh_mod.DATA_PATH = data
            refresh_mod._fetch_dump = lambda: _big_dump(6000)
            try:
                airports.invalidate_cache()
                buf = io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    rc = _cmd_airports(
                        self._ns(airports_action="refresh"),
                        self._ctx(home))
                finally:
                    sys.stdout = old_stdout
                self.assertEqual(rc, 0)
                self.assertIn("6000", buf.getvalue())
                self.assertEqual(airports.country_for_iata("AAA"),
                                 "AA")
            finally:
                airports.AIRPORTS_DATA_PATH = old_path
                refresh_mod.DATA_PATH = old_data
                refresh_mod._fetch_dump = old_fetch
                airports.invalidate_cache()


if __name__ == "__main__":
    unittest.main()
