"""Corpora ports (lane 118) — census §3A.25–A27.

The three storytold test corpora, vendored as oracles:

* A25 photocraft-corpus — 50 PSDs in tests/fixtures/photocraft-corpus/
  (plus the 4 lane-76 fixtures in tests/fixtures/photocraft/), run
  through the photo_editor oracle against Photoshop's own merged
  composite in each file. The full 258-file corpus stays in the
  mining checkout for the extended manual run; the vendored subset
  covers every mode/depth cell and every file the photo_editor
  tests name (including the same filename in different mode dirs).
  The gimp/ set (XCF) is NOT vendored: Devon has no
  XCF reader, so it would oracle nothing (survey remainder).
* A26 effectcraft-corpus — 16 .aep + 1 .aepx + 3 .ffx in
  tests/fixtures/effectcraft/aep/, read by the RIFX structure probe
  (nomorals/media/effects/aep.py). Every CASES.md composition name
  must be recovered from its file. Full project import is not
  built; the probe says so instead of pretending otherwise.
* A27 cadcraft-corpus — all 4 drawings already vendored by lane 106
  in tests/fixtures/cadcraft/; this lane pins their hashes against
  the upstream SHA256SUMS, records the LICENSES.tsv rows, and pins
  the DXF import counts.

Discipline, per the census: corpus absence is a test FAILURE, never
a silent skip. Every test here first requires its files present
(``_require`` names the missing file); the two skip-based
full-corpus tests in test_gapfix_photocraft.py /
test_gapfix_photocraft2.py were removed by this lane and their
gradient-map usages repointed at the vendored copy, so no test in
the suite skips on a missing corpus any more.
"""

from __future__ import annotations

import hashlib
import tomllib
import unittest
from collections import Counter
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = Path(__file__).resolve().parent.parent
PHOTO = FIXTURES / "photocraft"
PHOTO_CORPUS = FIXTURES / "photocraft-corpus"
EFFECT = FIXTURES / "effectcraft" / "aep"
CAD = FIXTURES / "cadcraft"
CLONES = Path.home() / "workspace" / "echelon-mining" / "repos" / "storytold"


def _load_toml(name: str) -> dict:
    with open(ROOT / name, "rb") as handle:
        return tomllib.load(handle)


ATTRIBUTION = _load_toml("corpora-attribution.toml")
PARITY = _load_toml("corpora-parity.toml")


def _require(paths: list[Path]) -> None:
    """Fail naming the file — never skip when a corpus file is gone."""
    for path in paths:
        if not path.is_file():
            raise AssertionError(f"missing corpus fixture {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _upstream_sums(repo: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (CLONES / repo / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(None, 1)
        out[rel.strip()] = digest
    return out


class AttributionTests(unittest.TestCase):
    def test_every_attributed_file_exists_and_hashes(self) -> None:
        assets = ATTRIBUTION["assets"]
        self.assertGreater(len(assets), 90)
        for asset in assets:
            path = ROOT / asset["path"]
            _require([path])
            self.assertEqual(
                _sha256(path), asset["sha256"],
                f"hash mismatch for {asset['path']}",
            )
            self.assertEqual(asset["license"], "MIT OR Apache-2.0")

    def test_upstream_manifest_hashes_match(self) -> None:
        """Vendored payload hashes equal the upstream SHA256SUMS."""
        checked = 0
        for repo in (
            "photocraft-corpus",
            "effectcraft-corpus",
            "cadcraft-corpus",
        ):
            sums = _upstream_sums(repo)
            for asset in ATTRIBUTION["assets"]:
                if asset["repo"] != f"storytold/{repo}":
                    continue
                upstream = asset["upstream_path"]
                if upstream in sums:
                    self.assertEqual(
                        asset["sha256"], sums[upstream],
                        f"{repo} {upstream}: vendored hash != manifest",
                    )
                    checked += 1
        self.assertEqual(checked, 78)  # 54 PSD + 20 AEP/FFX + 4 DXF/DWG

    def test_effectcraft_sidecars_carry_the_licence(self) -> None:
        payloads = sorted(EFFECT.rglob("*.aep")) + sorted(
            EFFECT.rglob("*.aepx")) + sorted(EFFECT.rglob("*.ffx"))
        self.assertEqual(len(payloads), 20)
        for payload in payloads:
            sidecar = payload.with_suffix(payload.suffix + ".attribution")
            _require([sidecar])
            text = sidecar.read_text()
            self.assertIn("license:      MIT OR Apache-2.0", text)
            self.assertIn("asset:", text)

    def test_cadcraft_licences_rows_cover_every_drawing(self) -> None:
        rows = (CAD / "LICENSES.tsv").read_text().splitlines()
        header, body = rows[0], rows[1:]
        self.assertTrue(header.startswith("path\tauthor"))
        by_path = {row.split("\t")[0]: row.split("\t") for row in body}
        for name in ("sample-bracket.dwg", "sample-bracket.dxf",
                     "sample-floorplan.dwg", "sample-floorplan.dxf"):
            key = f"cadcraft/{name}"
            self.assertIn(key, by_path, f"no LICENSES.tsv row for {key}")
            self.assertEqual(by_path[key][3], "MIT OR Apache-2.0")


class PhotocraftCorpusTests(unittest.TestCase):
    def test_vendored_oracle_matches_parity(self) -> None:
        from nomorals.media.photo_editor.oracle import run_oracle

        pinned = PARITY["photocraft"]
        # Require the PINNED list first: rglob only finds files that
        # exist, so a missing fixture must be caught here, by name.
        _require([ROOT / entry["path"] for entry in pinned["file"]])
        files = sorted(PHOTO.rglob("*.psd")) + sorted(
            PHOTO_CORPUS.rglob("*.psd"))
        self.assertEqual(len(files), pinned["files"])
        by_path = {
            entry["path"]: entry for entry in pinned["file"]
        }
        self.assertEqual(
            sorted(by_path),
            sorted(
                str(p.relative_to(ROOT)) for p in files
            ),
            "vendored files differ from the parity pin list",
        )
        report = run_oracle(str(PHOTO))
        corpus_report = run_oracle(str(PHOTO_CORPUS))
        summary = {
            key: report["summary"][key] + corpus_report["summary"][key]
            for key in (
                "files", "exact", "differs", "differs_adjustment",
                "differs_blend", "differs_unrendered", "failed",
                "byte_stable", "doc_byte_stable", "roundtrip_ok",
                "roundtrip_depth_reduced",
            )
        }
        for key, value in summary.items():
            self.assertEqual(
                value, pinned[key], f"photocraft oracle {key}: {value}")
        entries = {
            str(Path(entry["path"]).relative_to(ROOT)): entry
            for entry in report["files"] + corpus_report["files"]
        }
        for rel, pin in by_path.items():
            entry = entries[rel]
            self.assertEqual(
                entry["status"], pin["status"], rel)
            self.assertLessEqual(
                entry.get("max_diff", 0), pin["max_diff"],
                f"{rel}: max_diff {entry.get('max_diff')} > pin "
                f"{pin['max_diff']}",
            )

    def test_every_mode_depth_cell_has_a_file(self) -> None:
        from nomorals.media.photo_editor.psd import PsdFile

        cells: set[tuple[str, int]] = set()
        for path in sorted(PHOTO_CORPUS.rglob("*.psd")):
            psd = PsdFile.from_bytes(path.read_bytes())
            cells.add((psd.mode, psd.depth))
        for cell in (
            ("RGB", 8), ("RGB", 16), ("RGB", 32),
            ("CMYK", 8), ("CMYK", 16),
            ("L", 8), ("L", 16), ("L", 32),
            ("Lab", 8), ("Lab", 16),
        ):
            self.assertIn(cell, cells, f"no vendored PSD for {cell}")


class EffectcraftCorpusTests(unittest.TestCase):
    def test_every_project_probes_and_recovers_its_comps(self) -> None:
        from nomorals.media.effects.aep import (
            missing_compositions,
            probe_file,
        )

        parity = PARITY["effectcraft"]
        # Require the PINNED list first, by name (see photocraft).
        _require([EFFECT / rel for rel in parity])
        payloads = sorted(EFFECT.rglob("*.aep")) + sorted(
            EFFECT.rglob("*.aepx")) + sorted(EFFECT.rglob("*.ffx"))
        self.assertEqual(len(payloads), 20)
        self.assertEqual(
            sorted(str(p.relative_to(EFFECT)) for p in payloads),
            sorted(parity),
        )
        for payload in payloads:
            rel = str(payload.relative_to(EFFECT))
            pin = parity[rel]
            probe = probe_file(payload)
            self.assertEqual(probe.kind, pin["kind"], rel)
            self.assertEqual(probe.form, pin["form"], rel)
            self.assertEqual(probe.file_size, pin["file_size"], rel)
            self.assertEqual(probe.chunk_count, pin["chunk_count"], rel)
            self.assertEqual(probe.max_depth, pin["max_depth"], rel)
            self.assertEqual(
                probe.trailing_bytes, pin["trailing_bytes"], rel)
            self.assertEqual(
                probe.trailing_is_xmp, pin["trailing_is_xmp"], rel)
            self.assertFalse(probe.import_supported, rel)
            self.assertIn("aep_import_not_implemented",
                          probe.import_refusal, rel)
            self.assertEqual(
                missing_compositions(probe, pin.get("comps", [])), [],
                rel,
            )
            for marker in pin.get("markers", []):
                self.assertTrue(
                    any(marker in name for name in probe.names),
                    f"{rel}: marker {marker!r} not recovered",
                )
            if probe.kind == "aepx":
                self.assertEqual(list(probe.items), pin["items"], rel)
                self.assertEqual(probe.item_count, pin["item_count"], rel)

    def test_probe_refuses_garbage_and_truncation_by_name(self) -> None:
        from nomorals.media.effects.aep import (
            AepError,
            _probe_rifx,
            parse_aepx_items,
        )

        with self.assertRaises(AepError):
            _probe_rifx("garbage.aep", b"not a rifx file at all",
                        kind="aep", form="Egg!")
        head = (EFFECT / "layers-comps.aep").read_bytes()[:100]
        with self.assertRaises(AepError) as caught:
            _probe_rifx("truncated.aep", head, kind="aep", form="Egg!")
        self.assertIn("truncated", str(caught.exception))
        with self.assertRaises(AepError):
            parse_aepx_items(b"<broken><xml", what="aepx broken.aepx")

    def test_aepx_item_names_are_the_case_comps(self) -> None:
        from nomorals.media.effects.aep import probe_file

        probe = probe_file(EFFECT / "aepx-pair.aepx")
        for name in ("000-base", "058-key-opacity-2key",
                     "060-expr-opacity-wiggle", "110-text-A",
                     "124-precomp-Pre"):
            self.assertIn(name, probe.items)


class CadcraftCorpusTests(unittest.TestCase):
    def test_dxf_import_counts_match_parity(self) -> None:
        from nomorals.media.cad.codec import import_dxf

        for name, pin in PARITY["cadcraft"].items():
            if not name.endswith(".dxf"):
                continue
            _require([CAD / name])
            doc = import_dxf((CAD / name).read_bytes())
            self.assertEqual(len(doc.entities), pin["entities"], name)
            kinds = Counter(
                type(entity).__name__ for entity in doc.entities.values()
            )
            self.assertEqual(dict(kinds), pin["kinds"], name)
            layers = sorted({e.layer for e in doc.entities.values()})
            self.assertEqual(layers, pin["layers"], name)

    def test_dwg_refused_honestly_for_both_drawings(self) -> None:
        from nomorals.media.cad.codec import import_drawing

        for name in PARITY["cadcraft"]["dwg"]["files"]:
            _require([CAD / name])
            with self.assertRaises(Exception) as caught:
                import_drawing(name, (CAD / name).read_bytes())
            self.assertIn(
                PARITY["cadcraft"]["dwg"]["refusal"], str(caught.exception)
            )


class AbsenceFailsTests(unittest.TestCase):
    def test_require_names_the_missing_file(self) -> None:
        with self.assertRaises(AssertionError) as caught:
            _require([FIXTURES / "no-such-corpus-file.psd"])
        self.assertIn("no-such-corpus-file.psd", str(caught.exception))

    def test_oracle_over_an_empty_dir_reports_zero_files(self) -> None:
        """The guard that turns zero files into a failure, not a skip."""
        import tempfile

        from nomorals.media.photo_editor.oracle import run_oracle

        with tempfile.TemporaryDirectory() as tmp:
            report = run_oracle(tmp)
        self.assertEqual(report["summary"]["files"], 0)
        # The parity test above asserts files == 54, so an empty
        # corpus dir fails it; nothing here skips.


if __name__ == "__main__":
    unittest.main()
