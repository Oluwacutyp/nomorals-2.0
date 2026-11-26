"""Indd-Utils port (census §3A.24) — storytold/indd-utils in Devon.

Devon already has a documents engine with a measured-fidelity corpus
(nomorals/documents/corpus.py — "The indd-utils discipline, applied"),
but zero INDD/IDML handling: no InDesign file can be opened, read or
converted. This test pins the port: the full INDD pipeline (header →
container → database → object cursor → model → IDML writer) and the
measured-fidelity differ (paired-file diffing, coverage % as headline),
both proven on three real .indd fixtures spanning InDesign 3.0
big-endian, 4.0 big-endian and 20.3 little-endian.
"""

from __future__ import annotations

import hashlib
import io
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "inddutils"
_BLUE = _FIXTURES / "BlueSquare.indd"
_NEDDY = _FIXTURES / "Neddy.indd"
_MINI = _FIXTURES / "minizine.indd"

_BLUE_SHA = "5e4d1db27034f91fdc915c5fed77767935ae57015963c83ada25033860e144e0"
_NEDDY_SHA = "8f206d618c0161cfddf4c18aa3932ed99d9e500bf1260c2551e316b12a4e3ed2"
_MINI_SHA = "189021fc8c1d051e9528f155348bbf63bc90eac856f33668180bdfa8972b8cd8"

from nomorals.documents.indd import (  # noqa: E402
    InddError,
    convert,
    convert_file,
    measure_fidelity,
    read_header,
)
from nomorals.documents.indd.attribution import (  # noqa: E402
    verify_attribution,
)
from nomorals.documents.indd.parity import run_parity  # noqa: E402
from nomorals.tools.registry import ToolRegistry  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parts(idml: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(idml)) as zf:
        return {name: zf.read(name) for name in zf.namelist()}


def _story_texts(idml: bytes) -> list[str]:
    parts = _parts(idml)
    # Join in the designmap's StoryList order — the package's own
    # document order. Filename order is NOT document order: Neddy's
    # StoryList starts "ud0 uf1 u17c ..." and story uf1 is non-empty,
    # so a lexicographic join can never equal the source join.
    designmap = ET.fromstring(parts["designmap.xml"])
    order = (designmap.get("StoryList") or "").split()
    texts = []
    for uid in order:
        name = f"Stories/Story_{uid}.xml"
        if name not in parts:
            continue
        root = ET.fromstring(parts[name])
        texts.append("".join(
            node.text or "" for node in root.iter("Content")))
    return texts


class FixtureTest(unittest.TestCase):
    def test_fixtures_are_the_manifest_bytes(self):
        self.assertEqual(_sha(_BLUE), _BLUE_SHA)
        self.assertEqual(_sha(_NEDDY), _NEDDY_SHA)
        self.assertEqual(_sha(_MINI), _MINI_SHA)

    def test_fixtures_have_the_indd_signature(self):
        sig = bytes([0x06, 0x06, 0xED, 0xF5, 0xD8, 0x1D, 0x46, 0xE5,
                     0xBD, 0x31, 0xEF, 0xE7, 0xFE, 0x74, 0xB7, 0x1D])
        for path in (_BLUE, _NEDDY, _MINI):
            self.assertEqual(path.read_bytes()[:16], sig)


class HeaderTest(unittest.TestCase):
    def test_blue_square_is_big_endian_indesign_4(self):
        header = read_header(_BLUE.read_bytes())
        self.assertEqual(header.byte_order.value, "big")
        self.assertEqual((header.version.major, header.version.minor),
                         (4, 0))
        self.assertEqual(header.kind, b"DOCUMENT")

    def test_neddy_is_big_endian_indesign_3(self):
        header = read_header(_NEDDY.read_bytes())
        self.assertEqual(header.byte_order.value, "big")
        self.assertEqual((header.version.major, header.version.minor),
                         (3, 0))

    def test_minizine_is_little_endian_indesign_20(self):
        header = read_header(_MINI.read_bytes())
        self.assertEqual(header.byte_order.value, "little")
        self.assertEqual((header.version.major, header.version.minor),
                         (20, 3))

    def test_refusals_name_the_problem(self):
        with self.assertRaises(InddError) as ctx:
            read_header(b"%PDF-1.5 a short file of another kind")
        self.assertEqual(ctx.exception.kind, "not_indd")
        with self.assertRaises(InddError) as ctx:
            read_header(bytes([0x06, 0x06, 0xED, 0xF5]))
        self.assertEqual(ctx.exception.kind, "truncated")
        bad_order = bytearray(_BLUE.read_bytes()[:0x25])
        bad_order[0x18] = 7
        with self.assertRaises(InddError) as ctx:
            read_header(bytes(bad_order))
        self.assertEqual(ctx.exception.kind, "unknown_byte_order")


class ContainerTest(unittest.TestCase):
    def test_blue_square_masters_and_pages(self):
        from nomorals.documents.indd.container import Container
        container = Container.parse(_BLUE.read_bytes())
        self.assertEqual(container.active, 0)
        self.assertEqual(container.master().db_pages, 182)
        self.assertEqual(container.contig_start(), 182 * 4096)

    def test_neddy_and_minizine_masters(self):
        from nomorals.documents.indd.container import Container
        neddy = Container.parse(_NEDDY.read_bytes())
        self.assertEqual(neddy.active, 0)
        self.assertEqual(neddy.master().db_pages, 364)
        mini = Container.parse(_MINI.read_bytes())
        self.assertEqual(mini.active, 0)
        self.assertEqual(mini.master().db_pages, 337)

    def test_contiguous_objects_iterate_and_xmp_found(self):
        from nomorals.documents.indd.container import Container
        container = Container.parse(_BLUE.read_bytes())
        objects = list(container.contig_objects())
        self.assertTrue(len(objects) >= 1)
        packet = container.xmp()
        self.assertIsNotNone(packet)
        self.assertTrue(packet.startswith(b"<?xpacket begin="))


class DatabaseTest(unittest.TestCase):
    def _database(self, path: Path):
        from nomorals.documents.indd.container import Container
        from nomorals.documents.indd.database import Database
        return Database.open(Container.parse(path.read_bytes()))

    def test_blue_square_trees(self):
        db = self._database(_BLUE)
        self.assertEqual(len(db.entries), 338)
        self.assertEqual(len(db.classes), 187)
        self.assertEqual(len(db.unclassed), 0)
        self.assertEqual(len(list(db.uids())), 186)

    def test_neddy_trees(self):
        db = self._database(_NEDDY)
        self.assertEqual(len(db.entries), 1590)
        self.assertEqual(len(db.classes), 906)
        self.assertEqual(len(db.unclassed), 763)
        self.assertEqual(len(list(db.uids())), 1401)

    def test_minizine_trees(self):
        db = self._database(_MINI)
        self.assertEqual(len(db.entries), 538)
        self.assertEqual(len(db.classes), 271)
        self.assertEqual(len(db.unclassed), 60)
        self.assertEqual(len(list(db.uids())), 330)

    def test_object_bytes_round_trip_through_segments(self):
        db = self._database(_BLUE)
        obj = db.get(1)
        self.assertIsNotNone(obj)
        self.assertEqual(obj.uid, 1)
        self.assertEqual(obj.class_id, 0xE01)
        self.assertTrue(len(obj.bytes) > 0)
        self.assertIsNone(db.get(0xFFFFFF))

    def test_class_distribution_of_blue_square(self):
        from collections import Counter
        db = self._database(_BLUE)
        counts = Counter(c for _, c in db.classes)
        self.assertEqual(counts[0xE01], 1)   # DOCUMENT
        self.assertEqual(counts[0x501], 1)   # SPREAD
        self.assertEqual(counts[0x1401], 1)  # MASTER_SPREAD
        self.assertEqual(counts[0x50F], 3)   # PAGE
        self.assertEqual(counts[0x201], 1)   # STORY
        self.assertEqual(counts[0x205], 3)   # STYLE
        self.assertEqual(counts[0x1F05], 15)  # COLOR
        self.assertEqual(counts[0x1F07], 4)  # INK
        self.assertEqual(counts[0x1F11], 37)  # UI_COLOR
        self.assertEqual(counts[0x2D07], 36)  # LANGUAGE
        self.assertEqual(counts[0x3E03], 4)  # FONT_FAMILY
        self.assertEqual(counts[0x3301], 12)  # GUIDE
        self.assertEqual(counts[0x6201], 1)  # SPLINE_ITEM


class ObjectTest(unittest.TestCase):
    def test_chunks_split_and_cursor_decodes(self):
        from nomorals.documents.indd.object import Encoding
        enc = Encoding.default()
        data = (0x501).to_bytes(4, "little") + (2).to_bytes(4, "little")
        data += bytes([7, 8])
        data += (0x15B).to_bytes(4, "little") + (0).to_bytes(4, "little")
        chunks = enc.chunks(data)
        self.assertIsNotNone(chunks)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].id, 0x501)
        self.assertEqual(chunks[0].data, bytes([7, 8]))
        self.assertIsNone(enc.chunks(data[:-1]))

    def test_cursor_decodes_mixed_segments(self):
        from nomorals.documents.indd.object import Encoding
        enc = Encoding.default()
        data = bytes([0x05, 0x40, ord("W"), ord("O"), ord("M"), ord("E"),
                      ord("N"), 0x01, 0x80, 0x19, 0x20, 0x02, 0x40,
                      ord("S"), ord("\r")])
        self.assertEqual(enc.cursor(data).segments(8), "WOMEN\u2019S\r")

    def test_cursor_decodes_object_string(self):
        from nomorals.documents.indd.object import Encoding
        enc = Encoding.default()
        data = bytes([2, 0, 7, 0, 7, 0x40]) + b"Regular"
        self.assertEqual(enc.cursor(data).string(), "Regular")

    def test_big_endian_cursor(self):
        from nomorals.documents.indd.object import (
            ByteOrder,
            Encoding,
            Version,
        )
        enc = Encoding.new(ByteOrder.BIG,
                           Version(major=4, minor=0))
        data = bytes([2, 0, 0, 0, 5, 0x40, 5]) + b"Black"
        data += (10).to_bytes(4, "big")
        cursor = enc.cursor(data)
        self.assertEqual(cursor.flag(), 0)
        self.assertEqual(cursor.string(), "Black")
        self.assertEqual(cursor.u32(), 10)


class ModelTest(unittest.TestCase):
    def _document(self, path: Path):
        from nomorals.documents.indd.convert import read_document
        return read_document(path.read_bytes())

    def test_blue_square_document(self):
        doc = self._document(_BLUE)
        self.assertEqual((doc.version.major, doc.version.minor), (4, 0))
        self.assertEqual(len(doc.spreads), 1)
        self.assertEqual(len(doc.master_spreads), 1)
        # The 3 PAGE-class objects split 1 + 2: the spread holds its
        # own page, the master spread its two — as upstream's
        # Spread.pages does (its PAGE children, not the document's
        # pages; the writer lists all 3 in the Spread part).
        self.assertEqual(len(doc.spreads[0].pages), 1)
        self.assertEqual(len(doc.master_spreads[0].pages), 2)
        # Zero stories, not one: the DOC_STORIES chunk (0x222) reads a
        # u32 count of 0, and upstream's own converter emits no Story
        # parts for BlueSquare. The one story-class object (uid 124) is
        # the XML backing story, reached via chunk 0xBF14, not the
        # document's story list.
        self.assertEqual(len(doc.stories), 0)
        self.assertEqual(len(doc.layers), 2)
        self.assertEqual(len(doc.colors), 15)
        self.assertEqual(len(doc.fonts), 4)
        self.assertEqual(len(doc.languages), 36)
        self.assertEqual(len(doc.styles), 3)
        self.assertEqual(len(doc.sections), 1)

    def test_neddy_document(self):
        doc = self._document(_NEDDY)
        # 32, not 33: upstream emits 32 Story parts for Neddy — the
        # 33rd story-class object in the database is not listed by the
        # document, so the model leaves it out, as upstream does.
        self.assertEqual(len(doc.stories), 32)
        self.assertEqual(len(doc.spreads), 1)
        # As BlueSquare: the 3 PAGE-class objects split 1 + 2 between
        # the spread and the master spread.
        self.assertEqual(len(doc.spreads[0].pages), 1)
        self.assertEqual(len(doc.master_spreads[0].pages), 2)
        self.assertTrue(any(len(s.runs) > 0 for s in doc.stories))
        self.assertTrue(any(
            run.text.strip() for s in doc.stories for run in s.runs))
        # Upstream has no "graphic" item kind: a graphic is a shape
        # frame whose graphics list is non-empty, so the plain-string
        # kind is derived via kind_name(), not stored in kind.
        # Graphics live inside groups here — Neddy's top-level spread
        # items are four groups, so the walk must recurse.
        def _all_items(items):
            for item in items:
                yield item
                yield from _all_items(item.children)
        images = [item for spread in doc.spreads
                  for item in _all_items(spread.items)
                  if item.kind_name() == "graphic"]
        self.assertTrue(len(images) >= 1)

    def test_minizine_document(self):
        doc = self._document(_MINI)
        # 3, not 4: upstream emits 3 Story parts for minizine — the
        # 4th story-class object is not listed by the document.
        self.assertEqual(len(doc.stories), 3)
        self.assertTrue(any(
            run.text.strip() for s in doc.stories for run in s.runs))

    def test_story_text_is_real_text(self):
        doc = self._document(_NEDDY)
        joined = "".join(run.text for s in doc.stories for run in s.runs)
        self.assertTrue(len(joined) > 100)
        self.assertTrue(any(c.isalpha() for c in joined))

    def test_unreadable_objects_warn_instead_of_failing(self):
        doc = self._document(_BLUE)
        self.assertIsInstance(doc.warnings, list)


class ConvertTest(unittest.TestCase):
    def test_blue_square_converts_to_a_real_idml_package(self):
        conversion = convert(_BLUE.read_bytes(), "BlueSquare.indd")
        parts = _parts(conversion.idml)
        self.assertEqual(parts["mimetype"],
                         b"application/vnd.adobe.indesign-idml-package")
        self.assertIn("designmap.xml", parts)
        self.assertIn("META-INF/container.xml", parts)
        self.assertIn("Resources/Styles.xml", parts)
        self.assertIn("Resources/Fonts.xml", parts)
        self.assertIn("Resources/Preferences.xml", parts)
        self.assertIn("Resources/Graphic.xml", parts)
        self.assertTrue(any(n.startswith("Spreads/") for n in parts))
        self.assertTrue(any(n.startswith("MasterSpreads/")
                            for n in parts))
        # No Stories/ assertion here: BlueSquare's document lists zero
        # stories (see test_blue_square_document), so a real converter
        # emits no Story parts for it. Neddy and minizine cover them.
        for name, data in parts.items():
            if name.endswith(".xml"):
                ET.fromstring(data)

    def test_designmap_names_the_document_and_its_spreads(self):
        conversion = convert(_BLUE.read_bytes(), "BlueSquare.indd")
        root = ET.fromstring(_parts(conversion.idml)["designmap.xml"])
        self.assertEqual(root.tag, "Document")
        self.assertEqual(root.get("DOMVersion"), "4.0")
        # Spread/MasterSpread children carry the idPkg: prefix, so
        # the find needs the namespace.
        ns = "{http://ns.adobe.com/AdobeInDesign/idml/1.0/packaging}"
        self.assertTrue(root.find(f"{ns}Spread") is not None
                        or root.find(f"{ns}MasterSpread") is not None)

    def test_neddy_converts_with_32_stories(self):
        conversion = convert(_NEDDY.read_bytes(), "Neddy.indd")
        parts = _parts(conversion.idml)
        stories = [n for n in parts if n.startswith("Stories/")]
        # 32, not 33: upstream's own converter emits 32 Story parts.
        self.assertEqual(len(stories), 32)

    def test_minizine_converts_with_3_stories(self):
        conversion = convert(_MINI.read_bytes(), "minizine.indd")
        parts = _parts(conversion.idml)
        stories = [n for n in parts if n.startswith("Stories/")]
        # 3, not 4: upstream's own converter emits 3 Story parts.
        self.assertEqual(len(stories), 3)

    def test_story_text_survives_the_round_trip(self):
        from nomorals.documents.indd.convert import read_document
        doc = read_document(_NEDDY.read_bytes())
        source = "".join(run.text for s in doc.stories for run in s.runs)
        conversion = convert(_NEDDY.read_bytes(), "Neddy.indd")
        written = "".join(_story_texts(conversion.idml))
        # The writer's mapping, as upstream does it: a forced line break
        # is LF in the .indd and U+2028 in IDML, and CR is a paragraph
        # break — structural, never inside a <Content> element. The
        # comparison is made after that mapping, not on raw bytes.
        expected = source.replace("\n", "\u2028").replace("\r", "")
        self.assertEqual(written, expected)

    def test_convert_file_reads_from_disk(self):
        conversion = convert_file(_BLUE)
        self.assertTrue(len(conversion.idml) > 1000)

    def test_refusals(self):
        with self.assertRaises(InddError) as ctx:
            convert(b"not an indd file at all, far too short", "x.indd")
        self.assertEqual(ctx.exception.kind, "not_indd")
        with self.assertRaises(InddError) as ctx:
            convert(_BLUE.read_bytes()[:100], "cut.indd")
        self.assertEqual(ctx.exception.kind, "truncated")


class FidelityTest(unittest.TestCase):
    def test_self_diff_is_100_percent(self):
        conversion = convert(_BLUE.read_bytes(), "BlueSquare.indd")
        result = measure_fidelity(conversion.idml, conversion.idml)
        self.assertEqual(result.coverage, 1.0)
        self.assertEqual(result.missing, [])
        self.assertEqual(result.extra, [])

    def test_mutated_reference_names_what_is_missing(self):
        conversion = convert(_BLUE.read_bytes(), "BlueSquare.indd")
        parts = _parts(conversion.idml)
        # No element in the package has Self "u1" (the designmap's
        # Document is Self "d") — mutate the first element that really
        # carries a FillColor instead, which the differ must name.
        mutated_part = None
        for name, data in parts.items():
            if not name.endswith(".xml"):
                continue
            root = ET.fromstring(data)
            for element in root.iter():
                if "FillColor" in element.attrib:
                    element.set("FillColor", "Color/Red")
                    parts[name] = ET.tostring(root, encoding="unicode")
                    mutated_part = name
                    break
            if mutated_part is not None:
                break
        self.assertIsNotNone(mutated_part)
        mutated = io.BytesIO()
        with zipfile.ZipFile(mutated, "w", zipfile.ZIP_STORED) as zf:
            for name, data in parts.items():
                zf.writestr(name, data)
        result = measure_fidelity(conversion.idml, mutated.getvalue())
        self.assertLess(result.coverage, 1.0)
        self.assertTrue(any("FillColor" in m for m in result.missing))

    def test_fidelity_reports_counts_and_headline(self):
        conversion = convert(_BLUE.read_bytes(), "BlueSquare.indd")
        result = measure_fidelity(conversion.idml, conversion.idml)
        self.assertGreater(result.total, 0)
        self.assertEqual(result.reproduced, result.total)
        self.assertEqual(result.headline,
                         f"{result.reproduced}/{result.total} values, "
                         f"{result.coverage:.1%} coverage")

    def test_neddy_self_diff_is_100_percent(self):
        conversion = convert(_NEDDY.read_bytes(), "Neddy.indd")
        result = measure_fidelity(conversion.idml, conversion.idml)
        self.assertEqual(result.coverage, 1.0)


class ParityTest(unittest.TestCase):
    def test_parity_runs_green(self):
        result = run_parity()
        self.assertEqual(result["failed"], [])
        self.assertGreater(result["passed"], 0)
        self.assertEqual(result["skipped"], [])

    def test_attribution_verifies(self):
        result = verify_attribution()
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["skipped"], [])
        self.assertGreater(result["verified"], 0)


class ToolCountTest(unittest.TestCase):
    def test_tool_count_is_786_after_inddutils(self):
        # A new tool is added by this lane: the inddutils tool over
        # documents/indd. 734 at base 414ee6bb; lane 116 (Craft-I18n)
        # ships in parallel and adds no tool; 748 after the bughunter
        # port (lane 121) added 8 bughunter tools. 749 after the mocap port (lane 125) added the mocap tool. 774 after the secprobe wire-up added 25 tools (scope_load, scope_check, param_mine, param_confirm, inject_probe, jwt_attack, graphql_probe, graphql_amplify, race_wave, cors_probe, csrf_poc, plan_next, plan_add, plan_resolve, leaks_harvest, dossier_build, ban_report, ban_state, ban_set_state, ban_appeal, report_capture, report_amplify, replay_run, replay_diagnose, sec_chain). 775 after this lane (B.19) added scene_spawn. 776 after this lane (B.20) added cloudcam. 777 after this lane (B.22) added show. 778 after this lane (B.26) added scene_project. 779 after this lane (B.23) added facemcap. 780 after this lane (B.25) added facemesh. 781 after this lane (B.27) added vtuber. 782 after this lane (C.2) registered tonal_g2p; 783 after this lane (C.1) added smarthome; 784 after this lane (C.4) added camwall; 785 after this lane (C.8) added robot; 786 after this lane (audit N4) added build_swarm; 787 after gapfix74 added museum_library; 788 after this lane (goal-class) added goal_class; 789 after D5 (walabel) added whatsapp_account_label.
        registry = ToolRegistry(SimpleNamespace())
        registry.register_builtins()
        self.assertIn("inddutils", registry._tools)
        self.assertEqual(len(registry._tools), 789)


if __name__ == "__main__":
    unittest.main()
