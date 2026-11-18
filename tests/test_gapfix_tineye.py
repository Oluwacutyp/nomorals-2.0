"""Lane 35 — 52-websites #46 TinEye.

Extends the existing reverse-image organ (tools/imagedb.py) per report §46:
(a) a block-mean/gradient descriptor rung beside dHash, with robustness
MEASURED against a mutation corpus (crop/resize/recolor/watermark) and
pinned as floors — never asserted;
(b) a uniform ReverseMatch{engine, page_url, image_url, score, verified}
shape across engines (scraped results stay unverified unless the
verification rung downloads the candidate and fingerprints it);
(c) index growth by discovery — a landed download is fingerprinted
into the local image index automatically.
"""
import random
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.storage.db import Database
from nomorals.tools import imagedb
from nomorals.tools.imagedb import (
    BLOCK_MATCH_THRESHOLD,
    ReverseMatch,
    block_descriptor,
    block_distance,
    block_match,
    fingerprint,
    fingerprint_benchmark,
    fingerprint_distance,
    fingerprints_match,
    note_image_file,
    verify_reverse_matches,
)

try:
    from PIL import Image, ImageDraw, ImageEnhance

    PIL_OK = True
except ImportError:
    PIL_OK = False

_IMAGE_INDEX_DDL = (
    "CREATE TABLE IF NOT EXISTS image_index ("
    "hash TEXT PRIMARY KEY, path TEXT NOT NULL DEFAULT '', "
    "size INTEGER NOT NULL DEFAULT 0, mime TEXT NOT NULL DEFAULT '', "
    "first_seen REAL NOT NULL DEFAULT 0, last_seen REAL NOT NULL DEFAULT 0, "
    "seen_in TEXT NOT NULL DEFAULT '[]')"
)


def _make_db():
    db = Database(":memory:")
    db.execute(_IMAGE_INDEX_DDL)
    return db


def _make_context(db, home="~/.nomorals"):
    return SimpleNamespace(db=db, settings=SimpleNamespace(home=home, tools=None))


def _base_grad(path):
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = ((x * 3) % 256, (y * 2) % 256, ((x + y) % 256))
    img.save(path, "PNG")
    return path


def _base_stripes(path):
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            on = (x // 16) % 2 == 0
            px[x, y] = (220, 30, 30) if on else (20, 20, 220)
    img.save(path, "PNG")
    return path


def _base_rings(path):
    img = Image.new("RGB", (256, 256), (10, 10, 40))
    dr = ImageDraw.Draw(img)
    for r in range(120, 0, -20):
        dr.ellipse([128 - r, 128 - r, 128 + r, 128 + r],
                   fill=((r * 2) % 256, 120, 200 - r))
    img.save(path, "PNG")
    return path


def _base_noise(path):
    rnd = random.Random(42)
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            px[x, y] = (rnd.randrange(256), rnd.randrange(256), rnd.randrange(256))
    img = img.resize((32, 32), Image.LANCZOS).resize((256, 256), Image.NEAREST)
    img.save(path, "PNG")
    return path


def _base_scene(path):
    img = Image.new("RGB", (256, 256), (90, 160, 220))
    dr = ImageDraw.Draw(img)
    dr.rectangle([0, 170, 256, 256], fill=(40, 120, 50))
    dr.ellipse([180, 30, 230, 80], fill=(250, 220, 80))
    dr.rectangle([20, 120, 90, 170], fill=(120, 70, 40))
    img.save(path, "PNG")
    return path


def _base_check(path):
    img = Image.new("RGB", (256, 256))
    px = img.load()
    for y in range(256):
        for x in range(256):
            on = ((x // 32) + (y // 32)) % 2 == 0
            px[x, y] = (240, 240, 240) if on else (15, 15, 15)
    img.save(path, "PNG")
    return path


BASE_BUILDERS = {
    "grad": _base_grad,
    "stripes": _base_stripes,
    "rings": _base_rings,
    "noise": _base_noise,
    "scene": _base_scene,
    "check": _base_check,
}


def _mutate(img, kind, out_path):
    if kind == "resize50":
        img.resize((128, 128), Image.LANCZOS).save(out_path)
    elif kind == "resize200":
        img.resize((512, 512), Image.LANCZOS).save(out_path)
    elif kind == "crop80":
        w, h = img.size
        img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))).save(out_path)
    elif kind == "crop60":
        w, h = img.size
        img.crop((int(w * 0.2), int(h * 0.2), int(w * 0.8), int(h * 0.8))).save(out_path)
    elif kind == "hue":
        r, g, b = img.split()
        Image.merge("RGB", (b, r, g)).save(out_path)
    elif kind == "grayscale":
        img.convert("L").convert("RGB").save(out_path)
    elif kind == "bright+30":
        ImageEnhance.Brightness(img).enhance(1.3).save(out_path)
    elif kind == "bright-30":
        ImageEnhance.Brightness(img).enhance(0.7).save(out_path)
    elif kind == "contrast":
        ImageEnhance.Contrast(img).enhance(1.6).save(out_path)
    elif kind == "watermark_bar":
        m = img.copy()
        dr = ImageDraw.Draw(m, "RGBA")
        dr.rectangle([0, 220, 256, 256], fill=(255, 255, 255, 140))
        m.convert("RGB").save(out_path)
    elif kind == "watermark_corner":
        m = img.copy()
        dr = ImageDraw.Draw(m, "RGBA")
        dr.rectangle([180, 180, 256, 256], fill=(255, 255, 255, 160))
        m.convert("RGB").save(out_path)
    elif kind == "jpeg60":
        img.convert("RGB").save(out_path, "JPEG", quality=60)
    else:
        raise ValueError(kind)


MUTATIONS = ["resize50", "resize200", "crop80", "crop60", "hue", "grayscale",
             "bright+30", "bright-30", "contrast", "watermark_bar",
             "watermark_corner", "jpeg60"]


class DescriptorTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_block_descriptor_stable_and_versioned(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_desc_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _base_grad(root / "a.png")
        desc = block_descriptor(path)
        self.assertIsNotNone(desc)
        self.assertTrue(desc.startswith("bm1:"),
                        f"descriptor carries its version: {desc[:8]}")
        parts = desc.split(":")
        self.assertEqual(len(parts), 4)
        self.assertEqual(len(parts[1]), 64)  # 64 block-mean nibbles
        self.assertEqual(len(parts[2]), 16)  # horizontal gradient bits
        self.assertEqual(len(parts[3]), 16)  # vertical gradient bits
        self.assertEqual(block_descriptor(path), desc)
        self.assertEqual(block_distance(desc, desc), 0.0)

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_block_distance_is_a_real_distance(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_desc_"))
        self.addCleanup(shutil.rmtree, root, True)
        a = block_descriptor(_base_grad(root / "a.png"))
        b = block_descriptor(_base_stripes(root / "b.png"))
        dist = block_distance(a, b)
        self.assertIsNotNone(dist)
        self.assertGreater(dist, BLOCK_MATCH_THRESHOLD)
        self.assertLessEqual(dist, 1.0)
        self.assertEqual(block_distance(a, b), block_distance(b, a))

    def test_block_descriptor_missing_file_is_none(self):
        self.assertIsNone(block_descriptor("/nonexistent/nope.jpg"))
        self.assertIsNone(block_distance(None, "bm1:" + "0" * 64 + ":" + "0" * 16 + ":" + "0" * 16))
        self.assertIsNone(block_distance("garbage", "alsogarbage"))

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_block_descriptor_graceful_without_pillow(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_desc_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _base_grad(root / "a.png")
        with mock.patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            self.assertIsNone(block_descriptor(path))
            self.assertIsNone(fingerprint(path))

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_fingerprint_carries_both_rungs(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_fp_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _base_grad(root / "a.png")
        fp = fingerprint(path)
        self.assertIsNotNone(fp)
        self.assertEqual(set(fp), {"dhash", "block"})
        self.assertEqual(fp["dhash"], imagedb.dhash(path))
        self.assertEqual(fp["block"], block_descriptor(path))
        dists = fingerprint_distance(fp, fp)
        self.assertEqual(dists, {"dhash": 0, "block": 0.0})
        verdicts = fingerprints_match(fp, fp)
        self.assertEqual(verdicts, {"dhash": True, "block": True,
                                     "either": True, "both": True})

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_fingerprints_match_names_the_rung(self):
        # grad vs its 80% center crop: dHash (threshold 10) misses it,
        # the block rung catches it — that is the rung's whole job.
        root = Path(tempfile.mkdtemp(prefix="tineye_fp_"))
        self.addCleanup(shutil.rmtree, root, True)
        orig = _base_grad(root / "orig.png")
        with Image.open(orig) as im:
            _mutate(im, "crop80", root / "crop.png")
        a, b = fingerprint(orig), fingerprint(root / "crop.png")
        verdicts = fingerprints_match(a, b)
        self.assertFalse(verdicts["dhash"])
        self.assertTrue(verdicts["block"])
        self.assertTrue(verdicts["either"])
        self.assertFalse(verdicts["both"])
        self.assertTrue(block_match(a["block"], b["block"]))


class BenchmarkTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_benchmark_floors_are_measured_not_asserted(self):
        """Per-rung match rates on the mutation corpus, pinned as
        floors from the measured run (6 bases x 12 mutations):

        block rung: 69/72 overall, 5/6 on crop80, 4/6 on crop60,
        6/6 on every other mutation. dHash: 61/72, 2/6 on crop80,
        0/6 on crop60 — the crop fragility §46 names. Distinct
        pairs: block distance min 0.349 (> 0.25 threshold), dHash
        hamming min 24 (> 10)."""
        report = fingerprint_benchmark()
        self.assertEqual(report["corpus"]["bases"], 6)
        self.assertEqual(report["corpus"]["mutations"], len(MUTATIONS))
        self.assertEqual(report["thresholds"]["dhash"], 10)
        self.assertEqual(report["thresholds"]["block"], BLOCK_MATCH_THRESHOLD)
        block = report["rungs"]["block"]
        dhash_rung = report["rungs"]["dhash"]
        self.assertGreaterEqual(block["overall"], 0.90)
        self.assertGreaterEqual(block["crop80"], 0.66)
        self.assertGreaterEqual(block["crop60"], 0.50)
        for kind in MUTATIONS:
            if kind not in ("crop80", "crop60"):
                self.assertGreaterEqual(block[kind], 0.90, kind)
        # the block rung strictly beats dHash exactly where §46 says
        # dHash is fragile: crops.
        self.assertGreater(block["crop80"], dhash_rung["crop80"])
        self.assertGreater(block["crop60"], dhash_rung["crop60"])
        # distinct images stay separate on both rungs.
        sep = report["separation"]
        self.assertGreater(sep["block_min"], BLOCK_MATCH_THRESHOLD)
        self.assertGreater(sep["dhash_min"], 10)

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_benchmark_is_deterministic(self):
        first = fingerprint_benchmark()
        second = fingerprint_benchmark()
        self.assertEqual(first["rungs"], second["rungs"])
        self.assertEqual(first["separation"], second["separation"])


class ReverseMatchTests(unittest.TestCase):
    BING_HTML = (
        b'<html><body><div class="iusc" m="{x}">'
        b'<a mediaurl="https://example.com/first.jpg" href="#">x</a>'
        b'<a mediaurl="https://example.org/second.png" href="#">y</a>'
        b"</div></body></html>"
    )
    YANDEX_HTML = (
        b'<html><body><script>'
        b'{"originalImage":{"url":"https://avatars.yandex.net/img1.jpg"},"x":1}'
        b"</script>"
        b'<img class="serp-item__thumb" src="https://example.org/hit.png" alt="">'
        b"</body></html>"
    )

    class _FakeClient:
        def __init__(self, bodies):
            self.bodies = bodies

        def get(self, url):
            for key, body in self.bodies.items():
                if key in url:
                    return SimpleNamespace(body=body)
            raise RuntimeError(f"no canned body for {url}")

    def test_reverse_match_shape_is_uniform(self):
        m = ReverseMatch(engine="bing", page_url="https://www.bing.com/x",
                         image_url="https://example.com/a.jpg", score=1.0,
                         verified=False)
        d = m.to_dict()
        self.assertEqual(d, {"engine": "bing",
                             "page_url": "https://www.bing.com/x",
                             "image_url": "https://example.com/a.jpg",
                             "score": 1.0, "verified": False})

    def test_scrape_matches_uniform_shape_across_engines(self):
        client = self._FakeClient({"bing.com": self.BING_HTML,
                                   "yandex.com": self.YANDEX_HTML})
        out = imagedb._scrape_matches(client, "https://img.test/x.jpg")
        self.assertIn("bing", out)
        self.assertIn("yandex", out)
        for engine, matches in out.items():
            for match in matches:
                self.assertIsInstance(match, ReverseMatch)
                self.assertEqual(match.engine, engine)
                self.assertTrue(match.page_url.startswith("https://"))
                self.assertTrue(match.image_url.startswith("https://"))
                self.assertFalse(match.verified,
                                 "scraped results stay unverified")
                self.assertGreater(match.score, 0.0)
                self.assertLessEqual(match.score, 1.0)
        bing = out["bing"]
        self.assertEqual([m.image_url for m in bing],
                         ["https://example.com/first.jpg",
                          "https://example.org/second.png"])
        # rank score: first result scores highest, strictly decreasing.
        scores = [m.score for m in bing]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(scores[0], 1.0)
        self.assertIn("bing.com", bing[0].page_url)

    def test_scrape_matches_engine_failure_isolated(self):
        client = self._FakeClient({"yandex.com": self.YANDEX_HTML})
        out = imagedb._scrape_matches(client, "https://img.test/x.jpg")
        self.assertEqual(out["bing"], [])
        self.assertTrue(out["yandex"])

    def test_reverse_impl_adds_matches_without_changing_old_keys(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_rev_"))
        self.addCleanup(shutil.rmtree, root, True)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

        class URLClient:
            def get(self, url):
                if "bing.com" in url:
                    return SimpleNamespace(body=self.BING_HTML)
                if "yandex.com" in url:
                    return SimpleNamespace(body=self.YANDEX_HTML)
                return SimpleNamespace(body=png)

        URLClient.BING_HTML = self.BING_HTML
        URLClient.YANDEX_HTML = self.YANDEX_HTML
        ctx = SimpleNamespace(db=_make_db(),
                              settings=SimpleNamespace(home=str(root), tools=None))
        with mock.patch("nomorals.tools.imagedb.HttpClient",
                        return_value=URLClient()):
            res = imagedb._reverse_impl(ctx, "https://img.test/photo.jpg")
        self.assertTrue(res["ok"])
        # old payload byte-identical in shape.
        self.assertTrue(res["unverified_by_engine"]["bing"])
        self.assertTrue(res["unverified_by_engine"]["yandex"])
        self.assertTrue(res["unverified_matches"])
        # new uniform shape beside it.
        self.assertIn("matches", res)
        self.assertIn("matches_by_engine", res)
        flat = res["matches"]
        self.assertTrue(all(isinstance(m, dict) for m in flat))
        self.assertEqual({m["engine"] for m in flat}, {"bing", "yandex"})
        self.assertTrue(all(m["verified"] is False for m in flat))
        self.assertEqual(
            sorted(m["image_url"] for m in res["matches_by_engine"]["bing"]),
            sorted(res["unverified_by_engine"]["bing"]))
        self.assertEqual(sorted(m["image_url"] for m in flat),
                         sorted(res["unverified_matches"]))

    def test_reverse_impl_local_path_has_no_matches_key(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_rev_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "local.png"
        if PIL_OK:
            _base_grad(path)
        else:
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        res = imagedb._reverse_impl(_make_context(_make_db()), str(path))
        self.assertTrue(res["ok"])
        self.assertNotIn("matches", res)
        self.assertNotIn("unverified_matches", res)


class VerifyTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_verify_marks_a_real_copy_verified(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_verify_"))
        self.addCleanup(shutil.rmtree, root, True)
        query = _base_grad(root / "query.png")
        # the candidate URL serves a cropped copy of the query image.
        with Image.open(query) as im:
            _mutate(im, "crop80", root / "copy.png")
        copy_bytes = (root / "copy.png").read_bytes()
        other = _base_stripes(root / "other.png")
        other_bytes = other.read_bytes()

        class CopyClient:
            def get(self, url):
                if "copy" in url:
                    return SimpleNamespace(body=copy_bytes)
                if "other" in url:
                    return SimpleNamespace(body=other_bytes)
                raise RuntimeError(f"no canned body for {url}")

        matches = [
            ReverseMatch(engine="bing", page_url="https://www.bing.com/x",
                         image_url="https://cdn.test/copy.jpg", score=1.0),
            ReverseMatch(engine="bing", page_url="https://www.bing.com/x",
                         image_url="https://cdn.test/other.jpg", score=0.9),
            ReverseMatch(engine="yandex", page_url="https://yandex.com/x",
                         image_url="https://cdn.test/gone.jpg", score=1.0),
        ]
        ctx = _make_context(_make_db(), home=str(root))
        out = verify_reverse_matches(ctx, query, matches, client=CopyClient())
        self.assertTrue(out[0].verified,
                        "a cropped copy of the query verifies")
        self.assertFalse(out[1].verified,
                         "a different image never verifies")
        self.assertFalse(out[2].verified,
                         "an unreachable candidate never verifies")
        # the originals are untouched (frozen dataclass, new objects).
        self.assertFalse(matches[0].verified)

    def test_verify_never_raises(self):
        matches = [ReverseMatch(engine="bing", page_url="https://x",
                                image_url="https://cdn.test/a.jpg", score=1.0)]
        ctx = _make_context(_make_db())
        out = verify_reverse_matches(ctx, "/nonexistent/nope.jpg", matches)
        self.assertEqual(out, matches)


class IndexGrowthTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_note_image_file_indexes_with_fingerprints(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_note_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _base_grad(root / "a.png")
        ctx = _make_context(_make_db())
        first = note_image_file(ctx, path)
        self.assertTrue(first["ok"])
        self.assertFalse(first["seen_before"])
        self.assertEqual(first["dhash"], imagedb.dhash(path))
        self.assertEqual(first["block"], block_descriptor(path))
        second = note_image_file(ctx, path)
        self.assertTrue(second["ok"])
        self.assertTrue(second["seen_before"])
        rows = ctx.db.query("SELECT hash, path, seen_in FROM image_index")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seen_in"], "download")
        self.assertEqual(rows[0]["path"], str(path))

    def test_note_image_file_never_raises(self):
        self.assertFalse(note_image_file(_make_context(_make_db()),
                                         "/nonexistent/nope.jpg")["ok"])
        self.assertFalse(note_image_file(SimpleNamespace(db=None),
                                         "/whatever/x.jpg")["ok"])
        self.assertFalse(note_image_file(None, "/whatever/x.jpg")["ok"])

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_find_dupes_clusters_crop_copy_via_block_rung(self):
        # grad vs its 80% crop: dHash distance 16 (> 6 threshold) so
        # the old rung alone separates them; the block rung (0.081)
        # is what clusters them — the new rung's whole job.
        root = Path(tempfile.mkdtemp(prefix="tineye_dupes_"))
        self.addCleanup(shutil.rmtree, root, True)
        orig = _base_grad(root / "orig.png")
        with Image.open(orig) as im:
            _mutate(im, "crop80", root / "crop.png")
        other = _base_stripes(root / "other.png")
        ctx = _make_context(_make_db())
        for p in (orig, root / "crop.png", other):
            note_image_file(ctx, p)
        res = imagedb.find_dupes(ctx, threshold=6)
        self.assertTrue(res["ok"])
        grouped = [set(g["paths"]) for g in res["groups"]]
        self.assertIn({str(orig), str(root / "crop.png")}, grouped)
        self.assertNotIn({str(orig), str(other)}, grouped)
        self.assertNotIn({str(root / "crop.png"), str(other)}, grouped)

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_lookup_near_duplicates_names_the_rung(self):
        root = Path(tempfile.mkdtemp(prefix="tineye_lookup_"))
        self.addCleanup(shutil.rmtree, root, True)
        orig = _base_grad(root / "orig.png")
        with Image.open(orig) as im:
            _mutate(im, "crop80", root / "crop.png")
        ctx = _make_context(_make_db(), home=str(root))
        imagedb._lookup_impl(ctx, str(orig), "test")
        res = imagedb._lookup_impl(ctx, str(root / "crop.png"), "test")
        self.assertTrue(res["ok"])
        block_hits = [n for n in res["near_duplicates"]
                      if n.startswith(str(root / "orig.png"))
                      and "(block " in n]
        self.assertTrue(block_hits, res["near_duplicates"])


class RegistryTests(unittest.TestCase):
    def test_image_status_reports_fingerprint_ladder(self):
        from nomorals.media.source_registry import SourceRegistry

        status = SourceRegistry().image_status()
        self.assertIn("fingerprint", status)
        fp = status["fingerprint"]
        self.assertEqual(fp["rungs"], ["dhash", "block"])
        self.assertEqual(fp["thresholds"]["dhash"], 10)
        self.assertEqual(fp["thresholds"]["block"], BLOCK_MATCH_THRESHOLD)
        # the measured robustness table is IN the status, not asserted
        # in a comment somewhere.
        self.assertIn("measured", fp)
        self.assertGreaterEqual(fp["measured"]["block"]["overall"], 0.90)
        self.assertIn("separation", fp["measured"])

    def test_image_status_still_lists_engines(self):
        from nomorals.media.source_registry import SourceRegistry

        status = SourceRegistry().image_status()
        self.assertEqual([e["engine"] for e in status["engines"]],
                         ["bing", "yandex"])
        self.assertEqual({e["engine"] for e in status["link_only"]},
                         {"google_lens", "tineye"})


class DownloaderHookTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_landed_image_is_fingerprinted_into_the_index(self):
        import time
        from unittest import mock as _mock

        from nomorals.media import downloader as dl

        root = Path(tempfile.mkdtemp(prefix="tineye_hook_"))
        self.addCleanup(shutil.rmtree, root, True)
        img = _base_grad(root / "landed.png")
        db = _make_db()
        ctx = SimpleNamespace(db=db, settings=SimpleNamespace(
            home=str(root), tools=None))
        md = dl.MediaDownloader(
            context=ctx,
            source_registry=_mock.Mock(for_url=_mock.Mock(return_value=None)),
            ledger=SimpleNamespace(append=_mock.Mock()),
        )
        report = dl.DownloadReport(
            url="https://example.com/landed.png", ok=True, path=str(img),
            size_bytes=img.stat().st_size,
            stages=[dl.StageLog(stage="direct", url="https://example.com/landed.png",
                                ok=True)])
        md._succeed(report, time.perf_counter())
        rows = db.query("SELECT hash, path, seen_in FROM image_index")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["path"], str(img))
        self.assertEqual(rows[0]["seen_in"], "download")

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_landed_non_image_is_not_indexed(self):
        import time
        from unittest import mock as _mock

        from nomorals.media import downloader as dl

        root = Path(tempfile.mkdtemp(prefix="tineye_hook_"))
        self.addCleanup(shutil.rmtree, root, True)
        blob = root / "landed.bin"
        blob.write_bytes(b"not an image at all" * 100)
        db = _make_db()
        ctx = SimpleNamespace(db=db, settings=SimpleNamespace(
            home=str(root), tools=None))
        md = dl.MediaDownloader(
            context=ctx,
            source_registry=_mock.Mock(for_url=_mock.Mock(return_value=None)),
            ledger=SimpleNamespace(append=_mock.Mock()),
        )
        report = dl.DownloadReport(
            url="https://example.com/landed.bin", ok=True, path=str(blob),
            size_bytes=blob.stat().st_size,
            stages=[dl.StageLog(stage="direct", url="https://example.com/landed.bin",
                                ok=True)])
        md._succeed(report, time.perf_counter())
        rows = db.query("SELECT hash FROM image_index")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
