"""God-tier upgrade tests for nomorals/tools/imagedb.py.

Covers:
- EXIF extraction on a generated JPEG with EXIF (skipped when Pillow missing)
- dominant color palette on a generated solid-color PNG
- dhash/hamming sanity
- find_dupes clusters near-identical images, separates different ones
- image_stats counts
- reverse-search engine parsing against canned HTML (no live network),
  incl. per-engine isolation when one engine fails
- graceful degradation when Pillow is unavailable (mocked import failure)
- tool registration of the new tools
"""

import hashlib
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
    dhash,
    find_dupes,
    hamming,
    image_colors,
    image_exif,
    image_stats,
    sniff_format,
)

try:
    from PIL import Image
    from PIL.TiffImagePlugin import IFDRational

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


def _make_context(db):
    return SimpleNamespace(db=db, settings=SimpleNamespace(home="~/.nomorals"))


def _solid(path, color=(255, 0, 0), size=(64, 64), fmt="PNG"):
    img = Image.new("RGB", size, color)
    img.save(path, fmt)
    return path


def _gradient(path, seed_variant=0):
    """Structurally textured 64x64 image — dHash actually discriminates these."""
    img = Image.new("RGB", (64, 64))
    px = img.load()
    for y in range(64):
        for x in range(64):
            px[x, y] = ((x * 4 + seed_variant) % 256,
                        (y * 4 + seed_variant * 7) % 256,
                        ((x + y) * 2 + seed_variant * 13) % 256)
    img.save(path, "PNG")
    return path


def _stripes(path, vertical=True):
    img = Image.new("RGB", (64, 64))
    px = img.load()
    for y in range(64):
        for x in range(64):
            on = (x // 8) % 2 == 0 if vertical else (y // 8) % 2 == 0
            px[x, y] = (220, 30, 30) if on else (20, 20, 220)
    img.save(path, "PNG")
    return path


def _tweaked_copy(src, dst):
    """Near-duplicate: same image with a small region nudged — flips a few
    dHash bits (non-zero but small distance) instead of none at all."""
    img = Image.open(src).copy()
    px = img.load()
    base = px[0, 0]
    for y in range(12):
        for x in range(12):
            px[x, y] = tuple(min(255, v + 40) for v in base)
    img.save(dst, "PNG")
    return dst


def _index(ctx, path):
    data = Path(path).read_bytes()
    digest = hashlib.blake2b(data, digest_size=16).hexdigest()
    imagedb._index_row(ctx, digest, Path(path), len(data), sniff_format(data), "test")
    return digest


class ExifTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_exif_on_generated_jpeg(self):
        root = Path(tempfile.mkdtemp(prefix="imgexif_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "cam.jpg"
        exif = Image.Exif()
        exif[0x010F] = "TestMake"  # Make
        exif[0x0110] = "TestModel 5000"  # Model
        exif[0x0132] = "2026:10:01 10:30:00"  # DateTime
        exif[0x8825] = {  # GPSInfo
            0: b"\x02\x03\x00\x00",
            1: "N",
            2: (IFDRational(40, 1), IFDRational(26, 1), IFDRational(30, 1)),
            3: "W",
            4: (IFDRational(74, 1), IFDRational(0, 1), IFDRational(0, 1)),
        }
        Image.new("RGB", (64, 64), (10, 20, 30)).save(path, "JPEG", exif=exif)

        meta = image_exif(path)
        self.assertEqual(meta.get("make"), "TestMake")
        self.assertEqual(meta.get("model"), "TestModel 5000")
        self.assertEqual(meta.get("datetime"), "2026:10:01 10:30:00")
        gps = meta.get("gps")
        self.assertIsNotNone(gps)
        self.assertAlmostEqual(gps["lat"], 40 + 26 / 60 + 30 / 3600, places=4)
        self.assertAlmostEqual(gps["lon"], -74.0, places=4)

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_exif_empty_without_exif(self):
        root = Path(tempfile.mkdtemp(prefix="imgexif_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _solid(root / "plain.png")
        self.assertEqual(image_exif(path), {})

    def test_exif_missing_file_is_empty(self):
        self.assertEqual(image_exif("/nonexistent/nope.jpg"), {})


class ColorTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_solid_color_palette(self):
        root = Path(tempfile.mkdtemp(prefix="imgcol_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _solid(root / "red.png", color=(255, 0, 0))
        colors = image_colors(path, n=3)
        self.assertEqual(colors, ["#ff0000"])

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_two_tone_palette(self):
        root = Path(tempfile.mkdtemp(prefix="imgcol_"))
        self.addCleanup(shutil.rmtree, root, True)
        img = Image.new("RGB", (64, 64), (0, 0, 255))
        # paint the top half red -> red should dominate
        for y in range(32):
            for x in range(64):
                img.putpixel((x, y), (255, 0, 0))
        path = root / "halves.png"
        img.save(path, "PNG")
        colors = image_colors(path, n=4)
        self.assertIn("#ff0000", colors)
        self.assertIn("#0000ff", colors)
        self.assertLessEqual(len(colors), 4)

    def test_colors_missing_file_is_empty(self):
        self.assertEqual(image_colors("/nonexistent/nope.png"), [])


class HashTests(unittest.TestCase):
    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_dhash_stable_and_64bit(self):
        root = Path(tempfile.mkdtemp(prefix="imghash_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _solid(root / "a.png")
        h1 = dhash(path)
        h2 = dhash(path)
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)
        int(h1, 16)  # valid hex

    def test_hamming_sanity(self):
        self.assertEqual(hamming("ff", "ff"), 0)
        self.assertEqual(hamming("00", "ff"), 8)
        self.assertEqual(hamming("f0", "0f"), 8)
        self.assertIsNone(hamming("abc", "ab"))  # length mismatch
        self.assertIsNone(hamming("", "ff"))
        self.assertIsNone(hamming("zz", "ff"))  # not hex


@unittest.skipUnless(PIL_OK, "Pillow not installed")
class DupeTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="imgdupe_"))
        self.addCleanup(shutil.rmtree, self.root, True)
        self.db = _make_db()
        self.ctx = _make_context(self.db)

    def test_find_dupes_clusters_near_identical(self):
        a = _gradient(self.root / "a.png")
        b = _tweaked_copy(a, self.root / "b.png")
        c = _stripes(self.root / "c.png")  # structurally different

        ha, hb, hc = dhash(a), dhash(b), dhash(c)
        self.assertIsNotNone(ha)
        self.assertIsNotNone(hb)
        self.assertIsNotNone(hc)
        dist_ab = hamming(ha, hb)
        dist_ac = hamming(ha, hc)
        self.assertGreater(dist_ab, 0, f"fixture too identical: distance {dist_ab}")
        self.assertLessEqual(dist_ab, 6, f"fixture drifted: distance {dist_ab}")
        self.assertGreater(dist_ac, 6, f"fixture drifted: distance {dist_ac}")

        _index(self.ctx, a)
        _index(self.ctx, b)
        _index(self.ctx, c)

        result = find_dupes(self.ctx, threshold=6)
        self.assertTrue(result["ok"])
        self.assertEqual(result["scanned"], 3)
        groups = result["groups"]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["count"], 2)
        self.assertIn(str(a), groups[0]["paths"])
        self.assertIn(str(b), groups[0]["paths"])
        self.assertNotIn(str(c), groups[0]["paths"])

    def test_find_dupes_separates_distinct_images(self):
        _index(self.ctx, _gradient(self.root / "g1.png"))
        _index(self.ctx, _gradient(self.root / "g2.png", seed_variant=100))
        _index(self.ctx, _stripes(self.root / "s.png"))
        result = find_dupes(self.ctx, threshold=6)
        self.assertEqual(result["groups"], [])

    def test_find_dupes_empty_library(self):
        result = find_dupes(self.ctx)
        self.assertTrue(result["ok"])
        self.assertEqual(result["scanned"], 0)
        self.assertEqual(result["groups"], [])

    def test_find_dupes_ignores_missing_files(self):
        gone = self.root / "gone.png"
        _solid(gone)
        _index(self.ctx, gone)
        gone.unlink()
        result = find_dupes(self.ctx)
        self.assertTrue(result["ok"])
        self.assertEqual(result["scanned"], 0)

    def test_image_stats_counts(self):
        a = _solid(self.root / "a.png")
        b = _solid(self.root / "b.png", color=(0, 0, 255))
        jpg = self.root / "c.jpg"
        Image.new("RGB", (32, 32), (9, 9, 9)).save(jpg, "JPEG")
        _index(self.ctx, a)
        _index(self.ctx, b)
        _index(self.ctx, jpg)

        stats = image_stats(self.ctx)
        self.assertTrue(stats["ok"])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["formats"].get("png"), 2)
        self.assertEqual(stats["formats"].get("jpeg"), 1)
        self.assertGreater(stats["total_bytes"], 0)
        self.assertIn("dupe_groups", stats)

    def test_lookup_near_duplicates_still_works(self):
        a = _gradient(self.root / "a.png")
        b = _tweaked_copy(a, self.root / "b.png")
        _index(self.ctx, a)
        res = imagedb._lookup_impl(self.ctx, str(b), "test")
        self.assertTrue(res["ok"])
        self.assertTrue(any(str(a) in n for n in res["near_duplicates"]),
                        f"expected {a} in {res['near_duplicates']}")


class ReverseSearchTests(unittest.TestCase):
    BING_HTML = (
        b'<html><body><div class="iusc" m="{x}">'
        b'<a mediaurl="https://example.com/first.jpg" href="#">x</a>'
        b'<a mediaurl="https://example.org/second.png" href="#">y</a>'
        b'<a mediaurl="https://example.com/first.jpg" href="#">dup</a>'
        b"</div></body></html>"
    )
    YANDEX_HTML = (
        b'<html><body><script>'
        b'{"originalImage":{"url":"https://avatars.yandex.net/img1.jpg"},"x":1}'
        b"</script>"
        b'<img class="serp-item__thumb" src="https://example.org/hit.png" alt="">'
        b'<img src="https://example.com/thumb.gif?x=1" alt="">'
        b"</body></html>"
    )

    class _FakeClient:
        def __init__(self, bodies):
            self.bodies = bodies  # url-substring -> bytes

        def get(self, url):
            for key, body in self.bodies.items():
                if key in url:
                    return SimpleNamespace(body=body)
            raise RuntimeError(f"no canned body for {url}")

    def test_bing_parse(self):
        client = self._FakeClient({"bing.com": self.BING_HTML})
        urls = imagedb._scrape_bing(client, "https://img.test/x.jpg")
        self.assertEqual(urls, ["https://example.com/first.jpg",
                                "https://example.org/second.png"])

    def test_yandex_parse(self):
        client = self._FakeClient({"yandex.com": self.YANDEX_HTML})
        urls = imagedb._scrape_yandex(client, "https://img.test/x.jpg")
        self.assertIn("https://avatars.yandex.net/img1.jpg", urls)
        self.assertIn("https://example.org/hit.png", urls)
        self.assertIn("https://example.com/thumb.gif?x=1", urls)

    def test_engine_isolation_one_fails(self):
        # Bing explodes; Yandex must still deliver.
        client = self._FakeClient({"yandex.com": self.YANDEX_HTML})
        out = imagedb._scrape_engines(client, "https://img.test/x.jpg")
        self.assertEqual(out["bing"], [])
        self.assertTrue(out["yandex"])
        self.assertIn("bing", out)
        self.assertIn("yandex", out)

    def test_bing_scrape_failure_returns_empty(self):
        class Boom:
            def get(self, url):
                raise ConnectionError("blocked")

        self.assertEqual(imagedb._scrape_bing(Boom(), "https://x"), [])
        self.assertEqual(imagedb._scrape_yandex(Boom(), "https://x"), [])

    def test_reverse_impl_local_path_has_no_public_scrape(self):
        root = Path(tempfile.mkdtemp(prefix="imgrev_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = root / "local.png"
        if PIL_OK:
            _solid(path)
        else:
            path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
        res = imagedb._reverse_impl(_make_context(_make_db()), str(path))
        self.assertTrue(res["ok"])
        self.assertIn("note", res)
        self.assertNotIn("unverified_matches", res)

    def test_reverse_impl_url_runs_both_engines(self):
        root = Path(tempfile.mkdtemp(prefix="imgrev_"))
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
        ctx = SimpleNamespace(
            db=_make_db(),
            settings=SimpleNamespace(home=str(root), tools=None),
        )
        with mock.patch("nomorals.tools.imagedb.HttpClient",
                        return_value=URLClient()):
            res = imagedb._reverse_impl(ctx, "https://img.test/photo.jpg")
        self.assertTrue(res["ok"])
        by_engine = res["unverified_by_engine"]
        self.assertTrue(by_engine["bing"])
        self.assertTrue(by_engine["yandex"])
        # backward-compat flat list still present
        self.assertTrue(res["unverified_matches"])
        self.assertIn("google_lens", res["lookup_links"])


class PillowAbsentTests(unittest.TestCase):
    def _block_pil(self):
        return mock.patch.dict(sys.modules,
                               {"PIL": None, "PIL.Image": None,
                                "PIL.ExifTags": None})

    def test_exif_graceful_without_pillow(self):
        with self._block_pil():
            self.assertEqual(image_exif("/whatever/x.jpg"), {})

    def test_colors_graceful_without_pillow(self):
        with self._block_pil():
            self.assertEqual(image_colors("/whatever/x.png"), [])

    def test_dhash_graceful_without_pillow(self):
        with self._block_pil():
            self.assertIsNone(dhash("/whatever/x.png"))

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_find_dupes_graceful_without_pillow(self):
        # hashes can't be computed -> nothing scannable, but no crash
        root = Path(tempfile.mkdtemp(prefix="imgnopil_"))
        self.addCleanup(shutil.rmtree, root, True)
        path = _solid(root / "a.png")
        db = _make_db()
        ctx = _make_context(db)
        _index(ctx, path)
        with self._block_pil():
            result = find_dupes(ctx)
        self.assertTrue(result["ok"])
        self.assertEqual(result["scanned"], 0)
        self.assertEqual(result["groups"], [])


class RegisterTests(unittest.TestCase):
    def test_new_tools_registered(self):
        from nomorals.tools.registry import ToolRegistry

        db = _make_db()
        reg = ToolRegistry(context=_make_context(db))
        imagedb.register(reg)
        names = set(reg._tools)
        for expected in ("image_lookup", "reverse_image_search", "image_exif",
                         "image_colors", "find_dupes", "image_stats"):
            self.assertIn(expected, names, f"tool {expected} not registered")

    @unittest.skipUnless(PIL_OK, "Pillow not installed")
    def test_registered_exif_tool_end_to_end(self):
        from nomorals.tools.registry import ToolRegistry

        root = Path(tempfile.mkdtemp(prefix="imgreg_"))
        self.addCleanup(shutil.rmtree, root, True)
        exif = Image.Exif()
        exif[0x010F] = "RegCam"
        path = root / "r.jpg"
        Image.new("RGB", (16, 16), (1, 2, 3)).save(path, "JPEG", exif=exif)

        reg = ToolRegistry(context=_make_context(_make_db()))
        imagedb.register(reg)
        res = reg._tools["image_exif"].fn(str(path))
        self.assertTrue(res["ok"])
        self.assertEqual(res["exif"].get("make"), "RegCam")


if __name__ == "__main__":
    unittest.main()
