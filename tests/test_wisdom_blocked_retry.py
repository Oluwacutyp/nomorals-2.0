"""Retry of the 4 network-blocked WisdomKeeper corpus texts (2026-11-14).

On 2026-10-03 the corpus stood at 43/47: Gospel of Mary, Gospel of
Philip, Zohar, Philokalia were blocked at the network level (sites
refuse all connections) and marked in the manifest for retry. This
file pins the retry's outcome:

* Philip moved to the gospels.net mirror (Mark M. Mattison
  public-domain translation, verified HTTP 200 + ingested 2026-11-14);
  gnosis.org refuses all connections from this network (empty reply).
* Mary already pointed at gospels.net (switched 2026-10-19) but had
  never been ingested; the mirror ingests through the real path.
* Zohar and Philokalia keep their sacred-texts URLs: the sources are
  alive, the block is Cloudflare IP-reputation on this network (the
  Zohar mirror was HTTP 200 on 2026-10-19), and no single-file
  public-domain alternate exists for either (archive.org has no Manhar
  full text; the Philokalia 'Writings' copies are access-restricted
  401; the Faber complete text is copyrighted, ruled not usable
  2026-10-19). Both retry on EC2: ``nm wisdom ingest <slug>``.

All HTTP is mocked: the fake returns real excerpt bytes (slices of
the live gospels.net pages, stored under tests/fixtures) for the
mirror URL only and raises for anything else, simulating gnosis.org
refusing the connection. No test touches the network.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.core.errors import ProviderError
from nomorals.core.http import HttpResponse
from nomorals.wisdom import ArchiveIngestor

_FIXTURES = Path(__file__).parent / "fixtures"
PHILIP_URL = "https://www.gospels.net/philip"
MARY_URL = "https://www.gospels.net/mary"


def _ctx():
    tmp = tempfile.mkdtemp(prefix="wisdom-blocked-retry-test-")
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(settings=settings), tmp


class _FakeHttp:
    """Stand-in for HttpClient: serves fixture bytes for the mirror URL
    only; every other URL raises like a refused connection."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls: list[str] = []

    def get(self, url, **kw):
        self.calls.append(url)
        if url in self.mapping:
            return HttpResponse(status=200, body=self.mapping[url],
                                headers={}, url=url)
        raise ProviderError(f"connection refused by {url}",
                            retryable=False)


def _make_ingestor(mapping):
    ctx, tmp = _ctx()
    ing = ArchiveIngestor(ctx)
    ing._http = _FakeHttp(mapping)
    ing._robots = SimpleNamespace(allowed=lambda *a, **k: True)
    return ing, tmp


def _fixtures():
    return {
        PHILIP_URL: (_FIXTURES / "wisdom-gospel-philip-excerpt.html"
                     ).read_bytes(),
        MARY_URL: (_FIXTURES / "wisdom-gospel-mary-excerpt.html"
                   ).read_bytes(),
    }


class ManifestRetryTests(unittest.TestCase):
    def test_philip_points_at_gospels_net_mirror(self):
        ctx, _ = _ctx()
        ing, _tmp = _make_ingestor(_fixtures())
        ing.corpus.seed()
        entry = ing.corpus.get("gospel-of-philip")
        self.assertEqual(entry.source_url, PHILIP_URL)
        self.assertIn("Mattison", entry.translator)
        self.assertEqual(entry.license, "public-domain")

    def test_mary_mirror_unchanged(self):
        ctx, _ = _ctx()
        ing, _tmp = _make_ingestor(_fixtures())
        ing.corpus.seed()
        entry = ing.corpus.get("gospel-of-mary")
        self.assertEqual(entry.source_url, MARY_URL)
        self.assertIn("Mattison", entry.translator)
        self.assertEqual(entry.license, "public-domain")

    def test_zohar_and_philokalia_keep_sacred_texts_urls_with_retry_note(
            self):
        ctx, _ = _ctx()
        ing, _tmp = _make_ingestor(_fixtures())
        ing.corpus.seed()
        zohar = ing.corpus.get("zohar-selections")
        self.assertEqual(
            zohar.source_url,
            "https://archive.sacred-texts.com/jud/zdm/index.htm")
        self.assertIn("2026-11-14", zohar.notes)
        self.assertIn("retry on EC2", zohar.notes)
        philokalia = ing.corpus.get("philokalia-selections")
        self.assertEqual(
            philokalia.source_url,
            "https://www.sacred-texts.com/chr/philokalia/index.htm")
        self.assertIn("2026-11-14", philokalia.notes)
        self.assertIn("retry on EC2", philokalia.notes)


class IngestRetryTests(unittest.TestCase):
    def test_ingest_philip_from_mirror(self):
        ing, _tmp = _make_ingestor(_fixtures())
        ing.corpus.seed()
        entry = ing.ingest_entry(ing.corpus.get("gospel-of-philip"))
        state = ing.corpus.work_state(entry)
        self.assertTrue(state["ingested"])
        self.assertGreater(state["chars"], 200)
        ans = ing.corpus.ask("bridal chamber", top=3)
        slugs = [h.slug for h in ans.passages]
        self.assertIn("gospel-of-philip", slugs)

    def test_ingest_mary_from_mirror(self):
        ing, _tmp = _make_ingestor(_fixtures())
        ing.corpus.seed()
        entry = ing.ingest_entry(ing.corpus.get("gospel-of-mary"))
        state = ing.corpus.work_state(entry)
        self.assertTrue(state["ingested"])
        self.assertGreater(state["chars"], 200)
        ans = ing.corpus.ask("preach the gospel", top=3)
        slugs = [h.slug for h in ans.passages]
        self.assertIn("gospel-of-mary", slugs)


if __name__ == "__main__":
    unittest.main()
