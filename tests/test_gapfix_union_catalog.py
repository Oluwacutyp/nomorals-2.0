"""Union catalogue (52-websites #33, Anna's Archive): one record per
work, merged from many sources, provenance never discarded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomorals.books.union_catalog import (
    LAWFUL_SOURCES,
    Candidate,
    UnionCatalog,
    UnionCatalogStore,
    WorkRecord,
    format_catalog_reply,
    format_work,
    parse_catalog_request,
    register_source,
    search_archiveorg,
    search_gutenberg,
    search_openlibrary,
)

# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def home(tmp_path: Path) -> str:
    return str(tmp_path / "home")


@pytest.fixture()
def catalog(home: str) -> UnionCatalog:
    return UnionCatalog(home)


def _ol_pride() -> Candidate:
    return Candidate(
        title="Pride and Prejudice", authors=["Jane Austen"],
        identifiers={"isbn": "9780141439518",
                     "openlibrary": "/works/OL66554W"},
        source_id="openlibrary",
        source_ref="https://openlibrary.org/works/OL66554W",
        format="ebook")


def _gut_pride() -> Candidate:
    return Candidate(
        title="Pride and Prejudice", authors=["Austen, Jane"],
        identifiers={"gutenberg": "1342"},
        source_id="gutenberg",
        source_ref="https://www.gutenberg.org/ebooks/1342",
        format="Plain text, EPUB")


def _archive_pride() -> Candidate:
    return Candidate(
        title="Pride & Prejudice", authors=["Jane Austen"],
        identifiers={"archive": "prideprejudice00aust"},
        source_id="archiveorg",
        source_ref="https://archive.org/details/prideprejudice00aust",
        format="ebook (Archive.org)")


# ── merge / reconciliation ───────────────────────────────────────────────


class TestIdentifierMerge:
    def test_same_isbn_from_two_sources_is_one_record(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        result = catalog.ingest([_gut_pride()], "gutenberg", health="live")
        works = catalog.list_works()
        assert len(works) == 1
        assert result["merged"] >= 0
        work = works[0]
        assert work.title == "Pride and Prejudice"
        assert {h.source_id for h in work.holdings} == {
            "openlibrary", "gutenberg"}
        assert work.identifiers["isbn"] == "9780141439518"
        assert work.identifiers["gutenberg"] == "1342"

    def test_isbn10_converts_and_merges_with_isbn13(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        isbn10 = Candidate(
            title="Pride and Prejudice", authors=["Jane Austen"],
            identifiers={"isbn": "0-14-143951-3"},
            source_id="archiveorg",
            source_ref="https://archive.org/details/othercopy",
            format="ebook (Archive.org)")
        catalog.ingest([isbn10], "archiveorg", health="live")
        works = catalog.list_works()
        assert len(works) == 1
        assert works[0].identifiers["isbn"] == "9780141439518"
        assert len(works[0].holdings) == 2

    def test_same_title_different_author_is_not_a_merge(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        other = Candidate(
            title="Pride and Prejudice", authors=["Someone Else"],
            identifiers={"archive": "prideprejudice99other"},
            source_id="archiveorg",
            source_ref="https://archive.org/details/prideprejudice99other",
            format="ebook (Archive.org)")
        catalog.ingest([other], "archiveorg", health="live")
        assert len(catalog.list_works()) == 2

    def test_fuzzy_title_and_author_merges(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        result = catalog.ingest([_archive_pride()], "archiveorg",
                                health="live")
        works = catalog.list_works()
        assert len(works) == 1
        assert result["merged"] == 1
        assert {h.source_id for h in works[0].holdings} == {
            "openlibrary", "archiveorg"}

    def test_same_holding_is_never_duplicated(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        result = catalog.ingest([_ol_pride()], "openlibrary", health="live")
        works = catalog.list_works()
        assert len(works) == 1
        assert len(works[0].holdings) == 1
        assert result["holdings_added"] == 0

    def test_merge_record_is_reversible(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        catalog.ingest([_archive_pride()], "archiveorg", health="live")
        works = catalog.list_works()
        assert len(works) == 1
        canonical = works[0].canonical_id
        merges = catalog.store.merges_for(canonical)
        assert len(merges) == 1
        merged_id = merges[0].merged_id
        assert catalog.unmerge(canonical, merged_id) is True
        restored = catalog.list_works()
        assert len(restored) == 2
        by_id = {w.canonical_id: w for w in restored}
        assert merged_id in by_id
        reinstated = by_id[merged_id]
        # Provenance intact: the reinstated record keeps its own
        # holding, title and identifiers; the survivor keeps the rest.
        assert {h.source_id for h in reinstated.holdings} == {"archiveorg"}
        assert reinstated.identifiers["archive"] == "prideprejudice00aust"
        assert {h.source_id for h in by_id[canonical].holdings} == {
            "openlibrary"}
        assert catalog.unmerge(canonical, merged_id) is False

    def test_unmerge_unknown_ids_is_false(
            self, catalog: UnionCatalog) -> None:
        assert catalog.unmerge("work:nope", "work:alsono") is False


# ── ranking: dead holdings degrade, never vanish ─────────────────────────


class TestDeadHoldings:
    def test_dead_holding_still_listed_and_labelled(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_gut_pride()], "gutenberg", health="dead")
        works = catalog.find("pride prejudice", live=False)
        assert len(works) == 1
        reply = format_work(works[0])
        assert "unavailable right now" in reply
        assert "gutenberg.org/ebooks/1342" in reply

    def test_live_work_ranks_above_dead_only_work(
            self, catalog: UnionCatalog) -> None:
        # Two different works: one held only by a dead source, one
        # held live. (Same title + same author would merge into one
        # work — that is the catalogue doing its job.)
        dead = Candidate(
            title="Middlemarch", authors=["George Eliot"],
            identifiers={"gutenberg": "145"},
            source_id="gutenberg",
            source_ref="https://www.gutenberg.org/ebooks/145",
            format="Plain text")
        catalog.ingest([dead], "gutenberg", health="dead")
        live = Candidate(
            title="Pride and Prejudice", authors=["Jane Austen"],
            identifiers={"openlibrary": "/works/OL66554W"},
            source_id="openlibrary",
            source_ref="https://openlibrary.org/works/OL66554W",
            format="ebook")
        catalog.ingest([live], "openlibrary", health="live")
        works = catalog.list_works()
        assert len(works) == 2
        assert works[0].title == "Pride and Prejudice"
        assert works[0].live_holdings()
        assert works[1].title == "Middlemarch"
        assert not works[1].live_holdings()
        assert "unavailable right now" in format_work(works[1])

    def test_find_with_no_match_answers_honestly(
            self, catalog: UnionCatalog) -> None:
        assert catalog.find("zzq no such book", live=False) == []
        reply = format_catalog_reply("zzq no such book", [])
        assert "couldn't find" in reply
        assert "Nothing invented" in reply


# ── owner library leg ────────────────────────────────────────────────────


class _FakeLibrary:
    def __init__(self, books: list[dict[str, str]]) -> None:
        self._books = books

    def list_books(self) -> list[dict[str, str]]:
        return self._books


class TestOwnerLibrary:
    def test_owner_holding_ranks_first(self, home: str) -> None:
        library = _FakeLibrary([{
            "slug": "prince-of-persia",
            "title": "The Making of Prince of Persia",
            "author": "Jordan Mechner"}])
        catalog = UnionCatalog(home, library=library)
        works = catalog.find("making of prince of persia", live=False)
        assert len(works) == 1
        assert {h.source_id for h in works[0].holdings} == {"owner"}
        assert works[0].holdings[0].source_ref == "prince-of-persia"
        assert works[0].live_holdings()

    def test_broken_library_leg_degrades_to_empty(
            self, home: str) -> None:
        class _Broken:
            def list_books(self) -> list[dict[str, str]]:
                raise RuntimeError("library exploded")

        catalog = UnionCatalog(home, library=_Broken())
        assert catalog.find("pride prejudice", live=False) == []


# ── live legs with injected fetchers ─────────────────────────────────────


class TestLiveLegs:
    def test_openlibrary_fixture_yields_identifiers(self) -> None:
        def fetcher(url: str, timeout: float = 8.0) -> dict:
            assert "openlibrary.org" in url
            return {"docs": [{
                "title": "Pride and Prejudice",
                "author_name": ["Jane Austen"],
                "key": "/works/OL66554W",
                "isbn": ["9780141439518"]}]}

        found, status, _ = search_openlibrary(
            "pride prejudice", fetcher=fetcher)
        assert status == "live"
        assert len(found) == 1
        assert found[0].identifiers["isbn"] == "9780141439518"
        assert found[0].identifiers["openlibrary"] == "/works/OL66554W"

    def test_archiveorg_fixture(self) -> None:
        def fetcher(url: str, timeout: float = 8.0) -> dict:
            assert "archive.org" in url
            return {"response": {"docs": [{
                "identifier": "prideprejudice00aust",
                "title": "Pride and Prejudice",
                "creator": "Jane Austen"}]}}

        found, status, _ = search_archiveorg(
            "pride prejudice", fetcher=fetcher)
        assert status == "live"
        assert found[0].identifiers["archive"] == "prideprejudice00aust"

    def test_gutenberg_fixture(self) -> None:
        def fetcher(url: str, timeout: float = 8.0) -> dict:
            assert "gutendex.com" in url
            return {"results": [{
                "id": 1342, "title": "Pride and Prejudice",
                "authors": [{"name": "Austen, Jane"}],
                "formats": {
                    "text/plain; charset=utf-8": "https://example/x.txt",
                    "application/epub+zip": "https://example/x.epub"}}]}

        found, status, _ = search_gutenberg(
            "pride prejudice", fetcher=fetcher)
        assert status == "live"
        assert found[0].identifiers["gutenberg"] == "1342"
        assert "EPUB" in found[0].format

    def test_dead_fetcher_never_raises(self) -> None:
        def fetcher(url: str, timeout: float = 8.0) -> dict:
            raise ConnectionError("blackhole")

        for leg in (search_openlibrary, search_archiveorg,
                    search_gutenberg):
            found, status, _ = leg("pride prejudice", fetcher=fetcher)
            assert found == []
            assert status == "unverifiable"

    def test_empty_query_returns_nothing(self) -> None:
        for leg in (search_openlibrary, search_archiveorg,
                    search_gutenberg):
            found, status, _ = leg("   ")
            assert found == []
            assert status == "unknown"


# ── health refresh ───────────────────────────────────────────────────────


class TestRefreshHealth:
    def test_prober_live_and_dead(self, catalog: UnionCatalog) -> None:
        def prober(source_id: str) -> dict:
            return {"hits": 1} if source_id == "openlibrary" else {"hits": 0}

        results = catalog.refresh_health(prober=prober)
        assert results["openlibrary"]["status"] == "live"
        assert results["gutenberg"]["status"] == "dead"
        assert results["archiveorg"]["status"] == "dead"
        assert "owner" not in results

    def test_status_reports_size_and_health(
            self, catalog: UnionCatalog) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        status = catalog.status()
        assert status["works"] == 1
        assert status["holdings"] == 1
        assert status["sources"]["openlibrary"]["health"] == "live"
        assert status["sources"]["gutenberg"]["licence"] == (
            "Public domain (US)")


# ── store durability ─────────────────────────────────────────────────────


class TestStore:
    def test_corrupt_store_quarantined_and_reseeded(
            self, home: str) -> None:
        path = Path(home) / "data" / "union_catalog.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json", encoding="utf-8")
        store = UnionCatalogStore(path)
        assert store.list() == []
        assert len(list(path.parent.glob("union_catalog.json.corrupt-*"))) == 1
        store.put(WorkRecord(canonical_id="work:x", title="X"))
        store.save()
        assert UnionCatalogStore(path).get("work:x") is not None

    def test_round_trip_preserves_holdings_and_merges(
            self, catalog: UnionCatalog, home: str) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        catalog.ingest([_archive_pride()], "archiveorg", health="live")
        again = UnionCatalog(home)
        works = again.list_works()
        assert len(works) == 1
        assert len(works[0].holdings) == 2
        assert len(again.store.merges()) == 1


# ── lawful-sources discipline ────────────────────────────────────────────


class TestLawfulSources:
    def test_source_without_legal_basis_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no legal basis"):
            register_source("shady", label="Shady", licence="?",
                            legal_basis="")

    def test_registered_source_joins_the_table(self) -> None:
        row = register_source(
            "hathitrust-test", label="HathiTrust (test)",
            licence="open records only", legal_basis="open collection")
        assert row["legal_basis"] == "open collection"
        assert "hathitrust-test" in LAWFUL_SOURCES
        del LAWFUL_SOURCES["hathitrust-test"]


# ── export ───────────────────────────────────────────────────────────────


class TestExport:
    def test_jsonl_round_trip(self, catalog: UnionCatalog,
                              tmp_path: Path) -> None:
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        catalog.ingest([_gut_pride()], "gutenberg", health="live")
        out = tmp_path / "catalog.jsonl"
        receipt = catalog.export_aac(out)
        assert receipt["works"] == 1
        assert "zstd" in receipt
        lines = [json.loads(line) for line in
                 out.read_text(encoding="utf-8").splitlines() if line]
        assert len(lines) == 1
        assert lines[0]["title"] == "Pride and Prejudice"
        assert len(lines[0]["holdings"]) == 2


# ── request parsing ──────────────────────────────────────────────────────


class TestParseRequest:
    @pytest.mark.parametrize("text,query", [
        ("find me The Making of Prince of Persia, any format",
         "The Making of Prince of Persia"),
        ("find me Middlemarch any format", "Middlemarch"),
        ("find the book Middlemarch", "Middlemarch"),
        ("find me the book Dune any edition", "Dune"),
        ("where can I read Middlemarch", "Middlemarch"),
        ("where can I find Pride and Prejudice", "Pride and Prejudice"),
    ])
    def test_book_shaped_asks_parse(self, text: str, query: str) -> None:
        assert parse_catalog_request(text) == {"query": query}

    @pytest.mark.parametrize("text", [
        "find me a restaurant",
        "find me a good hotel near here",
        "find my phone",
        "/find me Middlemarch any format",
        "hello there",
        "find me",
    ])
    def test_non_book_asks_do_not_parse(self, text: str) -> None:
        assert parse_catalog_request(text) is None


# ── brain tool ───────────────────────────────────────────────────────────


class _FakeRegistry:
    def __init__(self, context: object) -> None:
        self.context = context
        self.tools: dict[str, object] = {}

    def register(self, name: str, **kwargs: object):  # noqa: ANN001, ANN202
        def deco(fn):  # noqa: ANN001, ANN202
            self.tools[name] = fn
            return fn

        return deco


class _FakeSettings:
    def __init__(self, home: str) -> None:
        self.home = home


class _FakeContext:
    def __init__(self, home: str) -> None:
        self.settings = _FakeSettings(home)


class TestTool:
    def test_find_show_list_status(self, home: str) -> None:
        from nomorals.tools.union_catalog import register

        catalog = UnionCatalog(home)
        catalog.ingest([_ol_pride()], "openlibrary", health="live")
        registry = _FakeRegistry(_FakeContext(home))
        register(registry)
        tool = registry.tools["union_catalog"]
        found = tool(action="find", query="pride prejudice", live="false")
        assert found["ok"] is True
        assert len(found["works"]) == 1
        assert "Open Library" in found["reply"]
        canonical = found["works"][0]["canonical_id"]
        shown = tool(action="show", id=canonical)
        assert shown["ok"] is True
        assert shown["work"]["title"] == "Pride and Prejudice"
        listed = tool(action="list")
        assert len(listed["works"]) == 1
        status = tool(action="status")
        assert status["works"] == 1
        assert tool(action="show", id="work:nope")["ok"] is False
        assert tool(action="bogus")["ok"] is False
        assert tool(action="find", query="")["ok"] is False

    def test_add_source_refuses_without_basis(self, home: str) -> None:
        from nomorals.tools.union_catalog import register

        registry = _FakeRegistry(_FakeContext(home))
        register(registry)
        tool = registry.tools["union_catalog"]
        refused = tool(action="add_source", id="shady", label="Shady",
                       licence="?", legal_basis="")
        assert refused["ok"] is False
        assert "legal basis" in refused["error"]


# ── runtime hook (unbound, on a context double) ──────────────────────────


class TestHook:
    def test_hook_answers_book_ask_and_ignores_other_chat(
            self, home: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import nomorals.books.union_catalog as uc
        from nomorals.agents.partner.runtime import PartnerRuntime

        # No live legs in this test: the hook must answer from the
        # seeded catalogue, never touch the network.
        monkeypatch.setattr(uc, "LIVE_LEGS", {})

        catalog = UnionCatalog(home)
        catalog.ingest([_ol_pride()], "openlibrary", health="live")

        class _Chat:
            platform = "telegram"

        class _Msg:
            text = "find me Pride and Prejudice, any format"
            incoming = True
            chat = _Chat()

        class _Ctx:
            settings = _FakeSettings(home)

        runtime = PartnerRuntime.__new__(PartnerRuntime)
        runtime.context = _Ctx()
        reply = PartnerRuntime._union_catalog_hook(runtime, _Msg())
        assert reply is not None
        assert "Pride and Prejudice" in reply
        assert "Open Library" in reply

        class _Other(_Msg):
            text = "find me a restaurant"

        assert PartnerRuntime._union_catalog_hook(runtime, _Other()) is None
