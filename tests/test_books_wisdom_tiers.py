"""BOOKS + WISDOM tiers — Tier 0–3 new modules.

No fabricated numbers: every assertion checks a value the code computed
from the fixture text. The bible.py facade is untouched; the golden-set
check proves the continuity engine still adapts a real StoryBible.
"""

import json
from types import SimpleNamespace

import pytest

# ── Tier 0: retrieval spine ────────────────────────────────────────────

from nomorals.retrieval.spine import (
    Hit, RetrievalSpine, HashEmbedder, mmr_rerank, rrf_fuse)


def test_rrf_fuse_weights():
    fused = rrf_fuse([["a", "b"], ["b", "a"]], weights=[2.0, 1.0])
    assert fused["a"] > fused["b"]


def test_mmr_demotes_duplicate():
    items = [Hit(key="a", text="cats dogs", score=1.0),
             Hit(key="b", text="cats dogs", score=0.9),
             Hit(key="c", text="quantum physics", score=0.8)]
    out = mmr_rerank(items, top=2, lambda_=0.6)
    assert [h.key for h in out] == ["a", "c"]


def test_spine_keyword_and_hybrid():
    docs = [
        {"id": "d1", "text": "cats sleep in the sun all afternoon"},
        {"id": "d2", "text": "dogs bark at the moon every night"},
        {"id": "d3", "text": "cats chase mice through the garden"},
    ]
    spine = RetrievalSpine(embedder=HashEmbedder())
    kw = spine.search("cats", docs, mode="keyword", top=2)
    assert {h.key for h in kw} == {"d1", "d3"}
    assert all("keyword" in h.scores for h in kw)
    hyb = spine.search("cats sleep", docs, mode="hybrid", top=2)
    assert hyb[0].key == "d1"  # both keyword + semantic agree
    assert "fused" in hyb[0].scores


def test_spine_similarity_floor():
    docs = [{"id": "d1", "text": "cats sleep in the sun"}]
    spine = RetrievalSpine(embedder=HashEmbedder())
    assert spine.search("quantum entanglement", docs, mode="semantic",
                        top=5, min_score=0.99) == []


def test_spine_rejects_empty_query():
    with pytest.raises(ValueError):
        RetrievalSpine(embedder=HashEmbedder()).search("", [], mode="keyword")


# ── Tier 1: continuity engine ──────────────────────────────────────────

from nomorals.books.continuity import ContinuityEngine, Entity, Thread


def test_continuity_lifecycle_and_retire():
    eng = ContinuityEngine()
    state = eng.new_state("work-1", "Work One")
    eng.add_entity(state, Entity(id="mara", name="Mara"))
    assert state.entity("mara").status == "alive"
    eng.retire_entity(state, "mara", reason="written out", chapter=3)
    assert state.entity("mara").status == "retired"
    # never deleted — still in state
    assert len(state.entities) == 1
    with pytest.raises(ValueError):
        eng.retire_entity(state, "mara", reason="", chapter=4)


def test_continuity_dead_entity_finding():
    eng = ContinuityEngine()
    state = eng.new_state("w")
    ent = eng.add_entity(state, Entity(id="mara", name="Mara"))
    ent.status = "dead"
    findings = eng.check_chapter(state, 5, "Mara walked into the room.")
    assert any(f.kind == "dead_entity_appears" and f.severity == "error"
               for f in findings)


def test_continuity_drift_reported_not_assumed():
    from nomorals.books.voice import vectorize
    eng = ContinuityEngine()
    state = eng.new_state("w")
    ref = ("Mara walked through the market. The stalls smelled of smoke "
           "and pepper. She counted her coins and kept walking.")
    state.voice.features = vectorize(ref).features
    same = eng.drift(state, 2, ref)
    assert same.similarity == 1.0 and same.drifted is False
    other = eng.drift(state, 3, '"What?" he shouted! "No! Never!" '
                      '"Run!" she screamed! "Now!"')
    assert other.similarity < 1.0
    assert eng.measure_drift(ref, ref) == 1.0


def test_bible_facade_golden_set():
    """Real StoryBible (untouched bible.py) adapts to engine state."""
    from nomorals.books.bible import (BibleCharacter, PlotThread,
                                      StoryBible)
    bible = StoryBible(
        story_slug="golden", title="Golden",
        characters=[BibleCharacter(name="Mara", role="protagonist"),
                    BibleCharacter(name="Jide", role="supporting")],
        threads=[PlotThread(id="t1", summary="the lost crown",
                            status="open", introduced=1, last_seen=1)],
        world_rules=["Magic drains blood."],
        chapters_digested=4)
    state = ContinuityEngine().from_story_bible(bible)
    assert state.work_slug == "golden"
    assert {e.name for e in state.entities} == {"Mara", "Jide"}
    assert state.open_threads()[0].id == "t1"
    assert len(state.rules) == 1


# ── Tier 1: critique engine ────────────────────────────────────────────

from nomorals.books.critique_engine import CritiqueEngine

GOOD_CHAPTER = """
Mara counted her coins at the market gate. The morning crowd pressed
around the yam stalls, and the air smelled of smoke and pepper and
rain that had not yet fallen. She had walked three days to reach the
city, and the gate guards had taken half her yams for the toll.

"We leave at dawn," she told the carrier. "The road to Ilesa eats slow
feet, and I have no more yams to pay another toll."

The carrier laughed and slung her bundle over his shoulder. He led
her through the narrow lanes where children chased chickens between
the stalls and women called out prices for cloth and pepper and
palm oil. Mara kept her hand on the knife at her belt and her eyes
on the crowd, because the city had already taken enough from her.

At the far end of the market, a boy ran toward them with a sealed
letter in his hand. He stopped in front of Mara, breathing hard, and
held the letter out. The seal was her father's. Mara broke it open
with shaking fingers and read the first line twice before she
understood it.

Would the gate still stand when they arrived?
"""

BAD_CHAPTER = (
    "He was sad. He really felt very tired. He was taken by the men. "
    "He saw the room. He felt the fear. He spoke softly. He was scared. "
    "He really felt very tired. He was taken by the men. "
    "He saw the room. He felt the fear."
)


def test_critique_good_beats_bad():
    eng = CritiqueEngine()
    good = eng.review(GOOD_CHAPTER)
    bad = eng.review(BAD_CHAPTER)
    assert good.overall > bad.overall
    assert good.verdict in ("ship", "revise")
    assert set(d.dimension for d in good.dimensions) == {
        "prose", "pacing", "dialogue", "hook", "structure", "continuity"}
    collab = good.to_collab()
    assert collab["verdict"] == good.verdict
    assert "score" in collab


def test_critique_floor():
    eng = CritiqueEngine(floor=0.99)
    report = eng.review(GOOD_CHAPTER)
    assert report.floor_passed is False
    assert report.verdict in ("revise", "rewrite")


# ── Tier 2: prose + voice ──────────────────────────────────────────────

from nomorals.books.prose import analyze as prose_analyze
from nomorals.books.prose import chapter_end_hook
from nomorals.books.voice import VoiceClaim, voice_similarity


def test_prose_metrics():
    rep = prose_analyze(GOOD_CHAPTER)
    assert rep.words >= 200
    assert rep.filter_density == 0.0  # fixture has no filter words
    assert rep.hook["kind"] == "question"
    assert rep.pov_drift is False
    bad = prose_analyze(BAD_CHAPTER)
    assert bad.filter_density > 0.015
    assert bad.adverb_density > 0.0
    assert bad.passive_count >= 2


def test_prose_hook_kinds():
    assert chapter_end_hook("Nothing happened. The end.").get("kind") == "none"
    assert chapter_end_hook("She opened the door. And then —").get(
        "kind") in ("cliffhanger", "trailing")


def test_voice_similarity():
    assert voice_similarity(GOOD_CHAPTER, GOOD_CHAPTER) == 1.0
    assert voice_similarity(GOOD_CHAPTER, BAD_CHAPTER) < 1.0
    claim = VoiceClaim()
    chapters = [(1, GOOD_CHAPTER), (2, GOOD_CHAPTER), (3, BAD_CHAPTER)]
    report = claim.report(chapters)
    assert len(report["per_chapter"]) == 3
    assert report["per_chapter"][0]["similarity"] > \
        report["per_chapter"][2]["similarity"]


# ── Tier 2: coherence ──────────────────────────────────────────────────

from nomorals.books.coherence import analyze_coherence
from nomorals.books.continuity import Rule


def test_coherence_all_five_checks():
    eng = ContinuityEngine()
    state = eng.new_state("w")
    state.entities.append(Entity(id="mara", name="Mara", kind="character",
                                 first_seen=1, mentions=5))
    state.entities.append(Entity(id="sunblade", name="Sunblade",
                                 kind="item", first_seen=1))
    state.threads.append(Thread(id="t1", summary="the lost crown",
                                introduced=1, last_seen=1,
                                keywords=["lost crown"]))
    state.rules.append(Rule(id="r1", statement="Magic drains blood.",
                            source_chapter=1))
    chapters = [(1, "Mara aged 30 lifted the Sunblade. Magic drains "
                    "blood, the elders said."),
                (2, "Mara aged 29 walked on.")] + \
               [(n, "Mara walked on through the quiet road.") for n in range(3, 11)]
    report = analyze_coherence(state, chapters, stall_after=5)
    kinds = {f.kind for f in report.findings}
    assert "timeline_age_regression" in kinds
    assert "unpaid_setup" in kinds
    assert "chekhov_gun" in kinds
    assert report.to_dict()["counts"]["total"] == len(report.findings)


def test_coherence_dead_speaks():
    chapters = [(1, "Mara died in the fire."),
                (2, "Mara said hello to the crowd.")]
    report = analyze_coherence(ContinuityEngine().new_state("w"), chapters)
    assert any(f.kind == "contradiction_dead_speaks"
               for f in report.contradictions)


# ── Tier 3: publishing extras ──────────────────────────────────────────

from nomorals.books.publish_ext import (
    ARCProgram, BookMetadata, SeriesManager, attribution_label,
    blurb, build_cover_brief, build_cover_prompt, generate_cover,
    release_notification)


def test_publish_ext_cover_blurb_metadata():
    brief = build_cover_brief("Crown of Pepper", "A. Writer",
                              genre="fantasy", mood="tense",
                              culture_pack="nigeria")
    assert brief.palette  # Nigeria pack palette applied
    assert "Crown of Pepper" in build_cover_prompt(brief)
    no_gen = generate_cover(brief)
    assert no_gen["ok"] is False and "prompt" in no_gen
    got = generate_cover(brief, image_generator=lambda p: f"img:{p[:10]}")
    assert got["ok"] is True
    assert "AI-generated" in got["attribution"]
    assert attribution_label() == "human-authored"
    text = "Mara counted her coins. She had walked three days. The gate loomed."
    assert blurb(text, max_chars=60).endswith("…") or \
        len(blurb(text, max_chars=60)) <= 60
    meta = BookMetadata(title="T", author="A", isbn="not-an-isbn")
    assert meta.validate()  # bad ISBN flagged
    assert BookMetadata(title="T", author="A").validate() == []


def test_publish_ext_arc_series_release(tmp_path):
    arc = ARCProgram("book-1", path=tmp_path / "arc.json")
    assert arc.add_reader("r1")["ok"] is True
    assert arc.record_feedback("ghost", 1, "x")["ok"] is False
    assert arc.record_feedback("r1", 1, "loved it", rating=5)["ok"] is True
    assert arc.stats()["avg_rating"] == 5.0
    series = SeriesManager(path=tmp_path / "series.json")
    assert series.add_volume("saga", 1, "book-1")["ok"] is True
    assert series.add_volume("saga", 1, "book-1")["ok"] is False
    assert series.next_unreleased("saga")["volume"] == 1
    note = release_notification({"title": "Saga", "followers": ["u1"],
                                 "announce_chat": "c1"},
                                chapter=2, blurb_text="Mara returns.")
    assert note["send_to"] == 1 and "chapter 2" in note["message"]


# ── Tier 3: reader continuity ──────────────────────────────────────────

from nomorals.books.reader_continuity import (
    catch_up_digest, chapter_summary, extractive_summary,
    group_by_series, return_after_break)


def test_reader_continuity(tmp_path):
    class FakeReader:
        def story_dir(self, slug):
            d = tmp_path / slug
            (d / "chapters").mkdir(parents=True, exist_ok=True)
            return d

        def _workspace(self):
            return tmp_path

        def _load(self, slug):
            return SimpleNamespace(current_chapter=1, title="Tale",
                                   total_chapters=4)

    reader = FakeReader()
    ch_dir = reader.story_dir("tale") / "chapters"
    body = ("Mara walked to the market. She bought yams and pepper. "
            "The guards took her toll. She swore she would return. "
            "The road home was long and dusty.")
    (ch_dir / "00002.md").write_text(body, encoding="utf-8")
    (ch_dir / "00003.md").write_text(body, encoding="utf-8")
    digest = catch_up_digest(reader, "tale")
    assert digest["ok"] is True
    assert digest["covered"] == [2, 3]
    assert digest["gaps_not_cached"] == [4]
    assert "Mara" in digest["combined"]  # extractive: real sentences
    ret = return_after_break(reader, "tale")
    assert ret["ok"] is True and ret["bible"] is False
    assert extractive_summary("") == ""
    assert chapter_summary(body, chapter=2)["words"] > 0
    groups = group_by_series([{"title": "Saga Book 1"},
                              {"title": "Saga Book 2"},
                              {"title": "Other"}])
    assert len(groups["Saga"]) == 2


# ── Tier 2: tradition packs ────────────────────────────────────────────

from nomorals.wisdom.traditions import (
    RestrictedCorpusError, check_ingest, corpus_coverage, nigeria_pack,
    pack_by_id)
from nomorals.wisdom.traditions import PackText, all_packs


def test_tradition_packs_nigeria_first():
    packs = all_packs()
    assert packs[0].id == "nigeria"
    assert nigeria_pack().validate() == []
    ifa = [t for t in nigeria_pack().texts if "Ifa" in t.title][0]
    assert ifa.restriction == "do-not-digitize"
    assert "verse" not in ifa.summary.lower() or "not verses" in ifa.summary.lower()
    with pytest.raises(RestrictedCorpusError):
        check_ingest(nigeria_pack(), ifa)  # STOP, no permission
    open_text = PackText(title="Public note", restriction="open")
    assert check_ingest(nigeria_pack(), open_text).restriction == "open"
    assert pack_by_id("kemet").region.startswith("Egypt")
    with pytest.raises(ValueError):
        pack_by_id("nope")


def test_corpus_coverage_measurable():
    cov = corpus_coverage([{"tradition": "nigeria", "region": ""},
                           {"tradition": "christianity", "region": ""},
                           {"tradition": "christianity", "region": ""}])
    assert cov["counts"] == {"sub_saharan_african": 1, "eurasian": 2,
                             "other": 0}
    assert cov["sub_saharan_share"] == 0.3333


# ── Tier 2: disagreements + lineage ────────────────────────────────────

from nomorals.wisdom.disagreements import DisagreementMap, Position
from nomorals.wisdom.lineage import (LineageEdge, LineageGraph,
                                     LineageNode)


def test_disagreements_never_averaged():
    dm = DisagreementMap()
    assert "self" in dm.topics()
    d = dm.compare("self")
    assert len(d.positions) == 3
    assert "not averaged" in d.to_dict()["note"]
    assert "Advaita" in d.render()
    with pytest.raises(ValueError):
        dm.register("bad", "q", [Position(tradition="X", claim="c",
                                          basis="", sources=["s"])])
    with pytest.raises(ValueError):
        dm.compare("no-such-topic")


def test_disagreements_with_corpus():
    class FakeHit:
        def to_dict(self):
            return {"work": "Upanishads", "tradition": "Advaita Vedanta",
                    "snippet": "x"}

    class FakeCorpus:
        corpus = SimpleNamespace(_manifest={})

        def ask(self, query, top=5, mode="keyword"):
            return SimpleNamespace(passages=[FakeHit()])

    d = DisagreementMap().compare_with_corpus("self", FakeCorpus())
    assert d.passages["Advaita Vedanta"][0]["work"] == "Upanishads"


def test_lineage_graph():
    g = LineageGraph()
    g.add_node(LineageNode(id="socrates", kind="person", name="Socrates"))
    g.add_node(LineageNode(id="plato", kind="person", name="Plato"))
    g.add_node(LineageNode(id="gita", kind="text", name="Bhagavad Gita"))
    g.add_node(LineageNode(id="gita-en", kind="translation",
                           name="Gita (Arnold)"))
    g.add_edge(LineageEdge(from_id="socrates", to_id="plato",
                           relation="teacher_of",
                           source="standard history of philosophy"))
    g.add_edge(LineageEdge(from_id="gita-en", to_id="gita",
                           relation="translation_of",
                           source="translator's preface"))
    assert g.teachers_of("plato")[0].name == "Socrates"
    assert g.students_of("socrates")[0].name == "Plato"
    assert len(g.translation_chain("gita")) == 1
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge(from_id="plato", to_id="socrates",
                               relation="teacher_of", source=""))
    with pytest.raises(ValueError):
        g.add_edge(LineageEdge(from_id="plato", to_id="ghost",
                               relation="teacher_of", source="x"))
    tl = LineageGraph.from_timeline([
        {"start": -500, "end": -400, "tradition": "taoism",
         "title": "Tao Te Ching", "summary": "s", "sources": ["x"]},
        {"start": -400, "end": -300, "tradition": "taoism",
         "title": "Chuang Tzu", "summary": "s", "sources": ["x"]}])
    assert len(tl.edges) == 1
    assert tl.edges[0].relation == "tradition_sequence"


# ── Tier 3: oral ingest ────────────────────────────────────────────────

from nomorals.wisdom.oral import (
    MissingTranscriberError, OralIngest, PermissionRecord, align_words,
    split_turns)


def test_oral_pipeline(tmp_path):
    assert [t.speaker for t in split_turns(
        "Ada: Hello.\nAda: Second line.\nplain continuation")] == [
        "Ada", "Ada"]
    assert align_words(split_turns("Ada: Hello world."))[0]["approximate"] is True
    oral = OralIngest(tmp_path / "oral")
    with pytest.raises(MissingTranscriberError):
        oral.transcribe("audio.mp3")
    assert oral.transcribe("audio.mp3",
                           transcriber=lambda ref: "Ada: Hi.") == "Ada: Hi."
    with pytest.raises(RestrictedCorpusError):
        oral.submit_draft("d-blocked", transcript="Elder: secret.",
                          restriction="initiated")
    draft = oral.submit_draft(
        "d1", transcript="Ada: Hello world.", language="Yoruba",
        tradition_pack_id="nigeria", restriction="community-permission",
        permission=PermissionRecord(granted_by="community elder",
                                    scope="one recording"))
    with pytest.raises(Exception):
        oral.ingest("d1")  # not verified yet
    oral.verify("d1", owner="owner", approve=True)
    out = oral.ingest("d1")
    assert out["ok"] is True and out["words"] == 2
    assert json.loads((tmp_path / "oral" / "d1.json").read_text())["status"] \
        == "verified"


# ── Tier 3: practice memory ────────────────────────────────────────────

from nomorals.wisdom.practice_memory import (
    MissingSourceError, PracticeMemory)


def test_practice_memory_sm2(tmp_path):
    mem = PracticeMemory(tmp_path / "mem.json")
    with pytest.raises(MissingSourceError):
        mem.add_item("What is ori?", "x", source={})
    item = mem.add_item("What is ori?", "pack: nigeria",
                        source={"tradition": "Yoruba",
                                "work": "Nigeria pack"})
    mem.review(item.id, 5, now=0.0)
    assert mem.get(item.id).interval_days == 1
    mem.review(item.id, 5, now=86400.0)
    assert mem.get(item.id).interval_days == 6
    mem.review(item.id, 5, now=7 * 86400.0)
    assert mem.get(item.id).interval_days == 16  # 6 × 2.6 easiness
    mem.review(item.id, 1, now=23 * 86400.0)
    assert mem.get(item.id).repetitions == 0
    assert mem.get(item.id).lapses == 1
    assert mem.stats(now=100 * 86400.0)["due"] == 1


# ── Tier 0: completeness audit ─────────────────────────────────────────

from nomorals.wisdom.completeness import wisdom_completeness_report


def test_completeness_all_done():
    report = wisdom_completeness_report()
    assert report["all_done"] is True, report["items"]
