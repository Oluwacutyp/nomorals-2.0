"""Wave 85 core upgrades: source trust, permanently-on reasoning, the
reasoning critique in the research loop, and course correction in the
project runner.  All hermetic: canned pages, fake routers, in-memory DBs.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from tests.test_search_and_model import _FakeRouter

from nomorals.agents.context import build_context
from nomorals.agents.projects import ProjectManager
from nomorals.agents.researcher import ResearchAgent
from nomorals.agents.search.engine import SearchEngine
from nomorals.agents.search.trust import SourceTrust, domain_tier
from nomorals.core.config import load_settings


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _make_context(tmp: str, **kw) -> Any:
    context = build_context(_settings(tmp.name), with_executor=False,
                            with_tools=False, with_router=False,
                            with_memory=False, **kw)
    context.router = _FakeRouter()
    return context


# ── 1. the domain ladder ────────────────────────────────────────────────────


class TrustTierTests(unittest.TestCase):
    LADDER = [
        ("https://www.cia.gov/the-library/", 0.95, "government"),
        ("https://osha.gov/laws-regs", 0.95, "government"),
        ("https://www.ox.ac.uk/research", 0.90, "academic"),
        ("https://arxiv.org/abs/2401.00001", 0.90, "academic"),
        ("https://docs.python.org/3/library/asyncio.html", 0.85, "official docs"),
        ("https://github.com/oven-sh/bun", 0.85, "source / project"),
        ("https://www.reuters.com/world/news", 0.80, "established press"),
        ("https://stackoverflow.com/questions/1", 0.75, "established Q&A"),
        ("https://en.wikipedia.org/wiki/No_morals", 0.70, "reference (crowd)"),
        ("https://medium.com/@bob/idea", 0.45, "blog / aggregator"),
        ("https://www.reddit.com/r/programming/", 0.40, "forum"),
        ("https://example.com/page", 0.55, "general web"),
        ("https://spam-site.xyz/free-money", 0.20, "low-reputation TLD"),
        ("not a url at all", 0.30, "unresolved host"),
    ]

    def test_domain_ladder_tiers(self) -> None:
        for url, tier, note in self.LADDER:
            got_tier, got_note = domain_tier(url)
            self.assertAlmostEqual(got_tier, tier, msg=url)
            self.assertEqual(got_note, note, msg=url)

    def test_no_false_suffix_match(self) -> None:
        # 'nytimes.com.example.org' must NOT inherit the press tier
        tier, _ = domain_tier("https://nytimes.com.example.org/x")
        self.assertEqual(tier, 0.55)

    def test_corroboration_raises_and_caps(self) -> None:
        class Ctx:  # no db — pure tier math
            db = None

        t = SourceTrust(Ctx())
        base = t.score("https://example.com/x")
        boosted = t.score("https://example.com/x", corroborated_by=2)
        self.assertGreater(boosted, base)
        self.assertLessEqual(t.score("https://example.com/x",
                                     corroborated_by=99), 0.99)

    def test_staleness_decays(self) -> None:
        class Ctx:
            db = None

        t = SourceTrust(Ctx())
        fresh = t.score("https://example.com/x")
        stale = t.score("https://example.com/x", age_hours=360)
        self.assertLess(stale, fresh)


# ── 2. persistent fetch learning ────────────────────────────────────────────


class TrustFeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-trust-")
        self.context = _make_context(self.tmp)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_failures_sink_the_source(self) -> None:
        t = SourceTrust(self.context)
        url = "https://flaky.example.org/page"
        tier, _ = domain_tier(url)
        for _ in range(3):
            t.feedback(url, ok=False)
        self.assertLess(t.score(url), tier)
        self.assertIn("failing source", t.note(url))

    def test_penalty_persists_across_instances(self) -> None:
        url = "https://flaky.example.org/page"
        SourceTrust(self.context).feedback(url, ok=False)
        # a NEW instance (simulating a restart) still sees the penalty
        tier, _ = domain_tier(url)
        self.assertLess(SourceTrust(self.context).score(url), tier)

    def test_successes_claw_back(self) -> None:
        url = "https://flaky.example.org/page"
        t = SourceTrust(self.context)
        t.feedback(url, ok=False)
        sank = t.score(url)
        for _ in range(3):
            t.feedback(url, ok=True)
        self.assertGreater(t.score(url), sank)

    def test_annotate_idempotent_and_ranked_stable(self) -> None:
        t = SourceTrust(self.context)
        results = [{"url": "https://www.reddit.com/r/x", "title": "forum"},
                   {"url": "https://www.reuters.com/y", "title": "press"},
                   {"url": "https://example.com/z", "title": "web"}]
        t.annotate(results)
        first = [r["trust"] for r in results]
        t.annotate(results)  # idempotent
        self.assertEqual(first, [r["trust"] for r in results])
        self.assertTrue(all("trust_note" in r for r in results))
        ranked = t.ranked(results)
        self.assertEqual(ranked[0]["title"], "press")  # highest tier first
        # within-band order preserved: forum+web stay below press,
        # original relative order kept
        self.assertEqual([r["title"] for r in ranked[1:]], ["forum", "web"])


# ── 3. trust wired into the search engine ───────────────────────────────────


class SearchTrustIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-search-trust-")
        self.context = _make_context(self.tmp)
        self.context.router = _FakeRouter()
        self.engine = SearchEngine(self.context)
        self.engine.search = lambda sub, max_results=8: [
            {"url": "https://www.gov.uk/topic", "title": "Gov guide",
             "snippet": "official guidance"},
            {"url": "https://www.reddit.com/r/topic", "title": "Thread",
             "snippet": "what people say"},
        ]
        self.reads: dict[str, int] = {}
        self.engine.read = self._canned_read

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _canned_read(self, url, max_chars=40000):
        self.reads[url] = self.reads.get(url, 0) + 1
        if url.endswith("reddit.com/r/topic") and self.reads[url] == 1:
            return None  # the forum link dies on the first read
        return {"url": url, "title": url.split("/")[2],
                "domain": url.split("/")[2], "text": "content " * 40,
                "chars": 320}

    def test_results_carry_trust(self) -> None:
        report = self.engine.run("some topic", mode="quick", pages=3)
        for r in report["results"]:
            self.assertIn("trust", r)
            self.assertIn("trust_note", r)
        by_url = {r["url"]: r for r in report["results"]}
        self.assertGreater(by_url["https://www.gov.uk/topic"]["trust"],
                           by_url["https://www.reddit.com/r/topic"]["trust"])

    def test_failed_read_earns_persistent_penalty(self) -> None:
        self.engine.run("some topic", mode="quick", pages=3)
        t = SourceTrust(self.context)
        self.assertLess(t.score("https://www.reddit.com/r/topic"), 0.4)
        # the gov link, which delivered, keeps its tier
        self.assertGreaterEqual(t.score("https://www.gov.uk/topic"), 0.9)


# ── 4. permanently-on reasoning ─────────────────────────────────────────────


class ReasoningAlwaysOnTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-reason-")
        self.context = _make_context(self.tmp)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def test_default_mode_is_always(self) -> None:
        self.assertEqual(self.context.settings.reasoning_mode, "always")
        from nomorals.agents.reasoning import reasoning_enabled

        self.assertTrue(reasoning_enabled(self.context))

    def test_off_mode_still_respected(self) -> None:
        self.context.settings.reasoning_mode = "off"
        from nomorals.agents.reasoning import reasoning_enabled

        self.assertFalse(reasoning_enabled(self.context))

    def test_auto_mode_keys_on_complexity(self) -> None:
        self.context.settings.reasoning_mode = "auto"
        from nomorals.agents.reasoning import reasoning_enabled

        self.assertFalse(reasoning_enabled(self.context, complex_ok=False))
        self.assertTrue(reasoning_enabled(self.context, complex_ok=True))


# ── 5. the reasoning critique in the research loop ─────────────────────────


class ResearchCritiqueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-research-")
        self.context = _make_context(self.tmp)
        self.agent = ResearchAgent(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _fake_cycle(self, *, verdict: str):
        """One run_cycle with canned search + score + critique."""
        from nomorals.agents.reasoning import ReasoningAgent

        report = {"summary": "a digest of the topic with some signal",
                  "pages_read": ["https://example.com/1"],
                  "results": [{"url": "https://example.com/1",
                               "title": "t"}],
                  "model_summary": False}
        scores = {"total": 0.9, "novelty": 0.8, "actionability": 0.9,
                  "signal": 0.9, "freshness": 0.9}
        with mock.patch.object(SearchEngine, "run",
                               return_value=report) as run_mock, \
             mock.patch("nomorals.agents.researcher.score_idea",
                        return_value=scores), \
             mock.patch.object(ReasoningAgent, "advise",
                               return_value=verdict) as advise_mock:
            out = self.agent.run_cycle(domain="tech")
        return out, run_mock, advise_mock

    def test_ok_verdict_passes(self) -> None:
        out, _run, advise = self._fake_cycle(verdict="ok")
        self.assertTrue(out["ok"])
        self.assertEqual(out["status"], "pending")  # no notifier wired → pending
        self.assertIn("ok", advise.call_args[0][0] or advise.call_args.kwargs.get("decision", ""))

    def test_weak_verdict_demotes(self) -> None:
        out, _run, _advise = self._fake_cycle(
            verdict="too vague and generic — no concrete action")
        self.assertTrue(out["ok"])
        # the -0.2 penalty must land in the stored score detail
        row = self.context.db.query_one(
            "SELECT score_detail FROM research_log ORDER BY created_at DESC "
            "LIMIT 1")
        import json

        detail = json.loads(row["score_detail"])
        self.assertIn("reasoning_note", detail)
        self.assertAlmostEqual(detail["total"], 0.7, places=3)


# ── 6. course correction in the project runner ─────────────────────────────


class ProjectCourseCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-projects-")
        self.context = _make_context(self.tmp)
        self.manager = ProjectManager(self.context)

    def tearDown(self) -> None:
        self.context.close()
        self.tmp.cleanup()

    def _make_project(self) -> str:
        p = self.manager.create("do the thing", "do the thing",
                                steps=["the fragile step"])
        return p.id

    def test_course_pivot_empty_when_reasoning_off(self) -> None:
        self.context.settings.reasoning_mode = "off"
        self.assertEqual(self.manager._course_pivot("step", "err", 1), {})

    def test_failed_step_follows_the_pivot(self) -> None:
        pid = self.make_project_id()
        pivot = {"should_pivot": True, "pivot": "use the sandboxed runner",
                 "rationale": "direct execution keeps failing"}
        with mock.patch.object(self.manager, "_course_pivot",
                               return_value=pivot), \
             mock.patch.object(self.manager, "_revise_step",
                               wraps=self.manager._revise_step) as rev:
            self.manager.advance(pid, executor=lambda d: (_ for _ in ()).throw(
                RuntimeError("boom")))
        # the step was reworded FOLLOWING the pivot (hint passed through)
        _args, kwargs = rev.call_args
        self.assertEqual(kwargs.get("hint", _args[2] if len(_args) > 2 else ""),
                         "use the sandboxed runner")
        p = self.manager._load(pid)
        self.assertEqual(p.status, "running")
        self.assertEqual(p.steps[0].status, "pending")
        self.assertTrue(p.steps[0].revised)

    def test_stop_verdict_abandons_early(self) -> None:
        pid = self.make_project_id()
        pivot = {"should_pivot": True, "pivot": "stop — infeasible",
                 "rationale": "three different families already failed"}
        with mock.patch.object(self.manager, "_course_pivot",
                               return_value=pivot):
            self.manager.advance(pid, executor=lambda d: (_ for _ in ()).throw(
                RuntimeError("boom")))
        p = self.manager._load(pid)
        # one attempt burned, but the project gave up EARLY (attempts < 3)
        self.assertEqual(p.status, "failed")
        self.assertEqual(p.steps[0].status, "failed")
        self.assertEqual(p.steps[0].attempts, 1)
        self.assertIn("abandoned", p.steps[0].result)

    def make_project_id(self) -> str:
        return self._make_project()


# ── 7. advanced browsing: the trust-ranked multi-page walk ───────────────────


class BrowserWalkTests(unittest.TestCase):
    """A canned 4-page site: the walk must follow relevance + trust, stay
    in-domain by default, and skip low-reputation off-domain links."""

    def setUp(self) -> None:
        import copy

        self.SITE = copy.deepcopy(self._SITE)

    _SITE = {
        "https://site.example/guide": {
            "title": "The Guide",
            "text": "The guide explains the fundamentals of the system.",
            "links": [
                ("https://site.example/advanced", "advanced techniques"),
                ("https://spam.xyz/paid", "cheap deals inside"),
                ("https://other.org/report", "independent report"),
            ],
        },
        "https://site.example/advanced": {
            "title": "Advanced Techniques",
            "text": "Advanced techniques for the system, in depth.",
            "links": [
                ("https://site.example/guide", "back to the guide"),
                ("https://other.org/report", "the report"),
            ],
        },
        "https://other.org/report": {
            "title": "Independent Report",
            "text": "The report measured the system across twelve sites.",
            "links": [],
        },
        "https://spam.xyz/paid": {
            "title": "BUY NOW",
            "text": "click click click",
            "links": [],
        },
    }

    def _session(self) -> Any:
        from nomorals.tools.browser import BrowserSession, parse_html

        session = BrowserSession()
        opened: list[str] = []

        def fake_open(url: str = "", **_kw: Any) -> dict[str, Any]:
            if url not in self.SITE:
                raise RuntimeError(f"404 for {url}")
            page = self.SITE[url]
            opened.append(url)
            body = page["text"]
            anchor = "".join(f'<a href="{u}">{t}</a>' for u, t in page["links"])
            session.url = url
            session.title = page["title"]
            session._raw = body
            session.dom = parse_html(
                f"<html><head><title>{page['title']}</title></head>"
                f"<body><p>{body}</p>{anchor}</body></html>")
            if url not in session._history:
                session._history.append(url)
            return {"ok": True, "url": url, "status": 200,
                    "title": page["title"], "chars": len(body)}

        session.open = fake_open
        session.opened = opened
        return session

    def test_in_domain_walk_follows_relevance(self) -> None:
        session = self._session()
        out = session.walk("https://site.example/guide", max_pages=4,
                           focus="advanced techniques")
        self.assertTrue(out["ok"])
        urls = [p["url"] for p in out["pages"]]
        # the relevance hit (advanced techniques) is the FIRST hop, and the
        # walk never left the seed domain
        self.assertEqual(urls[0], "https://site.example/guide")
        self.assertEqual(urls[1], "https://site.example/advanced")
        for u in urls:
            self.assertTrue(u.startswith("https://site.example/"))
        for p in out["pages"]:
            self.assertIn("trust", p)
            self.assertTrue(p["trust"] > 0)
        self.assertIn("Advanced Techniques", out["digest"])

    def test_off_domain_only_from_credible_sources(self) -> None:
        session = self._session()
        out = session.walk("https://site.example/guide", max_pages=4,
                           in_domain=False)
        urls = [p["url"] for p in out["pages"]]
        # other.org (0.55) is allowed off-domain; spam.xyz (0.20) never is
        self.assertIn("https://other.org/report", urls)
        self.assertNotIn("https://spam.xyz/paid", urls)
        # the walk still terminates (finite site, no revisits)
        self.assertLessEqual(out["count"], 4)

    def test_dead_link_does_not_kill_the_walk(self) -> None:
        session = self._session()
        # the only remaining link from the advanced page is a dead one
        self.SITE["https://site.example/advanced"]["links"] = [
            ("https://broken.example/nope", "the only way out")]
        out = session.walk("https://site.example/guide", max_pages=3,
                           focus="advanced", in_domain=False)
        self.assertTrue(out["ok"])
        unreadable = [p for p in out["pages"]
                      if p["url"] == "https://broken.example/nope"]
        self.assertTrue(unreadable)
        self.assertIn("unreadable", unreadable[0]["excerpt"])


if __name__ == "__main__":
    unittest.main()
