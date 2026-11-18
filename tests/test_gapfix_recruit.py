"""Lane 78 — the recruitment module (B42): per-vacancy jobs + tiering + /status board.

Pins:

* the recruitment store (``nomorals/recruitment/store.py``): per-vacancy
  independent jobs — each vacancy carries its OWN queue, scraper state,
  stats and CV flow; candidates in one vacancy never leak into another's
  list or stats; intake runs the tiering rules and records the tier +
  the reasons; duplicates (same contact/CV in one vacancy) are refused
  as the existing candidate, never queued twice; a closed vacancy
  refuses new candidates in words; persistence round-trips through the
  careers-shape JSON store (storeguard read, atomic write);
* tiering (``nomorals/recruitment/tiering.py``): pure rules in code —
  CV + skills overlap + experience + contact + location make a 0–100
  score; bands A/B/C/D; no contact is always D with the reason named;
  no CV caps at C with the reason named; every tier carries its
  deciding reasons, never a bare letter;
* the scraper (``nomorals/recruitment/scraper.py``): fetches the
  vacancy's source through the house HttpClient seam; JSON feeds and
  plain HTML applicant lists both intake real candidates; a re-scrape
  intakes only the new ones; an unreachable source, a non-2xx status
  or an unparseable body records a FAILED scrape with the reason —
  never fabricates a candidate;
* surfaces: the /status board shows every vacancy with queue depth,
  tier counts and scrape state live from the store; an empty module
  renders a designed empty state; vacancy resolution is id / #n /
  unique substring, mirroring /careers;
* the catalog: /recruit registered everywhere /careers is, and
  owner-only like it; /status itself stays her live-state command —
  the recruitment board is /recruit status, not a hijack of /status;
* the brain tool: ``recruit`` registers (status/list/add/scrape/
  candidate/tier) and the status action answers from the real store;
* the chat surface: /recruit add → candidate → status board end to
  end through a real PartnerRuntime, and a scrape through an injected
  fake client lands real candidates on the board.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.recruitment import surface as recruit_surface
from nomorals.recruitment.store import (
    Candidate,
    RecruitmentError,
    RecruitmentStore,
    Vacancy,
)
from nomorals.recruitment.tiering import tier_candidate
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    parse_control,
)

UID = "owner"
REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = 86400.0
T0 = 1_700_000_000.0

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402


def _vacancy(**kwargs: Any) -> Vacancy:
    base: dict[str, Any] = {
        "id": "vac_test",
        "title": "Senior Developer",
        "employer": "Acme",
        "deadline": None,
        "source_url": "",
        "source_kind": "auto",
        "required_skills": ["python", "postgres"],
        "min_years": 3,
        "location": "Lagos",
        "status": "open",
        "created_at": T0,
        "updated_at": T0,
        "last_scrape_at": None,
        "last_scrape_result": "never",
        "last_scrape_error": "",
        "scraped_count": 0,
    }
    base.update(kwargs)
    return Vacancy(**base)


def _candidate(**kwargs: Any) -> Candidate:
    base: dict[str, Any] = {
        "id": "cand_test",
        "vacancy_id": "vac_test",
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "",
        "location": "Lagos",
        "years": 5.0,
        "skills": ["python", "postgres"],
        "cv_text": "5 years building backends.",
        "cv_artifact_id": "",
        "cv_url": "",
        "source": "manual",
        "score": 0.0,
        "tier": "",
        "tier_reasons": [],
        "history": [],
        "created_at": T0,
        "updated_at": T0,
    }
    base.update(kwargs)
    return Candidate(**base)


class _StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.store = RecruitmentStore(
            Path(self._tmp.name) / "vacancies.json")

    def tearDown(self) -> None:
        self._tmp.cleanup()


class StoreTests(_StoreCase):
    def test_add_vacancy_defaults(self) -> None:
        vac = self.store.add_vacancy(
            "Senior Developer", "Acme", now=T0)
        self.assertTrue(vac.id.startswith("vac_"))
        self.assertEqual(vac.status, "open")
        self.assertEqual(vac.last_scrape_result, "never")
        self.assertEqual(vac.required_skills, [])
        self.assertEqual(self.store.get_vacancy(vac.id), vac)

    def test_add_vacancy_requires_title_and_employer(self) -> None:
        with self.assertRaises(RecruitmentError):
            self.store.add_vacancy("", "Acme", now=T0)
        with self.assertRaises(RecruitmentError):
            self.store.add_vacancy("Dev", "", now=T0)

    def test_vacancies_are_independent_jobs(self) -> None:
        first = self.store.add_vacancy("Dev", "Acme", now=T0)
        second = self.store.add_vacancy("Designer", "Beta", now=T0)
        self.store.add_candidate(
            first.id, name="Ada", email="ada@example.com",
            skills=["python"], years=5, cv_text="cv", now=T0)
        self.store.add_candidate(
            second.id, name="Bob", email="bob@example.com",
            skills=["figma"], years=4, cv_text="cv", now=T0)
        self.assertEqual(
            [c.name for c in self.store.list_candidates(first.id)],
            ["Ada"])
        self.assertEqual(
            [c.name for c in self.store.list_candidates(second.id)],
            ["Bob"])
        self.assertEqual(
            self.store.stats(first.id)["queue_depth"], 1)
        self.assertEqual(
            self.store.stats(second.id)["queue_depth"], 1)

    def test_intake_tiers_and_records_reasons(self) -> None:
        vac = self.store.add_vacancy(
            "Dev", "Acme", required_skills=["python", "postgres"],
            min_years=3, location="Lagos", now=T0)
        cand, is_new = self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            location="Lagos", years=5,
            skills=["python", "postgres"], cv_text="5y backends",
            now=T0)
        self.assertTrue(is_new)
        self.assertEqual(cand.tier, "A")
        self.assertGreaterEqual(cand.score, 75)
        self.assertTrue(cand.tier_reasons)
        self.assertTrue(any("CV" in r for r in cand.tier_reasons))
        self.assertEqual(cand.history[0]["event"], "intake")
        self.assertEqual(cand.history[-1]["event"], "tiered")

    def test_duplicate_contact_is_the_same_candidate(self) -> None:
        vac = self.store.add_vacancy("Dev", "Acme", now=T0)
        first, new1 = self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            cv_text="cv", now=T0)
        second, new2 = self.store.add_candidate(
            vac.id, name="Ada L.", email="ADA@example.com",
            cv_text="cv again", now=T0)
        self.assertTrue(new1)
        self.assertFalse(new2)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(self.store.list_candidates(vac.id)), 1)

    def test_closed_vacancy_refuses_new_candidates(self) -> None:
        vac = self.store.add_vacancy("Dev", "Acme", now=T0)
        self.store.set_vacancy_status(vac.id, "closed", now=T0)
        with self.assertRaises(RecruitmentError):
            self.store.add_candidate(
                vac.id, name="Ada", email="ada@example.com",
                cv_text="cv", now=T0)

    def test_scrape_state_is_per_vacancy(self) -> None:
        first = self.store.add_vacancy("Dev", "Acme", now=T0)
        second = self.store.add_vacancy("Designer", "Beta", now=T0)
        self.store.record_scrape(
            first.id, "ok", error="", new_count=3, now=T0)
        self.store.record_scrape(
            second.id, "failed", error="connection refused", new_count=0,
            now=T0)
        one = self.store.get_vacancy(first.id)
        two = self.store.get_vacancy(second.id)
        assert one is not None and two is not None
        self.assertEqual(one.last_scrape_result, "ok")
        self.assertEqual(one.scraped_count, 3)
        self.assertEqual(two.last_scrape_result, "failed")
        self.assertEqual(two.last_scrape_error, "connection refused")

    def test_stats_count_tiers(self) -> None:
        vac = self.store.add_vacancy(
            "Dev", "Acme", required_skills=["python"], min_years=3,
            now=T0)
        self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            skills=["python"], years=5, cv_text="cv", now=T0)
        self.store.add_candidate(
            vac.id, name="NoContact", email="", phone="",
            skills=[], years=0, cv_text="", now=T0)
        stats = self.store.stats(vac.id)
        self.assertEqual(stats["queue_depth"], 2)
        self.assertEqual(stats["tier_counts"]["A"], 1)
        self.assertEqual(stats["tier_counts"]["D"], 1)

    def test_persistence_round_trips(self) -> None:
        vac = self.store.add_vacancy(
            "Dev", "Acme", required_skills=["python"], now=T0)
        cand, _ = self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            skills=["python"], years=5, cv_text="cv", now=T0)
        reopened = RecruitmentStore(
            Path(self._tmp.name) / "vacancies.json")
        again = reopened.get_vacancy(vac.id)
        assert again is not None
        self.assertEqual(again.title, "Dev")
        kept = reopened.list_candidates(vac.id)
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0].id, cand.id)
        self.assertEqual(kept[0].tier, cand.tier)
        self.assertEqual(kept[0].tier_reasons, cand.tier_reasons)

    def test_remove_vacancy_drops_its_queue(self) -> None:
        vac = self.store.add_vacancy("Dev", "Acme", now=T0)
        self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            cv_text="cv", now=T0)
        self.assertTrue(self.store.remove_vacancy(vac.id))
        self.assertIsNone(self.store.get_vacancy(vac.id))
        self.assertEqual(self.store.list_candidates(vac.id), [])
        self.assertFalse(self.store.remove_vacancy(vac.id))


class TieringTests(unittest.TestCase):
    def test_strong_candidate_is_a_with_named_reasons(self) -> None:
        result = tier_candidate(_candidate(), _vacancy())
        self.assertEqual(result.tier, "A")
        self.assertGreaterEqual(result.score, 75)
        self.assertTrue(any("CV" in r for r in result.reasons))
        self.assertTrue(any("python" in r for r in result.reasons))

    def test_no_contact_is_always_d(self) -> None:
        cand = _candidate(email="", phone="")
        result = tier_candidate(cand, _vacancy())
        self.assertEqual(result.tier, "D")
        self.assertTrue(
            any("no contact" in r for r in result.reasons))

    def test_no_cv_caps_at_c(self) -> None:
        cand = _candidate(cv_text="")
        result = tier_candidate(cand, _vacancy())
        self.assertIn(result.tier, ("C", "D"))
        self.assertTrue(any("no CV" in r for r in result.reasons))

    def test_skills_fraction_moves_the_score(self) -> None:
        full = tier_candidate(_candidate(), _vacancy())
        half = tier_candidate(
            _candidate(skills=["python"]), _vacancy())
        none = tier_candidate(_candidate(skills=[]), _vacancy())
        self.assertGreater(full.score, half.score)
        self.assertGreater(half.score, none.score)

    def test_junior_for_senior_role_scores_lower(self) -> None:
        senior = tier_candidate(_candidate(years=5), _vacancy())
        junior = tier_candidate(_candidate(years=1), _vacancy())
        self.assertGreater(senior.score, junior.score)
        # Experience alone moves the tier when the rest is thin: a
        # 1y candidate with no matching skills drops below the
        # 5y candidate's band, not just its score.
        thin_junior = tier_candidate(
            _candidate(years=1, skills=[]), _vacancy())
        self.assertNotEqual(senior.tier, thin_junior.tier)


class _FakeResponse:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300


class _FakeClient:
    def __init__(self, response: Any = None,
                 error: BaseException | None = None) -> None:
        self.response = response
        self.error = error
        self.seen_urls: list[str] = []

    def get(self, url: str, **kwargs: Any) -> Any:
        self.seen_urls.append(url)
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class ScraperTests(_StoreCase):
    def test_json_feed_intakes_real_candidates(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/acme-dev.json",
            source_kind="json", required_skills=["python"],
            now=T0)
        feed = json.dumps({"candidates": [
            {"name": "Ada", "email": "ada@example.com",
             "skills": ["python"], "years": 5,
             "cv_text": "5y backends"},
            {"name": "Bob", "email": "bob@example.com",
             "skills": ["go"], "years": 2, "cv_text": "2y systems"},
        ]})
        out = scrape_vacancy(
            self.store, vac, client=_FakeClient(
                _FakeResponse(200, feed)), now=T0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["new"], 2)
        names = [c.name for c in self.store.list_candidates(vac.id)]
        self.assertEqual(names, ["Ada", "Bob"])
        again = self.store.get_vacancy(vac.id)
        assert again is not None
        self.assertEqual(again.last_scrape_result, "ok")
        self.assertEqual(again.scraped_count, 2)

    def test_rescrape_intakes_only_the_new(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/feed.json",
            source_kind="json", now=T0)
        feed = json.dumps({"candidates": [
            {"name": "Ada", "email": "ada@example.com",
             "cv_text": "cv"},
        ]})
        client = _FakeClient(_FakeResponse(200, feed))
        first = scrape_vacancy(self.store, vac, client=client, now=T0)
        second = scrape_vacancy(
            self.store, vac, client=client, now=T0 + DAY)
        self.assertEqual(first["new"], 1)
        self.assertEqual(second["new"], 0)
        self.assertEqual(second["seen"], 1)
        self.assertEqual(len(self.store.list_candidates(vac.id)), 1)

    def test_html_list_intakes_emailed_candidates(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/acme-dev",
            source_kind="html", now=T0)
        page = (
            "<html><body>"
            "<div class='applicant'><h3>Ada Lovelace</h3>"
            "<a href='mailto:ada@example.com'>ada@example.com</a>"
            "<p>5 years Python</p></div>"
            "<div class='applicant'><h3>Bob Stone</h3>"
            "<a href='mailto:bob@example.com'>bob@example.com</a>"
            "</div></body></html>")
        out = scrape_vacancy(
            self.store, vac, client=_FakeClient(
                _FakeResponse(200, page)), now=T0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["new"], 2)
        names = [c.name for c in self.store.list_candidates(vac.id)]
        self.assertIn("Ada Lovelace", names)
        self.assertIn("Bob Stone", names)

    def test_http_failure_records_failed_never_fabricates(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/feed.json",
            source_kind="json", now=T0)
        out = scrape_vacancy(
            self.store, vac, client=_FakeClient(
                _FakeResponse(404, "not found")), now=T0)
        self.assertFalse(out["ok"])
        self.assertIn("404", out["error"])
        self.assertEqual(self.store.list_candidates(vac.id), [])
        again = self.store.get_vacancy(vac.id)
        assert again is not None
        self.assertEqual(again.last_scrape_result, "failed")
        self.assertIn("404", again.last_scrape_error)

    def test_unreachable_source_records_failed(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/feed.json",
            source_kind="json", now=T0)
        out = scrape_vacancy(
            self.store, vac,
            client=_FakeClient(error=ConnectionError("refused")),
            now=T0)
        self.assertFalse(out["ok"])
        self.assertIn("refused", out["error"])
        self.assertEqual(self.store.list_candidates(vac.id), [])

    def test_no_source_url_refuses_honestly(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy("Dev", "Acme", now=T0)
        out = scrape_vacancy(self.store, vac, client=_FakeClient(
            _FakeResponse(200, "{}")), now=T0)
        self.assertFalse(out["ok"])
        self.assertIn("no source URL", out["error"])

    def test_unparseable_body_records_failed(self) -> None:
        from nomorals.recruitment.scraper import scrape_vacancy

        vac = self.store.add_vacancy(
            "Dev", "Acme",
            source_url="https://jobs.example.com/feed.json",
            source_kind="json", now=T0)
        out = scrape_vacancy(
            self.store, vac, client=_FakeClient(
                _FakeResponse(200, "{not json")), now=T0)
        self.assertFalse(out["ok"])
        self.assertEqual(self.store.list_candidates(vac.id), [])


class SurfaceTests(_StoreCase):
    def test_empty_board_is_a_designed_state(self) -> None:
        board = recruit_surface.render_status_board(
            self.store, now=T0)
        self.assertIn("no vacancies yet", board)
        self.assertIn("/recruit add", board)

    def test_board_shows_depth_tiers_and_scrape_state(self) -> None:
        vac = self.store.add_vacancy(
            "Dev", "Acme", required_skills=["python"], min_years=3,
            source_url="https://jobs.example.com/feed.json",
            now=T0)
        self.store.add_candidate(
            vac.id, name="Ada", email="ada@example.com",
            skills=["python"], years=5, cv_text="cv", now=T0)
        self.store.record_scrape(
            vac.id, "failed", error="connection refused", new_count=0,
            now=T0)
        board = recruit_surface.render_status_board(
            self.store, now=T0)
        self.assertIn("Acme", board)
        self.assertIn("queue 1", board)
        self.assertIn("A 1", board)
        self.assertIn("failed", board)
        self.assertIn("connection refused", board)

    def test_resolve_vacancy(self) -> None:
        vac = self.store.add_vacancy("Dev", "Acme", now=T0)
        found, _ = recruit_surface.resolve_vacancy(self.store, vac.id)
        assert found is not None
        self.assertEqual(found.id, vac.id)
        found, _ = recruit_surface.resolve_vacancy(self.store, "#1")
        assert found is not None
        self.assertEqual(found.id, vac.id)
        found, _ = recruit_surface.resolve_vacancy(self.store, "acme")
        assert found is not None
        self.assertEqual(found.id, vac.id)
        _, cands = recruit_surface.resolve_vacancy(self.store, "zzz")
        self.assertEqual(cands, [vac])


class CatalogTests(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("recruit", CONTROL_COMMANDS)
        self.assertIn("recruit", COMMAND_DETAILS)
        self.assertIn("recruit", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("recruit"), 1)
        self.assertIn("/recruit", COMMAND_DETAILS["recruit"]["usage"])

    def test_owner_only(self) -> None:
        self.assertNotIn("recruit", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("recruit", False))
        self.assertTrue(caller_may_run("recruit", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/recruit add Dev | Acme")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "recruit")
        self.assertEqual(cmd.arg, "add")

    def test_status_stays_her_live_state(self) -> None:
        # The recruitment board is /recruit status — /status itself
        # is her live-state command and must not become the board.
        self.assertIn("status", CONTROL_COMMANDS)
        self.assertIn("live state", COMMAND_DETAILS["status"]["what"])


class ToolTests(unittest.TestCase):
    def test_recruit_tool_registers(self) -> None:
        from nomorals.tools import recruitment as recruit_tool
        from nomorals.tools.registry import ToolRegistry

        reg = ToolRegistry()
        recruit_tool.register(reg)
        self.assertIn("recruit", reg._tools)

    def test_tool_status_answers_from_the_store(self) -> None:
        from nomorals.tools import recruitment as recruit_tool
        from nomorals.tools.registry import ToolRegistry

        with tempfile.TemporaryDirectory(
                dir=os.environ.get("NM_TEST_TMPDIR") or None) as td:
            store = RecruitmentStore(
                Path(td) / "recruitment" / "vacancies.json")
            store.add_vacancy("Dev", "Acme", now=T0)
            reg = ToolRegistry(SimpleNamespace(
                settings=SimpleNamespace(home_path=Path(td))))
            recruit_tool.register(reg)
            out = reg._tools["recruit"].fn(action="status")
            self.assertTrue(out.get("ok"))
            self.assertIn("Acme", out.get("text", ""))


class ChatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-recruit-78-")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        self.store = RecruitmentStore(
            Path(self.tmp.name) / "recruitment" / "vacancies.json")
        self.runtime._recruitment_store = lambda: self.store

    def tearDown(self) -> None:
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        reply = self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))
        return reply.text if hasattr(reply, "text") else reply

    def test_add_candidate_board_end_to_end(self) -> None:
        out = self._run(
            "/recruit add Senior Developer | Acme "
            "| skills: python, postgres | min: 3")
        self.assertIn("Senior Developer", out)
        self.assertIn("Acme", out)
        vac = self.store.list_vacancies()[0]
        out = self._run(
            f"/recruit candidate {vac.id} | Ada Lovelace "
            "| ada@example.com | years: 5 "
            "| skills: python, postgres | cv: 5 years backends")
        self.assertIn("tier A", out)
        board = self._run("/recruit status")
        self.assertIn("Acme", board)
        self.assertIn("queue 1", board)
        self.assertIn("A 1", board)

    def test_scrape_through_injected_client_lands_candidates(self) -> None:
        out = self._run(
            "/recruit add Dev | Acme "
            "| https://jobs.example.com/feed.json")
        vac = self.store.list_vacancies()[0]
        feed = json.dumps({"candidates": [
            {"name": "Ada", "email": "ada@example.com",
             "skills": ["python"], "years": 5, "cv_text": "5y"},
        ]})
        self.runtime._recruitment_http_client = _FakeClient(
            _FakeResponse(200, feed))
        out = self._run(f"/recruit scrape {vac.id}")
        self.assertIn("1 new", out)
        board = self._run("/recruit status")
        self.assertIn("queue 1", board)
        self.assertIn("A 1", board)
        detail = self._run(f"/recruit status {vac.id}")
        self.assertIn("Ada", detail)

    def test_unknown_vacancy_answers_honestly(self) -> None:
        out = self._run("/recruit status zzz-no-such")
        self.assertIn("no single vacancy matches", out)


if __name__ == "__main__":
    unittest.main()
