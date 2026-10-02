"""Wave G2 — ticket quality: schema, gate, boilerplate detection, audit.

The strengthened ``gate_ticket`` requires every upgrade ticket to carry
a substantive ``problem`` (what's wrong, with evidence), a
``proposed_change`` (the approach, tied to concrete files), a ``risk``
(honest blast-radius/reversibility assessment), and ``verify_steps``
(concrete tests/commands proving the fix — distinct from the
acceptance_criteria done-definition). Long-form fields are checked for
boilerplate: thin content, title restatements, and banned filler
phrases are all rejected with specific reasons.

The audit section builds 8 tickets from REAL repo observations (all
verified by reading code before the claims were written) through the
real ``ResearchDigest.upgrade_ticket`` pipeline and asserts each passes
the strengthened gate WITH substance — not just field presence.
"""
import unittest

from nomorals.agents.research_digest import (
    ResearchClaim,
    ResearchDigest,
    _content_words,
    _filler_hits,
    _substance_reasons,
    gate_ticket,
)
from nomorals.core.ids import new_id


def _claim(cid, text, domain="general", confidence=0.85, query="g2 audit",
           angle="code reading"):
    return ResearchClaim(
        id=cid, claim=text, domain=domain, angle=angle,
        sources=[{"url": "https://example.com/src",
                  "title": "repo code read", "trust": 0.9}],
        confidence=confidence, query=query)


def _concrete_ticket(**overrides):
    ticket = ResearchDigest.upgrade_ticket(
        _claim("seed", "apply_unified_diff is implemented in three separate "
                       "modules with no shared canonical version"))
    ticket.update(overrides)
    return ticket


# ── boilerplate detection unit tests ──────────────────────────────────────

class FillerDetectionTests(unittest.TestCase):
    def test_filler_phrases_are_caught(self):
        for phrase in ("improve code quality", "add tests",
                       "enhance functionality", "as described above",
                       "make it better"):
            self.assertTrue(_filler_hits(
                f"We should {phrase} across the whole system eventually"),
                phrase)

    def test_honest_unknown_is_not_filler(self):
        # an honest "unknown — needs human review" is substance, not filler
        self.assertEqual(_filler_hits(
            "Blast radius: unknown — the file map is heuristic, so this "
            "needs human review before any edit"), [])

    def test_substantive_text_has_no_reasons(self):
        self.assertEqual(_substance_reasons(
            "problem",
            "Redis connection pooling is missing in the cache layer: "
            "nomorals/storage/cache.py opens a fresh TCP connection per "
            "lookup, which adds 2ms latency per call under load",
            _content_words("cache latency ticket")), [])

    def test_thin_text_rejected(self):
        reasons = _substance_reasons("problem", "It is broken and bad",
                                     _content_words("ticket"))
        self.assertTrue(any("too vague" in r for r in reasons), reasons)

    def test_title_restatement_rejected(self):
        title = "Cache lookups are slow under load"
        # problem that just rephrases the title carries no novel content
        problem = (title + " which means the cache lookups are slow "
                           "when under load")
        reasons = _substance_reasons(
            "problem", problem, _content_words(title))
        self.assertTrue(any("restates" in r for r in reasons), reasons)


# ── gate: new-field rejections ─────────────────────────────────────────────

class GateNewFieldsTests(unittest.TestCase):
    def test_missing_problem(self):
        ticket = _concrete_ticket(problem="")
        problems = gate_ticket(ticket)
        self.assertTrue(any("problem statement too vague" in p
                            for p in problems), problems)

    def test_boilerplate_problem_rejected(self):
        ticket = _concrete_ticket(
            problem="This change will improve code quality and add tests "
                    "for better performance across the modules involved.")
        problems = gate_ticket(ticket)
        self.assertTrue(any("boilerplate" in p for p in problems), problems)

    def test_restated_problem_rejected(self):
        ticket = _concrete_ticket()
        title = ticket["title"]
        ticket["problem"] = (title + " — yes, that is the problem, "
                                    "as described above.")
        problems = gate_ticket(ticket)
        self.assertTrue(any("restates" in p or "boilerplate" in p
                            for p in problems), problems)

    def test_missing_proposed_change(self):
        ticket = _concrete_ticket(proposed_change="")
        problems = gate_ticket(ticket)
        self.assertTrue(any("no proposed approach" in p for p in problems),
                        problems)

    def test_proposed_change_must_name_a_file(self):
        ticket = _concrete_ticket(
            proposed_change="Approach: encode the claim as a failing test "
                            "first, then implement the smallest change that "
                            "makes it pass, re-running the module suite.")
        problems = gate_ticket(ticket)
        self.assertTrue(any("names no suggested file" in p for p in problems),
                        problems)

    def test_boilerplate_proposed_change_rejected(self):
        ticket = _concrete_ticket(
            proposed_change="We will refactor code in nomorals/tools/repo_index.py "
                            "to improve code quality and enhance functionality.")
        problems = gate_ticket(ticket)
        self.assertTrue(any("boilerplate" in p for p in problems), problems)

    def test_missing_risk(self):
        ticket = _concrete_ticket(risk="")
        problems = gate_ticket(ticket)
        self.assertTrue(any("no risk assessment" in p for p in problems),
                        problems)

    def test_thin_risk_rejected(self):
        ticket = _concrete_ticket(risk="Should be fine, low risk.")
        problems = gate_ticket(ticket)
        self.assertTrue(problems, "thin risk must be rejected")

    def test_missing_verify_steps(self):
        ticket = _concrete_ticket(verify_steps=[])
        problems = gate_ticket(ticket)
        self.assertTrue(any("no verify steps" in p for p in problems),
                        problems)

    def test_verify_steps_without_concrete_test_rejected(self):
        ticket = _concrete_ticket(
            verify_steps=["Check that everything works well and looks "
                          "correct in the running system end to end"])
        problems = gate_ticket(ticket)
        self.assertTrue(any("no concrete test/command" in p for p in problems),
                        problems)

    def test_vague_verify_step_rejected(self):
        ticket = _concrete_ticket(verify_steps=["run tests"])
        problems = gate_ticket(ticket)
        self.assertTrue(any("verify step too vague" in p for p in problems),
                        problems)

    def test_old_ticket_gets_clear_new_rejections(self):
        # backward compat: old tickets (no new fields) fail loudly with
        # specific reasons instead of crashing anything downstream
        old = {
            "title": "[systems] cache lookups are slow under load",
            "rationale": "research finding: the cache layer opens a fresh "
                         "connection per lookup",
            "domain": "systems",
            "suggested_files": ["nomorals/core/logging_setup.py"],
            "expected_tests": ["test_cache_pool_reuses_connection"],
            "acceptance_criteria": ["test passes"],
            "claim_ids": ["c-old"],
        }
        problems = gate_ticket(old)
        joined = " ".join(problems)
        self.assertIn("problem statement too vague", joined)
        self.assertIn("no proposed approach", joined)
        self.assertIn("no risk assessment", joined)
        self.assertIn("no verify steps", joined)


# ── upgrade_ticket populates the new fields with substance ─────────────────

class UpgradeTicketSubstanceTests(unittest.TestCase):
    def test_all_domains_populate_substantive_fields(self):
        for domain in ("security", "competitors", "ml", "systems",
                       "tooling", "product", "general"):
            claim = _claim(
                new_id("c"), "Observed behavior in this area needs an "
                             "encoded test and a minimal owning-module fix "
                             "so regressions are caught",
                domain=domain, confidence=0.82)
            ticket = ResearchDigest.upgrade_ticket(claim)
            self.assertEqual(gate_ticket(ticket), [], domain)
            baseline = _content_words(
                ticket["title"] + " " + ticket["rationale"])
            for field in ("problem", "proposed_change", "risk"):
                text = ticket[field]
                self.assertEqual(_filler_hits(text), [], f"{domain}:{field}")
                self.assertEqual(
                    _substance_reasons(field, text, baseline), [],
                    f"{domain}:{field}")
            steps = ticket["verify_steps"]
            self.assertGreaterEqual(len(steps), 2, domain)
            joined = " ".join(steps)
            self.assertRegex(joined, r"test_[a-z0-9_]+|pytest|error_scan")

    def test_risk_is_honest_not_invented(self):
        ticket = ResearchDigest.upgrade_ticket(
            _claim(new_id("c"), "Some claim about cache behavior",
                   domain="systems"))
        risk = ticket["risk"].lower()
        self.assertIn("unknown", risk)  # honest about the heuristic map
        self.assertIn("revert", risk)   # reversibility stated
        self.assertIn("human review", risk)

    def test_problem_carries_evidence(self):
        ticket = ResearchDigest.upgrade_ticket(
            _claim(new_id("c"), "Redis is the fastest in-memory cache",
                   domain="systems", query="cache survey", angle="core",
                   confidence=0.9))
        problem = ticket["problem"]
        self.assertIn("cache survey", problem)
        self.assertIn("90%", problem)
        self.assertIn("repo code read", problem)  # source title
        self.assertIn("nomorals/", problem)       # tied to real files


# ── audit: 8 real repo observations → tickets → gate with substance ────────
#
# Every claim below was verified by reading the repo first:
#   1. apply_unified_diff duplicated in 3 modules (grep: def apply_unified_diff)
#   2. estimate_tokens reimplemented in 3 modules (grep: def estimate_tokens)
#   3. build_repo_map in 2 modules (grep: def build_repo_map)
#   4. nomorals/native/ ships prebuilt .so files (ls: libbpe.so, libmemextract.so)
#   5. error_scan is "descriptive, not a gate" (its own module docstring)
#   6. propose_from_ticket drops the ticket's risk field (upgrade_queue.py:226-243)
#   7. OpenRouter catalog pull lives in memory notes, not the repo (memory)
#   8. 328/348 modules lack an obviously-named test file (measured via os.walk)

_AUDIT_CLAIMS = [
    ("tooling",
     "apply_unified_diff is implemented in three separate modules "
     "(nomorals/agents/patch.py, nomorals/agents/skill_evolution.py, "
     "nomorals/core/diff.py) with no shared canonical version — a "
     "diff-parsing bug fixed in one copy stays broken in the other two."),
    ("ml",
     "estimate_tokens is reimplemented in nomorals/llm/base.py, "
     "nomorals/tools/repo_index.py and nomorals/tools/repo_context.py — "
     "three token-counting heuristics that can disagree on context "
     "budgets, so one module's 'fits' is another's overflow."),
    ("tooling",
     "build_repo_map exists in both nomorals/agents/repo_map.py and "
     "nomorals/tools/repo_index.py — two repo-map builders maintained "
     "separately that can drift out of sync on ignore rules and "
     "entry formats."),
    ("systems",
     "nomorals/native/ ships prebuilt shared objects (libbpe.so, "
     "libmemextract.so were found on disk) with no recorded test proving "
     "the pure-Python fallback path works when a .so fails to load on an "
     "unsupported platform."),
    ("security",
     "error_scan detects swallowed exceptions (E103) but its own "
     "docstring says it is 'descriptive, not a gate: it reports, it never "
     "rewrites' — a swallowed exception in an auth/trust path "
     "(nomorals/core/trust.py, nomorals/core/policy.py) can hide a "
     "security failure with no blocking check anywhere in the pipeline."),
    ("product",
     "/upgrade show renders risk as 'unstated in the ticket' when the "
     "plan carries none (upgrade_chat.py), and propose_from_ticket only "
     "queues title/rationale/patch_plan/files/tests/claim_ids — the "
     "ticket's risk field is dropped on the floor, so approved proposals "
     "never show the blast radius the ticket author wrote."),
    ("competitors",
     "The 2026-10-01 OpenRouter catalog pull (464 models, 21 free, zero "
     "uncensored) lives only in memory notes, not in the repo — the bot "
     "brain fallback chain (HF -> Groq -> OpenRouter) can go stale "
     "silently when the free catalog changes."),
    ("general",
     "328 of 348 repo modules have no obviously-named test file "
     "(measured by module↔test filename pairing over nomorals/) — "
     "coverage gaps are invisible because nothing in the repo measures "
     "which modules lack tests."),
]


def _assert_ticket_substance(testcase, ticket, label):
    problems = gate_ticket(ticket)
    testcase.assertEqual(problems, [], f"{label}: {problems}")
    baseline = _content_words(ticket["title"] + " " + ticket["rationale"])
    for field in ("problem", "proposed_change", "risk"):
        text = str(ticket[field])
        testcase.assertEqual(_filler_hits(text), [],
                             f"{label}:{field} has filler")
        testcase.assertGreaterEqual(len(_content_words(text)), 5,
                                    f"{label}:{field} too thin")
        testcase.assertGreaterEqual(
            len(_content_words(text) - baseline), 3,
            f"{label}:{field} restates title/rationale")
    steps = ticket["verify_steps"]
    testcase.assertGreaterEqual(len(steps), 2, f"{label}: steps")
    for s in steps:
        testcase.assertGreaterEqual(len(s), 12, f"{label}: step {s!r}")
    joined = " ".join(steps)
    testcase.assertRegex(joined, r"test_[a-z0-9_]+|pytest|error_scan",
                         f"{label}: no concrete test/command in steps")


class AuditRealTicketsTests(unittest.TestCase):
    def test_audit_tickets_pass_gate_with_substance(self):
        for i, (domain, text) in enumerate(_AUDIT_CLAIMS):
            with self.subTest(i=i, domain=domain):
                claim = _claim(new_id("audit"), text, domain=domain,
                               confidence=0.85,
                               query="repo code reading audit",
                               angle="weakness scan")
                ticket = ResearchDigest.upgrade_ticket(claim)
                _assert_ticket_substance(self, ticket, f"audit-{i}")
                # the ticket still carries the original schema fields
                for key in ("title", "rationale", "domain", "suggested_files",
                            "verified_files", "expected_tests",
                            "acceptance_criteria", "test_plan", "patch_plan",
                            "claim_ids", "confidence"):
                    self.assertIn(key, ticket, f"audit-{i}:{key}")


if __name__ == "__main__":
    unittest.main()
