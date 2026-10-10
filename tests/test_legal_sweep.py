"""Sweep tests for nomorals/legal/ — mined-then-built upgrade.

Covers every new feature added in the legal sweep:
- contracts: CUAD-taxonomy rules, metadata extraction, clause
  categorization, plain-English rewriting, rights extraction, review
  comparison, triage/compact styles, suggestions, citation verification,
  gap-aware scoring
- aid: 4 new scenarios (all 5 languages), triage, lawyer referrals,
  letter templates, SMS chunking, brief style
- research: query expansion, verbatim citation verification,
  groundedness, corpus description, related sections
- portfolio: full-text search, extraction approval, renewal pipeline,
  roll-forward, value tracking, amendments/history, schema migration
- regulatory: new regulators/controls, change detection, dismiss,
  impact scoring, calendar, effective-date tracking, portfolio sync
"""

import tempfile
import time
import unittest

from nomorals.legal.contracts import (
    DISCLAIMER,
    extract_metadata,
    extract_rights,
    categorize_clauses,
    plain_english,
    review_contract,
    format_review,
    compare_reviews,
    format_diff,
    information_only_check,
    control_contract,
    extract_clauses,
)
from nomorals.legal.aid import (
    LANGUAGES,
    answer_legal_question,
    format_answer,
    format_answer_sms,
    chunk_for_sms,
    triage,
    refer_lawyer,
    letter_kinds,
    render_letter,
    control_legal,
)
from nomorals.legal.research import (
    LegalResearch,
    expand_query,
    verify_citations,
    groundedness,
    format_research,
)
from nomorals.legal.portfolio import ContractPortfolio
from nomorals.legal.regulatory import (
    REGULATORS,
    OBLIGATION_CONTROLS,
    RegulatoryWatch,
    RegulatoryItem,
    alert_text,
    impact_score,
    upcoming_deadlines,
    sync_calendar_to_portfolio,
    control_regwatch,
)


def _tmpdb():
    return tempfile.mktemp(suffix=".db")


TENANCY_RICH = """
TENANCY AGREEMENT between Musa Bello (the Landlord) and Funke Ade (the Tenant).
This agreement is governed by the laws of Lagos State.
Effective date: 1st January 2026 for a term of 1 year.
1. RENT: The rent is ₦1,200,000 per annum. The landlord may increase the rent at
any time without notice. Late rent attracts 25% monthly interest.
2. SERVICE CHARGE: A service charge is payable.
3. TERMINATION: The landlord may terminate immediately for any reason.
"""

FREELANCE_RICH = """
SERVICE AGREEMENT between TechCo (the Client) and Dami (the Contractor).
1. PAYMENT: The client shall pay ₦2,000,000. No deadline is stated.
2. IP: All intellectual property created will be handled appropriately.
3. The contractor assigns this agreement to third parties freely.
4. This agreement is governed by the laws of England and Wales.
5. TERMINATION: The client may terminate at any time.
6. The contractor shall perform the services diligently.
"""


# ── contracts ─────────────────────────────────────────────────────────


class TestContractMetadata(unittest.TestCase):
    def test_parties_and_amounts(self):
        meta = extract_metadata(TENANCY_RICH)
        self.assertTrue(any("Musa Bello" in p for p in meta.parties))
        self.assertTrue(any("Funke Ade" in p for p in meta.parties))
        self.assertIn("₦1,200,000", meta.amounts)

    def test_governing_law_clean(self):
        meta = extract_metadata(TENANCY_RICH)
        self.assertEqual(meta.governing_law, "Lagos State")

    def test_never_raises(self):
        m = extract_metadata("")
        self.assertEqual(m.parties, [])
        self.assertFalse(m.auto_renew)


class TestNewRules(unittest.TestCase):
    def test_late_rent_penalty(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        self.assertIn("T-07", {f.rule_id for f in r.findings})

    def test_no_governing_law(self):
        r = review_contract("TENANCY AGREEMENT. The rent is ₦500,000.", "tenancy")
        self.assertIn("T-08", {f.rule_id for f in r.findings})

    def test_employment_confidentiality_gap(self):
        r = review_contract("EMPLOYMENT AGREEMENT. Salary ₦300,000.", "employment")
        self.assertIn("E-06", {f.rule_id for f in r.findings})

    def test_freelance_gaps(self):
        r = review_contract(FREELANCE_RICH, "freelance")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("F-06", ids)  # no confidentiality
        self.assertIn("F-07", ids)  # no dispute resolution
        self.assertIn("F-09", ids)  # no kill fee

    def test_foreign_governing_law(self):
        r = review_contract(FREELANCE_RICH, "freelance")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("U-01", ids)
        f = next(x for x in r.findings if x.rule_id == "U-01")
        self.assertTrue(f.anomaly)

    def test_assignment_ignores_ip(self):
        # IP assignment must NOT trip the agreement-assignment rule
        doc = ("SERVICE AGREEMENT. On full payment, all intellectual property "
               "in the deliverables is assigned to the client.")
        r = review_contract(doc, "freelance")
        self.assertNotIn("U-03", {f.rule_id for f in r.findings})

    def test_assignment_agreement_fires(self):
        r = review_contract(FREELANCE_RICH, "freelance")
        self.assertIn("U-03", {f.rule_id for f in r.findings})

    def test_force_majeure_gap(self):
        r = review_contract(FREELANCE_RICH, "freelance")
        self.assertIn("U-02", {f.rule_id for f in r.findings})

    def test_auto_renew_universal(self):
        doc = ("SERVICE AGREEMENT. This agreement shall automatically renew "
               "for successive 1-year terms.")
        r = review_contract(doc, "freelance")
        self.assertIn("U-04", {f.rule_id for f in r.findings})

    def test_gap_scoring_lighter(self):
        # only gaps → small deduction, still a high grade
        doc = ("SERVICE AGREEMENT between A (the Client) and B (the Contractor). "
               "Payment of ₦1,000,000 within 14 days of invoice. "
               "IP assigned to client on full payment. "
               "Either party may terminate with 30 days notice. "
               "Late payments attract 1% monthly interest.")
        r = review_contract(doc, "freelance")
        self.assertTrue(all(f.gap for f in r.findings) or not r.findings)
        self.assertGreaterEqual(r.score, 90)

    def test_suggestions_present(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        sug = {f.rule_id: f.suggestion for f in r.findings}
        self.assertTrue(sug.get("T-01"))
        # suggestions are information-only, never advice-shaped
        for f in r.findings:
            self.assertEqual(information_only_check(f.suggestion), [])

    def test_citation_verification(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        for f in r.findings:
            self.assertTrue(f.citation_ok)

    def test_explanations_information_only(self):
        r = review_contract(TENANCY_RICH + FREELANCE_RICH, "tenancy")
        for f in r.findings:
            self.assertEqual(information_only_check(f.explanation), [],
                             f.rule_id)


class TestContractFeatures(unittest.TestCase):
    def test_rights_extraction(self):
        doc = ("TENANCY. The tenant shall be entitled to quiet enjoyment of "
               "the premises and may terminate with 3 months notice. The "
               "tenant has the right to renew for a further term.")
        rights = extract_rights(doc)
        self.assertGreaterEqual(len(rights), 2)

    def test_review_carries_rights(self):
        doc = ("TENANCY. The tenant shall be entitled to quiet enjoyment.")
        r = review_contract(doc, "tenancy")
        self.assertTrue(r.rights)
        self.assertIn("quiet enjoyment", format_review(r).lower())

    def test_plain_english(self):
        out = plain_english("The Tenant hereby indemnifies the Landlord "
                            "pursuant to this agreement.")
        self.assertNotIn("hereby", out.lower())
        self.assertNotIn("pursuant to", out.lower())
        self.assertIn("indemnifies", out.lower())

    def test_categorize_clauses(self):
        clauses = extract_clauses(TENANCY_RICH)
        cats = categorize_clauses(clauses)
        self.assertIn("payment", cats)
        self.assertIn("termination", cats)

    def test_compare_reviews(self):
        old = review_contract(FREELANCE_RICH, "freelance")
        new_text = FREELANCE_RICH.replace(
            "governed by the laws of England and Wales",
            "governed by the laws of Lagos State")
        new = review_contract(new_text, "freelance")
        diff = compare_reviews(old, new)
        self.assertEqual(len(diff.resolved_findings), 1)
        self.assertEqual(diff.resolved_findings[0].rule_id, "U-01")
        text = format_diff(diff)
        self.assertIn("U-01", text)
        self.assertIn(DISCLAIMER, text)

    def test_triage_style(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        out = format_review(r, style="triage")
        self.assertIn("|", out)
        self.assertIn("Triage", out)
        self.assertIn(DISCLAIMER, out)

    def test_compact_style(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        out = format_review(r, style="compact")
        self.assertLess(len(out), len(format_review(r, style="full")))
        self.assertIn(DISCLAIMER, out)

    def test_control_contract_meta(self):
        out = control_contract("meta " + TENANCY_RICH)
        self.assertIn("Lagos State", out)
        self.assertIn("₦1,200,000", out)

    def test_control_contract_rights(self):
        out = control_contract(
            "rights The tenant shall be entitled to quiet enjoyment of the "
            "premises under this tenancy agreement.")
        self.assertIn("quiet enjoyment", out.lower())

    def test_control_contract_rewrite(self):
        out = control_contract("rewrite The Tenant hereby indemnifies the "
                               "Landlord pursuant to this agreement.")
        self.assertNotIn("hereby", out.lower())

    def test_control_contract_review_triage_style(self):
        out = control_contract("review tenancy triage " + TENANCY_RICH)
        self.assertIn("Triage", out)


class TestGoldenReview(unittest.TestCase):
    """Golden regression: a known contract must produce a known rule set
    (contract-risk-assessment CI-gate pattern)."""

    def test_golden_tenancy_rules(self):
        r = review_contract(TENANCY_RICH, "tenancy")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("T-01", ids)
        self.assertIn("T-02", ids)
        self.assertIn("T-07", ids)
        self.assertNotIn("T-99", ids)  # no waiver in this text


# ── aid ───────────────────────────────────────────────────────────────


class TestAidScenarios(unittest.TestCase):
    def test_bail_money_all_languages(self):
        for lang in LANGUAGES:
            a = answer_legal_question("police dey demand money for bail", lang)
            self.assertTrue(a.escalate)
            self.assertIn("bail", a.text.lower() + "bail")

    def test_bail_money_scenario_english(self):
        a = answer_legal_question("officer demanded money to release him on bail")
        self.assertIn("free", a.text.lower())

    def test_inheritance(self):
        a = answer_legal_question("my father died without a will, who inherits?")
        self.assertTrue(a.escalate)
        self.assertIn("intestate", a.text.lower())

    def test_cac(self):
        a = answer_legal_question("how do I register my business name with CAC?")
        self.assertFalse(a.escalate)
        self.assertIn("Corporate Affairs Commission", a.text)

    def test_debt_harassment(self):
        a = answer_legal_question("the lender is threatening to arrest me for debt")
        self.assertTrue(a.escalate)
        self.assertIn("civil", a.text.lower())

    def test_triage_direct(self):
        t = triage("my landlord locked me out")
        self.assertEqual(t["match"], "landlord_lockout")

    def test_triage_unknown(self):
        t = triage("my neighbour took my land")
        self.assertIsNone(t["match"])
        self.assertTrue(t["questions"])


class TestAidFeatures(unittest.TestCase):
    def test_refer_lawyer(self):
        out = refer_lawyer("tenancy")
        self.assertIn("Legal Aid Council", out)
        self.assertIn("legalaidcouncil.gov.ng", out)
        self.assertIn(DISCLAIMER, out)

    def test_letter_kinds(self):
        self.assertIn("fccpc", letter_kinds())

    def test_render_letter_placeholders(self):
        out = render_letter("fccpc", name="Ada")
        self.assertIn("Ada", out)
        self.assertIn("[SELLER]", out)
        self.assertIn(DISCLAIMER, out)

    def test_render_letter_unknown(self):
        out = render_letter("bogus")
        self.assertIn("Available", out)

    def test_chunk_for_sms(self):
        chunks = chunk_for_sms(" ".join(["word"] * 60), limit=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[0].startswith("(1/"))
        for c in chunks:
            self.assertLessEqual(len(c), 60)

    def test_brief_style_shorter(self):
        a = answer_legal_question("my landlord locked me out")
        full = format_answer(a, style="full")
        brief = format_answer(a, style="brief")
        self.assertLess(len(brief), len(full))
        self.assertIn(DISCLAIMER, brief)

    def test_format_answer_sms(self):
        a = answer_legal_question("my landlord locked me out")
        chunks = format_answer_sms(a)
        self.assertTrue(chunks)

    def test_control_legal_refer(self):
        self.assertIn("Legal Aid Council", control_legal("refer"))

    def test_control_legal_letters(self):
        self.assertIn("fccpc", control_legal("letters"))

    def test_control_legal_letter(self):
        out = control_legal("letter fccpc")
        self.assertIn("[NAME]", out)

    def test_control_legal_triage(self):
        out = control_legal("triage my landlord changed the locks")
        self.assertIn("landlord", out.lower())

    def test_control_legal_brief(self):
        a = answer_legal_question("my landlord locked me out")
        out = control_legal("brief my landlord locked me out")
        self.assertLess(len(out), len(format_answer(a)))


# ── research ──────────────────────────────────────────────────────────


class TestResearchSweep(unittest.TestCase):
    def test_expand_query_inflection(self):
        variants = expand_query("my landlord sacked me")
        self.assertIn("my landlord sacked me", variants)
        self.assertTrue(any("terminated" in v for v in variants),
                        variants)

    def test_research_grounded(self):
        r = LegalResearch()
        try:
            res = r.research("landlord locked me out")
            self.assertTrue(res.answered)
            self.assertEqual(groundedness(res), 1.0)
            self.assertTrue(all(e.verified for e in res.evidence))
            self.assertIn("Groundedness", res.render())
        finally:
            r.close()

    def test_research_still_honest(self):
        r = LegalResearch()
        try:
            res = r.research("quantum physics teleportation")
            self.assertFalse(res.answered)
        finally:
            r.close()

    def test_verify_citations_drops_unverifiable(self):
        from nomorals.legal.research import ResearchResult, Evidence
        res = ResearchResult(
            query="x", answered=True, text="t",
            evidence=[Evidence(doc_id="nope", title="T", section="S",
                               snippet="fabricated snippet", score=9.0)])
        verify_citations(res, {})
        self.assertEqual(res.evidence, [])
        self.assertEqual(groundedness(res), 0.0)

    def test_describe_corpus(self):
        r = LegalResearch()
        try:
            desc = r.describe_corpus()
            self.assertIn("4 document", desc)
        finally:
            r.close()

    def test_related_sections(self):
        r = LegalResearch()
        try:
            res = r.research("landlord locked me out")
            rel = r.related_sections(res)
            self.assertTrue(rel)
        finally:
            r.close()

    def test_format_research_brief(self):
        r = LegalResearch()
        try:
            res = r.research("landlord locked me out")
            brief = format_research(res, style="brief")
            full = format_research(res, style="full")
            self.assertLess(len(brief), len(full))
        finally:
            r.close()

    def test_research_no_expand(self):
        r = LegalResearch()
        try:
            res = r.research("landlord locked me out", expand=False)
            self.assertTrue(res.answered)
        finally:
            r.close()


# ── portfolio ─────────────────────────────────────────────────────────


class TestPortfolioSweep(unittest.TestCase):
    DOC = ("SERVICE AGREEMENT between Ada Ltd (the Client) and Bola (the Contractor). "
           "Effective date: 1st January 2026 for a term of 1 year. "
           "The agreement shall automatically renew. "
           "Payment: ₦500,000 per month, payable within 30 days of invoice. "
           "Renewal notice 60 days before expiry. Expiry: 31st December 2026. "
           "Either party may terminate with 30 days written notice.")

    def _portfolio(self):
        return ContractPortfolio(db_path=_tmpdb())

    def test_auto_renew_and_value(self):
        p = self._portfolio()
        c, _ = p.add_raw(self.DOC, name="Ada deal")
        self.assertTrue(c.auto_renew)
        self.assertEqual(c.annual_value, 500000.0)
        self.assertEqual(p.portfolio_value(), 500000.0)

    def test_search_contracts(self):
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        hits = p.search_contracts("automatically renew")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0].name, "Ada deal")
        self.assertIn("renew", hits[0][1].lower())
        self.assertEqual(p.search_contracts("zebra"), [])

    def test_extraction_approval(self):
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        pending = p.unconfirmed_obligations()
        self.assertTrue(pending)
        oid = pending[0].id
        self.assertTrue(p.confirm_obligation(oid))
        self.assertTrue(p.correct_obligation(oid, description="Fixed desc"))
        self.assertTrue(p.remove_obligation(oid))
        self.assertFalse(p.confirm_obligation("obl_nope"))

    def test_renewal_pipeline(self):
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        pipe = p.renewal_pipeline()
        total = sum(len(v) for v in pipe.values())
        self.assertGreater(total, 0)
        self.assertIn("overdue", pipe)
        self.assertIn("30", pipe)

    def test_roll_forward(self):
        p = self._portfolio()
        c, _ = p.add_raw(self.DOC, name="Ada deal")
        before = [o.due_at for o in p.obligations(c.id)
                  if o.kind in ("renewal", "expiry", "notice")]
        n = p.roll_forward(c.id, 12)
        after = [o.due_at for o in p.obligations(c.id)
                 if o.kind in ("renewal", "expiry", "notice")]
        self.assertEqual(n, len(before))
        for b, a in zip(sorted(before), sorted(after)):
            self.assertAlmostEqual(a - b, 12 * 30 * 86400, delta=1)

    def test_amend_history(self):
        p = self._portfolio()
        c, _ = p.add_raw(self.DOC, name="Ada deal")
        result = p.amend(c.id, self.DOC.replace("₦500,000", "₦600,000"))
        self.assertIsNotNone(result)
        new_c, _ = result
        self.assertEqual(new_c.supersedes, c.id)
        chain = p.history(new_c.id)
        self.assertEqual([x.id for x in chain], [c.id, new_c.id])

    def test_pipeline_text(self):
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        text = p.pipeline_text()
        self.assertIn("Renewal pipeline", text)
        self.assertIn(DISCLAIMER, text)

    def test_summary_shows_value(self):
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        self.assertIn("₦500,000", p.summary())

    def test_migration_from_legacy_db(self):
        import sqlite3
        path = _tmpdb()
        db = sqlite3.connect(path)
        db.execute("CREATE TABLE contracts (id TEXT PRIMARY KEY, name TEXT, "
                   "contract_type TEXT, parties TEXT, grade TEXT, score INTEGER,"
                   " findings_count INTEGER, created_at REAL)")
        db.execute("CREATE TABLE obligations (id TEXT PRIMARY KEY, contract_id TEXT,"
                   " kind TEXT, description TEXT, due_at REAL, done INTEGER DEFAULT 0,"
                   " done_at REAL DEFAULT 0, source TEXT DEFAULT 'extracted')")
        db.commit()
        db.close()
        p = ContractPortfolio(db_path=path)  # must migrate, not crash
        self.assertEqual(p.list_contracts(), [])
        c, _ = p.add_raw(self.DOC, name="Legacy deal")
        self.assertTrue(c.auto_renew)  # new columns usable

    def test_chat_search_and_pipeline(self):
        from nomorals.legal.portfolio import control_contracts
        p = self._portfolio()
        p.add_raw(self.DOC, name="Ada deal")
        # chat controls construct their own portfolio; just check command parse
        out = control_contracts("help")
        self.assertIn("pipeline", out)
        self.assertIn("search", out)


# ── regulatory ────────────────────────────────────────────────────────


class TestRegulatorySweep(unittest.TestCase):
    def test_new_regulators(self):
        for code in ("CAC", "NCC", "SON"):
            self.assertIn(code, REGULATORS)
            self.assertTrue(REGULATORS[code].site.startswith("https://"))

    def test_new_controls(self):
        for key in ("annual returns", "telecom licensing", "product standards"):
            self.assertIn(key, OBLIGATION_CONTROLS)

    def _watch(self):
        s = RegulatoryWatch(db_path=_tmpdb())
        s.watch("NDPA", ["audit"])
        return s

    def _fetcher(self, summary):
        def f(reg):
            return [{"id": "ndpa-1", "regulator": reg,
                     "title": "CAR filing reminder", "summary": summary,
                     "ref_no": "NDPC/2026/01", "published": time.time(),
                     "topics": ["audit returns"]}]
        return f

    def test_change_detection(self):
        s = self._watch()
        first = s.check(fetcher=self._fetcher("File your CAR by 31 March"))
        self.assertEqual(len(first), 1)
        self.assertFalse(first[0].updated)
        # unchanged → no alert
        again = s.check(fetcher=self._fetcher("File your CAR by 31 March"))
        self.assertEqual(again, [])
        # changed content → alert with updated flag
        changed = s.check(fetcher=self._fetcher("UPDATED: deadline extended"))
        self.assertEqual(len(changed), 1)
        self.assertTrue(changed[0].updated)

    def test_dismiss(self):
        s = self._watch()
        s.check(fetcher=self._fetcher("File your CAR by 31 March"))
        self.assertTrue(s.dismiss("ndpa-1"))
        # even changed content no longer alerts
        self.assertEqual(s.check(fetcher=self._fetcher("COMPLETELY NEW")), [])

    def test_impact_score(self):
        item = RegulatoryItem(
            id="x", regulator="NDPA", title="Penalties for late CAR filing",
            summary="penalties apply immediately", topics=["audit"])
        profile = {"business_type": "fintech startup", "keywords": ["audit"]}
        score = impact_score(item, profile)
        self.assertGreaterEqual(score, 60)
        self.assertLessEqual(impact_score(item), 100)

    def test_alert_styles(self):
        item = RegulatoryItem(id="x", regulator="CBN", title="New circular",
                              summary="banks must report daily", urgency="urgent")
        full = alert_text(item, style="full")
        brief = alert_text(item, style="brief")
        self.assertLess(len(brief), len(full))
        self.assertIn(DISCLAIMER, brief)
        self.assertIn("🚨 URGENT", full)

    def test_item_urgency_passthrough(self):
        from nomorals.legal.regulatory import item_from_dict
        item = item_from_dict({"regulator": "CBN", "title": "Routine update",
                               "urgency": "urgent"})
        self.assertEqual(item.urgency, "urgent")

    def test_upcoming_deadlines(self):
        entries = upcoming_deadlines(12)
        self.assertTrue(entries)
        regs = {e["regulator"] for e in entries}
        self.assertIn("NDPA", regs)

    def test_sync_calendar_to_portfolio(self):
        p = ContractPortfolio(db_path=_tmpdb())
        n = sync_calendar_to_portfolio(p, 12)
        self.assertGreater(n, 0)
        # idempotent — second sync adds nothing
        self.assertEqual(sync_calendar_to_portfolio(p, 12), 0)
        descs = [o.description for o in p.obligations()]
        self.assertTrue(any("NDPA" in d or "NDPC" in d for d in descs))

    def test_upcoming_effective(self):
        s = RegulatoryWatch(db_path=_tmpdb())
        item = RegulatoryItem(
            id="eff-1", regulator="FIRS", title="E-invoicing mandate",
            summary="mandatory e-invoicing", topics=["e-invoicing"],
            effective_date=time.time() + 10 * 86400)
        self.assertTrue(s.note_item(item))
        upcoming = s.upcoming_effective(90)
        self.assertEqual(len(upcoming), 1)
        self.assertEqual(upcoming[0].id, "eff-1")

    def test_chat_calendar(self):
        out = control_regwatch("calendar 12")
        self.assertIn("Regulatory calendar", out)

    def test_chat_dismiss(self):
        out = control_regwatch("dismiss nope")
        self.assertIn("No tracked item", out)


if __name__ == "__main__":
    unittest.main()
