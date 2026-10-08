"""Build-map #76: letter-grade contract risk scores + Nigerian playbooks.

All offline. Verifies: tenancy/employment/freelance reviews, grade
calculation, anomaly flagging, disclaimer presence, no-advice language,
type detection, clause extraction, chat control, never-raises.
"""

from __future__ import annotations

import unittest

from nomorals.legal.contracts import (
    DISCLAIMER,
    CONTRACT_TYPES,
    _PLAYBOOKS,
    detect_contract_type,
    extract_clauses,
    format_review,
    information_only_check,
    review_contract,
    control_contract,
)

TENANCY_BAD = """
TENANCY AGREEMENT

1. RENT
The rent is ₦2,500,000 per annum. The landlord may increase the rent
at any time at his sole discretion.

2. SERVICE CHARGE
The tenant shall pay a service charge as determined by the landlord.

3. NOTICE TO QUIT
Either party may terminate this yearly tenancy with 1 month's notice to quit.

4. AGENCY FEE
An agency fee of 15% of the annual rent is payable to the agent.

5. GENERAL
The tenant hereby waives all rights under the Lagos Tenancy Law.
The tenant shall be liable without limit for any damage whatsoever.
"""

TENANCY_CLEAN = """
TENANCY AGREEMENT

1. RENT
The rent is ₦2,500,000 per annum, payable yearly in advance.

2. RENT REVIEW
The landlord may review the rent after 12 months by giving 3 months'
written notice of any increase.

3. SERVICE CHARGE
The service charge is ₦150,000 per annum, fixed for the first year.

4. NOTICE TO QUIT
This yearly tenancy may be terminated with 3 months' notice to quit
by either party.

5. REPAIRS
The landlord handles structural repairs; the tenant handles minor
internal maintenance.

6. AGENCY FEE
An agency fee of 10% of the annual rent is payable.
"""

EMPLOYMENT_BAD = """
EMPLOYMENT CONTRACT

1. PROBATION
The employee shall serve a probation period of 12 months.

2. TERMINATION
Employment may be terminated by the company at any time without notice.

3. NON-COMPETE
The employee shall not compete with the company anywhere in the world
for a period of 24 months after leaving.

4. SALARY
The salary is ₦300,000 per month. The company may deduct any amounts
it deems fit from the salary.
"""

EMPLOYMENT_CLEAN = """
EMPLOYMENT CONTRACT

1. PROBATION
The employee shall serve a probation period of 3 months.

2. TERMINATION
Either party may terminate with 1 month's written notice after
confirmation.

3. PENSION
The company operates the Contributory Pension Scheme with a licensed
PFA: employer contributes 10% and the employee 8% of monthly emoluments.

4. SALARY
The salary is ₦300,000 per month. Statutory deductions (PAYE, pension)
apply as required by law.
"""

FREELANCE_BAD = """
SERVICE AGREEMENT

1. SERVICES
The freelancer shall design the company website, including unlimited
revisions until the client is fully satisfied.

2. PAYMENT
The client shall pay ₦1,500,000 for the project.

3. INTELLECTUAL PROPERTY
All intellectual property created will be handled appropriately.

4. TERMINATION
The client may terminate this agreement at any time.
"""

FREELANCE_CLEAN = """
SERVICE AGREEMENT

1. SCOPE OF WORK
The freelancer shall design the company website: homepage plus 5
inner pages. Up to 2 rounds of revisions included; extra work billed
separately at an agreed rate.

2. PAYMENT
The client shall pay ₦1,500,000 within 14 days of each invoice.
Late payments attract 2% monthly interest.

3. INTELLECTUAL PROPERTY
On full payment, all intellectual property in the deliverables is
assigned to the client.

4. TERMINATION
Either party may terminate with 14 days' written notice.
"""


class TestTypeDetection(unittest.TestCase):
    def test_tenancy(self):
        self.assertEqual(detect_contract_type(TENANCY_BAD), "tenancy")

    def test_employment(self):
        self.assertEqual(detect_contract_type(EMPLOYMENT_BAD), "employment")

    def test_freelance(self):
        self.assertEqual(detect_contract_type(FREELANCE_BAD), "freelance")

    def test_empty_defaults_tenancy(self):
        self.assertEqual(detect_contract_type(""), "tenancy")
        self.assertEqual(detect_contract_type(None), "tenancy")


class TestClauseExtraction(unittest.TestCase):
    def test_numbered_headings_split(self):
        clauses = extract_clauses(TENANCY_BAD)
        self.assertGreaterEqual(len(clauses), 4)

    def test_never_raises(self):
        self.assertEqual(len(extract_clauses(None)), 1)
        self.assertEqual(len(extract_clauses("")), 1)


class TestTenancyReview(unittest.TestCase):
    def test_bad_tenancy_flagged(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("T-01", ids)  # rent increase, no notice
        self.assertIn("T-02", ids)  # service charge, no amount
        self.assertIn("T-03", ids)  # 1-month quit notice on yearly tenancy
        self.assertIn("T-04", ids)  # 15% agency fee
        self.assertIn("T-99", ids)  # waiver of statutory rights (critical)
        self.assertIn("T-98", ids)  # unlimited liability

    def test_bad_tenancy_grade_low(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        self.assertLessEqual(r.score, 40)
        self.assertIn(r.grade, ("F", "D-", "D", "D+"))

    def test_clean_tenancy_scores_high(self):
        r = review_contract(TENANCY_CLEAN, "tenancy")
        self.assertGreaterEqual(r.score, 85)
        self.assertTrue(r.grade.startswith(("A", "B")))

    def test_auto_detect_tenancy(self):
        r = review_contract(TENANCY_BAD)
        self.assertEqual(r.contract_type, "tenancy")


class TestEmploymentReview(unittest.TestCase):
    def test_bad_employment_flagged(self):
        r = review_contract(EMPLOYMENT_BAD, "employment")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("E-01", ids)  # 12-month probation
        self.assertIn("E-03", ids)  # 24-month worldwide non-compete
        self.assertIn("E-04", ids)  # no pension mention
        self.assertIn("E-05", ids)  # broad deductions

    def test_clean_employment_scores_high(self):
        r = review_contract(EMPLOYMENT_CLEAN, "employment")
        self.assertGreaterEqual(r.score, 90)


class TestFreelanceReview(unittest.TestCase):
    def test_bad_freelance_flagged(self):
        r = review_contract(FREELANCE_BAD, "freelance")
        ids = {f.rule_id for f in r.findings}
        self.assertIn("F-01", ids)  # no payment deadline
        self.assertIn("F-02", ids)  # ambiguous IP
        self.assertIn("F-03", ids)  # one-sided termination
        self.assertIn("F-04", ids)  # no late fees
        self.assertIn("F-05", ids)  # unlimited revisions

    def test_clean_freelance_scores_high(self):
        r = review_contract(FREELANCE_CLEAN, "freelance")
        self.assertGreaterEqual(r.score, 90)


class TestGrading(unittest.TestCase):
    def test_grade_bands(self):
        from nomorals.legal.contracts import _grade
        self.assertEqual(_grade(98), "A+")
        self.assertEqual(_grade(95), "A")
        self.assertEqual(_grade(91), "A-")
        self.assertEqual(_grade(88), "B+")
        self.assertEqual(_grade(84), "B")
        self.assertEqual(_grade(81), "B-")
        self.assertEqual(_grade(78), "C+")
        self.assertEqual(_grade(74), "C")
        self.assertEqual(_grade(71), "C-")
        self.assertEqual(_grade(68), "D+")
        self.assertEqual(_grade(64), "D")
        self.assertEqual(_grade(61), "D-")
        self.assertEqual(_grade(30), "F")
        self.assertEqual(_grade(0), "F")

    def test_findings_sorted_by_severity(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sev = [order[f.severity] for f in r.findings]
        self.assertEqual(sev, sorted(sev))


class TestAnomaly(unittest.TestCase):
    def test_waiver_is_critical_anomaly(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        w = [f for f in r.findings if f.rule_id == "T-99"]
        self.assertEqual(len(w), 1)
        self.assertEqual(w[0].severity, "critical")
        self.assertTrue(w[0].anomaly)

    def test_needs_attention_count(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        self.assertGreaterEqual(r.needs_attention, 4)


class TestDisclaimerAndTone(unittest.TestCase):
    def test_disclaimer_in_output(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        out = format_review(r)
        self.assertIn("not legal advice", out)
        self.assertIn(DISCLAIMER, out)

    def test_disclaimer_on_clean_review_too(self):
        r = review_contract(TENANCY_CLEAN, "tenancy")
        self.assertIn(DISCLAIMER, format_review(r))

    def test_no_advice_language_anywhere(self):
        # Every playbook explanation + every formatted review must be
        # information-only.
        for ctype, rules in _PLAYBOOKS.items():
            for rule in rules:
                f = rule.check("placeholder", [])
                if f is not None:
                    self.assertEqual(
                        information_only_check(f.explanation), [],
                        f"advice language in {rule.rule_id}: {f.explanation[:60]}")
        for doc, ctype in ((TENANCY_BAD, "tenancy"), (EMPLOYMENT_BAD, "employment"),
                           (FREELANCE_BAD, "freelance"), (TENANCY_CLEAN, "tenancy")):
            out = format_review(review_contract(doc, ctype))
            self.assertEqual(information_only_check(out), [],
                             f"advice language in formatted {ctype} review")

    def test_grade_headline_format(self):
        r = review_contract(TENANCY_BAD, "tenancy")
        out = format_review(r)
        self.assertIn("Grade:", out)
        self.assertIn("need attention", out)


class TestControl(unittest.TestCase):
    def test_usage(self):
        self.assertIn("/contract", control_contract(""))

    def test_types(self):
        out = control_contract("types")
        for t in CONTRACT_TYPES:
            self.assertIn(t, out)
        self.assertIn("not legal advice", out)

    def test_review_flow(self):
        out = control_contract("review tenancy " + TENANCY_BAD)
        self.assertIn("Grade:", out)
        self.assertIn("not legal advice", out)

    def test_review_auto_type(self):
        out = control_contract("review " + EMPLOYMENT_BAD)
        self.assertIn("employment", out.lower())

    def test_too_short(self):
        out = control_contract("review tenancy short text")
        self.assertIn("Paste the contract", out)

    def test_never_raises(self):
        self.assertIsInstance(control_contract(None), str)
        self.assertIsInstance(review_contract(None).grade, str)
        self.assertIsInstance(review_contract("   ").grade, str)


if __name__ == "__main__":
    unittest.main()
