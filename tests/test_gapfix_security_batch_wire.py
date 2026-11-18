"""Lane 56: wire in the owner's nine security commits.

- checkers.py: service adapters — verdicts from cookies/redirects/
  status, never body text.
- combo_checker.py: combos against many services, rate gates, lockout
  retry, resumable JSONL.
- social_enum.py: username/email enumeration, 404-or-marker discipline.
- recon.py: passive surface discovery — CT, DoH, AXFR, takeover leads.
- subbrute.py: DNS subdomain brute, wildcard suppression, permutations.
- evidence.py: hash-chained evidence packs, secrets masked at rest.
- disclose.py: severity rubric, RFC 9116 contacts, disclosure draft.
- report.py: engagement report from a case's chain + inventory.

Repairs proven here (all execution-proven, all in the owner's files):
- report.py register() raised NameError (Capability never imported).
- report.py rendered a broken chain with a warning while its own
  description promises refusal — now refuses (ToolError).
- disclose.py find_contacts stopped at the first security.txt that
  merely mentioned "contact" without a mailto: — now falls through.
- disclose.py + report.py told the reader to run "nm evidence verify",
  which is the research CLI, not the chain verifier — now "/chain verify".
- evidence.py Finding.masked() masked every header value (identical
  ternary branches); only secret headers are masked now.
- evidence.py export() on a fresh case raised FileNotFoundError.
- evidence.py evidence_verify slugified differently from EvidenceChain
  ("My Case!" -> chain "my-case", verify "my-case-") — one _slug now.
- evidence.py __all__ omitted Finding (the commit names it).
- checkers.py FormAdapter's free_marker branch was dead (sub is bool
  by then) — free-only specs now grade VALID_FREE.
- checkers.py EbayAdapter carried a dead sub block (always None).
- recon.py _read_name looped forever on a self-referential compression
  pointer — capped at 10 jumps.
- social_enum.py enum_email carried a dead body_ok branch.

Everything network-bound runs against fakes or a local loopback
server; no live target, no real credentials.
"""
from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.security.combo_checker as cc_mod
import nomorals.security.disclose as disclose_mod
import nomorals.security.evidence as evidence_mod
import nomorals.security.social_enum as enum_mod
import nomorals.security.subbrute as subbrute_mod
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.core.errors import ToolError
from nomorals.security import (
    EvidenceChain,
    Finding,
    Verdict,
    build_engagement_report,
)
from nomorals.security import (
    checkers as checkers_mod,
)
from nomorals.security.checkers import (
    FormAdapter,
    get_adapter,
)
from nomorals.security.combo_checker import ComboChecker, parse_combo
from nomorals.security.disclose import find_contacts, score
from nomorals.security.evidence import mask, verify_chain
from nomorals.security.recon import _read_name
from nomorals.security.report import VERDICT_KIND
from nomorals.security.social_enum import enum_email, enum_username
from nomorals.security.subbrute import SubBrute, permutations
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
)
from nomorals.tools.registry import ToolRegistry

# importlib, not `import ... as`: the security package's bare `recon`
# name would shadow the submodule on attribute lookup.
recon_mod = importlib.import_module("nomorals.security.recon")

NEW_TOOLS = (
    "combo_check", "social_enum", "recon_domain", "subbrute",
    "evidence_add", "evidence_verify", "evidence_export",
    "severity_score", "disclose_contacts", "disclose_draft",
    "report_build",
)
NEW_VERBS = ("combo", "enum", "recon", "subbrute", "chain", "disclose")


class FakeResp:
    def __init__(self, text="", status=200, location=""):
        self.text = text
        self.status = status
        self.location = location

    @property
    def ok(self):
        return 200 <= self.status < 400


class FakeClient:
    """Method-aware fake HttpClient: {(method, substring): FakeResp}."""

    def __init__(self, routes=None, cookies=None, **kw):
        self.routes = routes or {}
        self.cookies = cookies or {}
        self.calls = []

    def _hit(self, method, url):
        self.calls.append((method, url))
        for (m, key), resp in self.routes.items():
            if m == method and key in url:
                return resp
        return FakeResp("", 404)

    def get(self, url, headers=None, timeout=None):
        return self._hit("GET", url)

    def post(self, url, data=None, headers=None, timeout=None):
        return self._hit("POST", url)


def _ebay_client(cookie=False, post_body="", post_status=200):
    signin = '<input name="srt" value="abc"><input name="ru" value="/">'
    routes = {
        ("GET", "SignIn"): FakeResp(signin, 200),
        ("POST", "SignIn"): FakeResp(post_body, post_status),
    }
    return FakeClient(routes, cookies={"ebay": "1"} if cookie else {})


class CheckersTests(unittest.TestCase):
    def test_ebay_cookie_is_valid(self):
        probe = get_adapter("ebay").check(
            "a@b.com", "pw", _ebay_client(cookie=True))
        self.assertEqual(probe.verdict, Verdict.VALID)

    def test_ebay_form_rerender_is_invalid(self):
        probe = get_adapter("ebay").check(
            "a@b.com", "pw",
            _ebay_client(post_body='<form><input name="pass"></form>'))
        self.assertEqual(probe.verdict, Verdict.INVALID)

    def test_ebay_challenge_is_two_fa(self):
        probe = get_adapter("ebay").check(
            "a@b.com", "pw",
            _ebay_client(post_body="verify your identity otp"))
        self.assertEqual(probe.verdict, Verdict.TWO_FA)

    def test_ebay_429_is_rate_limited(self):
        probe = get_adapter("ebay").check(
            "a@b.com", "pw",
            _ebay_client(post_body="slow down", post_status=429))
        self.assertEqual(probe.verdict, Verdict.RATE_LIMITED)

    def test_ebay_captcha_is_captcha(self):
        probe = get_adapter("ebay").check(
            "a@b.com", "pw",
            _ebay_client(post_body="please solve the captcha"))
        self.assertEqual(probe.verdict, Verdict.CAPTCHA)

    def test_form_adapter_subscribed_and_free(self):
        spec = {"id": "t", "login_url": "https://t.example/login",
                "auth_cookies": ("sess",),
                "subscribed_marker": r"pro plan",
                "invalid_marker": r"wrong password"}
        fa = FormAdapter(spec)
        c = FakeClient({("GET", "login"): FakeResp("", 200),
                        ("POST", "login"): FakeResp("welcome pro plan", 200)},
                       cookies={"sess": "x"})
        self.assertEqual(fa.check("a@b.com", "pw", c).verdict,
                         Verdict.VALID_SUBSCRIBED)
        c = FakeClient({("GET", "login"): FakeResp("", 200),
                        ("POST", "login"): FakeResp("wrong password", 200)})
        self.assertEqual(fa.check("a@b.com", "pw", c).verdict,
                         Verdict.INVALID)

    def test_form_adapter_free_only_marker_grades_free(self):
        # The free_marker branch was dead (sub is bool by then); a
        # free-only spec must grade VALID_FREE, not plain VALID.
        spec = {"id": "t2", "login_url": "https://t.example/login",
                "auth_cookies": ("sess",), "free_marker": r"free tier"}
        fa = FormAdapter(spec)
        c = FakeClient({("GET", "login"): FakeResp("", 200),
                        ("POST", "login"): FakeResp("welcome free tier", 200)},
                       cookies={"sess": "x"})
        probe = fa.check("a@b.com", "pw", c)
        self.assertEqual(probe.verdict, Verdict.VALID_FREE)
        self.assertIs(probe.subscription, False)

    def test_unknown_service_is_tool_error(self):
        with self.assertRaises(ToolError):
            get_adapter("nope")

    def test_auth_cookie_names(self):
        names = checkers_mod.auth_cookie_names()
        self.assertIn("ebay", names)
        self.assertIn("netflixid", names)
        self.assertIn("sid", names)

    def test_verdict_good(self):
        self.assertTrue(Verdict.VALID_SUBSCRIBED.good)
        self.assertFalse(Verdict.INVALID.good)
        self.assertFalse(Verdict.UNKNOWN.good)


class ComboCheckerTests(unittest.TestCase):
    def test_parse_combo(self):
        self.assertEqual(parse_combo("a@b.com:secret").password, "secret")
        self.assertEqual(parse_combo("a@b.com|pw").password, "pw")
        self.assertIsNone(parse_combo("no-at-sign"))
        self.assertIsNone(parse_combo("# comment"))
        self.assertIsNone(parse_combo("   "))
        self.assertEqual(parse_combo("a@b.com").password, "")

    def test_no_parseable_combos_raises(self):
        with self.assertRaises(ToolError):
            ComboChecker(["###", "junk"], services=["ebay"])

    def test_unknown_service_raises(self):
        with self.assertRaises(ToolError):
            ComboChecker(["a@b.com:pw"], services=["nope"])

    def test_run_with_fake_http(self):
        tmp = Path(tempfile.mkdtemp())
        with mock.patch.object(cc_mod, "HttpClient", FakeClient):
            checker = ComboChecker(
                ["a@b.com:pw1"], services=["ebay"], workers=1,
                per_service_interval=0, quiet=True,
                out_file=str(tmp / "o.jsonl"))
            res = checker.run()
        self.assertEqual(res["counts"], {"unknown": 1})
        self.assertEqual(res["attempts"], 1)
        rows = [json.loads(line) for line in (tmp / "o.jsonl").read_text().splitlines()]
        self.assertEqual(rows[0]["verdict"], "unknown")

    def test_scripted_adapter_valid_and_lock_retry(self):
        from nomorals.security.checkers import Probe
        from nomorals.security.checkers import Verdict as V

        class Scripted:
            def __init__(self):
                self.calls = 0

            def check(self, email, password, client):
                self.calls += 1
                if self.calls == 1:
                    return Probe(V.LOCKED, "account locked")
                return Probe(V.VALID, "cookie proof")

        scripted = Scripted()
        tmp = Path(tempfile.mkdtemp())
        with (mock.patch.object(cc_mod, "get_adapter",
                                return_value=scripted),
              mock.patch("time.sleep", return_value=None)):
            checker = ComboChecker(
                ["a@b.com:pw"], services=["ebay"], workers=1,
                per_service_interval=0, quiet=True, retry_on_lock=1,
                out_file=str(tmp / "o.jsonl"),
                found_file=str(tmp / "found.txt"))
            res = checker.run()
        self.assertEqual(res["counts"], {"valid": 1})
        self.assertEqual(res["attempts"], 2)  # locked, then retried
        self.assertIn("a@b.com", (tmp / "found.txt").read_text())

    def test_resume_skips_done(self):
        tmp = Path(tempfile.mkdtemp())
        out = tmp / "o.jsonl"
        out.write_text(json.dumps({"email": "a@b.com",
                                   "service": "ebay"}) + "\n")
        with mock.patch.object(cc_mod, "HttpClient", FakeClient):
            checker = ComboChecker(
                ["a@b.com:pw"], services=["ebay"], workers=1,
                per_service_interval=0, quiet=True, out_file=str(out))
            res = checker.run()
        self.assertEqual(res["attempts"], 0)
        self.assertEqual(res["counts"], {})


class SocialEnumTests(unittest.TestCase):
    def test_probe_states(self):
        taken = FakeClient({("GET", "github.com/u1"):
                            FakeResp("<title>u1 (@u1)</title>", 200)})
        free = FakeClient({("GET", "github.com/u2"): FakeResp("x", 404)})
        incon = FakeClient({("GET", "github.com/u3"):
                            FakeResp("generic page", 200)})
        with mock.patch.object(enum_mod, "HttpClient",
                              lambda timeout=15, proxy_url="": taken):
            from nomorals.security.social_enum import _probe
            self.assertEqual(
                _probe("github", "https://github.com/u1",
                       r"<title>[^<]*\(@", "u1", 15, None).state, "taken")
        with mock.patch.object(enum_mod, "HttpClient",
                              lambda timeout=15, proxy_url="": free):
            from nomorals.security.social_enum import _probe
            self.assertEqual(
                _probe("github", "https://github.com/u2",
                       r"<title>[^<]*\(@", "u2", 15, None).state, "free")
        with mock.patch.object(enum_mod, "HttpClient",
                              lambda timeout=15, proxy_url="": incon):
            from nomorals.security.social_enum import _probe
            self.assertEqual(
                _probe("github", "https://github.com/u3",
                       r"<title>[^<]*\(@", "u3", 15, None).state,
                "inconclusive")

    def test_enum_username_two_sites(self):
        def fake_client(timeout=15, proxy_url=""):
            return FakeClient({
                ("GET", "github.com/alice"): FakeResp("<title>a (@a)</title>", 200),
                ("GET", "gitlab.com/alice"): FakeResp("x", 404),
            })
        with mock.patch.object(enum_mod, "HttpClient", fake_client):
            hits = enum_username("alice", sites=["github", "gitlab"],
                                 workers=2)
        by_site = {h.site: h.state for h in hits}
        self.assertEqual(by_site, {"github": "taken", "gitlab": "free"})

    def test_enum_email_gravatar(self):
        def fake_client(timeout=15, proxy_url=""):
            return FakeClient({
                ("GET", "gravatar.com/alice"): FakeResp("profile page", 200),
                ("GET", "github.com/search"): FakeResp("no users", 200),
            })
        with mock.patch.object(enum_mod, "HttpClient", fake_client):
            hits = enum_email("alice@example.com")
        by_site = {h.site: h.state for h in hits}
        self.assertEqual(by_site["gravatar"], "taken")
        self.assertEqual(by_site["github-search"], "inconclusive")


class ReconTests(unittest.TestCase):
    def test_crt_sh_parses_names(self):
        body = json.dumps([{"name_value": "*.example.com\nexample.com"},
                           {"name_value": "other.org"}])
        with mock.patch.object(recon_mod, "HttpClient",
                              lambda timeout=30: FakeClient(
                                  {("GET", "crt.sh"): FakeResp(body, 200)})):
            names = recon_mod.crt_sh("example.com")
        self.assertEqual(names, ["example.com"])

    def test_dns_lookup_parses_answers(self):
        body = json.dumps({"Answer": [{"type": 1, "data": "1.2.3.4"},
                                      {"type": 5, "data": "x.example.com."}]})
        with mock.patch.object(recon_mod, "HttpClient",
                              lambda timeout=10: FakeClient(
                                  {("GET", "dns-query"): FakeResp(body, 200)})):
            self.assertEqual(recon_mod.dns_lookup("example.com", "A"),
                             ["1.2.3.4"])
            self.assertEqual(recon_mod.dns_lookup("example.com", "CNAME"),
                             ["x.example.com"])

    def test_dns_lookup_bad_type_raises(self):
        with self.assertRaises(ToolError):
            recon_mod.dns_lookup("example.com", "NOPE")

    def test_read_name_roundtrip_and_pointer_guard(self):
        name = "www.example.com"
        buf = recon_mod._enc_name(name) + b"\x00\x01\x00\x01"
        got, end = _read_name(buf, 0)
        self.assertEqual(got, name)
        # Self-referential pointer: must stop, never loop forever.
        got, end = _read_name(bytes([0xC0, 0x00]), 0)
        self.assertEqual(got, "")

    def test_takeover_s3_likely_and_cname_only(self):
        with (mock.patch.object(recon_mod, "dns_lookup",
                                return_value=["bucket.s3.amazonaws.com"]),
              mock.patch.object(recon_mod, "HttpClient",
                                lambda timeout=15: FakeClient(
                                    {("GET", "example.com/"):
                                     FakeResp("NoSuchBucket", 200)}))):
            hit = recon_mod.takeover_check("example.com")
        self.assertEqual(hit["state"], "likely")
        self.assertEqual(hit["provider"], "AWS S3")
        with (mock.patch.object(recon_mod, "dns_lookup",
                                return_value=["bucket.s3.amazonaws.com"]),
              mock.patch.object(recon_mod, "HttpClient",
                                lambda timeout=15: FakeClient(
                                    {("GET", "example.com/"):
                                     FakeResp("real site", 200)}))):
            hit = recon_mod.takeover_check("example.com")
        self.assertEqual(hit["state"], "cname-only")

    def test_recon_orchestrator(self):
        fake = {("www.example.com", "A"): ["1.1.1.1"],
                ("api.example.com", "A"): ["2.2.2.2"],
                ("example.com", "NS"): []}
        with (mock.patch.object(recon_mod, "crt_sh",
                                return_value=["www.example.com",
                                              "api.example.com"]),
              mock.patch.object(recon_mod, "dns_lookup",
                                lambda n, t="A", timeout=10:
                                fake.get((n, t), [])),
              mock.patch.object(recon_mod, "takeover_check",
                                return_value=None),
              mock.patch.object(recon_mod, "wildcard_control",
                                return_value=None)):
            res = recon_mod.recon("example.com", do_axfr=False)
        self.assertEqual(res["host_count"], 3)
        self.assertIsNone(res["wildcard"])

    def test_recon_bad_domain_raises(self):
        with self.assertRaises(ToolError):
            recon_mod.recon("not a domain!!!")


class SubBruteTests(unittest.TestCase):
    def test_permutations(self):
        got = permutations(["www.example.com"], "example.com")
        self.assertIn("dev-www.example.com", got)
        self.assertIn("www2.example.com", got)
        self.assertNotIn("example.com", got)

    def test_run_with_fake_dns(self):
        fake_dns = {"www.example.com": ["1.1.1.1"],
                    "api.example.com": ["2.2.2.2"]}
        with (mock.patch.object(subbrute_mod, "dns_lookup",
                                lambda h, t="A", timeout=8:
                                fake_dns.get(h, []) if t == "A" else []),
              mock.patch.object(subbrute_mod, "wildcard_control",
                                return_value=None)):
            brute = SubBrute("example.com", workers=2, rate=1000,
                             rounds=1, progress=False)
            out = brute.run()
        self.assertEqual(out["found"], 2)
        self.assertEqual(out["wildcard_matches"], 0)
        hosts = {h["host"] for h in out["hits"]}
        self.assertEqual(hosts, {"www.example.com", "api.example.com"})

    def test_wildcard_matches_are_tagged_not_counted(self):
        fake_dns = {"www.example.com": ["9.9.9.9"]}
        with (mock.patch.object(subbrute_mod, "dns_lookup",
                                lambda h, t="A", timeout=8:
                                fake_dns.get(h, []) if t == "A" else []),
              mock.patch.object(subbrute_mod, "wildcard_control",
                                return_value="9.9.9.9")):
            brute = SubBrute("example.com", workers=2, rate=1000,
                             rounds=1, progress=False)
            out = brute.run()
        self.assertEqual(out["found"], 0)
        self.assertEqual(out["wildcard_matches"], 1)

    def test_bad_domain_and_missing_wordlist(self):
        with self.assertRaises(ToolError):
            SubBrute("not a domain!!!")
        with self.assertRaises(ToolError):
            SubBrute("example.com", wordlist="/no/such/words.txt")


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def _chain(self, case="demo"):
        return EvidenceChain(self.tmp, case)

    def _finding(self, verdict="takeover"):
        return Finding(
            module="recon", target="example.com", verdict=verdict,
            request={"method": "GET", "url": "https://example.com/",
                     "headers": {"Authorization": "Bearer abc.def.ghi",
                                 "Accept": "application/json"}},
            response={"status": 200, "body": "NoSuchBucket"},
            detail={"note": "cname to s3"})

    def test_add_verify_and_tamper_names_index(self):
        chain = self._chain()
        chain.add(self._finding())
        chain.add(self._finding("secret_leak"))
        self.assertEqual(verify_chain(chain.chain_path)["state"], "intact")
        lines = chain.chain_path.read_text().splitlines()
        lines[0] = lines[0].replace("takeover", "takeovex", 1)
        chain.chain_path.write_text("\n".join(lines) + "\n")
        bad = verify_chain(chain.chain_path)
        self.assertEqual(bad["state"], "broken")
        self.assertEqual(bad["broken_at"], 1)

    def test_mask_selective_headers(self):
        # Only secret headers are masked; a long value in a plain
        # header stays readable (the old code masked both branches).
        # mask() tails real secret shapes (JWT / long key), it does not
        # invent a "<redacted>" literal — assert the secret itself
        # never survives verbatim.
        jwt = ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIn0."
               "0123456789abcdef0123456789abcdef")
        f = Finding(module="m", target="t", verdict="v",
                    request={"headers": {"X-Trace": "a" * 40,
                                        "Authorization": f"Bearer {jwt}"}},
                    response={}, detail={})
        masked = f.masked()
        self.assertEqual(masked["request"]["headers"]["X-Trace"], "a" * 40)
        self.assertNotIn(jwt, masked["request"]["headers"]["Authorization"])
        self.assertNotEqual(masked["request"]["headers"]["Authorization"],
                            f"Bearer {jwt}")
        self.assertNotIn(jwt, mask(f"token {jwt}"))

    def test_export_fresh_case_is_empty_not_an_error(self):
        chain = self._chain("fresh")
        out = chain.export()
        self.assertEqual(out["records"], 0)
        self.assertEqual(out["verify"], "missing")
        self.assertTrue(Path(out["dir"], "manifest.json").is_file())

    def test_slug_agreement_through_the_tools(self):
        # EvidenceChain and the evidence_verify tool must agree on the
        # slug: "My Case!" used to verify as "my-case-" (missing).
        home = self.tmp
        ctx = SimpleNamespace(home=str(home))
        reg = ToolRegistry(ctx)
        evidence_mod.register(reg)
        out = reg.call("evidence_add", case="My Case!", module="recon",
                       target="example.com", verdict="takeover",
                       request={"method": "GET", "url": "https://e.com/"},
                       response={"status": 200, "body": "x"})
        self.assertTrue(out.ok, out.error)
        self.assertTrue((home / "data" / "evidence" / "my-case").is_dir())
        verified = reg.call("evidence_verify", case="My Case!")
        self.assertTrue(verified.ok, verified.error)
        self.assertEqual(verified.value["state"], "intact")
        self.assertEqual(verified.value["records"], 1)

    def test_finding_is_exported(self):
        import nomorals.security as sec
        self.assertIs(sec.Finding, Finding)
        self.assertIn("Finding", evidence_mod.__all__)


class DiscloseTests(unittest.TestCase):
    def test_score_table(self):
        r = score("rce", sensitivity="financial", reach="unauth",
                  scale="bulk")
        self.assertEqual((r["score"], r["tier"]), (100, "critical"))
        self.assertEqual(score("info")["tier"], "low")
        self.assertEqual(score("xss_reflected")["score"], 15 + 0 + 20 + 0)
        self.assertEqual(score("unknown-kind")["breakdown"]["kind"], 5)

    def test_find_contacts_falls_through_contactless_security_txt(self):
        # First security.txt mentions "contact" but carries no mailto:;
        # the old code stopped there and reported nothing.
        routes = {
            ("GET", "/.well-known/security.txt"): FakeResp(
                "Contact: https://example.com/contact\n", 200),
            ("GET", "/security.txt"): FakeResp(
                "Contact: mailto:second@example.com\n", 200),
        }
        with mock.patch.object(disclose_mod, "HttpClient",
                              lambda timeout=15: FakeClient(routes)):
            found = find_contacts("example.com")
        self.assertEqual([c["email"] for c in found["contacts"]],
                         ["second@example.com"])

    def test_find_contacts_empty_is_honest(self):
        with mock.patch.object(disclose_mod, "HttpClient",
                              lambda timeout=15: FakeClient({})):
            found = find_contacts("example.com")
        self.assertEqual(found["contacts"], [])
        self.assertIn("registrar", found["note"])

    def test_draft_from_real_chain(self):
        tmp = Path(tempfile.mkdtemp())
        chain = EvidenceChain(tmp, "demo")
        chain.add(Finding(module="recon", target="example.com",
                          verdict="takeover",
                          request={"method": "GET",
                                   "url": "https://example.com/"},
                          response={"status": 200, "body": "NoSuchBucket"},
                          detail={"note": "cname to s3"}))
        out = disclose_mod.draft("demo", chain.chain_path,
                                 domain="example.com")
        text = Path(out["file"]).read_text()
        self.assertIn("/chain verify demo", text)
        self.assertNotIn("nm evidence verify", text)
        self.assertIn("takeover", text)

    def test_draft_refuses_broken_chain(self):
        tmp = Path(tempfile.mkdtemp())
        chain = EvidenceChain(tmp, "demo")
        chain.add(Finding(module="recon", target="example.com",
                          verdict="takeover",
                          request={"method": "GET",
                                   "url": "https://example.com/"},
                          response={"status": 200, "body": "x"},
                          detail={}))
        lines = chain.chain_path.read_text().splitlines()
        lines[0] = lines[0].replace("takeover", "takeovex", 1)
        chain.chain_path.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ToolError):
            disclose_mod.draft("demo", chain.chain_path,
                               domain="example.com")


class ReportTests(unittest.TestCase):
    def _case_dir(self):
        tmp = Path(tempfile.mkdtemp())
        chain = EvidenceChain(tmp, "demo")
        chain.add(Finding(module="recon", target="example.com",
                          verdict="takeover",
                          request={"method": "GET",
                                   "url": "https://example.com/"},
                          response={"status": 200, "body": "NoSuchBucket"},
                          detail={"note": "cname to s3"}))
        (tmp / "evidence" / "demo" / "recon.json").write_text(json.dumps(
            {"wildcard": None, "axfr": {},
             "assets": [{"host": "example.com",
                         "records": {"A": ["1.2.3.4"]},
                         "sources": ["crt.sh"]}]}))
        return tmp / "evidence" / "demo"

    def test_build_counts_takeover_high(self):
        out = build_engagement_report(self._case_dir(),
                                      domain="example.com")
        self.assertEqual(out["integrity"], "intact")
        self.assertEqual(out["counts"]["high"], 1)
        self.assertIn("Engagement report", Path(out["file"]).read_text())
        self.assertNotIn("nm evidence verify",
                         Path(out["file"]).read_text())
        self.assertIn("/chain verify demo",
                      Path(out["file"]).read_text())

    def test_build_refuses_broken_chain(self):
        case_dir = self._case_dir()
        p = case_dir / "chain.jsonl"
        lines = p.read_text().splitlines()
        lines[0] = lines[0].replace("takeover", "takeovex", 1)
        p.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ToolError):
            build_engagement_report(case_dir, domain="example.com")

    def test_verdict_kind_bridge(self):
        self.assertEqual(VERDICT_KIND["takeover"], "exposed_git")
        self.assertEqual(VERDICT_KIND["valid"], "info")


class RegisterTests(unittest.TestCase):
    def test_register_builtins_clean_and_new_tools_present(self):
        reg = ToolRegistry(SimpleNamespace())
        reg.register_builtins()
        failed = [f["module"] for f in reg.failed_modules()]
        self.assertNotIn("security", failed)
        for name in NEW_TOOLS:
            self.assertIn(name, reg.names(), f"tool {name} missing")

    def test_each_module_registers(self):
        import nomorals.security.combo_checker as cc
        import nomorals.security.disclose as dc
        import nomorals.security.evidence as ev
        rc = importlib.import_module("nomorals.security.recon")
        import nomorals.security.report as rp
        import nomorals.security.social_enum as se
        import nomorals.security.subbrute as sb
        for mod in (cc, se, rc, sb, ev, dc, rp):
            self.assertTrue(hasattr(mod, "register"),
                            f"{mod.__name__} has no register()")

    def test_security_package_exports(self):
        import nomorals.security as sec
        for name in ("Verdict", "Probe", "ServiceAdapter", "ADAPTERS",
                     "get_adapter", "ComboChecker", "ComboResult",
                     "parse_combo", "SITES", "enum_username", "enum_email",
                     "crt_sh", "dns_lookup", "axfr",
                     "takeover_check", "SubBrute", "BruteHit",
                     "SEED_WORDS", "permutations", "EvidenceChain",
                     "Finding", "mask", "verify_chain", "score", "Tier",
                     "find_contacts", "draft", "EngagementReport",
                     "VERDICT_KIND", "build_engagement_report"):
            self.assertTrue(hasattr(sec, name), f"security.{name} missing")
        self.assertIn("Finding", sec.__all__)
        # The recon() function is deliberately NOT exported bare from
        # the package (a bare `recon` would shadow the submodule);
        # it lives on its module.
        from nomorals.security.recon import recon as recon_fn
        self.assertTrue(callable(recon_fn))


class _FakeRuntime:
    """Minimal stand-in for PartnerRuntime for the _control_* verbs."""

    def __init__(self, tools, home):
        self.context = SimpleNamespace(tools=tools)
        self.settings = SimpleNamespace(home_path=home)


class ChatSurfaceTests(unittest.TestCase):
    def test_verbs_registered_owner_only(self):
        for kind in NEW_VERBS:
            self.assertIn(kind, CONTROL_COMMANDS)
            page = COMMAND_DETAILS[kind]
            for field in ("what", "usage", "example", "related"):
                self.assertTrue(str(page.get(field, "")).strip(),
                                f"/{kind} DETAILS.{field} is empty")
            self.assertTrue(LIST_ONELINERS.get(kind, "").strip())
            self.assertNotIn(kind, PUBLIC_COMMANDS)
        grouped = [k for _, kinds in LIST_GROUPS for k in kinds]
        for kind in NEW_VERBS:
            self.assertEqual(grouped.count(kind), 1,
                             f"/{kind} must sit in exactly one /list group")

    def test_usage_on_empty_input(self):
        rt = _FakeRuntime(ToolRegistry(SimpleNamespace()), "/nonexistent")
        for verb, method in (
                ("combo", RuntimeSystemMixin._control_combo),
                ("enum", RuntimeSystemMixin._control_enum),
                ("recon", RuntimeSystemMixin._control_recon),
                ("subbrute", RuntimeSystemMixin._control_subbrute),
                ("chain", RuntimeSystemMixin._control_chain),
                ("disclose", RuntimeSystemMixin._control_disclose)):
            with self.subTest(verb=verb):
                reply = method(rt, "")
                self.assertTrue(reply.startswith("usage:"),
                                f"/{verb} empty input must answer usage")

    def test_enum_roundtrip_free_everywhere(self):
        def fake_client(timeout=15, proxy_url=""):
            return FakeClient({})
        reg = ToolRegistry(SimpleNamespace())
        reg.register_builtins()
        with mock.patch.object(enum_mod, "HttpClient", fake_client):
            rt = _FakeRuntime(reg, "/nonexistent")
            reply = RuntimeSystemMixin._control_enum(rt, "newname123")
        self.assertIn("newname123:", reply)
        self.assertIn("free on:", reply)

    def test_recon_roundtrip(self):
        fake = {("www.example.com", "A"): ["1.1.1.1"],
                ("example.com", "NS"): []}
        reg = ToolRegistry(SimpleNamespace())
        reg.register_builtins()
        with (mock.patch.object(recon_mod, "crt_sh",
                                return_value=["www.example.com"]),
              mock.patch.object(recon_mod, "dns_lookup",
                                lambda n, t="A", timeout=10:
                                fake.get((n, t), [])),
              mock.patch.object(recon_mod, "takeover_check",
                                return_value=None),
              mock.patch.object(recon_mod, "wildcard_control",
                                return_value=None)):
            rt = _FakeRuntime(reg, "/nonexistent")
            reply = RuntimeSystemMixin._control_recon(rt, "example.com")
        self.assertIn("example.com: 2 host(s)", reply)
        self.assertIn("www.example.com", reply)

    def test_subbrute_roundtrip(self):
        fake_dns = {"www.example.com": ["1.1.1.1"]}
        reg = ToolRegistry(SimpleNamespace())
        reg.register_builtins()
        with (mock.patch.object(subbrute_mod, "dns_lookup",
                                lambda h, t="A", timeout=8:
                                fake_dns.get(h, []) if t == "A" else []),
              mock.patch.object(subbrute_mod, "wildcard_control",
                                return_value=None)):
            rt = _FakeRuntime(reg, "/nonexistent")
            reply = RuntimeSystemMixin._control_subbrute(rt, "example.com 1")
        self.assertIn("1 found", reply)
        self.assertIn("www.example.com", reply)

    def test_chain_roundtrip(self):
        tmp = Path(tempfile.mkdtemp())
        chain = EvidenceChain(tmp / "data", "demo")
        chain.add(Finding(module="recon", target="example.com",
                          verdict="takeover",
                          request={"method": "GET",
                                   "url": "https://example.com/"},
                          response={"status": 200, "body": "NoSuchBucket"},
                          detail={}))
        reg = ToolRegistry(SimpleNamespace(home=str(tmp)))
        reg.register_builtins()
        rt = _FakeRuntime(reg, str(tmp))
        self.assertIn("intact",
                      RuntimeSystemMixin._control_chain(rt, "demo verify"))
        self.assertIn("exported 1 record",
                      RuntimeSystemMixin._control_chain(rt, "demo export"))
        self.assertIn("report at",
                      RuntimeSystemMixin._control_chain(
                          rt, "demo report example.com"))

    def test_disclose_roundtrip(self):
        tmp = Path(tempfile.mkdtemp())
        chain = EvidenceChain(tmp / "data", "demo")
        chain.add(Finding(module="recon", target="example.com",
                          verdict="takeover",
                          request={"method": "GET",
                                   "url": "https://example.com/"},
                          response={"status": 200, "body": "NoSuchBucket"},
                          detail={"note": "cname to s3"}))
        reg = ToolRegistry(SimpleNamespace(home=str(tmp)))
        reg.register_builtins()
        rt = _FakeRuntime(reg, str(tmp))
        self.assertIn("critical",
                      RuntimeSystemMixin._control_disclose(
                          rt, "score rce financial unauth bulk"))
        with mock.patch.object(disclose_mod, "HttpClient",
                              lambda timeout=15: FakeClient({
                                  (("GET", "security.txt")):
                                  FakeResp("Contact: mailto:s@example.com\n",
                                           200)})):
            reply = RuntimeSystemMixin._control_disclose(
                rt, "contacts example.com")
        self.assertIn("s@example.com", reply)
        reply = RuntimeSystemMixin._control_disclose(
            rt, "draft demo example.com")
        self.assertIn("disclosure draft at", reply)

    def test_combo_roundtrip(self):
        tmp = Path(tempfile.mkdtemp())
        reg = ToolRegistry(SimpleNamespace(home=str(tmp)))
        reg.register_builtins()
        with mock.patch.object(cc_mod, "HttpClient", FakeClient):
            rt = _FakeRuntime(reg, str(tmp))
            reply = RuntimeSystemMixin._control_combo(
                rt, "a@b.com:pw1 ebay")
        self.assertIn("combo: 1 check(s)", reply)
        self.assertIn("unknown 1", reply)
        self.assertTrue((tmp / "data" / "combos_checked.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
