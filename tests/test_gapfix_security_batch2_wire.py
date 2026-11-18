"""Lane 57: wire in the owner's three newest security commits.

- seed_sources.py: keyless seed sources — passive DNS, URL history,
  scan indexes, host intel, email reputation; harvested URLs into
  brute words for subbrute.
- cards.py: BIN intel, Luhn generation, structural + gateway checks.
- engage.py: the engagement chain (seed -> recon -> brute -> probe ->
  evidence -> report) as an `nm engage` command.

Repairs proven here (all execution-proven, all in the owner's files):
- cards.py check_gateway called HttpClient.post(), which does not
  exist — the only path that earns a LIVE verdict always answered
  "unknown" with a transport AttributeError. Now post_form(); a fake
  Stripe token create earns "live", a card_declined earns "dead".
- seed_sources.py documented kinds as host/url/email/port/vuln while
  the gravatar fetcher returns kind "profile" — the doc now says so.
- engage.py EvidenceChain(case_dir.parent, ...) doubled the path:
  the chain landed at .../data/evidence/evidence/<case>, invisible
  to /chain verify and to report.build(case_dir), so the report stage
  errored on every single run. Root is now the data dir.
- engage.py imported pool_from_context from tools.proxy_rotate,
  which does not exist — the pool was always None. Now imported
  from tools.proxy_rotation, where it lives.
- engage.py _case_root read getattr(context, "home", ".") — the CLI
  context is an AgentContext, which has .settings and no .home, so
  case dirs landed in the caller's working directory. Now resolved
  through settings.home_path / settings.home.
- engage.py built ExposureScanner(context) — the scanner takes a
  Fetcher, not a context, so every scan() raised AttributeError and
  the probe stage silently probed zero hosts. Now ExposureScanner().
- engage.py crashed a host's whole probe loop when a finding carried
  evidence=None (the _finding default) — the pack is now guarded.
- engage.py recorded a severity per probe finding, but audit_web
  findings carry no severity key at all — none is recorded now.
- engage.py imported Verdict and never used it (F401) — removed.
- engage.py shipped add_parser()/handle(), a shape nothing in the
  repo calls — `nm engage` was unreachable. Now a real _cmd_engage
  with a parser.py entry, a CLI_ALIASES entry and a dispatch entry.

No live target, no real credentials, no real card numbers: every
network-bound check runs against fakes.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.security.cards as cards_mod
import nomorals.security.seed_sources as seed_mod
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.cmdline.commands import engage as engage_mod
from nomorals.cmdline.dispatch import _COMMANDS
from nomorals.cmdline.parser import CLI_ALIASES, _parser
from nomorals.security import verify_chain
from nomorals.security.evidence import EvidenceChain, Finding
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
)
from nomorals.tools.registry import ToolRegistry

NEW_TOOLS = (
    "seed_harvest", "seed_wordlist", "ip_intel",
    "bin_lookup", "bin_generate", "card_check", "card_spam",
)
NEW_VERBS = ("seed", "bin")


class FakeResp:
    def __init__(self, text="", status=200):
        self.text = text
        self.status = status

    @property
    def ok(self):
        return 200 <= self.status < 400


class FakeClient:
    """Method-aware fake HttpClient: {(method, substring): FakeResp}."""

    def __init__(self, *a, **k):
        self.posts = []

    def get(self, url, **kw):
        if "otx.alienvault.com" in url:
            return FakeResp(json.dumps({"passive_dns": [
                {"hostname": "a.example.com", "first": "2020",
                 "last": "2024"},
                {"hostname": "a.example.com", "first": "2020",
                 "last": "2024"},
                {"hostname": "other.com", "first": "2020",
                 "last": "2024"}]}))
        if "hackertarget.com" in url:
            return FakeResp("b.example.com,1.2.3.4\nc.example.com,1.2.3.5\n")
        if "rapiddns.io" in url:
            return FakeResp("<td>d.example.com</td><td>e.example.com</td>")
        if "web.archive.org" in url:
            return FakeResp(json.dumps([["original"],
                                        ["http://example.com/admin/login"],
                                        ["http://example.com/blog/post"]]))
        if "urlscan.io" in url:
            return FakeResp(json.dumps({"results": [
                {"page": {"url": "http://example.com/x",
                          "domain": "example.com", "ip": "1.1.1.1",
                          "asnname": "AS1"}}]}))
        if "index.commoncrawl.org/collinfo.json" in url:
            return FakeResp(json.dumps([{"id": "CC-MAIN-2024-01"}]))
        if "index.commoncrawl.org" in url:
            return FakeResp(json.dumps({"url": "http://example.com/y"})
                            + "\n")
        if "internetdb.shodan.io" in url:
            return FakeResp(json.dumps({"ports": [80, 443],
                                        "vulns": ["CVE-2021-1"]}))
        if "emailrep.io" in url:
            return FakeResp(json.dumps({"reputation": "high",
                                        "suspicious": False,
                                        "details": {"breaches": 1,
                                                    "profiles": ["gh"]}}))
        if "gravatar.com" in url:
            return FakeResp(json.dumps({"entry": [
                {"profileUrl": "https://x",
                 "name": {"formatted": "A"},
                 "accounts": [{"shortname": "gh"}]}]}))
        if "lookup.binlist.net" in url:
            return FakeResp(json.dumps({
                "scheme": "visa", "type": "debit", "brand": "Visa",
                "bank": {"name": "Test Bank"},
                "country": {"name": "Nigeria", "alpha2": "NG"},
                "prepaid": False}))
        return FakeResp("", 404)

    def post_form(self, url, form, **kw):
        self.posts.append((url, form))
        if form.get("card[number]") == "4242424242424242":
            return FakeResp(json.dumps({"id": "tok_live"}))
        return FakeResp(json.dumps({"error": {
            "code": "card_declined", "message": "do not honor"}}), 402)


class FakeScanner:
    """ExposureScanner stand-in: one finding with no evidence pack
    (the _finding default) and one with a pack — the old probe loop
    died on the first."""

    def __init__(self, *a, **k):
        pass

    def scan(self, url):
        return {"findings": [
            {"kind": "exposed", "detail": "open dir",
             "evidence": None},
            {"kind": "info", "detail": "headers ok",
             "evidence": {"request": {"method": "GET", "url": url},
                          "response": {"status": 200}}}]}


class RegisterTests(unittest.TestCase):
    def test_register_builtins_clean_and_new_tools_present(self):
        reg = ToolRegistry(SimpleNamespace())
        reg.register_builtins()
        failed = [f["module"] for f in reg.failed_modules()]
        self.assertNotIn("security", failed)
        for name in NEW_TOOLS:
            self.assertIn(name, reg.names(), f"tool {name} missing")

    def test_each_module_registers(self):
        for mod in (seed_mod, cards_mod):
            self.assertTrue(hasattr(mod, "register"),
                            f"{mod.__name__} has no register()")

    def test_security_package_exports(self):
        import nomorals.security as sec
        for name in ("SOURCES", "KEYED", "harvest", "wordlist",
                     "bin_check", "BRANDS", "Card", "luhn_ok",
                     "luhn_complete", "brand_of", "bin_lookup",
                     "generate", "check_structural", "check_gateway",
                     "spam"):
            self.assertTrue(hasattr(sec, name), f"security.{name} missing")
        self.assertIn("harvest", sec.__all__)
        self.assertIn("Card", sec.__all__)
        # No shadowing: the package attributes `cards` / `seed_sources`
        # are still the submodules, not the exported functions.
        self.assertTrue(hasattr(sec.cards, "register"))
        self.assertTrue(hasattr(sec.seed_sources, "register"))

    def test_all_promises_exist(self):
        for mod in (seed_mod, cards_mod):
            missing = [n for n in mod.__all__ if not hasattr(mod, n)]
            self.assertEqual(missing, [], f"{mod.__name__}: {missing}")


class SeedSourcesTests(unittest.TestCase):
    def setUp(self):
        self._patch = mock.patch.object(seed_mod, "HttpClient", FakeClient)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()

    def test_harvest_dedupes_and_buckets(self):
        data = seed_mod.harvest("example.com", email="a@example.com",
                                ips=["1.2.3.4"])
        self.assertEqual(data["sources_run"], 9)
        self.assertEqual(data["sources_failed"], [])
        self.assertEqual(sorted(h["value"] for h in data["host"]),
                         ["a.example.com", "b.example.com",
                          "c.example.com", "d.example.com",
                          "e.example.com"])
        self.assertEqual(len(data["url"]), 4)
        self.assertEqual([p["value"] for p in data["port"]],
                         ["80", "443"])
        self.assertEqual([v["value"] for v in data["vuln"]],
                         ["CVE-2021-1"])
        self.assertEqual(data["email"][0]["extra"]["reputation"], "high")
        self.assertEqual(data["profile"][0]["extra"]["name"], "A")
        self.assertIn("hibp_account", data["keyed_not_used"])

    def test_harvest_only_filter(self):
        data = seed_mod.harvest("example.com", only=["otx"])
        self.assertEqual(data["sources_run"], 1)
        self.assertEqual([h["value"] for h in data["host"]],
                         ["a.example.com"])

    def test_wordlist_from_harvested_urls(self):
        data = seed_mod.harvest("example.com")
        self.assertEqual(sorted(seed_mod.wordlist(data["url"])),
                         ["admin", "blog", "login", "post", "x", "y"])

    def test_bin_check_is_ip_intel(self):
        out = seed_mod.bin_check("1.2.3.4")
        self.assertEqual(out["ip"], "1.2.3.4")
        self.assertEqual(len(out["items"]), 3)


class CardsTests(unittest.TestCase):
    def test_luhn_and_brand(self):
        self.assertTrue(cards_mod.luhn_ok("4242424242424242"))
        self.assertFalse(cards_mod.luhn_ok("4242424242424241"))
        self.assertTrue(cards_mod.luhn_ok("378282246310005"))
        self.assertEqual(cards_mod.luhn_complete("424242424242424"),
                         "4242424242424242")
        self.assertEqual(cards_mod.brand_of("4242424242424242"),
                         ("visa", 16, 3))
        self.assertEqual(cards_mod.brand_of("378282246310005"),
                         ("amex", 15, 4))
        self.assertIsNone(cards_mod.brand_of("9999999999999999"))

    def test_generate_is_luhn_valid_and_brand_shaped(self):
        cards = cards_mod.generate("424242", 3)
        self.assertEqual(len(cards), 3)
        for card in cards:
            self.assertTrue(cards_mod.luhn_ok(card.number))
            self.assertEqual(card.brand, "visa")
            self.assertEqual(len(card.cvv), 3)

    def test_structural_verdicts(self):
        good = cards_mod.check_structural(cards_mod.Card(
            number="4242424242424242", month=12, year=2099))
        self.assertEqual((good.verdict, good.reason),
                         ("well_formed", "structural only"))
        dead = cards_mod.check_structural(cards_mod.Card(
            number="4242424242424242", month=1, year=2001))
        self.assertEqual((dead.verdict, dead.reason), ("dead", "expired"))
        unknown = cards_mod.check_structural(cards_mod.Card(
            number="9999999999999999"))
        self.assertEqual(unknown.verdict, "dead")

    def test_gateway_live_and_dead_via_post_form(self):
        with mock.patch.object(cards_mod, "HttpClient", FakeClient):
            live = cards_mod.check_gateway(cards_mod.Card(
                number="4242424242424242", month=12, year=2099,
                cvv="123"), "pk_test_x")
            self.assertEqual((live.verdict, live.reason),
                             ("live", "gateway accepted"))
            dead = cards_mod.check_gateway(cards_mod.Card(
                number="4000000000000002", month=12, year=2099,
                cvv="123"), "pk_test_x")
            self.assertEqual(dead.verdict, "dead")
            self.assertIn("card_declined", dead.reason)


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
                ("seed", RuntimeSystemMixin._control_seed),
                ("bin", RuntimeSystemMixin._control_bin)):
            out = method(rt, "")
            self.assertIn("usage: /" + verb, out)

    def test_seed_and_bin_round_trips(self):
        with mock.patch.object(seed_mod, "HttpClient", FakeClient), \
                mock.patch.object(cards_mod, "HttpClient", FakeClient):
            reg = ToolRegistry(SimpleNamespace())
            reg.register_builtins()
            rt = _FakeRuntime(reg, "/nonexistent")
            out = RuntimeSystemMixin._control_seed(rt, "example.com")
            self.assertIn("example.com:", out)
            self.assertIn("a.example.com", out)
            self.assertIn("9 item(s)", out)
            out = RuntimeSystemMixin._control_seed(rt, "ip 1.2.3.4")
            self.assertIn("80", out)
            self.assertIn("CVE-2021-1", out)
            out = RuntimeSystemMixin._control_seed(
                rt, "words http://example.com/admin/login")
            self.assertIn("admin", out)
            out = RuntimeSystemMixin._control_bin(rt, "424242")
            self.assertIn("Test Bank", out)
            self.assertIn("Nigeria", out)
            out = RuntimeSystemMixin._control_bin(rt, "gen 424242 2")
            self.assertIn("2 card(s)", out)
            lines = [ln for ln in out.splitlines() if "|" in ln]
            self.assertEqual(len(lines), 2)
            for ln in lines:
                self.assertTrue(cards_mod.luhn_ok(ln.split("|")[0].strip()))
            out = RuntimeSystemMixin._control_bin(
                rt, "check 4242424242424242 12 2099 123")
            self.assertIn("well_formed", out)
            self.assertIn("structural only", out)
            out = RuntimeSystemMixin._control_bin(rt, "spam 424242 2")
            self.assertIn("well_formed 2", out)


def _fake_harvest(domain, **kw):
    return {"domain": domain, "email": "", "sources_run": 2,
            "sources_failed": [], "total": 3,
            "host": [{"kind": "host", "value": "a.example.com",
                      "source": "otx", "extra": {}},
                     {"kind": "host", "value": "b.example.com",
                      "source": "hackertarget", "extra": {}}],
            "url": [{"kind": "url",
                     "value": "http://example.com/admin",
                     "source": "wayback", "extra": {}}],
            "keyed_not_used": []}


def _fake_recon(domain, **kw):
    return {"domain": domain, "wildcard": None, "host_count": 1,
            "axfr": {}, "takeover_candidates": [],
            "assets": [{"host": "a.example.com", "records": {},
                        "cname_chain": [], "takeover": None,
                        "sources": ["otx"]}]}


class _FakeSubBrute:
    def __init__(self, domain, **kw):
        self.domain = domain

    def run(self):
        return {"domain": self.domain, "wildcard": None, "queries": 5,
                "seconds": 0.1, "found": 1, "wildcard_matches": 0,
                "hits": [{"host": "c.example.com", "addresses": [],
                          "cname": [], "wildcard": False,
                          "via": "brute"}]}


class EngageTests(unittest.TestCase):
    def _ctx(self, home):
        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(home), home=str(home)))

    def test_case_root_uses_settings_home(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            got = engage_mod._case_root(self._ctx(home), "Demo Case!")
            self.assertEqual(got, home / "data" / "evidence" / "demo-case")

    def test_chain_lands_at_canonical_path_and_report_builds(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            ctx = self._ctx(home)
            with mock.patch.object(engage_mod.seed_sources, "harvest",
                                  _fake_harvest), \
                    mock.patch.object(engage_mod.recon, "recon",
                                      _fake_recon), \
                    mock.patch.object(engage_mod.subbrute, "SubBrute",
                                      _FakeSubBrute), \
                    mock.patch.object(engage_mod.disclose, "find_contacts",
                                      lambda d: {"contacts": [
                                          {"email": "security@example.com",
                                           "source": "security.txt"}],
                                          "count": 1}), \
                    mock.patch("nomorals.security.audit_web.ExposureScanner",
                               FakeScanner):
                res = engage_mod.run_engage(ctx, "example.com")
            case_dir = home / "data" / "evidence" / "example-com"
            # The old code wrote the chain one level deeper, where
            # nothing else looks; the doubled dir must not exist.
            self.assertFalse(
                (home / "data" / "evidence" / "evidence").exists())
            self.assertTrue((case_dir / "chain.jsonl").is_file())
            self.assertTrue((case_dir / "seed.json").is_file())
            self.assertTrue((case_dir / "recon.json").is_file())
            self.assertTrue((case_dir / "subbrute.json").is_file())
            self.assertTrue((case_dir / "engage.json").is_file())
            self.assertNotIn("error", res["stages"]["report"])
            self.assertEqual(res["integrity"]["state"], "intact")
            # 2 seed records + 6 probe records (2 findings x 3 hosts —
            # seed hosts a/b plus brute hit c; the first finding carries
            # evidence=None, which the old probe loop died on, so 8 —
            # not 2 — pins that repair).
            self.assertEqual(res["integrity"]["records"], 8)
            self.assertEqual(res["stages"]["probe"]["hosts_probed"], 3)
            self.assertEqual(res["contacts"][0]["email"],
                             "security@example.com")
            # The same chain, read the way /chain verify reads it.
            self.assertEqual(verify_chain(case_dir / "chain.jsonl")
                             ["state"], "intact")
            report_file = res["stages"]["report"].get("file")
            self.assertTrue(report_file and Path(report_file).is_file())

    def test_dry_run_skips_network_stages_honestly(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            res = engage_mod.run_engage(self._ctx(home), "example.com",
                                        dry_run=True)
            for stage in ("seed", "recon", "brute", "probe"):
                self.assertEqual(res["stages"][stage]["skipped"],
                                 "dry run")
            # No chain exists yet, so the report stage refuses rather
            # than rendering an empty case as fact.
            self.assertIn("error", res["stages"]["report"])
            self.assertIn("no evidence chain",
                          res["stages"]["report"]["error"])
            self.assertEqual(res["integrity"]["state"], "missing")

    def test_chain_verify_names_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            chain = EvidenceChain(home / "data", "demo")
            chain.add(Finding(module="seed", target="example.com",
                              verdict="info", request={}, response={},
                              detail={}))
            path = home / "data" / "evidence" / "demo" / "chain.jsonl"
            self.assertEqual(verify_chain(path)["state"], "intact")
            lines = path.read_text().splitlines()
            row = json.loads(lines[0])
            row["verdict"] = "pwned"
            lines[0] = json.dumps(row)
            path.write_text("\n".join(lines) + "\n")
            bad = verify_chain(path)
            self.assertEqual(bad["state"], "broken")
            self.assertEqual(bad["broken_at"], 1)


class CliTests(unittest.TestCase):
    def test_engage_in_parser_aliases_and_dispatch(self):
        self.assertIn("engage", CLI_ALIASES)
        args = _parser().parse_args(
            ["engage", "example.com", "--case", "demo", "--dry-run",
             "--json"])
        self.assertEqual(args.domain, "example.com")
        self.assertEqual(args.case, "demo")
        self.assertTrue(args.dry_run)
        self.assertEqual(args.start_from, "seed")
        self.assertEqual(args.until, "report")
        self.assertIn("engage", _COMMANDS)
        self.assertTrue(callable(_COMMANDS["engage"].handler))

    def test_cmd_engage_full_run_returns_zero(self):
        with tempfile.TemporaryDirectory() as td:
            home = Path(td) / "home"
            ctx = self._ctx(home)
            args = _parser().parse_args(["engage", "example.com"])
            with mock.patch.object(engage_mod.seed_sources, "harvest",
                                  _fake_harvest), \
                    mock.patch.object(engage_mod.recon, "recon",
                                      _fake_recon), \
                    mock.patch.object(engage_mod.subbrute, "SubBrute",
                                      _FakeSubBrute), \
                    mock.patch.object(engage_mod.disclose, "find_contacts",
                                      lambda d: {"contacts": [], "count": 0}), \
                    mock.patch("nomorals.security.audit_web.ExposureScanner",
                               FakeScanner):
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = engage_mod._cmd_engage(args, ctx)
            self.assertEqual(rc, 0)
            self.assertIn("integrity: intact", buf.getvalue())

    def _ctx(self, home):
        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(home), home=str(home)))


if __name__ == "__main__":
    unittest.main()
