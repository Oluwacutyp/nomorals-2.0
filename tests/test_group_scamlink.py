"""The scam-link OSINT sensor (DESIGN §4 beyond-parity / §9.4):
links posted in claimed groups get dns/whois/look checks against
the keyless threat feeds; a known-bad link is auto-deleted +
warned on the shared escalation spine, a suspicious link stages
a `scam` flag proposal (a human confirms — the sensor brands
nobody), a clean link passes, and a dead feed fails open.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore; the RuntimeGroupMixin stub rig from the reputation
suite; the sensor's network is a fake transport — no live feed
is ever called from a test.
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.core.errors import ToolError
from nomorals.security.payload_inspection import normalize_envelope
from nomorals.social.chat.group_defense import DefenseConfig
from nomorals.social.chat.group_scamlink import (
    LinkVerdict,
    check_link,
    check_scamlink,
    extract_links,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface, default_surface

KEY = "telegram-bot:-1001234567"
NOW = 1_700_000_000.0
DAY = 86400.0

EVIL = "https://evil.example.com/prize"
SHADY = "https://fresh-scam.example.net/login"
CLEAN = "https://example.org/about"


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def _env(text="", **kw):
    return normalize_envelope(kw.pop("raw", None) or kw or None,
                              text=text)


def _cfg(**over):
    base = DefenseConfig()
    for key, value in over.items():
        setattr(base, key, value)
    return base


# ── fake feeds ──────────────────────────────────────────────────


class _FakeFetch:
    """(method, url-needle) -> payload dict or exception.  Records
    every call so tests can prove early exit and caching."""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple[str, str]] = []

    def __call__(self, method, url, payload=None, timeout=2.5):
        self.calls.append((method, url))
        for want_method, needle, out in self.routes:
            if want_method == method and needle in url:
                if isinstance(out, Exception):
                    raise out
                return out
        raise AssertionError(f"unexpected fetch {method} {url}")


def _urlhaus_ok(threat="malware_download"):
    return {"query_status": "ok", "threat": threat,
            "url_count": 3, "tags": ["malware"]}


def _urlhaus_miss():
    return {"query_status": "no_results"}


def _threatfox_hit(confidence=90):
    # The real ThreatFox v1 shape (the repo's own osint.py
    # parses it the same way): data[] with an integer
    # confidence_level 0-100.
    return {"query_status": "ok",
            "data": [{"ioc": EVIL, "confidence_level": confidence,
                      "threat_type_desc": "phishing"}]}


def _threatfox_miss():
    return {"query_status": "no_results"}


def _urlscan_clean():
    return {"total": 0, "results": []}


def _urlscan_malicious():
    # The real urlscan.io search shape: results[].verdicts
    # .overall.malicious.
    return {"total": 1, "results": [{
        "page": {"url": EVIL, "domain": "evil.example.com"},
        "verdicts": {"overall": {"malicious": True,
                                 "score": 100}},
        "_id": "abc123"}]}


def _dns_ok(domain, timeout=2.0):
    return ["93.184.216.34"]


def _dns_dead(domain, timeout=2.0):
    return []


def _dns_down(domain, timeout=2.0):
    raise ToolError("DNS query failed")


def _rdap_old(domain, timeout=2.5):
    return {"domain": domain, "registration": "2019-04-02",
            "registrar": "Example Registrar"}


def _rdap_young(domain, timeout=2.5):
    young = time.strftime("%Y-%m-%d", time.gmtime(NOW - 5 * DAY))
    return {"domain": domain, "registration": young,
            "registrar": "Cheap Registrar"}


def _rdap_down(domain, timeout=2.5):
    raise ToolError("RDAP lookup failed")


def _clean_fetch():
    return _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_miss()),
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/host/",
         _urlhaus_miss()),
        ("POST_JSON", "threatfox-api.abuse.ch",
         _threatfox_miss()),
        ("GET", "urlscan.io", _urlscan_clean()),
    ])


def _check(url, *, fetch=None, dns=None, rdap=None, **kw):
    return check_link(url, fetch=fetch or _clean_fetch(),
                      dns=dns or _dns_ok, rdap=rdap or _rdap_old,
                      now=NOW, **kw)


# ── link extraction ─────────────────────────────────────────────


def test_extract_links_from_entities_and_bare_text():
    env = _env("see https://example.org/about and this one",
               raw={"entities": [
                   {"type": "url", "offset": 4, "length": 25},
                   {"type": "text_link", "offset": 34, "length": 8,
                    "url": "https://t.me/somechannel"}]})
    links = extract_links(env)
    urls = [u for u, _ in links]
    assert "https://example.org/about" in urls
    assert "https://t.me/somechannel" in urls
    assert len(urls) == len(set(urls))


def test_extract_links_dedupes_and_skips_non_http():
    env = _env("tg://resolve?domain=x https://example.org/a "
               "https://example.org/a.")
    urls = [u for u, _ in extract_links(env)]
    assert urls == ["https://example.org/a"]


def test_extract_links_empty_without_links():
    assert extract_links(_env("just words, no links")) == []


# ── verdict ladder ──────────────────────────────────────────────


def test_urlhaus_url_hit_is_bad_and_exits_early():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_ok("phishing")),
    ])
    verdict = _check(EVIL, fetch=fetch)
    assert verdict.verdict == "bad"
    assert verdict.sources == ["urlhaus"]
    assert "phishing" in verdict.detail
    assert len(fetch.calls) == 1  # no further feed consulted


def test_urlhaus_host_hit_is_bad():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_miss()),
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/host/",
         _urlhaus_ok()),
    ])
    verdict = _check(EVIL, fetch=fetch)
    assert verdict.verdict == "bad"
    assert "host listed" in verdict.detail


def test_threatfox_high_confidence_is_bad():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_miss()),
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/host/",
         _urlhaus_miss()),
        ("POST_JSON", "threatfox-api.abuse.ch",
         _threatfox_hit(90)),
    ])
    verdict = _check(EVIL, fetch=fetch)
    assert verdict.verdict == "bad"
    assert verdict.sources == ["threatfox"]


def test_threatfox_medium_confidence_is_not_bad():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_miss()),
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/host/",
         _urlhaus_miss()),
        ("POST_JSON", "threatfox-api.abuse.ch",
         _threatfox_hit(50)),
        ("GET", "urlscan.io", _urlscan_clean()),
    ])
    verdict = _check(EVIL, fetch=fetch)
    assert verdict.verdict == "clean"


def test_urlscan_malicious_is_bad():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/url/",
         _urlhaus_miss()),
        ("POST_FORM", "urlhaus-api.abuse.ch/v1/host/",
         _urlhaus_miss()),
        ("POST_JSON", "threatfox-api.abuse.ch",
         _threatfox_miss()),
        ("GET", "urlscan.io", _urlscan_malicious()),
    ])
    verdict = _check(EVIL, fetch=fetch)
    assert verdict.verdict == "bad"
    assert verdict.sources == ["urlscan"]


def test_unresolvable_domain_is_suspicious_not_bad():
    verdict = _check(SHADY, dns=_dns_dead)
    assert verdict.verdict == "suspicious"
    assert "does not resolve" in verdict.detail


def test_young_domain_is_suspicious_not_bad():
    verdict = _check(SHADY, rdap=_rdap_young)
    assert verdict.verdict == "suspicious"
    assert "registered" in verdict.detail
    assert "days ago" in verdict.detail


def test_old_clean_domain_is_clean():
    verdict = _check(CLEAN)
    assert verdict.verdict == "clean"
    assert verdict.sources  # feeds answered


def test_all_feeds_down_is_unknown_and_fails_open():
    fetch = _FakeFetch([
        ("POST_FORM", "urlhaus-api.abuse.ch", ToolError("down")),
        ("POST_JSON", "threatfox-api.abuse.ch",
         ToolError("down")),
        ("GET", "urlscan.io", ToolError("down")),
    ])
    verdict = _check(CLEAN, fetch=fetch, dns=_dns_down,
                     rdap=_rdap_down)
    assert verdict.verdict == "unknown"


# ── the sensor: config, cache, verdicts ─────────────────────────


def _bad_checker(url, **kw):
    return LinkVerdict("bad", url, "evil.example.com",
                       ["urlhaus"], "listed by urlhaus as phishing")


def _suspect_checker(url, **kw):
    return LinkVerdict("suspicious", url, "fresh-scam.example.net",
                       ["rdap"], "domain registered 5 days ago")


def _clean_checker(url, **kw):
    return LinkVerdict("clean", url, "example.org",
                       ["urlhaus", "threatfox", "urlscan", "dns",
                        "rdap"], "no feed fired")


def test_sensor_disabled_returns_nothing():
    env = _env(f"look {EVIL}")
    assert check_scamlink(env, _cfg(scamlink_enabled=False),
                          checker=_bad_checker) is None


def test_sensor_ignores_messages_without_links():
    env = _env("just words")
    assert check_scamlink(env, _cfg(), checker=_bad_checker) is None


def test_bad_link_returns_delete_warn_hit():
    env = _env(f"free money {EVIL}")
    hit = check_scamlink(env, _cfg(), checker=_bad_checker, now=NOW)
    assert hit is not None
    assert hit.rule == "scamlink"
    assert hit.action == "delete_warn"
    assert hit.signal["verdict"] == "bad"
    assert hit.signal["url"] == EVIL
    assert "phishing" in hit.reason


def test_suspicious_link_returns_flag_hit_not_delete():
    env = _env(f"new site {SHADY}")
    hit = check_scamlink(env, _cfg(), checker=_suspect_checker,
                         now=NOW)
    assert hit is not None
    assert hit.rule == "scamlink"
    assert hit.action == "flag"
    assert hit.signal["verdict"] == "suspicious"


def test_clean_and_unknown_links_return_nothing():
    env = _env(f"fine {CLEAN}")
    assert check_scamlink(env, _cfg(), checker=_clean_checker,
                          now=NOW) is None

    def _unknown(url, **kw):
        return LinkVerdict("unknown", url, "example.org", [],
                           "all feeds failed")

    assert check_scamlink(env, _cfg(), checker=_unknown,
                          now=NOW) is None


def test_bad_beats_suspicious_when_both_present():
    def _mixed(url, **kw):
        if "evil" in url:
            return _bad_checker(url)
        return _suspect_checker(url)

    env = _env(f"{SHADY} then {EVIL}")
    hit = check_scamlink(env, _cfg(), checker=_mixed, now=NOW)
    assert hit is not None
    assert hit.signal["verdict"] == "bad"


def test_verdict_cache_stops_repeat_checks():
    calls: list[str] = []

    def _counting(url, **kw):
        calls.append(url)
        return _bad_checker(url)

    env = _env(f"again {EVIL}")
    cache: dict = {}
    first = check_scamlink(env, _cfg(), checker=_counting,
                           cache=cache, now=NOW)
    second = check_scamlink(env, _cfg(), checker=_counting,
                            cache=cache, now=NOW + 60)
    assert first is not None and second is not None
    assert calls == [EVIL]  # second message served from cache


def test_cache_expires_and_rechecks():
    calls: list[str] = []

    def _counting(url, **kw):
        calls.append(url)
        return _clean_checker(url)

    env = _env(f"again {CLEAN}")
    cache: dict = {}
    check_scamlink(env, _cfg(), checker=_counting, cache=cache,
                   now=NOW)
    check_scamlink(env, _cfg(), checker=_counting, cache=cache,
                   now=NOW + 25 * 3600)  # clean TTL is 24h
    assert len(calls) == 2


def test_max_links_caps_checks_per_message():
    calls: list[str] = []

    def _counting(url, **kw):
        calls.append(url)
        return _clean_checker(url)

    text = " ".join(
        f"https://site{i}.example.org/x" for i in range(5))
    env = _env(text)
    check_scamlink(env, _cfg(scamlink_max_links=3),
                   checker=_counting, now=NOW)
    assert len(calls) == 3


# ── config: the scamlink block reads both ways ──────────────────


def test_defense_config_defaults_carry_scamlink():
    cfg = DefenseConfig()
    assert cfg.scamlink_enabled is True
    assert cfg.scamlink_action == "delete_warn"
    assert cfg.scamlink_suspicious == "flag"
    assert cfg.scamlink_max_links == 3
    assert cfg.scamlink_young_days == 30


def test_defense_config_reads_stored_scamlink_block():
    settings = {"defense": {"scamlink": {
        "enabled": False, "action": "warn", "suspicious": "off",
        "max_links": 1, "young_days": 7}}}
    cfg = DefenseConfig.from_settings(settings)
    assert cfg.scamlink_enabled is False
    assert cfg.scamlink_action == "warn"
    assert cfg.scamlink_suspicious == "off"
    assert cfg.scamlink_max_links == 1
    assert cfg.scamlink_young_days == 7
    # the escalation-block branch reads it too
    settings["escalation"] = {"warn_limit": 7,
                              "final_action": "ban", "window_days": 7,
                              "intervals": []}
    cfg2 = DefenseConfig.from_settings(settings)
    assert cfg2.scamlink_enabled is False
    assert cfg2.scamlink_young_days == 7


# ── surface: stage 3 hook ──────────────────────────────────────


def test_bad_link_deleted_struck_and_audited(tmp_path):
    surface = _surface(tmp_path)
    surface.link_checker = _bad_checker
    out = surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"free money {EVIL}", message_id="m1", ts=NOW)
    kinds = [(a.get("kind"), a.get("rule")) for a in out["actions"]]
    assert ("delete", "scamlink") in kinds
    hit_action = next(a for a in out["actions"]
                      if a.get("rule") == "scamlink")
    assert hit_action.get("url") == EVIL
    member = surface.store.group(KEY)["members"]["7"]
    assert len(member.get("strikes") or []) == 1
    assert member["strikes"][0]["rule"] == "scamlink"
    audits = [a for a in surface.audit(KEY, limit=50)
              if a.get("kind") == "scamlink"]
    assert audits and EVIL in str(audits[-1].get("reason") or "")


def test_suspicious_link_stages_a_scam_flag_not_a_delete(tmp_path):
    surface = _surface(tmp_path)
    surface.link_checker = _suspect_checker
    surface.record_join(KEY, "7", username="ada", name="Ada")
    out = surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"new site {SHADY}", message_id="m2", ts=NOW)
    assert not [a for a in out["actions"]
                if a.get("rule") == "scamlink"]
    member = surface.store.group(KEY)["members"]["7"]
    assert not member.get("strikes")
    props = surface.flag_proposals(KEY, now=NOW)
    assert len(props) == 1
    assert props[0]["category"] == "scam"
    assert props[0]["source"] == "sensor"
    assert SHADY in props[0]["note"]
    audits = [a for a in surface.audit(KEY, limit=50)
              if a.get("kind") == "scamlink_suspect"]
    assert audits


def test_suspicious_link_does_not_restage_a_duplicate(tmp_path):
    surface = _surface(tmp_path)
    surface.link_checker = _suspect_checker
    surface.record_join(KEY, "7", username="ada", name="Ada")
    for i in range(2):
        surface.handle_message(
            KEY, "7", username="ada", name="Ada",
            text=f"new site {SHADY}", message_id=f"m{i}",
            ts=NOW + i)
    assert len(surface.flag_proposals(KEY, now=NOW + 1)) == 1


def test_admin_link_still_deleted_but_never_struck(tmp_path):
    """House rule (test_admin_payload_still_deleted_but_never_
    struck): a known-bad link is a safety delete on anyone
    except the owner ladder — the strike is what privilege
    blocks, not the delete."""
    surface = _surface(tmp_path)
    surface.link_checker = _bad_checker
    out = surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"free money {EVIL}", message_id="m3", ts=NOW,
        sender_is_admin=True)
    kinds = [(a.get("kind"), a.get("rule")) for a in out["actions"]]
    assert ("delete", "scamlink") in kinds
    assert surface.strikes_for(KEY, "7", now=NOW)["count"] == 0
    audits = [a for a in surface.audit(KEY, limit=50)
              if a.get("kind") == "scamlink"]
    assert audits


def test_admin_suspicious_link_is_audited_never_proposed(tmp_path):
    """Suspicious is weak evidence — it never stages a scam
    proposal against an admin; the audit is the record."""
    surface = _surface(tmp_path)
    surface.link_checker = _suspect_checker
    surface.record_join(KEY, "7", username="ada", name="Ada")
    surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"new site {SHADY}", message_id="m3b", ts=NOW,
        sender_is_admin=True)
    assert surface.flag_proposals(KEY, now=NOW) == []
    audits = [a for a in surface.audit(KEY, limit=50)
              if a.get("kind") == "scamlink_suspect"]
    assert audits


def test_autonomy_off_is_observe_only(tmp_path):
    surface = _surface(tmp_path)
    surface.link_checker = _bad_checker
    surface.update_settings(KEY, defense={"autonomy": False})
    out = surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"free money {EVIL}", message_id="m4", ts=NOW)
    assert not [a for a in out["actions"]
                if a.get("rule") == "scamlink"]
    member = surface.store.group(KEY)["members"]["7"]
    assert not member.get("strikes")
    audits = [a for a in surface.audit(KEY, limit=50)
              if a.get("kind") == "scamlink"]
    assert audits and audits[-1].get("suppressed") == "autonomy_off"


def test_suspicious_off_stages_nothing(tmp_path):
    surface = _surface(tmp_path)
    surface.link_checker = _suspect_checker
    surface.record_join(KEY, "7", username="ada", name="Ada")
    surface.update_settings(
        KEY, defense={"scamlink": {"suspicious": "off"}})
    surface.handle_message(
        KEY, "7", username="ada", name="Ada",
        text=f"new site {SHADY}", message_id="m5", ts=NOW)
    assert surface.flag_proposals(KEY, now=NOW) == []


# ── runtime: ban path + grammar ─────────────────────────────────


class _Chat:
    def __init__(self, key=KEY, platform="telegram-bot",
                 kind="group", title="Room"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "99")
        self.sender = kw.get("sender", "Boss")
        self.sender_username = kw.get("sender_username", "boss")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = ""


class _Adapter:
    def __init__(self, statuses=None):
        self.calls = []
        self.statuses = statuses or {}

    def admin_member(self, chat, user_id):
        return {"status": self.statuses.get(str(user_id), "member")}


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter):
        self.sent = []
        self.gateway = SimpleNamespace(adapters={
            "telegram-bot": adapter, "telegram": adapter,
            "discord": adapter})
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return False


def _rig(tmp_path, adapter=None):
    adapter = adapter or _Adapter(statuses={"99": "administrator"})
    rt = _Runtime(tmp_path, adapter)
    return rt, rt._group_surface_obj, adapter


def test_sensor_ban_proposes_a_scam_flag_with_the_link(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    surface.record_join(KEY, "42", username="ada", name="Ada")
    rt._sensor_flag_proposal(surface, KEY, {
        "kind": "ban", "rule": "escalation",
        "_strike_rule": "scamlink", "user_id": "42",
        "url": EVIL})
    props = surface.flag_proposals(KEY)
    assert len(props) == 1
    assert props[0]["category"] == "scam"
    assert props[0]["source"] == "sensor"
    assert EVIL in props[0]["note"]
    assert surface.reputation.stats()["live"] == 0


def test_gsettings_scamlink_grammar(tmp_path):
    rt, surface, _adapter = _rig(tmp_path)
    msg = _Msg(sender_id="99", sender="Boss")
    out = rt._control_gsettings("scamlink off", msg)
    assert "scamlink off" in out
    assert surface.store.settings(KEY)["defense"]["scamlink"][
        "enabled"] is False
    out = rt._control_gsettings("scamlink on", msg)
    assert "scamlink on" in out
    out = rt._control_gsettings("scamlink 5", msg)
    assert "5" in out
    assert surface.store.settings(KEY)["defense"]["scamlink"][
        "max_links"] == 5
    out = rt._control_gsettings("scamlink suspicious off", msg)
    assert "suspicious" in out
    assert surface.store.settings(KEY)["defense"]["scamlink"][
        "suspicious"] == "off"
    panel = rt._control_gsettings("", msg)
    text = getattr(panel, "text", panel)
    assert "scamlink" in text
