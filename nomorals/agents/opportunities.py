"""Money-making opportunities hunter — every legit way to earn, not just tasks.

Devon already had a paid-task report (``/searchleads``) and a Nigeria shopping
deal hunter (``naija_deals``).  This module is the broader earner: it hunts
ALL kinds of money-making opportunities — paid microtasks (global), referral
programs, free courses with earning paths, bug bounties, hackathons/prizes,
freelance gigs, AI-training data work, user testing, cashback/deals arbitrage,
content-monetization openings — ranks them by expected value vs effort, dedupes
across finders, and remembers what it already showed you.

Usage (agent code)::

    from nomorals.agents.opportunities import OpportunityHunter
    hunter = OpportunityHunter()                      # stdlib only
    opps = hunter.scan()                               # uses SearchEngine when a context exists
    for o in opps[:10]:
        print(o.score, o.title, o.url)

Chat/CLI surface: ``/money`` (control command), ``nm money scan|list|profile``.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "KINDS",
    "Opportunity",
    "Finder",
    "PaidTaskFinder",
    "ReferralFinder",
    "FreeCourseFinder",
    "BountyFinder",
    "GigFinder",
    "FINDERS",
    "OpportunityHunter",
    "OpportunityProfile",
    "score_opportunity",
    "extract_payout_usd",
    "normalize_url",
    "handle_money_command",
    "register",
]

#: Opportunity kinds the hunter knows.
KINDS = (
    "paid_task",     # microtasks, user testing, data annotation, AI gigs
    "referral",      # refer-and-earn programs
    "course",        # free courses that lead to earning
    "bounty",        # bug bounties, hackathons, grants with prizes
    "gig",           # freelance boards, micro jobs
    "arbitrage",     # cashback / deals / resale openings
    "content",       # content-monetization openings
)

#: Rough FX used only for scoring comparisons (clearly approximate).
_NGN_PER_USD = 1500.0

#: Domains that get a small trust bonus in scoring (well-known platforms).
TRUSTED_DOMAINS = frozenset({
    "upwork.com", "fiverr.com", "freelancer.com", "toptal.com",
    "hackerone.com", "bugcrowd.com", "intigriti.com", "yeswehack.com",
    "usertesting.com", "userinterviews.com", "trymyui.com", "maze.co",
    "clickworker.com", "microworkers.com", "remota.sh", "appen.com",
    "centaurlabs.com",
    "coursera.org", "edx.org", "udacity.com", "freecodecamp.org",
    "kaggle.com", "devpost.com", "mlh.io",
    "swagbucks.com", "surveyjunkie.com", "prolific.com",
    "github.com",
})

_PAYOUT_RE = re.compile(
    r"(?:up to|earn|pay(?:s|ing)?|worth|prize(?:s)?(?: of)?|grant(?:s)?(?: of)?)\s*"
    r"[\$\₦]?\s?([\d,]+(?:\.\d{1,2})?)",
    re.IGNORECASE,
)
_DOLLAR_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
_NAIRA_RE = re.compile(r"₦\s?([\d,]+(?:\.\d{1,2})?)")
_PER_HOUR_RE = re.compile(r"\$\s?([\d,]+)\s*/\s*(?:hour|hr)", re.IGNORECASE)


def _num(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def extract_payout_usd(text: str) -> Optional[float]:
    """Best-effort payout extraction from free text → USD estimate.

    Understands ``$25``, ``earn up to $60``, ``$30/hour``, ``₦5000``.
    Returns None when no money figure is found.  Hourly figures are
    scaled to a rough task-equivalent (x4) so they compare sanely.
    """
    if not text:
        return None
    m = _PER_HOUR_RE.search(text)
    if m:
        v = _num(m.group(1))
        if v:
            return round(v * 4, 2)
    m = _DOLLAR_RE.search(text)
    if m:
        v = _num(m.group(1))
        if v:
            return v
    m = _PAYOUT_RE.search(text)
    if m:
        v = _num(m.group(1))
        if v:
            # ₦ amounts get converted; bare numbers after pay-words are USD.
            if "₦" in text[max(0, m.start() - 4):m.start()]:
                return round(v / _NGN_PER_USD, 2)
            return v
    m = _NAIRA_RE.search(text)
    if m:
        v = _num(m.group(1))
        if v:
            return round(v / _NGN_PER_USD, 2)
    return None


_LOW_EFFORT = ("5-min", "5 minute", "quick", "easy", "simple", "survey",
               "signup", "sign-up", "referral", "cashback")
_HIGH_EFFORT = ("full-time", "full time", "project", "contract", "build",
                "develop", "freelance project", "hackathon", "competition")


def infer_effort(text: str) -> str:
    t = (text or "").lower()
    if any(k in t for k in _LOW_EFFORT):
        return "low"
    if any(k in t for k in _HIGH_EFFORT):
        return "high"
    return "medium"


def infer_regions(text: str) -> list[str]:
    t = (text or "").lower()
    regions: list[str] = []
    if any(k in t for k in ("worldwide", "global", "remote", "anywhere", "international")):
        regions.append("global")
    if "nigeria" in t or "nigerian" in t or "₦" in (text or ""):
        regions.append("nigeria")
    if "africa" in t:
        regions.append("africa")
    if "usa" in t or "u.s." in t or "united states" in t:
        regions.append("usa")
    if "europe" in t or "eu " in t:
        regions.append("europe")
    return regions or ["global"]


_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "referral", "aff", "affiliate", "fbclid", "gclid",
})


def normalize_url(url: str) -> str:
    """Canonical URL for dedupe: lowercase host, strip tracking params."""
    try:
        p = urllib.parse.urlsplit((url or "").strip())
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query)
             if k.lower() not in _TRACKING_PARAMS]
        query = urllib.parse.urlencode(sorted(q))
        path = p.path.rstrip("/") or "/"
        return urllib.parse.urlunsplit(("https", host, path, query, ""))
    except Exception:  # noqa: BLE001 - never break a scan on a bad URL
        return (url or "").strip().lower()


def _title_tokens(title: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z0-9]{3,}", title.lower()))


def titles_similar(a: str, b: str) -> bool:
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta), len(tb)) >= 0.8


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class Opportunity:
    kind: str
    title: str
    source: str          # finder name / site name
    url: str
    payout_text: str = ""
    payout_usd: Optional[float] = None
    effort: str = "medium"          # low | medium | high
    regions: list[str] = field(default_factory=lambda: ["global"])
    skills: list[str] = field(default_factory=list)
    fetched_at: float = field(default_factory=time.time)
    first_seen: float = field(default_factory=time.time)
    score: float = 0.0
    is_new: bool = True

    def __post_init__(self) -> None:
        if not self.payout_usd and self.payout_text:
            self.payout_usd = extract_payout_usd(self.payout_text)
        if self.effort not in ("low", "medium", "high"):
            self.effort = "medium"
        if not self.regions:
            self.regions = ["global"]

    @property
    def id(self) -> str:
        return hashlib.sha1(normalize_url(self.url).encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["id"] = self.id
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Opportunity":
        d = dict(d)
        d.pop("id", None)
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Scoring — transparent, deterministic
# ---------------------------------------------------------------------------

_KIND_BASE = {
    "paid_task": 50.0,
    "gig": 55.0,
    "bounty": 60.0,
    "referral": 45.0,
    "arbitrage": 45.0,
    "content": 40.0,
    "course": 30.0,   # courses pay later; base reflects indirect value
}
_EFFORT_MULT = {"low": 1.25, "medium": 1.0, "high": 0.8}


def score_opportunity(opp: Opportunity) -> float:
    """Transparent expected-value-vs-effort score (deterministic).

    base(kind) × effort_mult + payout_bonus + trust_bonus + freshness_bonus.
    """
    base = _KIND_BASE.get(opp.kind, 40.0)
    score = base * _EFFORT_MULT.get(opp.effort, 1.0)
    if opp.payout_usd:
        # log-scale so a $10k prize doesn't nuke the scale
        import math
        score += 12.0 * math.log10(1.0 + opp.payout_usd)
    try:
        host = urllib.parse.urlsplit(opp.url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host in TRUSTED_DOMAINS:
            score += 8.0
    except Exception:  # noqa: BLE001
        pass
    age_h = max(0.0, (time.time() - opp.fetched_at) / 3600.0)
    score += max(0.0, 5.0 - age_h * 0.2)   # fresh finds get a small bump
    if opp.is_new:
        score += 3.0
    return round(score, 2)


# ---------------------------------------------------------------------------
# Finders — query → search → parse → normalize
# ---------------------------------------------------------------------------

class Finder:
    """One opportunity source.  Subclass per vertical."""
    name: str = "base"
    kind: str = "paid_task"
    queries: list[str] = []
    #: Curated entries that work with no network: list of dicts with
    #: title/source/url/payout_text/effort/regions/skills.
    curated: list[dict[str, Any]] = []

    def parse_results(self, results: list[dict[str, str]]) -> list[Opportunity]:
        opps: list[Opportunity] = []
        for r in results:
            url = (r.get("url") or "").strip()
            title = (r.get("title") or "").strip()
            if not url or not title:
                continue
            snippet = r.get("snippet") or ""
            host = urllib.parse.urlsplit(url).netloc or self.name
            opps.append(Opportunity(
                kind=self.kind,
                title=title,
                source=host,
                url=url,
                payout_text=snippet,
                effort=infer_effort(f"{title} {snippet}"),
                regions=infer_regions(f"{title} {snippet}"),
            ))
        return opps

    def curated_opportunities(self) -> list[Opportunity]:
        return [Opportunity(kind=self.kind, **c) for c in self.curated]


class PaidTaskFinder(Finder):
    name = "paid_tasks"
    kind = "paid_task"
    queries = [
        "best paid microtask websites worldwide 2026",
        "get paid for user testing websites 2026",
        "data annotation jobs remote worldwide no experience",
        "AI training data gigs remote 2026",
        "get paid to transcribe audio online",
        "paid online surveys that actually pay worldwide",
        "website testing jobs for beginners remote",
        "earn money labeling images online",
        "paid product testing sites 2026",
        "remote micro jobs no experience worldwide",
        "get paid to test apps 2026",
        "online tasks for money beginners",
        "paid focus groups online worldwide",
        "earn money with phone microtasks 2026",
    ]


class ReferralFinder(Finder):
    name = "referrals"
    kind = "referral"
    queries = [
        "best refer and earn programs 2026",
        "referral programs that pay cash worldwide",
        "fintech referral bonuses 2026",
        "get paid for referrals apps",
    ]
    curated = [
        {"title": "Wise referral — invite friends, earn on transfers",
         "source": "wise.com", "url": "https://wise.com/invite/",
         "payout_text": "earn rewards per referral", "effort": "low",
         "regions": ["global"], "skills": []},
        {"title": "Revolut referral program — cash per invite",
         "source": "revolut.com", "url": "https://www.revolut.com/referral/",
         "payout_text": "cash reward per qualified referral", "effort": "low",
         "regions": ["global"], "skills": []},
        {"title": "Binance referral — commission on trading fees",
         "source": "binance.com", "url": "https://www.binance.com/en/referral",
         "payout_text": "earn commission on referrals' trading fees",
         "effort": "low", "regions": ["global"], "skills": []},
        {"title": "Coinbase referral — earn crypto per invite",
         "source": "coinbase.com", "url": "https://www.coinbase.com/referrals",
         "payout_text": "earn crypto for each referral", "effort": "low",
         "regions": ["global"], "skills": []},
        {"title": "Fiverr affiliate — promote services, earn commission",
         "source": "fiverr.com", "url": "https://affiliates.fiverr.com/",
         "payout_text": "commission per first-time buyer", "effort": "medium",
         "regions": ["global"], "skills": ["marketing"]},
        {"title": "Swagbucks referrals — earn from friends' activity",
         "source": "swagbucks.com", "url": "https://www.swagbucks.com/refer",
         "payout_text": "earn a cut of referrals' earnings", "effort": "low",
         "regions": ["global"], "skills": []},
        {"title": "Payoneer referral — reward per signup",
         "source": "payoneer.com", "url": "https://www.payoneer.com/refer-a-friend/",
         "payout_text": "reward per qualified signup", "effort": "low",
         "regions": ["global"], "skills": []},
        {"title": "Grey.co referral — earn on African freelancer payouts",
         "source": "grey.co", "url": "https://grey.co/referral",
         "payout_text": "earn per referral", "effort": "low",
         "regions": ["global", "africa", "nigeria"], "skills": []},
    ]


class FreeCourseFinder(Finder):
    name = "free_courses"
    kind = "course"
    queries = [
        "free coding courses with certificate 2026",
        "free data annotation training course",
        "free AI skills courses that lead to jobs 2026",
        "free freelancing skills courses for beginners",
        "free digital marketing course certificate 2026",
        "free cybersecurity training for beginners",
        "free UI UX design course 2026",
        "learn Python free with certificate",
        "free video editing course for freelancers",
        "free courses to earn money online skills",
    ]


class BountyFinder(Finder):
    name = "bounties"
    kind = "bounty"
    queries = [
        "bug bounty programs for beginners 2026",
        "hackathons with cash prizes 2026",
        "coding competitions with prizes worldwide",
        "grants for developers Africa 2026",
        "Kaggle competitions prize money",
        "open source bounties get paid",
        "AI hackathon cash prize 2026",
        "startup grants no equity 2026",
    ]


class GigFinder(Finder):
    name = "gigs"
    kind = "gig"
    queries = [
        "freelance gigs for beginners no experience worldwide",
        "remote micro jobs worldwide 2026",
        "best freelance websites for beginners 2026",
        "get paid for writing articles online",
        "remote customer support jobs worldwide",
        "virtual assistant jobs remote no experience",
        "sell digital products online beginners",
        "print on demand for beginners 2026",
        "freelance video editing gigs remote",
        "AI services to sell as freelancer 2026",
    ]


class ArbitrageFinder(Finder):
    name = "arbitrage"
    kind = "arbitrage"
    queries = [
        "best cashback apps worldwide 2026",
        "retail arbitrage for beginners",
        "flipping items for profit online",
        "credit card signup bonuses worth it 2026",
    ]


class ContentFinder(Finder):
    name = "content_monetization"
    kind = "content"
    queries = [
        "content monetization programs 2026 creators",
        "get paid for short videos platforms 2026",
        "newsletter monetization for beginners",
        "earn from AI generated content 2026",
    ]


FINDERS: tuple[type[Finder], ...] = (
    PaidTaskFinder,
    ReferralFinder,
    FreeCourseFinder,
    BountyFinder,
    GigFinder,
    ArbitrageFinder,
    ContentFinder,
)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------

@dataclass
class OpportunityProfile:
    skills: list[str] = field(default_factory=list)
    regions: list[str] = field(default_factory=lambda: ["global"])
    min_payout_usd: float = 0.0
    kinds: list[str] = field(default_factory=lambda: list(KINDS))
    max_per_query: int = 6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpportunityProfile":
        d = dict(d)
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})

    def accepts(self, opp: Opportunity) -> bool:
        if opp.kind not in self.kinds:
            return False
        if self.min_payout_usd > 0:
            if not opp.payout_usd or opp.payout_usd < self.min_payout_usd:
                return False
        if self.regions and "global" not in self.regions:
            if not (set(opp.regions) & set(self.regions)):
                return False
        if self.skills:
            text = f"{opp.title} {' '.join(opp.skills)}".lower()
            if not any(s.lower() in text for s in self.skills):
                # skills filter is soft: keep no-skill-tagged opps
                if opp.skills:
                    return False
        return True


# ---------------------------------------------------------------------------
# Hunter
# ---------------------------------------------------------------------------

class OpportunityHunter:
    """Runs finders, dedupes, scores, persists seen opportunities."""

    def __init__(self, settings: Any = None,
                 search_fn: Any = None) -> None:
        self.settings = settings
        # search_fn(query, max_results) -> list[{"url","title","snippet"}]
        self.search_fn = search_fn
        self._data_dir: Optional[Path] = None

    # -- storage ---------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        if self._data_dir is None:
            if self.settings is not None and hasattr(self.settings, "resolve"):
                self._data_dir = Path(self.settings.resolve("data/opportunities"))
            else:
                self._data_dir = Path.home() / ".config" / "nomorals" / "opportunities"
            self._data_dir.mkdir(parents=True, exist_ok=True)
        return self._data_dir

    def _seen_path(self) -> Path:
        return self.data_dir / "seen.jsonl"

    def load_seen(self) -> dict[str, Opportunity]:
        seen: dict[str, Opportunity] = {}
        p = self._seen_path()
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    o = Opportunity.from_dict(json.loads(line))
                    seen[o.id] = o
                except Exception:  # noqa: BLE001 - skip corrupt lines
                    continue
        return seen

    def _profile_path(self) -> Path:
        return self.data_dir / "profile.json"

    def load_profile(self) -> OpportunityProfile:
        p = self._profile_path()
        if p.exists():
            try:
                return OpportunityProfile.from_dict(
                    json.loads(p.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001
                pass
        return OpportunityProfile()

    def save_profile(self, profile: OpportunityProfile) -> None:
        self._profile_path().write_text(
            json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    # -- scanning ---------------------------------------------------------
    def scan(self, profile: Optional[OpportunityProfile] = None,
             kinds: Optional[list[str]] = None,
             max_per_query: Optional[int] = None,
             use_curated: bool = True) -> list[Opportunity]:
        """Run all (or selected) finders → ranked, deduped opportunities."""
        profile = profile or self.load_profile()
        kinds = kinds or profile.kinds
        per_query = max_per_query or profile.max_per_query
        seen = self.load_seen()

        found: list[Opportunity] = []
        for finder_cls in FINDERS:
            finder = finder_cls()
            if finder.kind not in kinds:
                continue
            if use_curated:
                found.extend(finder.curated_opportunities())
            if self.search_fn:
                for q in finder.queries:
                    try:
                        results = self.search_fn(q, per_query) or []
                    except Exception as exc:  # noqa: BLE001 - one bad query never kills a scan
                        _log.warning("opportunity query failed %r: %s", q, exc)
                        continue
                    found.extend(finder.parse_results(results))

        # dedupe: exact URL first, then fuzzy title
        unique: list[Opportunity] = []
        seen_urls: set[str] = set()
        for o in found:
            nu = normalize_url(o.url)
            if nu in seen_urls:
                continue
            if any(titles_similar(o.title, u.title) and o.kind == u.kind
                   for u in unique):
                continue
            seen_urls.add(nu)
            unique.append(o)

        # mark new vs seen, persist
        fresh: list[Opportunity] = []
        with self._seen_path().open("a", encoding="utf-8") as fh:
            for o in unique:
                if o.id in seen:
                    o.is_new = False
                    o.first_seen = seen[o.id].first_seen
                else:
                    o.is_new = True
                    seen[o.id] = o
                    fh.write(json.dumps(o.to_dict()) + "\n")
                fresh.append(o)

        for o in fresh:
            o.score = score_opportunity(o)
        ranked = sorted(
            (o for o in fresh if profile.accepts(o)),
            key=lambda o: o.score, reverse=True)
        return ranked

    def list(self, profile: Optional[OpportunityProfile] = None,
             only_new: bool = False, limit: int = 30) -> list[Opportunity]:
        seen = self.load_seen()
        opps = sorted(seen.values(), key=lambda o: o.score, reverse=True)
        profile = profile or self.load_profile()
        out = [o for o in opps if profile.accepts(o)]
        if only_new:
            # "new" = seen in the last 7 days
            cutoff = time.time() - 7 * 86400
            out = [o for o in out if o.first_seen >= cutoff]
        return out[:limit]

    def render(self, opps: list[Opportunity], limit: int = 15) -> str:
        lines = [f"💰 money opportunities ({len(opps)} shown)"]
        for o in opps[:limit]:
            flag = "🆕" if o.is_new else "  "
            pay = f" · {o.payout_text}" if o.payout_text else ""
            lines.append(
                f"{flag} [{o.score:.0f}] ({o.kind}/{o.effort}) {o.title}{pay}\n"
                f"    ↳ {o.url}")
        if len(opps) > limit:
            lines.append(f"…and {len(opps) - limit} more — /money list")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat control wiring (partner_runtime adds: if kind == "money": ...)
# ---------------------------------------------------------------------------

def handle_money_command(tail: str, context: Any = None) -> str:
    """Entry point for ``/money``.  Returns the reply text.

    Verbs: ``/money`` | ``/money scan [kind]`` | ``/money list`` |
    ``/money new`` | ``/money profile``
    """
    from ..core.config import get_settings
    settings = getattr(context, "settings", None) or get_settings()

    search_fn = None
    tools = getattr(context, "tools", None) if context is not None else None
    if tools is not None:
        # Call the web_search tool directly: SearchEngine.search() has a
        # latent bug (passes freshness=, which the tool doesn't accept).
        def search_fn(q: str, n: int = 6):  # noqa: E306
            outcome = tools.call("web_search", query=q, max_results=n)
            if not outcome.ok:
                err = outcome.error.message if outcome.error else "unknown"
                raise RuntimeError(f"search failed: {err}")
            return list(outcome.unwrap().get("results") or [])

    hunter = OpportunityHunter(settings=settings, search_fn=search_fn)
    verb = (tail or "").strip().split()
    action = verb[0].lower() if verb else "list"

    if action == "scan":
        kinds = [verb[1].lower()] if len(verb) > 1 else None
        if kinds and kinds[0] not in KINDS:
            return (f"unknown kind {kinds[0]!r}. kinds: "
                    f"{', '.join(KINDS)}")
        opps = hunter.scan(kinds=kinds)
        if not opps:
            return ("no opportunities matched your profile right now. "
                    "try /money scan with no kind filter, or loosen "
                    "/money profile.")
        return hunter.render(opps)
    if action in ("list", "new"):
        opps = hunter.list(only_new=(action == "new"))
        if not opps:
            return ("nothing stored yet — run /money scan first.")
        return hunter.render(opps)
    if action == "profile":
        p = hunter.load_profile()
        return ("money profile:\n"
                f"  skills: {', '.join(p.skills) or 'any'}\n"
                f"  regions: {', '.join(p.regions)}\n"
                f"  min payout: ${p.min_payout_usd:.0f}\n"
                f"  kinds: {', '.join(p.kinds)}")
    return ("usage: /money scan [kind] · /money list · /money new · "
            "/money profile")


def register(registry: Any) -> None:
    """Tool-registry hook: ``money_scan`` for agent use."""
    from ..core.policy import Capability

    @registry.register(
        "money_scan",
        description=("Scan money-making opportunities (paid tasks, referrals, "
                     "free courses, bounties, gigs, arbitrage, content). "
                     "Curated sources by default; pass search=true with an "
                     "agent context for live web queries."),
        capability=Capability.NET_OUT,
        parameters={
            "kinds": "str — comma-separated kinds (optional)",
            "max_results": "int — cap (default 15)",
        },
    )
    def money_scan(kinds: str = "", max_results: int = 15) -> dict[str, Any]:
        from ..core.config import get_settings
        hunter = OpportunityHunter(settings=get_settings())
        ks = [k.strip() for k in kinds.split(",") if k.strip()] or None
        opps = hunter.scan(kinds=ks, use_curated=True)
        return {"count": len(opps),
                "opportunities": [o.to_dict() for o in opps[:max_results]]}
