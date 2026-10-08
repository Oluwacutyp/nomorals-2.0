"""No-access competitor intel — public data, no credentials (build-map #102).

Socialinsider pattern: benchmark ANY public account without access.
This module analyzes public posts only — it never logs in, never uses
OAuth, never touches credentialed endpoints. The scrape seam is
injectable (``scrape_fn``); the default is honestly empty.

What it produces per tracked account:
  * content pillars — what they post about (keyword-clustered topics),
  * posting cadence — posts/week, broken down by format,
  * engagement deltas — likes/comments/shares averages and trends.

Scheduled digests compare the latest window against the previous one
("competitor X posted 3x more video this week, here's the engagement
delta") and can be pushed to the owner DM or fed into the #45 content
pipeline.

The AEO extension compares what AI engines say about a competitor vs
the owner's brand (reuses #97's AEOTracker — engines, not accounts).

Never raises: every public function is wrapped.
"""

from __future__ import annotations

import re
import sqlite3
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable


# ── constants ─────────────────────────────────────────────────────────────

_DEFAULT_DB = "~/.nomorals/marketing/competitor.db"
_PLATFORMS = ("instagram", "tiktok", "x", "facebook", "youtube", "linkedin")

#: Tiny stopword set so pillar extraction stays dependency-free.
_STOPWORDS = frozenset(
    "the a an and or of to in on for with is are was were be been it its "
    "this that these those you your we our they their he she his her i me my "
    "at by from as not no yes do does did will would can could should have has "
    "had so if but just about into over after before up down out off very more "
    "most new now today here there when where how what why who which also get "
    "got like than then them us all any one two dm link bio check out".split()
)

_FORMATS = ("video", "image", "text", "carousel")


# ── data ──────────────────────────────────────────────────────────────────

@dataclass
class Post:
    """One public post. Only public fields — nothing credentialed."""

    platform: str = ""
    account: str = ""
    post_id: str = ""
    text: str = ""
    format: str = "text"          # video | image | text | carousel
    posted_at: float = 0.0       # epoch seconds
    likes: int = 0
    comments: int = 0
    shares: int = 0


@dataclass
class Pillar:
    name: str = ""
    posts: int = 0
    share: float = 0.0           # fraction of analyzed posts
    sample_terms: list[str] = field(default_factory=list)


@dataclass
class Cadence:
    posts_per_week: float = 0.0
    by_format: dict[str, float] = field(default_factory=dict)
    window_days: float = 0.0
    total_posts: int = 0


@dataclass
class Engagement:
    avg_likes: float = 0.0
    avg_comments: float = 0.0
    avg_shares: float = 0.0
    trend: str = "flat"          # rising | falling | flat (2nd half vs 1st half)
    trend_pct: float = 0.0


@dataclass
class CompetitorReport:
    report_id: str = ""
    account: str = ""
    platform: str = ""
    pillars: list[Pillar] = field(default_factory=list)
    cadence: Cadence = field(default_factory=Cadence)
    engagement: Engagement = field(default_factory=Engagement)
    post_count: int = 0
    created_at: float = 0.0
    source: str = "public-scrape"  # or "injected" / "no-source"

    def format(self) -> str:
        try:
            lines = [f"🔍 competitor intel — @{self.account} ({self.platform})"]
            if self.post_count == 0:
                lines.append("no public posts found — nothing to analyze.")
                return "\n".join(lines)
            c = self.cadence
            lines.append(
                f"📅 cadence: {c.posts_per_week:.1f} posts/week "
                f"({c.total_posts} posts over {c.window_days:.0f}d)")
            if c.by_format:
                fmt = ", ".join(f"{k}: {v:.1f}/wk"
                                for k, v in sorted(c.by_format.items()))
                lines.append(f"   by format: {fmt}")
            e = self.engagement
            lines.append(
                f"❤️ engagement: {e.avg_likes:.0f} likes · "
                f"{e.avg_comments:.0f} comments · {e.avg_shares:.0f} shares "
                f"(avg/post, trend: {e.trend} {e.trend_pct:+.0f}%)")
            if self.pillars:
                lines.append("📌 content pillars:")
                for p in self.pillars[:5]:
                    lines.append(
                        f"   • {p.name} — {p.posts} posts ({p.share:.0%})")
            lines.append("public data only — no account access used.")
            return "\n".join(lines)
        except Exception:
            return "couldn't format that competitor report."


# ── analysis ──────────────────────────────────────────────────────────────

def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [w.strip("'") for w in words
            if w.strip("'") and w not in _STOPWORDS and len(w) > 2]


def analyze_pillars(posts: list[Post], top_n: int = 5) -> list[Pillar]:
    """Cluster posts into content pillars by shared keywords. Never raises."""
    try:
        posts = [p for p in (posts or []) if p and p.text]
        if not posts:
            return []
        # Score candidate pillar terms by document frequency.
        doc_terms: list[set[str]] = []
        for p in posts:
            doc_terms.append(set(_terms(p.text)))
        df: Counter = Counter()
        for terms in doc_terms:
            df.update(terms)
        # Pick the top terms that appear in at least 2 posts.
        candidates = [t for t, c in df.most_common(12) if c >= 2]
        pillars: list[Pillar] = []
        claimed: set[int] = set()
        for term in candidates[:top_n]:
            members = [i for i, terms in enumerate(doc_terms)
                       if term in terms and i not in claimed]
            if not members:
                continue
            for i in members:
                claimed.add(i)
            # Sample co-occurring terms for flavor.
            co: Counter = Counter()
            for i in members:
                co.update(t for t in doc_terms[i] if t != term)
            pillars.append(Pillar(
                name=term, posts=len(members),
                share=len(members) / len(posts),
                sample_terms=[t for t, _ in co.most_common(3)]))
        if not pillars:
            pillars.append(Pillar(name="general", posts=len(posts),
                                  share=1.0, sample_terms=[]))
        return pillars
    except Exception:
        return []


def analyze_cadence(posts: list[Post]) -> Cadence:
    """Posts/week overall and by format. Never raises."""
    try:
        posts = [p for p in (posts or []) if p]
        if not posts:
            return Cadence()
        times = sorted(p.posted_at for p in posts if p.posted_at > 0)
        if len(times) >= 2:
            window_days = max(1.0, (times[-1] - times[0]) / 86400.0)
        else:
            window_days = 7.0
        weeks = max(window_days / 7.0, 1 / 7.0)
        by_format: dict[str, float] = {}
        for p in posts:
            fmt = (p.format or "text").lower()
            by_format[fmt] = by_format.get(fmt, 0) + 1
        return Cadence(
            posts_per_week=len(posts) / weeks,
            by_format={k: v / weeks for k, v in by_format.items()},
            window_days=window_days,
            total_posts=len(posts))
    except Exception:
        return Cadence()


def analyze_engagement(posts: list[Post]) -> Engagement:
    """Averages + 2nd-half vs 1st-half trend. Never raises."""
    try:
        posts = sorted([p for p in (posts or []) if p],
                       key=lambda p: p.posted_at or 0)
        if not posts:
            return Engagement()
        n = len(posts)
        avg = lambda f: sum(getattr(p, f, 0) or 0 for p in posts) / n
        avg_likes, avg_comments, avg_shares = (avg("likes"), avg("comments"),
                                              avg("shares"))
        trend, trend_pct = "flat", 0.0
        if n >= 4:
            half = n // 2
            first = posts[:half]
            second = posts[half:]

            def eng(ps: list[Post]) -> float:
                if not ps:
                    return 0.0
                return sum((p.likes or 0) + (p.comments or 0) * 3
                           + (p.shares or 0) * 2 for p in ps) / len(ps)
            e1, e2 = eng(first), eng(second)
            if e1 > 0:
                trend_pct = (e2 - e1) / e1 * 100.0
                if trend_pct >= 15:
                    trend = "rising"
                elif trend_pct <= -15:
                    trend = "falling"
        return Engagement(avg_likes=avg_likes, avg_comments=avg_comments,
                          avg_shares=avg_shares, trend=trend,
                          trend_pct=trend_pct)
    except Exception:
        return Engagement()


# ── store ─────────────────────────────────────────────────────────────────

def _resolve_db(db_path: str = "") -> str:
    import os
    p = (db_path or _DEFAULT_DB).strip() or _DEFAULT_DB
    if p.startswith("~"):
        p = os.path.expanduser(p)
    if p != ":memory:":
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    return p


class CompetitorStore:
    """Tracks public accounts; snapshots windows for delta digests."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            self._path = _resolve_db(db_path)
            self._db = sqlite3.connect(self._path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS tracked (
                       account TEXT, platform TEXT,
                       added_at REAL,
                       PRIMARY KEY (account, platform))""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS snapshots (
                       snap_id TEXT PRIMARY KEY, account TEXT, platform TEXT,
                       created_at REAL, post_count INTEGER,
                       posts_per_week REAL, video_per_week REAL,
                       avg_likes REAL, avg_comments REAL, avg_shares REAL)""")
            self._db.commit()
        except Exception:
            self._db = None

    # -- tracking ------------------------------------------------------

    def track(self, account: str, platform: str = "instagram") -> bool:
        """Start tracking a public account. Never raises."""
        try:
            account = (account or "").strip().lstrip("@").lower()
            platform = (platform or "instagram").strip().lower()
            if not account or platform not in _PLATFORMS:
                return False
            if self._db is None:
                return False
            self._db.execute(
                "INSERT OR IGNORE INTO tracked VALUES (?, ?, ?)",
                (account, platform, time.time()))
            self._db.commit()
            return True
        except Exception:
            return False

    def untrack(self, account: str, platform: str = "") -> bool:
        try:
            account = (account or "").strip().lstrip("@").lower()
            if not account or self._db is None:
                return False
            if platform:
                cur = self._db.execute(
                    "DELETE FROM tracked WHERE account=? AND platform=?",
                    (account, platform.strip().lower()))
            else:
                cur = self._db.execute(
                    "DELETE FROM tracked WHERE account=?", (account,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:
            return False

    def list_tracked(self) -> list[dict[str, Any]]:
        try:
            if self._db is None:
                return []
            return [dict(r) for r in self._db.execute(
                "SELECT account, platform, added_at FROM tracked "
                "ORDER BY added_at")]
        except Exception:
            return []

    # -- analysis --------------------------------------------------------

    def analyze(self, account: str, platform: str = "instagram",
                scrape_fn: Callable[..., list[Post]] | None = None
                ) -> CompetitorReport:
        """Public scrape → pillars + cadence + engagement. Never raises."""
        try:
            account = (account or "").strip().lstrip("@").lower()
            platform = (platform or "instagram").strip().lower()
            posts: list[Post] = []
            source = "no-source"
            if scrape_fn is not None:
                try:
                    posts = [p for p in (scrape_fn(account, platform) or [])
                             if isinstance(p, Post)]
                    source = "injected"
                except Exception:
                    posts = []
            report = CompetitorReport(
                report_id="cmp_" + uuid.uuid4().hex[:8],
                account=account, platform=platform,
                pillars=analyze_pillars(posts),
                cadence=analyze_cadence(posts),
                engagement=analyze_engagement(posts),
                post_count=len(posts),
                created_at=time.time(),
                source=source)
            self._snapshot(report)
            return report
        except Exception:
            return CompetitorReport(account=account or "",
                                    platform=platform or "")

    def _snapshot(self, report: CompetitorReport) -> None:
        try:
            if self._db is None or report.post_count == 0:
                return
            c = report.cadence
            e = report.engagement
            self._db.execute(
                "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("snap_" + uuid.uuid4().hex[:8], report.account,
                 report.platform, report.created_at, report.post_count,
                 c.posts_per_week, c.by_format.get("video", 0.0),
                 e.avg_likes, e.avg_comments, e.avg_shares))
            self._db.commit()
        except Exception:
            pass

    # -- digests -----------------------------------------------------------

    def digest(self, account: str = "", platform: str = "") -> str:
        """Latest window vs previous: 'posted 3x more video this week…'.
        Never raises."""
        try:
            if self._db is None:
                return "no competitor history yet."
            q = ("SELECT * FROM snapshots "
                 + ("WHERE account=? " if account else "")
                 + ("AND platform=? " if platform and account else
                    ("WHERE platform=? " if platform and not account else ""))
                 + "ORDER BY created_at DESC LIMIT 2")
            params: list[Any] = []
            if account:
                params.append(account.strip().lstrip("@").lower())
            if platform:
                params.append(platform.strip().lower())
            rows = [dict(r) for r in self._db.execute(q, params)]
            if len(rows) < 2:
                return ("not enough history for a digest yet — "
                        "track an account and re-run after the next window.")
            new, old = rows[0], rows[1]
            who = f"@{new['account']} ({new['platform']})"
            bits: list[str] = []

            def ratio(a: float, b: float) -> str:
                if b and b > 0:
                    return f"{a / b:.1f}x"
                return "new" if a > 0 else "—"

            bits.append(
                f"video: {new['video_per_week']:.1f}/wk "
                f"({ratio(new['video_per_week'], old['video_per_week'])} vs last window)")
            bits.append(
                f"posts: {new['posts_per_week']:.1f}/wk "
                f"({ratio(new['posts_per_week'], old['posts_per_week'])})")
            eng_new = (new['avg_likes'] or 0) + (new['avg_comments'] or 0) * 3
            eng_old = (old['avg_likes'] or 0) + (old['avg_comments'] or 0) * 3
            if eng_old > 0:
                pct = (eng_new - eng_old) / eng_old * 100.0
                bits.append(f"engagement delta: {pct:+.0f}%")
            return ("📊 competitor digest — " + who + "\n"
                    + "\n".join("• " + b for b in bits))
        except Exception:
            return "couldn't build that digest."


# ── AEO extension (#97) ───────────────────────────────────────────────────

def aeo_compare(brand: str, competitor: str,
                prompts: list[str] | None = None,
                tracker: Any = None) -> str:
    """'What do engines say about competitors vs you?' Never raises."""
    try:
        brand = (brand or "").strip()
        competitor = (competitor or "").strip()
        if not brand or not competitor:
            return "give me both brands — /competitor vs <them> <you>."
        if tracker is None:
            from .aeo import AEOTracker
            tracker = AEOTracker()
        own = tracker.track_visibility(brand, prompts)
        theirs = tracker.track_visibility(competitor, prompts)
        os_, ts_ = own.share, theirs.share
        verdict = ("you're ahead" if os_ > ts_ else
                   "they're ahead" if ts_ > os_ else "neck and neck")
        return (f"⚔️ AEO face-off — {brand} vs {competitor}\n"
                f"share of answer: you {os_:.0%} · them {ts_:.0%} "
                f"({own.confirmed_pairs}/{own.total_pairs} vs "
                f"{theirs.confirmed_pairs}/{theirs.total_pairs} pairs)\n"
                f"verdict: {verdict}.")
    except Exception:
        return "couldn't run that AEO comparison."


# ── chat ──────────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("/competitor track <account> [platform] — start tracking a public account\n"
            "/competitor report <account> [platform] — pillars + cadence + engagement\n"
            "/competitor digest [account] — latest window vs previous\n"
            "/competitor list — tracked accounts\n"
            "/competitor untrack <account> — stop tracking\n"
            "/competitor vs <competitor> <your-brand> — AEO share-of-answer face-off\n"
            "public data only — no logins, no credentials, ever.")


_store: CompetitorStore | None = None


def _get_store(**kwargs: Any) -> CompetitorStore:
    global _store
    override = kwargs.get("store")
    if isinstance(override, CompetitorStore):
        return override
    if _store is None:
        _store = CompetitorStore()
    return _store


def control_competitor(tail: str, context=None, chat=None,
                       **kwargs: Any) -> str:
    """/competitor — no-access competitor intel. Owner-only; never raises."""
    try:
        store = _get_store(**kwargs)
        scrape_fn = kwargs.get("scrape_fn")
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("track"):
            body = rest[5:].strip().split()
            if not body:
                return "track who? " + _usage()
            account, platform = body[0], (body[1] if len(body) > 1 else "instagram")
            if store.track(account, platform):
                return (f"🔍 tracking @{account.lstrip('@')} ({platform}) — "
                        "public posts only, no account access.")
            return "couldn't track that — check the account/platform."

        if low.startswith("untrack"):
            body = rest[7:].strip()
            if store.untrack(body):
                return f"stopped tracking @{body.lstrip('@')}."
            return "wasn't tracking that account."

        if low == "list" or low.startswith("list "):
            tracked = store.list_tracked()
            if not tracked:
                return "no competitors tracked — /competitor track <account>."
            return ("🔍 tracked competitors:\n" + "\n".join(
                f"• @{t['account']} ({t['platform']})" for t in tracked))

        if low.startswith("digest"):
            body = rest[6:].strip().split()
            account = body[0] if body else ""
            platform = body[1] if len(body) > 1 else ""
            return store.digest(account, platform)

        if low.startswith("vs "):
            body = rest[3:].strip().split(None, 1)
            if len(body) < 2:
                return "need both — /competitor vs <competitor> <your-brand>."
            return aeo_compare(body[1], body[0],
                               tracker=kwargs.get("aeo_tracker"))

        if low.startswith("report"):
            body = rest[6:].strip().split()
            if not body:
                return "report on who? " + _usage()
            account, platform = body[0], (body[1] if len(body) > 1 else "instagram")
            report = store.analyze(account, platform, scrape_fn=scrape_fn)
            return report.format()

        return "didn't catch that.\n" + _usage()
    except Exception:
        return "competitor intel hit a snag — try again."
