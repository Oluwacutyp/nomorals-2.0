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
    # ── sweep upgrade: per-pillar engagement (Socialinsider's key insight:
    # a quiet pillar often outperforms a loud one) ──
    avg_engagement: float = 0.0  # likes + 3*comments + 2*shares per post
    top_post_excerpt: str = ""


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
                # Rank pillars by engagement, not just volume.
                ranked = sorted(self.pillars,
                                key=lambda p: -p.avg_engagement)[:5]
                mx = max((p.avg_engagement for p in ranked), default=0) or 1
                lines.append("📌 content pillars (by engagement):")
                for p in ranked:
                    bar = "█" * max(1, int(round(p.avg_engagement / mx * 10)))
                    lines.append(
                        f"   • {p.name} — {p.posts} posts ({p.share:.0%}) "
                        f"{bar} {p.avg_engagement:.0f} eng/post")
                # The money insight: quiet pillars that punch above weight.
                quiet = [p for p in ranked
                         if p.share < 0.25 and p.avg_engagement > e.avg_likes]
                if quiet:
                    lines.append("💡 quality > frequency: " +
                                 ", ".join(f"'{p.name}' ({p.avg_engagement:.0f} eng/post, "
                                           f"only {p.share:.0%} of posts)"
                                           for p in quiet[:2]))
            lines.append("public data only — no account access used.")
            return "\n".join(lines)
        except Exception:
            return "couldn't format that competitor report."


# ── analysis ──────────────────────────────────────────────────────────────

def _terms(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return [w.strip("'") for w in words
            if w.strip("'") and w not in _STOPWORDS and len(w) > 2]


def _engagement_of(p: Post) -> float:
    try:
        return float(p.likes or 0) + float(p.comments or 0) * 3 + float(p.shares or 0) * 2
    except Exception:
        return 0.0


def analyze_pillars(posts: list[Post], top_n: int = 5) -> list[Pillar]:
    """Cluster posts into content pillars by shared keywords.

    Each pillar carries its own average engagement (Socialinsider pattern) —
    a pillar with fewer posts often outperforms a louder one.
    Never raises.
    """
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
            member_posts = [posts[i] for i in members]
            avg_eng = (sum(_engagement_of(p) for p in member_posts)
                       / max(1, len(member_posts)))
            top_post = max(member_posts, key=_engagement_of, default=None)
            pillars.append(Pillar(
                name=term, posts=len(members),
                share=len(members) / len(posts),
                sample_terms=[t for t, _ in co.most_common(3)],
                avg_engagement=round(avg_eng, 1),
                top_post_excerpt=((top_post.text[:90] + "…")
                                  if top_post and len(top_post.text) > 90
                                  else (top_post.text if top_post else ""))))
        if not pillars:
            avg_all = sum(_engagement_of(p) for p in posts) / len(posts)
            pillars.append(Pillar(name="general", posts=len(posts),
                                  share=1.0, sample_terms=[],
                                  avg_engagement=round(avg_all, 1)))
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


# ── deep intel (Socialinsider / Metricool patterns) ──────────────────────

def top_posts(posts: list[Post], n: int = 5) -> list[Post]:
    """Highest-engagement posts — 'which posts caused the spike'. Never raises."""
    try:
        ranked = sorted((p for p in (posts or []) if p),
                        key=_engagement_of, reverse=True)
        return ranked[:max(1, int(n or 5))]
    except Exception:
        return []


def viral_posts(posts: list[Post], multiple: float = 3.0) -> list[Post]:
    """Posts beating the mean engagement by ``multiple``x. Never raises."""
    try:
        posts = [p for p in (posts or []) if p]
        if not posts:
            return []
        mean = sum(_engagement_of(p) for p in posts) / len(posts)
        if mean <= 0:
            return []
        return [p for p in posts
                if _engagement_of(p) >= mean * max(1.5, float(multiple or 3.0))]
    except Exception:
        return []


def best_times(posts: list[Post]) -> dict:
    """Best posting times from public timestamps: top hours and weekdays by
    average engagement (Metricool pattern). Never raises."""
    out: dict = {"by_hour": [], "by_weekday": [], "best_hour": None,
                 "best_weekday": None}
    try:
        import datetime
        posts = [p for p in (posts or []) if p and p.posted_at > 0]
        if len(posts) < 3:
            return out
        hours: dict[int, list[float]] = {}
        days: dict[int, list[float]] = {}
        for p in posts:
            dt = datetime.datetime.fromtimestamp(p.posted_at)
            eng = _engagement_of(p)
            hours.setdefault(dt.hour, []).append(eng)
            days.setdefault(dt.weekday(), []).append(eng)
        avg = lambda vs: sum(vs) / len(vs)
        by_hour = sorted(((h, round(avg(vs), 1), len(vs))
                          for h, vs in hours.items() if len(vs) >= 1),
                         key=lambda t: -t[1])[:3]
        by_day = sorted(((d, round(avg(vs), 1), len(vs))
                         for d, vs in days.items() if len(vs) >= 1),
                        key=lambda t: -t[1])[:3]
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        out["by_hour"] = [{"hour": h, "avg_eng": a, "posts": c}
                          for h, a, c in by_hour]
        out["by_weekday"] = [{"weekday": names[d], "avg_eng": a, "posts": c}
                             for d, a, c in by_day]
        out["best_hour"] = by_hour[0][0] if by_hour else None
        out["best_weekday"] = names[by_day[0][0]] if by_day else None
        return out
    except Exception:
        return out


_HASHTAG_RE = re.compile(r"#([a-z0-9_]{2,40})", re.IGNORECASE)


def analyze_hashtags(posts: list[Post], n: int = 10) -> list[dict]:
    """Top hashtags with post counts and average engagement. Never raises."""
    try:
        agg: dict[str, list[float]] = {}
        for p in (posts or []):
            if not p or not p.text:
                continue
            tags = {t.lower() for t in _HASHTAG_RE.findall(p.text)}
            eng = _engagement_of(p)
            for t in tags:
                agg.setdefault(t, []).append(eng)
        ranked = sorted(agg.items(), key=lambda kv: (-len(kv[1]),
                                                     -(sum(kv[1]) / len(kv[1]))))
        return [{"tag": "#" + tag, "posts": len(v),
                 "avg_eng": round(sum(v) / len(v), 1)}
                for tag, v in ranked[:max(1, int(n or 10))]]
    except Exception:
        return []


def key_insights(report: CompetitorReport) -> str:
    """Socialinsider-style Key Insights: written summary + observations you
    can forward, not an export you have to interpret. Never raises."""
    try:
        if report.post_count == 0:
            return "no data — nothing to conclude."
        lines = [f"💡 key insights — @{report.account} ({report.platform})"]
        c, e = report.cadence, report.engagement
        lines.append(
            f"posts {c.posts_per_week:.1f}/week across {c.window_days:.0f} days; "
            f"avg {e.avg_likes:.0f} likes / {e.avg_comments:.0f} comments / "
            f"{e.avg_shares:.0f} shares per post; trend {e.trend} "
            f"({e.trend_pct:+.0f}%).")
        if report.pillars:
            ranked = sorted(report.pillars, key=lambda p: -p.avg_engagement)
            best = ranked[0]
            lines.append(
                f"strongest pillar: '{best.name}' at {best.avg_engagement:.0f} "
                f"eng/post ({best.share:.0%} of output).")
            quiet = [p for p in ranked[1:]
                     if p.share < 0.25 and p.avg_engagement > best.avg_engagement * 0.7]
            if quiet:
                q = quiet[0]
                lines.append(
                    f"opportunity: '{q.name}' punches at {q.avg_engagement:.0f} "
                    f"eng/post on only {q.share:.0%} of posts — frequency "
                    f"could increase here.")
            loud_weak = [p for p in ranked
                         if p.share >= 0.3 and p.avg_engagement < best.avg_engagement * 0.5]
            if loud_weak:
                lines.append(
                    f"watch: '{loud_weak[0].name}' is {loud_weak[0].share:.0%} of "
                    f"output but underperforms — volume without resonance.")
        if c.by_format:
            top_fmt = max(c.by_format.items(), key=lambda kv: kv[1])
            lines.append(f"format mix leans {top_fmt[0]} ({top_fmt[1]:.1f}/wk).")
        if e.trend == "falling":
            lines.append("engagement is falling — check their recent replies "
                         "for what changed before copying anything.")
        elif e.trend == "rising":
            lines.append("engagement is rising — their current mix is working; "
                         "mirror the pillars, not the posts.")
        return "\n".join(lines)
    except Exception:
        return "couldn't build key insights."


def benchmark(own_posts: list[Post], rival_posts: list[Post],
              own_name: str = "you", rival_name: str = "rival") -> str:
    """Side-by-side: your public posts vs theirs. Never raises."""
    try:
        def _stats(posts: list[Post]) -> dict:
            posts = [p for p in (posts or []) if p]
            n = len(posts)
            if not n:
                return {"n": 0, "eng": 0.0, "pw": 0.0}
            eng = sum(_engagement_of(p) for p in posts) / n
            cad = analyze_cadence(posts)
            return {"n": n, "eng": eng, "pw": cad.posts_per_week}
        a, b = _stats(own_posts), _stats(rival_posts)
        if not a["n"] and not b["n"]:
            return "no posts on either side to compare."
        lines = [f"⚔️ benchmark — {own_name} vs {rival_name}"]
        lines.append(f"posts analyzed: {a['n']} vs {b['n']}")
        if a["n"] and b["n"]:
            lead = "you lead" if a["eng"] >= b["eng"] else "they lead"
            gap = abs(a["eng"] - b["eng"]) / max(1.0, b["eng"])
            lines.append(f"avg engagement/post: {a['eng']:.0f} vs {b['eng']:.0f} "
                         f"({lead} by {gap:.0%})")
            lines.append(f"cadence: {a['pw']:.1f}/wk vs {b['pw']:.1f}/wk")
            if a["eng"] < b["eng"]:
                lines.append("verdict: they earn more per post — study their top "
                             "pillars, then out-teach them.")
            else:
                lines.append("verdict: you're ahead per post — press the advantage "
                             "with more of what works.")
        elif b["n"]:
            lines.append("verdict: no own posts to compare — feed yours in to benchmark.")
        return "\n".join(lines)
    except Exception:
        return "couldn't build that benchmark."


def content_gaps(report: CompetitorReport,
                 own_keywords: list[str] | None = None) -> list[dict]:
    """Pillars they cover that your keyword list doesn't — the gaps worth
    stealing (legitimately). Never raises."""
    try:
        own = {k.lower().strip() for k in (own_keywords or []) if k and k.strip()}
        gaps: list[dict] = []
        for p in (report.pillars or []):
            terms = {p.name.lower()} | {t.lower() for t in p.sample_terms}
            if not terms & own:
                gaps.append({"pillar": p.name, "their_posts": p.posts,
                             "their_avg_eng": p.avg_engagement,
                             "sample_terms": p.sample_terms[:3],
                             "why": (f"they earn {p.avg_engagement:.0f} eng/post "
                                     f"on '{p.name}' and you cover none of it")})
        return sorted(gaps, key=lambda g: -g["their_avg_eng"])
    except Exception:
        return []


def to_briefs(report: CompetitorReport, n: int = 3) -> list[dict]:
    """Turn their best pillars into content briefs for the #99 pipeline.
    Never raises."""
    try:
        ranked = sorted(report.pillars or [], key=lambda p: -p.avg_engagement)
        briefs: list[dict] = []
        for p in ranked[:max(1, int(n or 3))]:
            briefs.append({
                "title": f"Own the '{p.name}' conversation",
                "angle": (f"Competitor @{report.account} earns "
                          f"{p.avg_engagement:.0f} eng/post on '{p.name}' "
                          f"({p.posts} posts). Cover it deeper: their top post "
                          f"went: \"{p.top_post_excerpt[:80]}\""),
                "gap_pillar": p.name,
                "sample_terms": p.sample_terms[:5],
                "why": f"proven demand in your niche — {p.share:.0%} of their output",
            })
        return briefs
    except Exception:
        return []


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
            "/competitor insights <account> [platform] — written key insights + opportunities\n"
            "/competitor top <account> [platform] — their highest-engagement posts\n"
            "/competitor hashtags <account> [platform] — top hashtags by engagement\n"
            "/competitor besttime <account> [platform] — best posting hours/days\n"
            "/competitor gaps <account> <your keywords…> — pillars they own that you don't\n"
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

        def _posts_for(args: list[str]) -> tuple[list[Post], str, str]:
            account = args[0] if args else ""
            platform = args[1] if len(args) > 1 else "instagram"
            posts: list[Post] = []
            if scrape_fn is not None and account:
                try:
                    posts = [p for p in
                             (scrape_fn(account.strip().lstrip("@").lower(),
                                        platform.strip().lower()) or [])
                             if isinstance(p, Post)]
                except Exception:
                    posts = []
            return posts, account, platform

        if low.startswith("insights"):
            body = rest[8:].strip().split()
            if not body:
                return "insights on who? " + _usage()
            posts, account, platform = _posts_for(body)
            report = store.analyze(account, platform, scrape_fn=scrape_fn)
            if posts:
                # analyze() re-scrapes; prefer the posts we already have.
                report = CompetitorReport(
                    report_id=report.report_id, account=report.account,
                    platform=report.platform, pillars=analyze_pillars(posts),
                    cadence=analyze_cadence(posts),
                    engagement=analyze_engagement(posts),
                    post_count=len(posts), created_at=report.created_at,
                    source=report.source)
            return key_insights(report)

        if low.startswith("top"):
            body = rest[3:].strip().split()
            if not body:
                return "top posts of who? " + _usage()
            posts, account, _platform = _posts_for(body)
            tops = top_posts(posts, 5)
            if not tops:
                return f"no posts found for @{account.lstrip('@')}."
            lines = [f"🔥 top posts — @{account.lstrip('@')}"]
            for i, p in enumerate(tops, 1):
                excerpt = (p.text[:100] + "…") if len(p.text) > 100 else p.text
                lines.append(f"{i}. {_engagement_of(p):.0f} eng "
                             f"({p.likes}♥ {p.comments}💬 {p.shares}🔁) — {excerpt}")
            viral = viral_posts(posts)
            if viral:
                lines.append(f"🚨 {len(viral)} viral post(s) (>3x mean) in this window.")
            return "\n".join(lines)

        if low.startswith("hashtags"):
            body = rest[8:].strip().split()
            if not body:
                return "hashtags of who? " + _usage()
            posts, account, _platform = _posts_for(body)
            tags = analyze_hashtags(posts, 10)
            if not tags:
                return f"no hashtags found for @{account.lstrip('@')}."
            lines = [f"#️⃣ top hashtags — @{account.lstrip('@')}"]
            for t in tags:
                lines.append(f"• {t['tag']} — {t['posts']} posts, "
                             f"{t['avg_eng']:.0f} avg eng")
            return "\n".join(lines)

        if low.startswith("besttime"):
            body = rest[8:].strip().split()
            if not body:
                return "best time for who? " + _usage()
            posts, account, _platform = _posts_for(body)
            bt = best_times(posts)
            if not bt["by_hour"]:
                return f"not enough timestamped posts for @{account.lstrip('@')}."
            lines = [f"⏰ best posting times — @{account.lstrip('@')}"]
            lines.append("hours: " + ", ".join(
                f"{h['hour']:02d}:00 ({h['avg_eng']:.0f} eng)" for h in bt["by_hour"]))
            lines.append("days: " + ", ".join(
                f"{d['weekday']} ({d['avg_eng']:.0f} eng)" for d in bt["by_weekday"]))
            lines.append(f"sweet spot: {bt['best_weekday']}s around "
                         f"{bt['best_hour']:02d}:00.")
            return "\n".join(lines)

        if low.startswith("gaps"):
            body = rest[4:].strip().split()
            if len(body) < 2:
                return ("usage: /competitor gaps <account> [platform] <your keyword1 keyword2 …>\n"
                        "e.g. /competitor gaps rivalbrand fitness nutrition coaching")
            account = body[0]
            if len(body) > 2 and body[1].lower() in (
                    "instagram", "tiktok", "x", "facebook", "youtube", "linkedin"):
                platform, keywords = body[1].lower(), body[2:]
            else:
                platform, keywords = "instagram", body[1:]
            posts, account, platform = _posts_for([account, platform])
            report = CompetitorReport(
                report_id="cmp_gaps", account=account.strip().lstrip("@").lower(),
                platform=platform, pillars=analyze_pillars(posts),
                cadence=analyze_cadence(posts),
                engagement=analyze_engagement(posts),
                post_count=len(posts), created_at=time.time(),
                source="injected" if posts else "no-source")
            gaps = content_gaps(report, keywords)
            if not gaps:
                return (f"no clear gaps — your keywords already cover their "
                        f"pillars, @{account.lstrip('@')}.")
            lines = [f"🕳️ content gaps vs @{account.lstrip('@')} — pillars they own:"]
            for g in gaps[:5]:
                lines.append(f"• '{g['pillar']}' — {g['why']}")
            lines.append("feed the best into the content pipeline as briefs.")
            return "\n".join(lines)

        return "didn't catch that.\n" + _usage()
    except Exception:
        return "competitor intel hit a snag — try again."
