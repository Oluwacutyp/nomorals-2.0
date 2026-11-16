"""Status/story radar (census §6 candidate #3).

Pattern briefs from the status/story vault: who posts a lot, who
went quiet, what themes run through statuses, what is worth a look.
The vault already journals every observed status (``status_arrival``,
once per post, durable) and every saved story (``story``) — the radar
is pure organ work on those rows.  No model, no network, no new
verbs: deterministic counts over journal rows, in the owner's
timezone (WAT, UTC+1 year-round, a fixed offset — never tzdata).

The organ takes rows, never the runtime: the chat handler feeds it
``PersonalVault.journal()`` rows and the morning-briefing provider
feeds it the same rows off the journal file.  A post is counted once
— an arrival and its saved ``status`` row are the same post.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: Africa/Lagos is UTC+1 year-round (no DST) — a fixed offset, so
#: weekday/hour patterns are right on every host without tzdata.
WAT = timezone(timedelta(hours=1))

#: The journal kinds that are one post each.  Channel posts are NOT
#: here — that is /channeltexts territory.
POST_KINDS = ("status_arrival", "story", "status")

#: Kind priority when one post has several rows: the arrival is the
#: funnel's census record; a saved copy adds nothing.
_KIND_PRIORITY = {"status_arrival": 0, "story": 1, "status": 2}

_DAY = 86400.0
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday")

_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "if", "so", "to", "of", "in",
    "on", "at", "for", "with", "from", "by", "is", "am", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "i", "you", "he", "she", "we", "they", "my", "your", "his", "her",
    "our", "their", "me", "him", "us", "them", "as", "no", "yes", "not",
    "do", "does", "did", "have", "has", "had", "will", "would", "can",
    "could", "just", "very", "really", "more", "most", "all", "any",
    "than", "then", "there", "here", "when", "where", "who", "what",
    "how", "why",
])

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9']{2,}")


@dataclass(frozen=True)
class RadarPost:
    platform: str
    poster_id: str
    poster_name: str
    ts: float
    text: str
    media_kinds: tuple[str, ...] = ()
    source_id: str = ""
    account: str = ""


@dataclass
class PosterStat:
    name: str
    platform: str
    count: int


@dataclass
class QuietStat:
    name: str
    platform: str
    last_ts: float
    days_quiet: float
    baseline_count: int


@dataclass
class PatternStat:
    name: str
    platform: str
    weekday: str
    part: str
    count: int
    total: int


@dataclass
class ThemeStat:
    word: str
    count: int
    posters: list[str]


@dataclass
class WorthStat:
    name: str
    platform: str
    ts: float
    snippet: str


@dataclass
class RadarBrief:
    window_days: int
    window_start: float
    now: float
    total: int = 0
    posters: int = 0
    per_platform: dict[str, int] = field(default_factory=dict)
    top: list[PosterStat] = field(default_factory=list)
    new: list[PosterStat] = field(default_factory=list)
    quiet: list[QuietStat] = field(default_factory=list)
    patterns: list[PatternStat] = field(default_factory=list)
    themes: list[ThemeStat] = field(default_factory=list)
    worth: list[WorthStat] = field(default_factory=list)


def load_journal_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a journal.jsonl file, skipping bad lines (never raises
    on a corrupt line — a broken row must not sink the brief)."""
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _post_ts(row: dict[str, Any]) -> float:
    try:
        return float(row.get("post_ts") or row.get("ts") or 0)
    except (TypeError, ValueError):
        return 0.0


def normalize_posts(rows: list[dict[str, Any]]) -> list[RadarPost]:
    """Journal rows → one RadarPost per post, deduped.

    An arrival and its saved ``status`` row share (account,
    source_id) — the arrival wins.  Rows without a source id fall
    back to (account, kind, ts, counterpart) so they still count
    without ever merging two different posts.
    """
    best: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("kind") or "") not in POST_KINDS:
            continue
        account = str(row.get("account") or "")
        source_id = str(row.get("source_id") or "")
        if source_id:
            key = (account, source_id)
        else:
            key = (account, f"nosid:{row.get('kind')}:"
                   f"{_post_ts(row)}:"
                   f"{row.get('counterpart_id') or ''}")
        prio = _KIND_PRIORITY[str(row.get("kind") or "")]
        cur = best.get(key)
        if cur is None or prio < cur[0]:
            best[key] = (prio, row)
    posts: list[RadarPost] = []
    for _prio, row in best.values():
        ts = _post_ts(row)
        if ts <= 0:
            continue
        media: list[str] = []
        for m in row.get("media") or []:
            if isinstance(m, dict) and m.get("kind"):
                media.append(str(m["kind"]))
        if not media and row.get("media_kind"):
            media.append(str(row["media_kind"]))
        poster_id = str(row.get("counterpart_id") or "")
        posts.append(RadarPost(
            platform=str(row.get("platform") or ""),
            poster_id=poster_id,
            poster_name=str(row.get("poster_name") or "") or poster_id,
            ts=ts,
            text=str(row.get("text") or ""),
            media_kinds=tuple(k for k in media if k),
            source_id=str(row.get("source_id") or ""),
            account=str(row.get("account") or "")))
    posts.sort(key=lambda p: p.ts)
    return posts


def _part_of_day(hour: int) -> str:
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _ago(ts: float, now: float) -> str:
    delta = max(0.0, now - ts)
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < _DAY:
        return f"{int(delta // 3600)}h ago"
    return f"{int(delta // _DAY)}d ago"


def _window_label(window_days: int) -> str:
    if window_days == 1:
        return "last 24 hours"
    return f"last {window_days} days"


def build_brief(rows: list[dict[str, Any]], *, now: float,
                window_days: int = 7, quiet_days: float = 3.0,
                tz: timezone = WAT) -> RadarBrief:
    """The brief over one window of vault rows.

    ``window_days`` is the brief window; the baseline is the same
    length immediately before it.  Quiet means >=2 posts in the
    baseline, none in the window, silent for >= ``quiet_days``.
    A pattern needs a real concentration: the poster's modal
    (weekday, part of day) holds >=3 posts and >=50% of their
    window posts.  A theme word must show up across >=2 posters.
    """
    window_days = max(1, int(window_days))
    posts = normalize_posts(rows)
    window_start = now - window_days * _DAY
    baseline_start = now - 2 * window_days * _DAY
    window_posts = [p for p in posts
                    if window_start <= p.ts <= now]
    baseline_posts = [p for p in posts
                      if baseline_start <= p.ts < window_start]
    brief = RadarBrief(window_days=window_days,
                       window_start=window_start, now=now)
    brief.total = len(window_posts)
    if not window_posts and not baseline_posts:
        return brief
    by_key: dict[tuple[str, str], dict[str, Any]] = {}

    def _slot(post: RadarPost) -> dict[str, Any]:
        key = (post.platform, post.poster_id or post.poster_name)
        slot = by_key.get(key)
        if slot is None:
            slot = {"platform": post.platform,
                    "poster_id": post.poster_id,
                    "name": post.poster_name, "window": [],
                    "baseline": [], "first_ts": post.ts,
                    "last_ts": post.ts}
            by_key[key] = slot
        return slot

    for post in window_posts:
        slot = _slot(post)
        slot["window"].append(post)
        slot["name"] = post.poster_name or slot["name"]
        slot["first_ts"] = min(float(slot["first_ts"]), post.ts)
        slot["last_ts"] = max(float(slot["last_ts"]), post.ts)
    for post in baseline_posts:
        slot = _slot(post)
        slot["baseline"].append(post)
        slot["first_ts"] = min(float(slot["first_ts"]), post.ts)
        slot["last_ts"] = max(float(slot["last_ts"]), post.ts)
    brief.posters = len([s for s in by_key.values() if s["window"]])
    for post in window_posts:
        brief.per_platform[post.platform] = \
            brief.per_platform.get(post.platform, 0) + 1
    ranked = sorted(
        (s for s in by_key.values() if s["window"]),
        key=lambda s: (-len(s["window"]), str(s["name"]).lower()))
    brief.top = [PosterStat(name=str(s["name"]),
                            platform=str(s["platform"]),
                            count=len(s["window"]))
                 for s in ranked[:5]]
    brief.new = [PosterStat(name=str(s["name"]),
                            platform=str(s["platform"]),
                            count=len(s["window"]))
                 for s in ranked
                 if not s["baseline"]
                 and float(s["first_ts"]) >= window_start][:5]
    quiet: list[QuietStat] = []
    for s in by_key.values():
        if s["window"] or len(s["baseline"]) < 2:
            continue
        days = (now - float(s["last_ts"])) / _DAY
        if days < quiet_days:
            continue
        quiet.append(QuietStat(name=str(s["name"]),
                               platform=str(s["platform"]),
                               last_ts=float(s["last_ts"]),
                               days_quiet=days,
                               baseline_count=len(s["baseline"])))
    quiet.sort(key=lambda q: -q.days_quiet)
    brief.quiet = quiet[:5]
    patterns: list[PatternStat] = []
    for s in ranked:
        window = s["window"]
        if len(window) < 3:
            continue
        buckets: dict[tuple[str, str], int] = {}
        for post in window:
            dt = datetime.fromtimestamp(post.ts, tz=tz)
            bucket = (_WEEKDAYS[dt.weekday()],
                      _part_of_day(dt.hour))
            buckets[bucket] = buckets.get(bucket, 0) + 1
        (weekday, part), count = max(
            buckets.items(), key=lambda kv: (kv[1], kv[0]))
        if count >= 3 and count / len(window) >= 0.5:
            patterns.append(PatternStat(
                name=str(s["name"]), platform=str(s["platform"]),
                weekday=weekday, part=part, count=count,
                total=len(window)))
    patterns.sort(key=lambda p: (-p.count, p.name.lower()))
    brief.patterns = patterns[:5]
    word_posters: dict[str, set[str]] = {}
    word_counts: dict[str, int] = {}
    for post in window_posts:
        words = set(_TOKEN_RE.findall(post.text.lower()))
        words -= _STOPWORDS
        for word in words:
            word_posters.setdefault(word, set()).add(
                str(post.poster_name))
            word_counts[word] = word_counts.get(word, 0) + 1
    themes = [(word, word_counts[word], sorted(word_posters[word]))
              for word in word_posters
              if len(word_posters[word]) >= 2]
    themes.sort(key=lambda t: (-len(t[2]), -t[1], t[0]))
    brief.themes = [ThemeStat(word=w, count=c, posters=names)
                    for w, c, names in themes[:5]]
    recent = [p for p in window_posts if p.ts >= now - _DAY]
    recent.sort(key=lambda p: (
        0 if len([q for q in window_posts
                  if q.poster_id == p.poster_id
                  and q.platform == p.platform]) <= 2 else 1,
        -p.ts))
    brief.worth = []
    for post in recent[:5]:
        text = " ".join(str(post.text or "").split())
        snippet = text[:80] if text else (
            post.media_kinds[0] if post.media_kinds else "media")
        brief.worth.append(WorthStat(
            name=post.poster_name, platform=post.platform,
            ts=post.ts, snippet=snippet))
    return brief


def _plural(n: int, singular: str, plural: str | None = None) -> str:
    if n == 1:
        return f"1 {singular}"
    return f"{n} {plural or singular + 's'}"


def render_brief(brief: RadarBrief) -> str:
    """Plain-text brief.  The first line is the window summary; the
    sections follow only when they have something to say.  An empty
    vault briefs honestly instead of inventing activity."""
    label = _window_label(brief.window_days)
    if brief.total == 0 and not brief.quiet:
        return ("No statuses or stories in the vault in the "
                f"{label} — nothing observed to brief on.")
    per = " · ".join(
        f"{platform or 'unknown'} {count}"
        for platform, count in sorted(brief.per_platform.items()))
    lines = [f"{label} — {_plural(brief.total, 'post')} from "
             f"{_plural(brief.posters, 'person', 'people')}"
             f"{(' · ' + per) if per else ''}"]
    if brief.top:
        lines.append("")
        lines.append("Most active")
        for stat in brief.top:
            lines.append(f"• {stat.name} ({stat.platform}) — "
                         f"{stat.count}")
    if brief.new:
        lines.append("")
        lines.append("New voices")
        for stat in brief.new:
            lines.append(f"• {stat.name} ({stat.platform}) — first "
                         "post in the window")
    if brief.quiet:
        lines.append("")
        lines.append("Gone quiet")
        for stat in brief.quiet:
            lines.append(f"• {stat.name} ({stat.platform}) — last "
                         f"post {_ago(stat.last_ts, brief.now)} "
                         f"({stat.baseline_count} in the week before)")
    if brief.patterns:
        lines.append("")
        lines.append("Patterns")
        for pat in brief.patterns:
            lines.append(f"• {pat.name} posts most {pat.weekday}s, "
                         f"{pat.part} ({pat.count} of {pat.total})")
    if brief.themes:
        lines.append("")
        lines.append("Running themes")
        for theme in brief.themes:
            lines.append(f"• “{theme.word}” — "
                         f"{', '.join(theme.posters)} "
                         f"({_plural(theme.count, 'post')})")
    if brief.worth:
        lines.append("")
        lines.append("Worth a look")
        for item in brief.worth:
            lines.append(f"• {item.name} ({item.platform}) · "
                         f"{_ago(item.ts, brief.now)} — "
                         f"“{item.snippet}”")
    return "\n".join(lines)
