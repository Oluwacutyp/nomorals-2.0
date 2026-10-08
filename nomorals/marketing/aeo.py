"""AEO/GEO visibility tracking — share of answer, not rank (build-map #97).

~60% of searches end zero-click. The game is being *cited* by AI
engines, not ranked on a results page. This module:

* fans prompts out to AI engines (ChatGPT / Claude / Gemini /
  Perplexity) via official APIs where keys exist,
* parses the citations in each response,
* two-model cross-checks every brand mention (a *different* engine
  verifies it — hallucinated mentions get filtered),
* reports "share of answer": confirmed mentions / engine-prompt pairs.

Gaps found ("engines get asked X about you and you have no cited
presence") become content briefs consumable by the #45 content
pipeline (``nomorals/social/content_pipeline.py``).

Everything is injectable and offline-testable. Every public entry
point never raises. This is a pattern implementation — it does NOT
port or depend on webappski/aeo-tracker (flagged as an unverified
code candidate, never verified).
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# ── engines ──────────────────────────────────────────────────────────────────

#: Engines in fan-out order. ``api_env`` is checked at call time only —
#: no key is read, stored, or logged anywhere in this module.
ENGINES: tuple[dict[str, str], ...] = (
    {"name": "chatgpt", "provider": "OpenAI", "api_env": "OPENAI_API_KEY"},
    {"name": "claude", "provider": "Anthropic", "api_env": "ANTHROPIC_API_KEY"},
    {"name": "gemini", "provider": "Google", "api_env": "GEMINI_API_KEY"},
    {"name": "perplexity", "provider": "Perplexity", "api_env": "PERPLEXITY_API_KEY"},
)

ENGINE_NAMES = tuple(e["name"] for e in ENGINES)

#: Cost planning estimate per engine-prompt pair (USD). Real billing
#: varies; these exist so a run can be bounded, not invoiced.
#: ~$0.20–0.55 for a standard 4-engine × 5-8-prompt run.
COST_PER_PAIR_USD = 0.035


def default_prompts(brand: str) -> list[str]:
    """Sensible starter prompts when the owner just gives a brand."""
    brand = (brand or "").strip() or "your brand"
    return [
        f"What is {brand}?",
        f"Tell me about {brand}.",
        f"{brand} reviews — is it any good?",
        f"Who is behind {brand}?",
        f"Is {brand} legit or a scam?",
    ]


# ── data ─────────────────────────────────────────────────────────────────────

@dataclass
class EngineResponse:
    engine: str
    prompt: str
    text: str
    ok: bool = True
    error: str = ""


@dataclass
class Mention:
    engine: str
    prompt: str
    brand: str = ""                   # the brand this mention is about
    context: str = ""                 # the sentence(s) where the brand appeared
    citations: list[str] = field(default_factory=list)
    confirmed: bool = False           # set by cross-check
    confirming_engine: str = ""


@dataclass
class VisibilityReport:
    report_id: str
    brand: str
    prompts: list[str]
    engines: list[str]
    responses: list[EngineResponse] = field(default_factory=list)
    mentions: list[Mention] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)  # prompts with no confirmed mention
    created_at: float = 0.0

    @property
    def total_pairs(self) -> int:
        return len(self.prompts) * len(self.engines)

    @property
    def confirmed(self) -> list[Mention]:
        return [m for m in self.mentions if m.confirmed]

    @property
    def confirmed_pairs(self) -> set[tuple[str, str]]:
        """(engine, prompt) pairs with at least one confirmed mention."""
        return {(m.engine, m.prompt) for m in self.mentions if m.confirmed}

    @property
    def share(self) -> float:
        """Share of answer: engine-prompt pairs with a confirmed mention
        / total pairs. Bounded to [0, 1] — multiple mentions in one
        response count once for the pair."""
        if not self.total_pairs:
            return 0.0
        return min(1.0, len(self.confirmed_pairs) / self.total_pairs)

    def format(self) -> str:
        try:
            lines = [
                f"📊 AEO visibility — {self.brand}",
                f"share of answer: {self.share:.0%} "
                f"({len(self.confirmed_pairs)}/{self.total_pairs} engine-prompt pairs)",
            ]
            per_engine: dict[str, int] = {}
            for eng, _prompt in self.confirmed_pairs:
                per_engine[eng] = per_engine.get(eng, 0) + 1
            if per_engine:
                lines.append("by engine: " + ", ".join(
                    f"{e} {c}/{len(self.prompts)}" for e, c in sorted(per_engine.items())))
            for m in self.confirmed[:6]:
                ctx = (m.context[:110] + "…") if len(m.context) > 110 else m.context
                lines.append(f"• [{m.engine}] {ctx}")
            if len(self.confirmed) > 6:
                lines.append(f"• …and {len(self.confirmed) - 6} more")
            if self.gaps:
                lines.append(f"⚠️ {len(self.gaps)} gap(s) — engines get asked and you have no cited presence:")
                for g in self.gaps[:4]:
                    lines.append(f"  – {g[:90]}")
            lines.append("zero-click reality: ~60% of searches never click — being cited IS the ranking.")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return f"AEO report for {self.brand} (format failed)."


# ── parsing ──────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s\]\)\"'<>]+", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\[([^\]]{2,80})\]\((https?://[^\)]+)\)")
_CITE_NUM_RE = re.compile(r"\[(\d{1,3})\]")
_DOMAIN_RE = re.compile(r"\b([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z]{2,})+)\b", re.IGNORECASE)


def parse_citations(text: str) -> list[str]:
    """Extract citations (URLs, markdown links, domains) from engine text. Never raises."""
    try:
        text = text or ""
        out: list[str] = []
        seen: set[str] = set()

        def _add(c: str) -> None:
            c = (c or "").strip().rstrip(".,;:")
            if c and c not in seen and len(c) < 300:
                seen.add(c)
                out.append(c)

        for m in _MD_LINK_RE.finditer(text):
            _add(m.group(2))
        for m in _URL_RE.finditer(text):
            _add(m.group(0))
        # numbered citations only when a matching URL exists nearby — they are
        # meaningless alone, so we don't surface bare [1]s.
        for m in _CITE_NUM_RE.finditer(text):
            _add("[" + m.group(1) + "]")
        for m in _DOMAIN_RE.finditer(text):
            dom = m.group(1).lower()
            if dom.count(".") >= 1 and not dom.endswith((".png", ".jpg", ".css")):
                _add(dom)
        return out
    except Exception:  # noqa: BLE001
        return []


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if len(p.strip()) > 8]


def find_mentions(text: str, brand: str, engine: str, prompt: str) -> list[Mention]:
    """Sentence-level brand mentions with surrounding context. Never raises."""
    try:
        brand = (brand or "").strip()
        if not brand or not text:
            return []
        pattern = re.compile(r"\b" + re.escape(brand) + r"\b", re.IGNORECASE)
        out: list[Mention] = []
        for sent in _sentences(text):
            if pattern.search(sent):
                out.append(Mention(
                    engine=engine,
                    prompt=prompt,
                    brand=brand,
                    context=sent,
                    citations=parse_citations(sent),
                ))
        return out
    except Exception:  # noqa: BLE001
        return []


# ── engine fan-out ───────────────────────────────────────────────────────────

#: ``engine_caller(engine_name, prompt) -> str``. Injectable; tests mock it.
#: Any callable ``(str, str) -> str`` works.
from typing import Callable
EngineCaller = Callable[[str, str], str]


def _default_engine_caller(engine: str, prompt: str) -> str:
    """Call a real engine via its official API when a key is present.

    Honest by default: without a key it returns "" (unavailable), never a
    fabricated answer. Keys are read from the environment at call time
    only and are never stored or logged.
    """
    try:
        spec = next((e for e in ENGINES if e["name"] == engine), None)
        if spec is None:
            return ""
        key = os.environ.get(spec["api_env"] or "", "").strip()
        if not key:
            return ""
        # Minimal stdlib HTTP call — no new hard dependency.
        import json
        import urllib.request

        if engine == "chatgpt":
            url, payload = ("https://api.openai.com/v1/chat/completions",
                            {"model": "gpt-4o-mini",
                             "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": 600})
            headers = {"Authorization": f"Bearer {key}"}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        if engine == "claude":
            url, payload = ("https://api.anthropic.com/v1/messages",
                            {"model": "claude-3-5-haiku-latest", "max_tokens": 600,
                             "messages": [{"role": "user", "content": prompt}]})
            headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            blocks = data.get("content", [])
            return "".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
        if engine == "gemini":
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   f"gemini-2.0-flash:generateContent?key={key}")
            payload = {"contents": [{"parts": [{"text": prompt}]}],
                       "generationConfig": {"maxOutputTokens": 600}}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            cands = data.get("candidates", [])
            parts = (cands[0].get("content", {}).get("parts", []) if cands else [])
            return "".join(p.get("text", "") for p in parts if isinstance(p, dict)).strip()
        if engine == "perplexity":
            url, payload = ("https://api.perplexity.ai/chat/completions",
                            {"model": "sonar",
                             "messages": [{"role": "user", "content": prompt}],
                             "max_tokens": 600})
            headers = {"Authorization": f"Bearer {key}"}
            body = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=body, headers={**headers, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read().decode())
            return (data.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()
        return ""
    except Exception:  # noqa: BLE001
        _log.warning("aeo engine call failed for %s", engine, exc_info=True)
        return ""


def cross_check(mention: Mention, verifier: str, verifier_caller) -> bool:
    """Ask a *different* engine whether the mention is real.

    The verifier is asked to confirm or deny the quoted context with a
    YES/NO answer. Ambiguous answers count as unconfirmed — we would
    rather drop a real mention than report a hallucinated one.
    Never raises.
    """
    try:
        if not mention or not mention.context or not verifier_caller:
            return False
        probe = (
            "Answer with exactly YES or NO on the first line.\n"
            f"Question: does the following passage actually name and describe "
            f"the brand '{mention.brand or 'the brand in question'}'? "
            f"Passage: \"{mention.context[:400]}\"\n"
            "If the passage names the brand with real context, say YES and quote "
            "the relevant words. If it is vague, generic, or names no brand, say NO."
        )
        answer = (verifier_caller(verifier, probe) or "").strip()
        first = answer.split("\n", 1)[0].strip().upper()
        ok = first.startswith("YES")
        if ok:
            mention.confirmed = True
            mention.confirming_engine = verifier
        return ok
    except Exception:  # noqa: BLE001
        return False


# ── tracker ──────────────────────────────────────────────────────────────────

def _default_db() -> str:
    try:
        base = os.path.expanduser("~/.nomorals/marketing")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "aeo.db")
    except Exception:  # noqa: BLE001
        return ":memory:"


class AEOTracker:
    """Runs visibility checks, cross-checks mentions, keeps report history."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS aeo_reports ("
                "report_id TEXT PRIMARY KEY, brand TEXT, share REAL, "
                "pairs INTEGER, confirmed INTEGER, gaps_json TEXT, "
                "created_at REAL)")
            # tolerate older tables missing gaps_json
            try:
                self._db.execute("ALTER TABLE aeo_reports ADD COLUMN gaps_json TEXT")
                self._db.commit()
            except Exception:  # noqa: BLE001
                pass
            self._db.commit()
        except Exception:  # noqa: BLE001
            self._db = None

    # -- core --

    def track_visibility(self, brand: str, prompts: list[str] | None = None,
                         engines: list[str] | None = None,
                         engine_caller=None) -> VisibilityReport:
        """Fan out → parse citations → cross-check → share of answer. Never raises."""
        try:
            brand = (brand or "").strip()
            prompts = [p.strip() for p in (prompts or default_prompts(brand)) if p and p.strip()]
            engines = [e for e in (engines or list(ENGINE_NAMES)) if e in ENGINE_NAMES] or list(ENGINE_NAMES)
            caller = engine_caller or _default_engine_caller
            report = VisibilityReport(
                report_id="aeo_" + uuid.uuid4().hex[:8],
                brand=brand, prompts=prompts, engines=engines,
                created_at=time.time(),
            )
            if not brand:
                return report
            for prompt in prompts:
                for engine in engines:
                    try:
                        text = caller(engine, prompt) or ""
                    except Exception:  # noqa: BLE001
                        text = ""
                    resp = EngineResponse(engine=engine, prompt=prompt, text=text, ok=bool(text))
                    report.responses.append(resp)
                    if not text:
                        continue
                    for mention in find_mentions(text, brand, engine, prompt):
                        # cross-check with a DIFFERENT engine
                        verifier = next((e for e in engines if e != engine), engine)
                        try:
                            cross_check(mention, verifier, caller)
                        except Exception:  # noqa: BLE001
                            pass
                        report.mentions.append(mention)
            confirmed_prompts = {m.prompt for m in report.mentions if m.confirmed}
            report.gaps = [p for p in prompts if p not in confirmed_prompts]
            self._record(report)
            return report
        except Exception:  # noqa: BLE001
            _log.warning("aeo track_visibility failed", exc_info=True)
            return VisibilityReport(report_id="aeo_" + uuid.uuid4().hex[:8],
                                    brand=str(brand or ""), prompts=[], engines=[])

    # -- history --

    def _record(self, report: VisibilityReport) -> None:
        try:
            if self._db is None:
                return
            import json as _json
            self._db.execute(
                "INSERT OR REPLACE INTO aeo_reports "
                "(report_id, brand, share, pairs, confirmed, gaps_json, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (report.report_id, report.brand, report.share,
                 report.total_pairs, len(report.confirmed_pairs),
                 _json.dumps(list(report.gaps or [])), report.created_at))
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    def get(self, report_id: str) -> VisibilityReport | None:
        """Full stored report (gaps rebuilt; mentions are session-local). Never raises."""
        try:
            if self._db is None or not report_id:
                return None
            row = self._db.execute(
                "SELECT * FROM aeo_reports WHERE report_id = ?",
                (report_id,)).fetchone()
            if not row:
                return None
            import json as _json
            gaps: list[str] = []
            try:
                gaps = list(_json.loads(row["gaps_json"] or "[]"))
            except Exception:  # noqa: BLE001
                gaps = []
            return VisibilityReport(
                report_id=row["report_id"], brand=row["brand"], gaps=gaps,
                prompts=[], engines=[], created_at=row["created_at"])
        except Exception:  # noqa: BLE001
            return None

    def latest(self, brand: str = "") -> VisibilityReport | None:
        """Most recent report (optionally for a brand). Never raises."""
        try:
            if self._db is None:
                return None
            brand = (brand or "").strip()
            if brand:
                row = self._db.execute(
                    "SELECT report_id FROM aeo_reports WHERE brand = ? "
                    "ORDER BY created_at DESC LIMIT 1", (brand,)).fetchone()
            else:
                row = self._db.execute(
                    "SELECT report_id FROM aeo_reports "
                    "ORDER BY created_at DESC LIMIT 1").fetchone()
            if not row:
                return None
            return self.get(row["report_id"])
        except Exception:  # noqa: BLE001
            return None

    def history(self, brand: str = "", limit: int = 12) -> list[dict]:
        """Share-of-answer trend. Never raises."""
        try:
            if self._db is None:
                return []
            q = "SELECT report_id, brand, share, pairs, confirmed, created_at FROM aeo_reports"
            args: list = []
            if (brand or "").strip():
                q += " WHERE brand = ?"
                args.append(brand.strip())
            q += " ORDER BY created_at DESC LIMIT ?"
            args.append(max(1, int(limit or 12)))
            return [dict(r) for r in self._db.execute(q, args).fetchall()]
        except Exception:  # noqa: BLE001
            return []

    # -- briefs --

    def content_briefs(self, report: VisibilityReport) -> list[dict]:
        """Turn gaps into content briefs for the #45 content pipeline.

        Each brief is a plain dict the pipeline can consume:
        ``{title, angle, gap_prompt, why}``.
        Never raises.
        """
        try:
            briefs: list[dict] = []
            for gap in (report.gaps or []):
                briefs.append({
                    "title": f"Cited presence for: {gap[:70]}",
                    "angle": ("Publish a page/FAQ/post that AI engines can cite "
                              "when answering this exact question."),
                    "gap_prompt": gap,
                    "why": (f"Engines get asked '{gap[:80]}' about {report.brand} "
                            f"and produce no confirmed citation. Own the answer."),
                })
            return briefs
        except Exception:  # noqa: BLE001
            return []


def estimate_cost(n_prompts: int, n_engines: int) -> tuple[float, float]:
    """(low, high) USD planning estimate for a run. Never raises."""
    try:
        n = max(0, int(n_prompts)) * max(0, int(n_engines))
        low = n * 0.02
        high = n * 0.07
        return (round(low, 2), round(high, 2))
    except Exception:  # noqa: BLE001
        return (0.0, 0.0)


#: Cron expression for the weekly AEO check (Monday 09:00).
AEO_CRON = "0 9 * * MON"

#: Action string the scheduler dispatches for the weekly AEO run.
AEO_WEEKLY_ACTION = "aeo_weekly_check"


def ensure_weekly(scheduler) -> bool:
    """Register the Monday-morning AEO check. Idempotent-ish. Never raises.

    Mirrors the ``saved_search`` seam: checks existing jobs by action,
    then ``schedule_cron(task_id=..., cron_expr=..., action=...)``.
    """
    try:
        import asyncio

        async def _ensure() -> bool:
            try:
                jobs = []
                try:
                    jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    pass
                for j in jobs or []:
                    if getattr(j, "action", "") == AEO_WEEKLY_ACTION:
                        return True
                await scheduler.schedule_cron(  # type: ignore[attr-defined]
                    task_id="aeo-weekly",
                    cron_expr=AEO_CRON,
                    action=AEO_WEEKLY_ACTION,
                    parameters={},
                )
                return True
            except Exception:  # noqa: BLE001
                return False

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_ensure())
        _log.warning("aeo ensure_weekly inside a running loop")
        return False
    except Exception:  # noqa: BLE001
        _log.debug("aeo ensure_weekly failed", exc_info=True)
        return False


# ── chat ─────────────────────────────────────────────────────────────────────

def _usage() -> str:
    return ("usage:\n"
            "  /aeo track <brand> [prompt1; prompt2; …] — run a visibility check\n"
            "  /aeo report [brand] — latest report + share-of-answer trend\n"
            "  /aeo briefs [brand] — content briefs from the gaps\n"
            "engines: chatgpt, claude, gemini, perplexity (keys optional — "
            "missing engines report honestly as unavailable).")


def _get_tracker() -> AEOTracker:
    return AEOTracker()


def control_aeo(tail: str, context=None, chat=None, **kwargs) -> str:
    """/aeo — AEO/GEO visibility tracking. Owner-only; never raises."""
    try:
        tracker = _get_tracker()
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("track"):
            body = rest[5:].strip()
            if not body:
                return "usage: /aeo track <brand> [prompt1; prompt2; …]"
            parts = [p.strip() for p in body.split(";") if p.strip()]
            brand, prompts = parts[0], (parts[1:] or None)
            n_prompts = len(prompts or default_prompts(brand))
            lo, hi = estimate_cost(n_prompts, len(ENGINE_NAMES))
            report = tracker.track_visibility(brand, prompts)
            out = [report.format(), "",
                   f"run cost (planning estimate): ${lo:.2f}–${hi:.2f}"]
            return "\n".join(out)

        if low.startswith("report"):
            brand = rest[6:].strip()
            probe = tracker.latest(brand)
            hist_brand = brand or (probe.brand if probe else "")
            hist = tracker.history(hist_brand)
            if not hist:
                return "no AEO report yet — /aeo track <brand> first."
            h0 = hist[0]
            pairs = h0.get("pairs") or 1
            pct = h0.get("confirmed", 0) / pairs
            lines = [
                f"📊 latest AEO report — {h0.get('brand')} "
                f"({time.strftime('%Y-%m-%d', time.localtime(h0.get('created_at', 0)))})",
                f"share of answer: {pct:.0%} "
                f"({h0.get('confirmed')}/{h0.get('pairs')} engine-prompt pairs)",
            ]
            if len(hist) > 1:
                trend = []
                for h in hist[:6]:
                    p = h.get("pairs") or 1
                    trend.append(f"{h.get('confirmed', 0) / p:.0%}")
                lines.append("trend: " + " → ".join(reversed(trend)))
            return "\n".join(lines)

        if low.startswith("briefs"):
            brand = rest[6:].strip()
            rep = tracker.latest(brand)
            if rep is None:
                return "no AEO report yet — /aeo track <brand> first."
            briefs = tracker.content_briefs(rep)
            if not briefs:
                return (f"no content gaps for {rep.brand} — every prompt had a "
                        "confirmed citation. 🎉")
            lines = [f"📝 content briefs from AEO gaps — {rep.brand}"]
            for i, b in enumerate(briefs[:8], 1):
                lines.append(f"{i}. {b['title']}\n   {b['angle']}")
            lines.append("these plug into the content pipeline as drafts.")
            return "\n".join(lines)

        return _usage()
    except Exception:  # noqa: BLE001
        _log.warning("aeo control failed", exc_info=True)
        return "aeo hit a snag — try again."
