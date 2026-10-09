"""Wisdom organ — the corpus maintains itself.

WisdomKeeper could already answer questions, but the corpus only grew
when the owner ran ``/wisdom seed`` by hand. This organ makes it
autonomous:

- **Ingestion loop**: every tick ingests the next pending manifest
  entries (fetch → parse → index) inside a time budget. The 47 seed
  texts — and anything queued later — get ingested without commands.
- **Digest**: each newly ingested work gets an extractive digest (key
  passages chosen by term salience, no model needed). Digests are what
  briefings and cross-links work from.
- **Cross-link**: digests are matched against other traditions'
  passages; real thematic links (shared concepts, parallel teachings)
  are stored in ``wisdom_links``. This is the "connects across
  archives unprompted" behavior.
- **Streams**: research findings tagged esoteric arrive as organ events
  and are queued for ingestion; the corpus genuinely grows from the
  research organ's work.
- **Synthesis**: ``synthesize()`` fuses passages across traditions for
  one question — the brain calls it from plain language via the
  ``wisdom_synthesize`` spine tool.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any

from ..core.logging_setup import get_logger
from .. import organs as _bus

_log = get_logger(__name__)

_STOP = frozenset(
    "the a an and or of to in is are was were be been being on at for with "
    "by from as it its this that these those he she they we you i his her "
    "their our your my me him them us not no but if then than so such into "
    "out up down over under again once here there when where which who whom "
    "what how all any both each few more most other some only own same too "
    "very can will just should now".split()
)


def ensure_schema(db: Any) -> None:
    _bus.ensure_schema(db)
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS wisdom_digests (
            slug TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            tradition TEXT NOT NULL DEFAULT '',
            key_terms TEXT NOT NULL DEFAULT '[]',
            passages TEXT NOT NULL DEFAULT '[]',
            digested_at REAL NOT NULL
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS wisdom_links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug_a TEXT NOT NULL,
            slug_b TEXT NOT NULL,
            shared_terms TEXT NOT NULL DEFAULT '[]',
            strength REAL NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            UNIQUE(slug_a, slug_b)
        )
        """
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS wisdom_ingest_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            tradition TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'queued',
            queued_at REAL NOT NULL,
            done_at REAL NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT ''
        )
        """
    )


def _terms(text: str, top: int = 40) -> list[str]:
    words = re.findall(r"[a-z]{3,}", text.lower())
    counts = Counter(w for w in words if w not in _STOP)
    return [w for w, _ in counts.most_common(top)]


class WisdomOrgan:
    """One ``tick()`` = ingest → digest → cross-link, inside a budget."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        if self.db is None:
            raise ValueError("wisdom organ needs context.db")
        ensure_schema(self.db)

    def _keeper(self) -> Any:
        from .keeper import WisdomKeeper
        return WisdomKeeper(self.context)

    # ── queue ────────────────────────────────────────────────────────

    def queue_ingest(self, url: str, title: str,
                     tradition: str = "") -> int:
        url, title = (url or "").strip(), (title or "").strip()
        if not url or not title:
            raise ValueError("ingest queue needs a url and a title")
        row = self.db.query_one(
            "SELECT id FROM wisdom_ingest_queue WHERE url = ?"
            " AND status IN ('queued','failed')", (url,))
        if row:
            return int(row["id"])
        cur = self.db.execute(
            "INSERT INTO wisdom_ingest_queue (url, title, tradition, queued_at)"
            " VALUES (?, ?, ?, ?)",
            (url, title, tradition, time.time()))
        try:
            return int(cur.lastrowid or 0)
        except Exception:  # noqa: BLE001
            return 0

    # ── the autonomous tick ──────────────────────────────────────────

    def tick(self, budget_seconds: float = 900.0) -> dict[str, Any]:
        """Ingest → digest → cross-link inside ``budget_seconds``.

        Idempotent and resumable: every step records state, so a killed
        tick just continues next time.
        """
        deadline = time.time() + max(60.0, float(budget_seconds))
        report: dict[str, Any] = {
            "ingested": [], "digested": [], "links": 0,
            "events": 0, "errors": [],
        }
        try:
            report["events"] = self._drain_events()
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"events: {exc}")
        try:
            self._ingest_pending(report, deadline)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"ingest: {exc}")
        try:
            self._digest_new(report, deadline)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"digest: {exc}")
        try:
            report["links"] = self._cross_link(deadline)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"crosslink: {exc}")
        return report

    def _drain_events(self) -> int:
        n = 0
        for event in _bus.drain(self.db, "wisdom"):
            payload = event["payload"] or {}
            if event["kind"] == "finding.esoteric":
                url = (payload.get("url") or "").strip()
                title = (payload.get("title") or "").strip()
                if url and title:
                    try:
                        self.queue_ingest(url, title, tradition="research")
                        n += 1
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("wisdom queue failed: %s", exc)
            elif event["kind"] == "directive.ingest":
                try:
                    self.queue_ingest(
                        payload.get("url", ""), payload.get("title", ""),
                        payload.get("tradition", ""))
                    n += 1
                except Exception as exc:  # noqa: BLE001
                    _log.warning("wisdom directive failed: %s", exc)
        return n

    def _pending_manifest(self) -> list[Any]:
        keeper = self._keeper()
        manifest = getattr(keeper.corpus, "_manifest", {}) or {}
        return [e for e in manifest.values() if not e.ingested_at]

    def _ingest_pending(self, report: dict[str, Any],
                        deadline: float) -> None:
        from .ingestor import ArchiveIngestor, IngestError
        keeper = self._keeper()
        ingestor = ArchiveIngestor(self.context)
        # Manifest seeds first — the standing corpus commitment.
        for entry in self._pending_manifest():
            if time.time() >= deadline:
                break
            try:
                ingestor.ingest_entry(entry)
                report["ingested"].append(entry.slug)
                _log.info("wisdom ingested: %s", entry.slug)
            except Exception as exc:  # noqa: BLE001
                _log.warning("wisdom ingest failed for %s: %s",
                             entry.slug, exc)
        # Then the research-fed queue.
        rows = self.db.query(
            "SELECT id, url, title, tradition FROM wisdom_ingest_queue"
            " WHERE status = 'queued' ORDER BY queued_at ASC LIMIT 5")
        for row in rows or []:
            if time.time() >= deadline:
                break
            qid = int(row["id"])
            try:
                from .corpus import ManifestEntry
                import hashlib
                slug = "q-" + hashlib.sha256(
                    row["url"].encode()).hexdigest()[:10]
                entry = ManifestEntry(
                    slug=slug, title=row["title"],
                    tradition=row["tradition"] or "research",
                    canon_status="reference", source_url=row["url"])
                ingestor.ingest_entry(entry)
                self.db.execute(
                    "UPDATE wisdom_ingest_queue SET status='done', done_at=?"
                    " WHERE id = ?", (time.time(), qid))
                report["ingested"].append(slug)
            except Exception as exc:  # noqa: BLE001
                self.db.execute(
                    "UPDATE wisdom_ingest_queue SET status='failed', error=?"
                    " WHERE id = ?", (str(exc)[:300], qid))
                _log.warning("wisdom queue ingest failed: %s", exc)

    # ── digest (extractive — no model needed) ─────────────────────────

    def _digest_new(self, report: dict[str, Any], deadline: float) -> None:
        keeper = self._keeper()
        manifest = getattr(keeper.corpus, "_manifest", {}) or {}
        rows = self.db.query("SELECT slug FROM wisdom_digests")
        done = {r["slug"] for r in (rows or [])}
        for slug, entry in manifest.items():
            if time.time() >= deadline:
                break
            if slug in done or not entry.ingested_at:
                continue
            try:
                digest = self.digest_work(slug)
                if digest:
                    report["digested"].append(slug)
            except Exception as exc:  # noqa: BLE001
                _log.warning("wisdom digest failed for %s: %s", slug, exc)

    def digest_work(self, slug: str, top_passages: int = 8) -> dict[str, Any]:
        """Extractive digest: the work's most salient passages.

        Scores passages by overlap with the work's own top terms —
        representative content, chosen structurally, never by keyword
        dictionaries about what "wisdom" should sound like.
        """
        keeper = self._keeper()
        manifest = getattr(keeper.corpus, "_manifest", {}) or {}
        entry = manifest.get(slug)
        if entry is None or not entry.ingested_at:
            raise ValueError(f"no ingested work {slug!r}")
        # Pull a broad sample of the work's own passages via its title terms.
        seed_terms = _terms(entry.title, top=6) or [slug.replace("-", " ")]
        seen: dict[str, Any] = {}
        for term in seed_terms[:4]:
            try:
                ans = keeper.ask(term, top=25, mode="keyword")
            except Exception:  # noqa: BLE001
                continue
            for p in ans.passages:
                key = (p.work or "") + "::" + (p.snippet or "")[:80]
                if slug in (p.work or "") and key not in seen:
                    seen[key] = p
        passages = list(seen.values())
        if not passages:
            # Fall back: ask for the work by slug directly.
            try:
                ans = keeper.ask(slug.replace("-", " "), top=25, mode="keyword")
                passages = [p for p in ans.passages if slug in (p.work or "")]
            except Exception:  # noqa: BLE001
                passages = []
        if not passages:
            raise ValueError(f"no passages found for {slug!r}")
        corpus_terms = _terms(" ".join(
            (p.snippet or "") for p in passages[:50]), top=60)
        term_set = set(corpus_terms)
        scored = []
        for p in passages:
            snippet = p.snippet or ""
            words = set(re.findall(r"[a-z]{3,}", snippet.lower())) - _STOP
            overlap = len(words & term_set)
            # Prefer substantive passages; penalize stubs.
            length_bonus = min(1.0, len(snippet) / 600.0)
            scored.append((overlap * (0.5 + 0.5 * length_bonus), p))
        scored.sort(key=lambda t: -t[0])
        top = [p for _, p in scored[:top_passages]]
        digest = {
            "slug": slug, "title": entry.title,
            "tradition": entry.tradition,
            "key_terms": corpus_terms[:20],
            "passages": [{
                "section": p.section, "snippet": (p.snippet or "").strip()[:800],
                "url": p.url or "",
            } for p in top],
        }
        self.db.execute(
            "INSERT INTO wisdom_digests"
            " (slug, title, tradition, key_terms, passages, digested_at)"
            " VALUES (?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(slug) DO UPDATE SET"
            " key_terms=excluded.key_terms, passages=excluded.passages,"
            " digested_at=excluded.digested_at",
            (slug, entry.title, entry.tradition,
             json.dumps(digest["key_terms"]), json.dumps(digest["passages"]),
             time.time()))
        return digest

    def get_digest(self, slug: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT slug, title, tradition, key_terms, passages, digested_at"
            " FROM wisdom_digests WHERE slug = ?", (slug,))
        if not row:
            return None
        return {
            "slug": row["slug"], "title": row["title"],
            "tradition": row["tradition"],
            "key_terms": json.loads(row["key_terms"] or "[]"),
            "passages": json.loads(row["passages"] or "[]"),
            "digested_at": row["digested_at"],
        }

    def recent_digests(self, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT slug, title, tradition, digested_at FROM wisdom_digests"
            " ORDER BY digested_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in (rows or [])]

    # ── cross-link ───────────────────────────────────────────────────

    def _cross_link(self, deadline: float) -> int:
        """Link digests across traditions by shared salient terms."""
        keeper = self._keeper()
        rows = self.db.query(
            "SELECT slug, title, tradition, key_terms FROM wisdom_digests")
        digests = []
        for r in rows or []:
            digests.append({
                "slug": r["slug"], "title": r["title"],
                "tradition": r["tradition"] or "",
                "terms": set(json.loads(r["key_terms"] or "[]")),
            })
        made = 0
        for i, a in enumerate(digests):
            if time.time() >= deadline:
                break
            for b in digests[i + 1:]:
                if a["tradition"] and a["tradition"] == b["tradition"]:
                    continue  # cross-TRADITION links are the point
                shared = (a["terms"] & b["terms"]) - _STOP
                if len(shared) >= 4:
                    strength = len(shared) / max(
                        1, min(len(a["terms"]), len(b["terms"])))
                    pair = tuple(sorted((a["slug"], b["slug"])))
                    self.db.execute(
                        "INSERT INTO wisdom_links"
                        " (slug_a, slug_b, shared_terms, strength, created_at)"
                        " VALUES (?, ?, ?, ?, ?)"
                        " ON CONFLICT(slug_a, slug_b) DO UPDATE SET"
                        " shared_terms=excluded.shared_terms,"
                        " strength=excluded.strength",
                        (pair[0], pair[1], json.dumps(sorted(shared)),
                         round(strength, 3), time.time()))
                    made += 1
        # Emit the strongest new links so the brain can use them unprompted.
        if made:
            _bus.emit(self.db, "wisdom", "brain", "links.ready",
                      {"new_links": made})
        return made

    def links_for(self, slug: str, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT slug_a, slug_b, shared_terms, strength FROM wisdom_links"
            " WHERE slug_a = ? OR slug_b = ?"
            " ORDER BY strength DESC LIMIT ?", (slug, slug, limit))
        out = []
        for r in rows or []:
            other = r["slug_b"] if r["slug_a"] == slug else r["slug_a"]
            out.append({
                "work": other,
                "shared_terms": json.loads(r["shared_terms"] or "[]"),
                "strength": r["strength"],
            })
        return out

    # ── synthesis (cross-tradition, for the brain) ───────────────────

    def synthesize(self, query: str, top: int = 5) -> dict[str, Any]:
        """Ask across traditions and fuse: passages grouped by tradition
        with shared-concept notes from the link graph."""
        keeper = self._keeper()
        ans = keeper.ask(query, top=top * 3, mode="hybrid")
        by_tradition: dict[str, list[dict[str, Any]]] = {}
        for p in ans.passages[:top * 2]:
            manifest = getattr(keeper.corpus, "_manifest", {}) or {}
            tradition = ""
            for slug, entry in manifest.items():
                if slug in (p.work or ""):
                    tradition = entry.tradition
                    break
            by_tradition.setdefault(tradition or "unknown", []).append({
                "work": p.work, "section": p.section,
                "snippet": (p.snippet or "").strip()[:600],
                "url": p.url or "",
            })
        # Shared-concept notes: which traditions' digests link together.
        notes: list[str] = []
        seen_slugs = {p.work for p in ans.passages[:top]}
        for work in list(seen_slugs)[:4]:
            for link in self.links_for(work or "", limit=3):
                notes.append(
                    f"{work} ↔ {link['work']} share: "
                    f"{', '.join(link['shared_terms'][:6])}")
        return {
            "query": query,
            "by_tradition": {k: v[:top] for k, v in by_tradition.items()},
            "cross_links": notes[:8],
            "synthesis": ans.synthesis,
        }
