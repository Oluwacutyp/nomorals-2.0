"""Story reader — follow webnovel stories, track reading, keep bookmarks.

A *story* is a followed webnovel: its metadata, the chapters Devon has
fetched (cached on disk so re-reads never hit the network), reading
progress (chapter + char offset), and bookmarks.  Everything lives under
``workspace/books/reader/<story-slug>/`` as JSON + markdown — resumable
across restarts, greppable by the owner.

Fetch politeness is profile-gated: termux fetches fewer chapters per
sync with longer pauses; laptop/workstation go faster.

Following a story also means OWNING it: :meth:`StoryReader.ingest`
pulls every chapter from 1 to the source's current end into the
story's own library (resumable, progress persisted on the record,
reading position untouched), so continuation can anchor at the
story's true end instead of wherever the owner stopped reading.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.profiles import get_profile_kind
from .model import slugify
from .sources import (ADAPTERS, Chapter, SourceAdapter, SourceError,
                      StoryMeta, adapter_for_url, book_registry_for_context,
                      search_all)

_log = get_logger(__name__)

__all__ = ["StoryReader", "FollowedStory", "ReaderError"]


class ReaderError(Exception):
    """The reader could not do what was asked."""


@dataclass
class FollowedStory:
    slug: str
    title: str
    url: str
    source: str
    author: str = ""
    synopsis: str = ""
    genres: list[str] = field(default_factory=list)
    status: str = ""
    total_chapters: int = 0
    cover_url: str = ""
    # reading state
    current_chapter: int = 0
    current_offset: int = 0
    last_read_at: str = ""
    chapters_cached: int = 0
    followed_at: str = ""
    # library ingestion state — SEPARATE from reading progress: how much
    # of the story sits in the story's own library, ingested on follow.
    # ``ingested_chapters`` is the contiguous watermark (chapters
    # 1..N are all held locally); ``ingest_total`` is the source total
    # observed at the last ingest.
    ingested_chapters: int = 0
    ingest_total: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FollowedStory":
        from dataclasses import MISSING
        kwargs: dict[str, Any] = {}
        for k, f in cls.__dataclass_fields__.items():
            if k in d:
                kwargs[k] = d[k]
            elif f.default is not MISSING:
                kwargs[k] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                kwargs[k] = f.default_factory()
            else:
                kwargs[k] = ""
        return cls(**kwargs)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _norm_story_name(value: str) -> str:
    """Normalize a story name/slug/title for resolution: case-folded,
    underscores/spaces/punctuation all become single dashes."""
    s = (value or "").strip().lower().replace("_", "-")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-")


class StoryReader:
    """The owner's webnovel shelf."""

    def __init__(self, context: Any, *,
                 source_registry: Any = None) -> None:
        self.context = context
        # Source registry (Pass #89): the ONE media registry the book
        # class credits fetch outcomes to, takes search ordering
        # from, and grows through.  Tests may inject; otherwise the
        # shared DEFAULT_REGISTRY bound to the context's KV store —
        # and only when the context carries a database, so db-less
        # callers keep the pre-#89 behaviour exactly (#88 pattern).
        self._source_registry_injected = source_registry
        self._source_registry_cache: Any = None
        self._source_registry_resolved = False

    def _registry(self) -> Any:
        """The source registry for book outcomes, or None when this
        reader has none to speak to.  Never raises."""
        if self._source_registry_resolved:
            return self._source_registry_cache
        self._source_registry_resolved = True
        self._source_registry_cache = book_registry_for_context(
            self.context, injected=self._source_registry_injected)
        return self._source_registry_cache

    # ── book source status (Pass #89) ─────────────────────────────────
    def source_status(self) -> dict[str, Any]:
        """Book-source state from the ONE registry: seed vs grown
        sources, health, outcome counts, candidates pending.  With
        no registry (db-less context) this is the shared in-memory
        registry's book class — the state this process has learned."""
        reg = self._registry()
        if reg is None:
            from ..media.source_registry import DEFAULT_REGISTRY
            reg = DEFAULT_REGISTRY
        return reg.book_status()

    def source_status_text(self) -> str:
        """Human rendering of :meth:`source_status` for /novel sources."""
        st = self.source_status()
        lines = ["📚 book sources — story sites learn from fetch "
                 "outcomes (pass #89; one registry with downloads "
                 "and music):"]
        for row in st.get("sources") or []:
            lines.append(
                f"  📖 {row['source']} → {row['domain']} "
                f"[{row['origin']}/{row['health']}] ok={row['ok']} "
                f"fail={row['fail']} blocked={row['blocked']} "
                f"no-result={row['no_result']}")
        for g in st.get("grown") or []:
            lines.append(
                f"  📖 grown:{g['domain']} [grown/{g['health']}] "
                f"ok={g['ok']} fail={g['fail']} — promoted by the "
                "health loop, serving via the universal reader shape")
        lines.append(
            f"  book candidates pending verification: "
            f"{st.get('candidates_pending', 0)} (untrusted — they "
            "serve nothing until the health loop promotes them)")
        return "\n".join(lines)

    # ── paths ─────────────────────────────────────────────────────────────
    def _workspace(self) -> Path:
        settings = getattr(self.context, "settings", None)
        root = getattr(settings, "workspace_dir", None) if settings else None
        return Path(root) if root else Path.cwd() / "workspace"

    def reader_dir(self) -> Path:
        d = self._workspace() / "books" / "reader"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def story_dir(self, slug: str) -> Path:
        d = self.reader_dir() / slug
        d.mkdir(parents=True, exist_ok=True)
        (d / "chapters").mkdir(exist_ok=True)
        return d

    def _meta_path(self, slug: str) -> Path:
        return self.story_dir(slug) / "story.json"

    def _profile(self) -> dict[str, Any]:
        # chapters fetched per sync, chapter-list walk cap, politeness
        # pause, fetch timeout — profile-gated, never designed down.
        kind = get_profile_kind()
        lean = kind == "termux"
        return {
            "sync_chapters": 3 if lean else 8,
            "list_limit": 60 if lean else 400,
            "pause_s": 1.2 if lean else 0.4,
            "timeout": 20.0 if lean else 30.0,
        }

    # ── story registry ────────────────────────────────────────────────────
    def resolve_slug(self, name: str) -> str:
        """Resolve a user-typed story reference to the followed slug.

        Accepts the full slug, the slug without its source suffix, or
        the story title — all case/underscore/dash-insensitive — plus a
        unique prefix of any of those.  Ambiguous input raises
        ReaderError listing the candidate slugs; unknown input raises
        the honest "no followed story" error.  Never creates dirs.
        """
        raw = (name or "").strip()
        if raw and (self.reader_dir() / raw / "story.json").exists():
            return raw
        want = _norm_story_name(raw)
        if not want:
            raise ReaderError(f"no followed story {name!r}")
        stories: list[tuple[str, FollowedStory]] = []
        for meta_path in sorted(self.reader_dir().glob("*/story.json")):
            try:
                story = FollowedStory.from_dict(
                    json.loads(meta_path.read_text(encoding="utf-8")))
            except Exception:  # noqa: BLE001 - a corrupt entry never resolves
                continue
            stories.append((meta_path.parent.name, story))

        def keys(slug: str, story: FollowedStory) -> set[str]:
            full = _norm_story_name(slug)
            out = {full, _norm_story_name(story.title)}
            suffix = _norm_story_name(story.source or "")
            if suffix and full.endswith("-" + suffix):
                out.add(full[: -(len(suffix) + 1)])
            return {k for k in out if k}

        exact = [slug for slug, st in stories if want in keys(slug, st)]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            raise ReaderError(
                f"{name!r} is ambiguous — it matches more than one "
                f"followed story: {', '.join(sorted(exact))}")
        prefix = [slug for slug, st in stories
                  if any(k.startswith(want) for k in keys(slug, st))]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            raise ReaderError(
                f"{name!r} is ambiguous — it matches more than one "
                f"followed story: {', '.join(sorted(prefix))}")
        raise ReaderError(f"no followed story {name!r}")

    def _load_exact(self, slug: str) -> FollowedStory:
        """Load by canonical slug (no resolution)."""
        path = self.reader_dir() / slug / "story.json"
        if not path.exists():
            raise ReaderError(f"no followed story {slug!r}")
        return FollowedStory.from_dict(
            json.loads(path.read_text(encoding="utf-8")))

    def _load(self, slug: str) -> FollowedStory:
        return self._load_exact(self.resolve_slug(slug))

    def _save(self, story: FollowedStory) -> None:
        path = self._meta_path(story.slug)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(story.to_dict(), indent=2,
                                  ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _adapter(self, url: str) -> SourceAdapter:
        reg = self._registry()
        if reg is None:
            return adapter_for_url(url)
        try:
            adapter = adapter_for_url(url, source_registry=reg)
        except TypeError:  # a test double with the pre-#89 signature
            adapter = adapter_for_url(url)
        bind = getattr(adapter, "bind_book_registry", None)
        if callable(bind):
            bind(reg)
        return adapter

    def _search_all(self, query: str, *, limit: int = 10,
                    sources: list[str] | None = None) -> list[Any]:
        """search_all with this reader's registry, when it has one
        (Pass #89) — the module function's pre-#89 call otherwise."""
        reg = self._registry()
        if reg is None:
            return search_all(query, limit=limit, sources=sources)
        return search_all(query, limit=limit, sources=sources,
                          source_registry=reg)

    def follow(self, url_or_query: str, *,
               source: str = "") -> FollowedStory:
        """Follow a story by URL — or by title (searches verified sources)."""
        target = (url_or_query or "").strip()
        if not target:
            raise ReaderError("follow needs a URL or a story title")
        if not target.startswith("http"):
            hits = self._search_all(
                target, limit=8,
                sources=[source] if source else None)
            if not hits:
                raise ReaderError(
                    f"no story found for {target!r} on any verified source")
            target = hits[0].url
        adapter = self._adapter(target)
        meta: StoryMeta = adapter.novel(target)
        slug = slugify(f"{meta.title}-{adapter.name}", fallback="story")
        # exact-slug existence check (NOT _load: name resolution must
        # never let a new follow land on a different story's record)
        meta_path = self.reader_dir() / slug / "story.json"
        if meta_path.exists():
            story = FollowedStory.from_dict(
                json.loads(meta_path.read_text(encoding="utf-8")))
            # re-follow refreshes metadata, never wipes reading state
            story.title = meta.title or story.title
            story.url = meta.url
            story.author = meta.author or story.author
            story.total_chapters = meta.total_chapters or story.total_chapters
            story.status = meta.status or story.status
        else:
            story = FollowedStory(
                slug=slug, title=meta.title, url=meta.url,
                source=adapter.name, author=meta.author,
                synopsis=meta.synopsis, genres=meta.genres,
                status=meta.status, total_chapters=meta.total_chapters,
                cover_url=meta.cover_url, followed_at=_now())
        self._save(story)
        _log.info("following story %s (%s)", slug, adapter.name)
        return story

    def unfollow(self, slug: str, *, keep_cache: bool = True) -> dict[str, Any]:
        story = self._load(slug)
        slug = story.slug
        d = self.story_dir(slug)
        if keep_cache:
            # keep the chapters; drop the follow record
            meta = self._meta_path(slug)
            if meta.exists():
                meta.unlink()
        else:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
        return {"slug": slug, "title": story.title, "unfollowed": True,
                "cache_kept": keep_cache}

    def following(self) -> list[dict[str, Any]]:
        out = []
        for meta_path in sorted(self.reader_dir().glob("*/story.json")):
            try:
                story = self._load(meta_path.parent.name)
            except Exception:  # noqa: BLE001 - a corrupt entry never lists
                continue
            out.append({
                "slug": story.slug, "title": story.title,
                "author": story.author, "source": story.source,
                "status": story.status,
                "progress": f"{story.current_chapter}/{story.total_chapters or '?'}",
                "current_chapter": story.current_chapter,
                "total_chapters": story.total_chapters,
                "chapters_cached": story.chapters_cached,
                "ingested": story.ingested_chapters,
                "ingest_total": story.ingest_total,
                "library_complete": bool(story.ingest_total) and
                    story.ingested_chapters >= story.ingest_total,
                "last_read_at": story.last_read_at,
            })
        return out

    def search(self, query: str, *, limit: int = 10,
               sources: list[str] | None = None) -> list[dict[str, Any]]:
        return [h.to_dict() for h in
                self._search_all(query, limit=limit, sources=sources)]

    # ── chapters ──────────────────────────────────────────────────────────
    def _chapter_path(self, slug: str, number: int) -> Path:
        return self.story_dir(slug) / "chapters" / f"{number:05d}.md"

    def _store_chapter(self, slug: str, chapter: Chapter) -> Path:
        path = self._chapter_path(slug, chapter.number)
        body = f"# {chapter.title}\n\n*Chapter {chapter.number} — {chapter.source}*\n\n"
        body += "\n\n".join(chapter.paragraphs) + "\n"
        path.write_text(body, encoding="utf-8")
        meta = {"number": chapter.number, "title": chapter.title,
                "url": chapter.url, "prev_url": chapter.prev_url,
                "next_url": chapter.next_url, "words": chapter.words}
        path.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        return path

    def _cached_chapter(self, slug: str, number: int) -> Chapter | None:
        path = self._chapter_path(slug, number)
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
        text = path.read_text(encoding="utf-8")
        # strip the header we wrote
        parts = text.split("\n\n", 2)
        body = parts[2] if len(parts) > 2 else text
        return Chapter(number=meta.get("number", number),
                       title=meta.get("title", f"Chapter {number}"),
                       url=meta.get("url", ""),
                       paragraphs=[p for p in body.split("\n\n") if p.strip()],
                       prev_url=meta.get("prev_url", ""),
                       next_url=meta.get("next_url", ""),
                       source=meta.get("source", ""))

    def _chapter_url(self, story: FollowedStory, number: int) -> str:
        """Resolve a chapter URL: cache map → chapter list → next-link walk."""
        cached = self._cached_chapter(story.slug, number)
        if cached and cached.next_url and number >= 1:
            # prefer the authoritative list when we can get it cheaply
            pass
        adapter = self._adapter(story.url)
        profile = self._profile()
        try:
            links = adapter.chapter_list(story.url, limit=profile["list_limit"])
        except SourceError as exc:
            _log.warning("chapter list failed for %s: %s", story.slug, exc)
            links = []
        for n, _title, href in links:
            if n == number:
                return href
        # fall back to walking forward from the current chapter
        if number == story.current_chapter + 1:
            cur = self._cached_chapter(story.slug, story.current_chapter)
            if cur and cur.next_url:
                return cur.next_url
        # last resort: deterministic URL when the pattern is predictable
        direct = adapter.chapter_url(story.url, number)
        if direct:
            _log.info("chapter list missed ch%s of %s — trying %s",
                      number, story.slug, direct)
            return direct
        raise ReaderError(
            f"chapter {number} of {story.title!r}: no URL found "
            f"(source lists {len(links)} chapters)")

    def read(self, slug: str, chapter: int = 0) -> dict[str, Any]:
        """Read a chapter (cached when available); updates progress."""
        story = self._load(slug)
        slug = story.slug
        number = chapter or max(1, story.current_chapter or 1)
        if number < 1:
            number = 1
        ch = self._cached_chapter(slug, number)
        source_note = "cache"
        if ch is None:
            adapter = self._adapter(story.url)
            url = self._chapter_url(story, number)
            try:
                ch = adapter.fetch_chapter(url)
            except SourceError as exc:
                raise ReaderError(str(exc)) from exc
            # normalize the number when the source disagrees
            if not ch.number:
                ch.number = number
            self._store_chapter(slug, ch)
            story.chapters_cached = len(
                list((self.story_dir(slug) / "chapters").glob("*.md")))
            source_note = adapter.name
        story.current_chapter = ch.number
        story.current_offset = 0
        story.last_read_at = _now()
        self._save(story)
        return {
            "slug": slug, "title": story.title,
            "chapter": ch.number, "chapter_title": ch.title,
            "words": ch.words, "source": source_note,
            "prev": bool(ch.prev_url), "next": bool(ch.next_url),
            "text": ch.text[:12000],
            "truncated": len(ch.text) > 12000,
        }

    def next(self, slug: str) -> dict[str, Any]:
        story = self._load(slug)
        return self.read(slug, chapter=(story.current_chapter or 0) + 1)

    def prev(self, slug: str) -> dict[str, Any]:
        story = self._load(slug)
        number = max(1, (story.current_chapter or 2) - 1)
        return self.read(slug, chapter=number)

    def chapter_text(self, slug: str, number: int) -> str:
        """Full raw text of a chapter (fetching it first when needed)."""
        slug = self.resolve_slug(slug)
        ch = self._cached_chapter(slug, number)
        if ch is None:
            self.read(slug, chapter=number)
            ch = self._cached_chapter(slug, number)
        if ch is None:
            raise ReaderError(f"chapter {number} unavailable")
        return f"# {ch.title}\n\n{ch.text}"

    def sync(self, slug: str = "", *, chapters: int = 0) -> dict[str, Any]:
        """Fetch the next few unread chapters for one/all followed stories."""
        profile = self._profile()
        want = chapters or profile["sync_chapters"]
        targets = [self._load(slug)] if slug else [
            self._load(p.parent.name)
            for p in sorted(self.reader_dir().glob("*/story.json"))]
        report: list[dict[str, Any]] = []
        for story in targets:
            fetched: list[int] = []
            adapter = self._adapter(story.url)
            start = (story.current_chapter or 0) + 1
            for n in range(start, start + want):
                try:
                    url = self._chapter_url(story, n)
                    ch = adapter.fetch_chapter(url)
                except (SourceError, ReaderError) as exc:
                    _log.warning("sync %s ch%s: %s", story.slug, n, exc)
                    break
                if not ch.number:
                    ch.number = n
                self._store_chapter(story.slug, ch)
                fetched.append(ch.number)
                time.sleep(profile["pause_s"])
            story.chapters_cached = len(
                list((self.story_dir(story.slug) / "chapters").glob("*.md")))
            # refresh the known total from the source
            try:
                meta = adapter.novel(story.url)
                if meta.total_chapters:
                    story.total_chapters = meta.total_chapters
            except SourceError:
                pass
            self._save(story)
            report.append({"slug": story.slug, "title": story.title,
                           "fetched": fetched,
                           "new_total": story.total_chapters})
        return {"synced": report}

    # ── full library ingestion ────────────────────────────────────────────
    def _cached_numbers(self, slug: str) -> set[int]:
        """Chapter numbers held in the story's library (disk is truth)."""
        return {int(p.stem) for p in
                (self.story_dir(slug) / "chapters").glob("*.md")
                if p.stem.isdigit()}

    @staticmethod
    def _watermark(cached: set[int]) -> int:
        """Highest N such that chapters 1..N are ALL in the library."""
        n = 0
        while n + 1 in cached:
            n += 1
        return n

    def ingest_state(self, slug: str) -> dict[str, Any]:
        """Library completeness from disk — no network."""
        story = self._load(slug)
        cached = self._cached_numbers(story.slug)
        watermark = self._watermark(cached)
        total = story.ingest_total or story.total_chapters or 0
        return {"slug": story.slug, "title": story.title,
                "ingested": watermark, "cached": len(cached),
                "total": total,
                "complete": bool(total) and watermark >= total}

    def ingest(self, slug: str, *, max_chapters: int = 0) -> dict[str, Any]:
        """Ingest the WHOLE story into its own library, 1..source end.

        Resumable: chapters already cached are never re-fetched, and
        progress (contiguous watermark vs the source total observed)
        is persisted on the story record after every run.  Ingestion
        is not reading — ``current_chapter`` is never touched.

        Enumeration asks the adapter for its FULL list (``limit=0`` —
        the profile list cap applies to interactive reads, never to
        ingestion); numbers the listing misses are resolved via the
        previous chapter's ``next_url`` chain or the adapter's
        deterministic ``chapter_url``.  A chapter that fails is
        recorded, never silently skipped; three consecutive failures
        stop the run (the source is likely down) and the report says
        exactly how far the library got.
        """
        story = self._load(slug)
        slug = story.slug
        adapter = self._adapter(story.url)
        profile = self._profile()
        # refresh the known end from the source itself
        try:
            meta = adapter.novel(story.url)
            if meta.total_chapters:
                story.total_chapters = meta.total_chapters
            if meta.status:
                story.status = meta.status
        except SourceError as exc:
            _log.warning("ingest %s: meta refresh failed: %s", slug, exc)
        try:
            listing = adapter.chapter_list(story.url, limit=0)
        except SourceError as exc:
            # a library already complete on disk stays usable offline —
            # only a genuinely incomplete one fails here
            cached = self._cached_numbers(slug)
            watermark = self._watermark(cached)
            known_total = story.ingest_total or story.total_chapters or 0
            if known_total and watermark >= known_total:
                _log.warning("ingest %s: source unreachable (%s) — "
                             "library already complete on disk",
                             slug, exc)
                return {"slug": slug, "title": story.title, "fetched": [],
                        "failed": [], "ingested": watermark,
                        "cached": len(cached), "total": known_total,
                        "complete": True, "source_unreachable": True}
            raise ReaderError(
                f"could not list chapters of {story.title!r}: {exc}"
            ) from exc
        urls: dict[int, str] = {}
        for n, _title, href in listing:
            if n >= 1 and href:
                urls.setdefault(n, href)
        total = max([story.total_chapters or 0, *urls.keys()], default=0)
        if total <= 0:
            raise ReaderError(
                f"{story.title!r}: the source lists no chapters to ingest")
        cached = self._cached_numbers(slug)
        fetched: list[int] = []
        failed: list[dict[str, Any]] = []
        consecutive_failures = 0
        next_hint = ""
        for n in range(1, total + 1):
            if max_chapters and len(fetched) >= max_chapters:
                break
            if n in cached:
                ch = self._cached_chapter(slug, n)
                if ch and ch.next_url:
                    next_hint = ch.next_url
                continue
            url = urls.get(n) or next_hint or adapter.chapter_url(
                story.url, n)
            if not url:
                failed.append({"number": n,
                               "reason": "no URL in the chapter list, "
                                         "the next-link chain, or the "
                                         "adapter's URL pattern"})
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    break
                continue
            try:
                ch = adapter.fetch_chapter(url)
            except SourceError as exc:
                failed.append({"number": n, "reason": str(exc)[:160]})
                consecutive_failures += 1
                _log.warning("ingest %s ch%s: %s", slug, n, exc)
                if consecutive_failures >= 3:
                    break
                continue
            consecutive_failures = 0
            if ch.number != n:
                # the enumerated list is authoritative for numbering
                ch.number = n
            self._store_chapter(slug, ch)
            cached.add(n)
            fetched.append(n)
            if ch.next_url:
                next_hint = ch.next_url
            time.sleep(profile["pause_s"])
        watermark = self._watermark(cached)
        story.ingested_chapters = watermark
        story.ingest_total = total
        story.chapters_cached = len(cached)
        self._save(story)
        report = {"slug": slug, "title": story.title,
                  "fetched": fetched, "failed": failed,
                  "ingested": watermark, "cached": len(cached),
                  "total": total, "complete": watermark >= total}
        _log.info("ingest %s: %s/%s chapters (fetched %s, failed %s)",
                  slug, watermark, total, len(fetched), len(failed))
        return report

    # ── progress & bookmarks ──────────────────────────────────────────────
    def get_progress(self, slug: str) -> dict[str, Any]:
        story = self._load(slug)
        return {"slug": story.slug, "title": story.title,
                "chapter": story.current_chapter,
                "offset": story.current_offset,
                "total_chapters": story.total_chapters,
                "last_read_at": story.last_read_at}

    def set_progress(self, slug: str, chapter: int,
                     offset: int = 0) -> dict[str, Any]:
        story = self._load(slug)
        if chapter < 1:
            raise ReaderError("chapter must be >= 1")
        story.current_chapter = chapter
        story.current_offset = max(0, offset)
        story.last_read_at = _now()
        self._save(story)
        return self.get_progress(slug)

    def resume(self, slug: str) -> dict[str, Any]:
        story = self._load(slug)
        progress = self.get_progress(slug)
        progress["resume_chapter"] = max(1, story.current_chapter or 1)
        return progress

    def _bookmarks_path(self, slug: str) -> Path:
        return self.story_dir(slug) / "bookmarks.json"

    def _read_json_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:  # noqa: BLE001
            return []

    def add_bookmark(self, slug: str, chapter: int, offset: int = 0,
                     label: str = "") -> dict[str, Any]:
        story = self._load(slug)  # validates the story exists
        slug = story.slug
        path = self._bookmarks_path(slug)
        marks = self._read_json_list(path)
        mark_id = max([m.get("id", 0) for m in marks] + [0]) + 1
        mark = {"id": mark_id, "chapter": chapter, "offset": offset,
                "label": label or f"Chapter {chapter}", "at": _now()}
        marks.append(mark)
        path.write_text(json.dumps(marks, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return {"slug": slug, "bookmark": mark}

    def list_bookmarks(self, slug: str = "") -> list[dict[str, Any]]:
        if slug:
            slug = self.resolve_slug(slug)
            return [{"slug": slug, **m}
                    for m in self._read_json_list(self._bookmarks_path(slug))]
        out: list[dict[str, Any]] = []
        for meta_path in sorted(self.reader_dir().glob("*/story.json")):
            s = meta_path.parent.name
            out.extend({"slug": s, **m}
                       for m in self._read_json_list(self._bookmarks_path(s)))
        return out

    def remove_bookmark(self, slug: str, mark_id: int) -> dict[str, Any]:
        slug = self.resolve_slug(slug)
        path = self._bookmarks_path(slug)
        marks = self._read_json_list(path)
        kept = [m for m in marks if m.get("id") != mark_id]
        if len(kept) == len(marks):
            raise ReaderError(f"no bookmark #{mark_id} on {slug!r}")
        path.write_text(json.dumps(kept, indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return {"slug": slug, "removed": mark_id}


# ── updates, highlights, stats ─────────────────────────────────────────────
#
# Mined from FanFicFare (mark_new_chapters: only-new-chapter updates) and
# KOReader (highlights with colors + notes, exportable; reading stats).


_HIGHLIGHT_COLORS = ("yellow", "green", "blue", "pink", "orange")


def _highlights_path(self: "StoryReader", slug: str) -> Path:
    return self.story_dir(slug) / "highlights.json"


def add_highlight(self: "StoryReader", slug: str, chapter: int,
                  quote: str, note: str = "",
                  color: str = "yellow") -> dict[str, Any]:
    """KOReader-style highlight on a chapter passage."""
    story = self._load(slug)  # validates the story exists
    slug = story.slug
    quote = (quote or "").strip()
    if not quote:
        raise ReaderError("highlight needs quoted text")
    color = color if color in _HIGHLIGHT_COLORS else "yellow"
    path = _highlights_path(self, slug)
    marks = self._read_json_list(path)
    hid = max([m.get("id", 0) for m in marks] + [0]) + 1
    mark = {"id": hid, "chapter": chapter, "quote": quote[:2000],
            "note": (note or "").strip()[:2000], "color": color,
            "at": _now()}
    marks.append(mark)
    path.write_text(json.dumps(marks, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return {"slug": slug, "highlight": mark}


def list_highlights(self: "StoryReader",
                    slug: str = "") -> list[dict[str, Any]]:
    if slug:
        slug = self.resolve_slug(slug)
        return [{"slug": slug, **m}
                for m in self._read_json_list(_highlights_path(self, slug))]
    out: list[dict[str, Any]] = []
    for meta_path in sorted(self.reader_dir().glob("*/story.json")):
        s = meta_path.parent.name
        out.extend({"slug": s, **m}
                   for m in self._read_json_list(_highlights_path(self, s)))
    return out


def remove_highlight(self: "StoryReader", slug: str,
                     highlight_id: int) -> dict[str, Any]:
    slug = self.resolve_slug(slug)
    path = _highlights_path(self, slug)
    marks = self._read_json_list(path)
    kept = [m for m in marks if m.get("id") != highlight_id]
    if len(kept) == len(marks):
        raise ReaderError(f"no highlight #{highlight_id} on {slug!r}")
    path.write_text(json.dumps(kept, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return {"slug": slug, "removed": highlight_id}


def export_highlights(self: "StoryReader", slug: str = "",
                      format: str = "markdown") -> dict[str, Any]:
    """Export highlights + notes (KOReader-style export)."""
    fmt = (format or "markdown").lower()
    if fmt not in ("markdown", "json", "text"):
        raise ReaderError(f"format must be markdown|json|text, got {format!r}")
    marks = list_highlights(self, slug)
    if fmt == "json":
        return {"format": "json",
                "text": json.dumps(marks, indent=2, ensure_ascii=False)}
    by_story: dict[str, list[dict[str, Any]]] = {}
    for m in marks:
        by_story.setdefault(m["slug"], []).append(m)
    lines: list[str] = []
    for s in sorted(by_story):
        try:
            title = self._load(s).title
        except ReaderError:
            title = s
        lines.append(f"# {title}" if fmt == "markdown" else f"== {title} ==")
        for m in sorted(by_story[s], key=lambda x: (x["chapter"], x["id"])):
            note = f" — {m['note']}" if m.get("note") else ""
            if fmt == "markdown":
                lines.append(f"- ch.{m['chapter']} [{m['color']}] "
                             f"“{m['quote']}”{note}")
            else:
                lines.append(f"* [ch.{m['chapter']}] \"{m['quote']}\"{note}")
        lines.append("")
    return {"format": fmt, "text": "\n".join(lines).strip()}


def check_updates(self: "StoryReader",
                  slug: str = "") -> dict[str, Any]:
    """Compare cached chapters against the live chapter list.

    FanFicFare's mark_new_chapters idea: report exactly which chapters
    are new since the last sync, per story.
    """
    targets = [self._load(slug)] if slug else [
        self._load(p.parent.name)
        for p in sorted(self.reader_dir().glob("*/story.json"))]
    report: list[dict[str, Any]] = []
    for story in targets:
        cached = {int(p.stem) for p in
                  (self.story_dir(story.slug) / "chapters").glob("*.md")
                  if p.stem.isdigit()}
        new: list[int] = []
        live_total = story.total_chapters
        try:
            adapter = self._adapter(story.url)
            listing = adapter.chapter_list(
                story.url, limit=self._profile()["list_limit"])
            live_total = len(listing) or live_total
            new = [n for n, _t, _u in listing if n not in cached]
        except SourceError as exc:
            report.append({"slug": story.slug, "title": story.title,
                           "ok": False, "reason": str(exc)[:160]})
            continue
        report.append({"slug": story.slug, "title": story.title, "ok": True,
                       "cached": len(cached), "live_total": live_total,
                       "new_chapters": sorted(new)[:50],
                       "new_count": len(new)})
    return {"updates": report}


def catch_up(self: "StoryReader", slug: str,
             limit: int = 10) -> dict[str, Any]:
    """Read every unread cached chapter in order, advancing progress."""
    story = self._load(slug)
    slug = story.slug
    cached = sorted(
        int(p.stem) for p in
        (self.story_dir(slug) / "chapters").glob("*.md")
        if p.stem.isdigit())
    unread = [n for n in cached if n > (story.current_chapter or 0)][:limit]
    read: list[dict[str, Any]] = []
    for n in unread:
        r = self.read(slug, chapter=n)
        read.append({"chapter": r["chapter"], "title": r["chapter_title"],
                     "words": r["words"]})
    return {"slug": slug, "title": story.title, "read": read,
            "remaining": max(0, len(cached) - (story.current_chapter or 0)
                             - len(read))}


def reading_stats(self: "StoryReader",
                  slug: str = "") -> dict[str, Any]:
    """Per-story reading stats: progress %, unread, highlights, streak."""
    targets = [self._load(slug)] if slug else [
        self._load(p.parent.name)
        for p in sorted(self.reader_dir().glob("*/story.json"))]
    stories = []
    for story in targets:
        cached = sum(1 for _ in
                     (self.story_dir(story.slug) / "chapters").glob("*.md"))
        total = story.total_chapters or cached
        pct = round(100.0 * (story.current_chapter or 0) / total, 1) if total else 0
        stories.append({
            "slug": story.slug, "title": story.title,
            "chapter": story.current_chapter, "cached": cached,
            "total_chapters": total, "percent": pct,
            "unread_cached": max(0, cached - (story.current_chapter or 0)),
            "highlights": len(list_highlights(self, story.slug)),
            "bookmarks": len(self.list_bookmarks(story.slug)),
            "last_read_at": story.last_read_at,
        })
    return {"stories": stories}


# bind the new methods onto StoryReader (keeps the class body untouched)
StoryReader.add_highlight = add_highlight
StoryReader.list_highlights = list_highlights
StoryReader.remove_highlight = remove_highlight
StoryReader.export_highlights = export_highlights
StoryReader.check_updates = check_updates
StoryReader.catch_up = catch_up
StoryReader.reading_stats = reading_stats
