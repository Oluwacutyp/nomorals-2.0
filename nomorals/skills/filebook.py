"""Flat-file skillbook — Devon's ``.md`` skill library (L4).

One topic per Markdown file, with a small frontmatter header the brain
routes by.  Two roots:

* the **seed library** shipped in the repo (``nomorals/skills/library/``)
  — distilled organ knowledge, versioned with the code;
* the **user library** at ``$NM_HOME/skills/`` — distilled drafts land in
  ``drafts/`` and promote into the user root, which *shadows* the seed by
  id (a promoted revision of a seed skill never edits the repo file).

Routing is deterministic token scoring over topics / triggers / title /
summary — no model call, so the seam that assembles task context
(``nomorals.context.engine``) can consult it on every mission cheaply.
Every load is recorded in a JSONL usage log; outcomes are settled when
the mission terminates, and a draft is auto-promoted after it has been
loaded and the mission succeeded in
:data:`PROMOTE_AFTER_SUCCESSES` **distinct** missions (an explicit
``promote`` verb on ``/skills`` also promotes immediately).

This is the *knowledge* half of the skills system.  The executable
package registry in this same package (``registry.py``) is unchanged —
it runs tool chains; this one teaches the brain how.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PROMOTE_AFTER_SUCCESSES",
    "SkillDoc",
    "FileSkillLibrary",
    "parse_skill_text",
    "render_index",
    "seed_library_dir",
    "user_skills_dir",
    "default_library",
]

log = logging.getLogger(__name__)

#: A distilled draft promotes automatically once it has been loaded in
#: this many distinct missions that went on to succeed.  Below that it
#: stays a draft: routable, but labelled unproven wherever it surfaces.
PROMOTE_AFTER_SUCCESSES = 2

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "is", "are", "be", "do", "does", "how", "what", "when", "it", "this",
    "that", "from", "by", "at", "as", "into", "her", "his", "she",
})


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(str(text or "").lower())
            if t not in _STOPWORDS and len(t) > 1]


def seed_library_dir() -> Path:
    return Path(__file__).resolve().parent / "library"


def user_skills_dir() -> Path:
    home = os.environ.get("NM_HOME") or "~/.nomorals"
    return Path(os.path.expanduser(home)) / "skills"


@dataclass(frozen=True)
class SkillDoc:
    """One parsed skill file."""

    id: str
    title: str
    summary: str
    topics: tuple[str, ...] = ()
    triggers: tuple[str, ...] = ()
    layer: str = ""
    version: int = 1
    body: str = ""
    path: str = ""
    origin: str = "seed"          # seed | user | draft
    revises: str = ""             # drafts: the live skill id they amend
    # Craft packs (docs 22+23 Stage A P3): a pack is a skill doc whose
    # frontmatter declares ``kind: pack``; its body carries the pack
    # sections and its fixtures name the executable checks its
    # evaluators run.  Plain skills leave all three empty — the pack
    # schema lives in skills/packs.py, the lifecycle below is shared.
    kind: str = ""                # "" = plain skill | "pack"
    domain: str = ""              # packs: the domain the pack serves
    fixtures: tuple[str, ...] = ()  # packs: executable fixture paths

    @property
    def is_draft(self) -> bool:
        return self.origin == "draft"

    def header(self) -> str:
        label = " (draft skill — unproven)" if self.is_draft else ""
        return f"Skill: {self.title} [{self.id}]{label}"


def _split_list(raw: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in str(raw or "").split(",") if p.strip())


def parse_skill_text(text: str, *, path: str = "",
                     origin: str = "seed") -> SkillDoc | None:
    """Parse one ``.md`` skill file.  ``None`` when it has no usable
    frontmatter (an id is mandatory — the router keys on it)."""
    lines = str(text or "").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    meta: dict[str, str] = {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
        if ":" in lines[i]:
            key, _, value = lines[i].partition(":")
            meta[key.strip().lower()] = value.strip()
    if end is None or not meta.get("id"):
        return None
    body = "\n".join(lines[end + 1:]).strip()
    try:
        version = int(meta.get("version") or 1)
    except ValueError:
        version = 1
    return SkillDoc(
        id=meta["id"].strip(),
        title=meta.get("title") or meta["id"].strip(),
        summary=meta.get("summary") or "",
        topics=_split_list(meta.get("topics", "")),
        triggers=_split_list(meta.get("triggers", "")),
        layer=meta.get("layer") or "",
        version=version,
        body=body,
        path=path,
        origin=origin,
        revises=meta.get("revises") or "",
        kind=meta.get("kind") or "",
        domain=meta.get("domain") or "",
        fixtures=_split_list(meta.get("fixtures", "")),
    )


def render_skill_text(doc: SkillDoc) -> str:
    lines = ["---", f"id: {doc.id}", f"title: {doc.title}"]
    if doc.summary:
        lines.append(f"summary: {doc.summary}")
    if doc.topics:
        lines.append("topics: " + ", ".join(doc.topics))
    if doc.triggers:
        lines.append("triggers: " + ", ".join(doc.triggers))
    if doc.layer:
        lines.append(f"layer: {doc.layer}")
    lines.append(f"version: {doc.version}")
    if doc.revises:
        lines.append(f"revises: {doc.revises}")
    if doc.kind:
        lines.append(f"kind: {doc.kind}")
    if doc.domain:
        lines.append(f"domain: {doc.domain}")
    if doc.fixtures:
        lines.append("fixtures: " + ", ".join(doc.fixtures))
    lines.append("---")
    lines.append("")
    lines.append(doc.body.strip())
    lines.append("")
    return "\n".join(lines)


def render_index(docs: Iterable[SkillDoc]) -> str:
    """The root INDEX/map: one line per skill, grouped by layer.

    This exact text is what ``library/INDEX.md`` (seed) and
    ``$NM_HOME/skills/INDEX.md`` (user) contain — a consistency test
    regenerates it and pins the checked-in file to the render.
    """
    rows = sorted(docs, key=lambda d: (d.layer or "zzz", d.id))
    out = ["# Skill index — routed by the brain, loaded on demand", ""]
    current_layer = None
    for doc in rows:
        layer = doc.layer or "general"
        if layer != current_layer:
            current_layer = layer
            out.append(f"## {layer}")
            out.append("")
        topics = ", ".join(doc.topics)
        out.append(f"- `{doc.id}` — {doc.title}: {doc.summary} "
                   f"(topics: {topics})")
    out.append("")
    return "\n".join(out)


class FileSkillLibrary:
    """The routed, loadable skill library over the seed + user roots."""

    def __init__(self, *, seed_dir: Path | None = None,
                 user_dir: Path | None = None) -> None:
        self.seed_dir = Path(seed_dir) if seed_dir else seed_library_dir()
        self.user_dir = Path(user_dir) if user_dir else user_skills_dir()
        self._cache: list[SkillDoc] | None = None

    # -- scanning -----------------------------------------------------

    def _scan(self) -> list[SkillDoc]:
        if self._cache is not None:
            return self._cache
        docs: list[SkillDoc] = []
        for root, origin in ((self.seed_dir, "seed"), (self.user_dir, "user")):
            if root.is_dir():
                for path in sorted(root.glob("*.md")):
                    if path.name.upper() == "INDEX.MD":
                        continue
                    doc = self._read(path, origin)
                    if doc is not None:
                        docs.append(doc)
        drafts = self.user_dir / "drafts"
        if drafts.is_dir():
            for path in sorted(drafts.glob("*.md")):
                doc = self._read(path, "draft")
                if doc is not None:
                    docs.append(doc)
        self._cache = docs
        return docs

    @staticmethod
    def _read(path: Path, origin: str) -> SkillDoc | None:
        try:
            return parse_skill_text(path.read_text(encoding="utf-8"),
                                    path=str(path), origin=origin)
        except OSError:
            return None

    def invalidate(self) -> None:
        self._cache = None

    def docs(self, *, include_drafts: bool = True) -> list[SkillDoc]:
        all_docs = self._scan()
        if include_drafts:
            return list(all_docs)
        return [d for d in all_docs if not d.is_draft]

    def get(self, skill_id: str) -> SkillDoc | None:
        """The live doc for an id — a user copy shadows the seed copy."""
        best: SkillDoc | None = None
        for doc in self._scan():
            if doc.id == skill_id and not doc.is_draft \
                    and (best is None or doc.origin == "user"):
                best = doc
        return best

    def get_draft(self, skill_id: str) -> SkillDoc | None:
        for doc in self._scan():
            if doc.id == skill_id and doc.is_draft:
                return doc
        return None

    def drafts(self) -> list[SkillDoc]:
        return [d for d in self._scan() if d.is_draft]

    # -- routing ------------------------------------------------------

    def _score(self, doc: SkillDoc, query: str,
               qtokens: list[str]) -> float:
        qset = set(qtokens)
        score = 0.0
        lowered = query.lower()
        for trigger in doc.triggers:
            trig = trigger.strip().lower()
            if not trig:
                continue
            if " " in trig and trig in lowered:
                score += 5.0
            score += 2.0 * len(qset & set(_tokens(trig)))
        for topic in doc.topics:
            score += 3.0 * len(qset & set(_tokens(topic)))
        score += 2.0 * len(qset & set(_tokens(doc.title)))
        score += 1.0 * len(qset & set(_tokens(doc.summary)))
        score += 0.25 * min(8, len(qset & set(_tokens(doc.body))))
        if doc.is_draft:
            score *= 0.8
        return score

    def route(self, query: str, *, limit: int = 3,
              include_drafts: bool = True) -> list[tuple[SkillDoc, float]]:
        """Rank skills for a task text.  Deterministic: score desc,
        then id asc.  Zero-score docs are noise and never returned."""
        qtokens = _tokens(query)
        if not qtokens:
            return []
        scored = []
        for doc in self.docs(include_drafts=include_drafts):
            value = self._score(doc, query, qtokens)
            if value > 0:
                scored.append((doc, value))
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:max(0, limit)]

    def search(self, text: str) -> list[SkillDoc]:
        needle = str(text or "").lower()
        if not needle:
            return []
        hits = [d for d in self.docs()
                if needle in d.id.lower() or needle in d.title.lower()
                or needle in d.summary.lower()
                or any(needle in t.lower() for t in d.topics)]
        return sorted(hits, key=lambda d: d.id)

    def load(self, skill_id: str) -> SkillDoc | None:
        return self.get(skill_id) or self.get_draft(skill_id)

    # -- usage log + promotion ----------------------------------------

    @property
    def usage_path(self) -> Path:
        return self.user_dir / "usage.jsonl"

    def _append_usage(self, entry: dict[str, Any]) -> None:
        try:
            self.user_dir.mkdir(parents=True, exist_ok=True)
            with self.usage_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            log.warning("skill usage log write failed: %s", exc)

    def usage_entries(self) -> list[dict[str, Any]]:
        try:
            lines = self.usage_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = []
        for line in lines:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    def record_load(self, skill_ids: Iterable[str], *,
                    mission_id: str = "") -> None:
        """Journal that skills were loaded into a mission's context."""
        for skill_id in skill_ids:
            doc = self.load(skill_id)
            self._append_usage({
                "kind": "load", "skill_id": skill_id,
                "origin": doc.origin if doc else "",
                "mission_id": mission_id, "success": None,
                "ts": time.time(),
            })

    def settle_mission(self, mission_id: str, *, success: bool) -> dict:
        """Settle a finished mission's loads and run promotion checks.

        Returns ``{"settled": n, "promoted": [draft ids]}``.  A draft is
        promoted when the set of *distinct* missions that loaded it and
        settled successfully reaches :data:`PROMOTE_AFTER_SUCCESSES`.
        """
        if not mission_id:
            return {"settled": 0, "promoted": []}
        loaded: dict[str, str] = {}
        for entry in self.usage_entries():
            if entry.get("kind") == "load" \
                    and entry.get("mission_id") == mission_id:
                loaded[str(entry.get("skill_id"))] = str(
                    entry.get("origin") or "")
        for skill_id, origin in loaded.items():
            self._append_usage({
                "kind": "settle", "skill_id": skill_id, "origin": origin,
                "mission_id": mission_id, "success": bool(success),
                "ts": time.time(),
            })
        promoted: list[str] = []
        if success:
            for skill_id, origin in loaded.items():
                if origin != "draft":
                    continue
                if len(self._successful_missions(skill_id)) \
                        >= PROMOTE_AFTER_SUCCESSES:
                    result = self.promote(skill_id)
                    if result.get("ok"):
                        promoted.append(skill_id)
        return {"settled": len(loaded), "promoted": promoted}

    def _successful_missions(self, skill_id: str) -> set[str]:
        missions: set[str] = set()
        for entry in self.usage_entries():
            if entry.get("kind") == "settle" \
                    and entry.get("skill_id") == skill_id \
                    and entry.get("success") is True \
                    and entry.get("mission_id"):
                missions.add(str(entry["mission_id"]))
        return missions

    def draft_progress(self, skill_id: str) -> dict[str, Any]:
        """Promotion progress for a draft: successful missions so far."""
        return {"skill_id": skill_id,
                "successful_missions": len(
                    self._successful_missions(skill_id)),
                "needed": PROMOTE_AFTER_SUCCESSES}

    # -- writes ---------------------------------------------------------

    def write_draft(self, doc: SkillDoc) -> Path:
        """Write a distilled draft into the drafts area (journaled by
        the usage log as a ``distill`` entry)."""
        drafts = self.user_dir / "drafts"
        drafts.mkdir(parents=True, exist_ok=True)
        path = drafts / f"{doc.id}.md"
        draft = SkillDoc(**{**doc.__dict__, "origin": "draft",
                            "path": str(path)})
        path.write_text(render_skill_text(draft), encoding="utf-8")
        self._append_usage({"kind": "distill", "skill_id": doc.id,
                            "origin": "draft", "mission_id": "",
                            "success": None, "ts": time.time()})
        self.invalidate()
        return path

    def promote(self, skill_id: str) -> dict[str, Any]:
        """Promote a draft into the live user library.

        The promoted file lands in the user root, shadowing any seed doc
        with the same id; a draft that revises a live skill bumps its
        version past the live one.  The seed tree is never written.
        """
        draft = self.get_draft(skill_id)
        if draft is None:
            return {"ok": False, "reason": f"no draft named {skill_id!r}"}
        target_id = draft.revises or draft.id
        live = self.get(target_id)
        version = draft.version
        if live is not None and draft.revises:
            version = max(version, live.version + 1)
        promoted = SkillDoc(
            id=target_id, title=draft.title, summary=draft.summary,
            topics=draft.topics, triggers=draft.triggers,
            layer=draft.layer, version=version, body=draft.body,
            origin="user", kind=draft.kind, domain=draft.domain,
            fixtures=draft.fixtures)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        path = self.user_dir / f"{target_id}.md"
        promoted = SkillDoc(**{**promoted.__dict__, "path": str(path)})
        path.write_text(render_skill_text(promoted), encoding="utf-8")
        with contextlib.suppress(OSError):
            Path(draft.path).unlink()
        self._append_usage({"kind": "promote", "skill_id": target_id,
                            "origin": "user", "mission_id": "",
                            "success": True, "ts": time.time()})
        self.invalidate()
        self.write_user_index()
        return {"ok": True, "skill_id": target_id, "path": str(path),
                "version": version}

    def write_user_index(self) -> Path:
        self.user_dir.mkdir(parents=True, exist_ok=True)
        path = self.user_dir / "INDEX.md"
        live = [d for d in self.docs(include_drafts=False)]
        path.write_text(render_index(live), encoding="utf-8")
        return path


def default_library() -> FileSkillLibrary:
    """The process library: repo seed root + ``$NM_HOME/skills``."""
    return FileSkillLibrary()
