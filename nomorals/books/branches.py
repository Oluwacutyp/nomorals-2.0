"""BookForge story branches — what-if timelines.

Mined from AI Writing Studio's branch-per-story-arc, inkle's Ink (weaves,
gather points, knots/stitches, choice nodes, conditional text), and
AI-Dungeon-style story trees (cheap forks, multiple takes per node, a
visual tree view):

* main branch is canon; feature branches test major plot changes;
* choice nodes record interactive decision points with reader picks;
* tree() renders the whole branch graph;
* merge is explicit with a merge record (what came from where).
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


def _books_root() -> Path:
    return Path.home() / "workspace" / "devon" / "books_data"


class StoryBranch:
    """A what-if timeline forked from a story at a chapter."""

    def __init__(self, story_slug: str, branch_name: str) -> None:
        self.story_slug = story_slug
        self.branch_name = branch_name
        self._dir = _books_root() / story_slug / "branches" / branch_name

    @property
    def exists(self) -> bool:
        return (self._dir / "branch.json").exists()

    def fork(self, from_chapter: int, premise: str) -> dict[str, Any]:
        """Fork a new timeline from a chapter with a what-if premise."""
        story_dir = _books_root() / self.story_slug
        if not story_dir.exists():
            return {"ok": False, "reason": f"story '{self.story_slug}' not found"}
        self._dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "story": self.story_slug,
            "branch": self.branch_name,
            "forked_from_chapter": from_chapter,
            "premise": premise,
            "created": time.time(),
            "chapters": [],
            "status": "active",
        }
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "branch": self.branch_name,
                "forked_from": from_chapter, "premise": premise}

    def add_chapter(self, text: str, title: str = "") -> dict[str, Any]:
        """Add a chapter to this branch timeline."""
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        meta = json.loads((self._dir / "branch.json").read_text())
        n = len(meta["chapters"]) + 1
        ch_path = self._dir / f"chapter_{n:03d}.md"
        ch_path.write_text(f"# {title or f'Chapter {n}'}\n\n{text}")
        meta["chapters"].append({
            "n": n, "title": title or f"Chapter {n}",
            "words": len(text.split()), "path": str(ch_path),
        })
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "chapter": n, "words": len(text.split())}

    def chapters(self) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        return json.loads((self._dir / "branch.json").read_text())["chapters"]

    def read_chapter(self, n: int) -> dict[str, Any]:
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        ch_path = self._dir / f"chapter_{n:03d}.md"
        if not ch_path.exists():
            return {"ok": False, "reason": f"chapter {n} not found in branch"}
        return {"ok": True, "text": ch_path.read_text()}

    def merge(self, strategy: str = "append") -> dict[str, Any]:
        """Merge branch back to canon.

        Strategies: ``append`` (branch chapters continue the main story),
        ``replace-from`` (branch chapters overwrite canon from the fork
        point), ``interleave`` (alternate canon/branch chapters).
        """
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        if strategy not in ("append", "replace-from", "interleave"):
            return {"ok": False,
                    "reason": "strategy must be append | replace-from | interleave"}
        meta = json.loads((self._dir / "branch.json").read_text())
        meta["status"] = "merged"
        meta["merged_at"] = time.time()
        meta["merge_strategy"] = strategy
        # merge record: what came from where (auditable canon)
        meta["merge_record"] = {
            "strategy": strategy,
            "forked_from_chapter": meta["forked_from_chapter"],
            "branch_chapters": [c["n"] for c in meta["chapters"]],
            "choices": meta.get("choices", []),
            "premise": meta.get("premise", ""),
        }
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "branch": self.branch_name,
                "chapters_merged": len(meta["chapters"]), "strategy": strategy}

    def rename(self, new_name: str) -> dict[str, Any]:
        """Rename a branch."""
        new_name = (new_name or "").strip()
        if not new_name:
            return {"ok": False, "reason": "new name required"}
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        target = _books_root() / self.story_slug / "branches" / new_name
        if target.exists():
            return {"ok": False, "reason": f"branch {new_name!r} exists"}
        meta = json.loads((self._dir / "branch.json").read_text())
        self._dir.rename(target)
        meta["branch"] = new_name
        (target / "branch.json").write_text(json.dumps(meta, indent=2))
        self.branch_name = new_name
        self._dir = target
        return {"ok": True, "branch": new_name}

    # ── choice nodes (Ink-style interactive decisions) ──────────────────
    def add_choice(self, after_chapter: int, prompt: str,
                   options: list[str]) -> dict[str, Any]:
        """Add an interactive choice node after a branch chapter.

        Options are the weaves; the reader's pick is recorded.  A gather
        point (``gather=True``) collapses the weave back to one timeline.
        """
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        options = [o.strip() for o in (options or []) if o.strip()]
        if len(options) < 2:
            return {"ok": False, "reason": "a choice needs ≥2 options"}
        meta = json.loads((self._dir / "branch.json").read_text())
        choices = meta.setdefault("choices", [])
        node = {"id": len(choices) + 1, "after_chapter": after_chapter,
                "prompt": prompt.strip(), "options": options,
                "pick": None, "decided_at": None}
        choices.append(node)
        (self._dir / "branch.json").write_text(json.dumps(meta, indent=2))
        return {"ok": True, "choice_id": node["id"], "options": options}

    def decide(self, choice_id: int, option: str | int) -> dict[str, Any]:
        """Record the reader's pick for a choice node."""
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        meta = json.loads((self._dir / "branch.json").read_text())
        for node in meta.get("choices", []):
            if node["id"] == choice_id:
                opts = node["options"]
                pick = opts[int(option)] if isinstance(option, int) else option
                if pick not in opts:
                    return {"ok": False,
                            "reason": f"not an option: {option!r}"}
                node["pick"] = pick
                node["decided_at"] = time.time()
                (self._dir / "branch.json").write_text(
                    json.dumps(meta, indent=2))
                return {"ok": True, "choice_id": choice_id, "pick": pick}
        return {"ok": False, "reason": f"no choice #{choice_id}"}

    def list_choices(self) -> list[dict[str, Any]]:
        if not self.exists:
            return []
        meta = json.loads((self._dir / "branch.json").read_text())
        return meta.get("choices", [])

    # ── tree + diff ─────────────────────────────────────────────────────
    def diff(self) -> dict[str, Any]:
        """Chapter-by-chapter divergence of this branch vs canon."""
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        meta = json.loads((self._dir / "branch.json").read_text())
        story_dir = _books_root() / self.story_slug
        fork = meta["forked_from_chapter"]
        # canon chapters: files chapter_*.md next to story.json-style layout
        canon_words: dict[int, int] = {}
        for child in sorted(story_dir.glob("chapter_*.md")):
            try:
                n = int(child.stem.split("_")[1])
                canon_words[n] = len(child.read_text(
                    encoding="utf-8").split())
            except (IndexError, ValueError, OSError):
                continue
        rows = []
        for c in meta["chapters"]:
            canon_n = fork + c["n"]
            cw = canon_words.get(canon_n)
            branch_text = ""
            try:
                branch_text = (self._dir /
                               f"chapter_{c['n']:03d}.md").read_text(
                                   encoding="utf-8")
            except OSError:
                pass
            canon_text = ""
            if cw is not None:
                try:
                    canon_text = next(
                        p for p in story_dir.glob("chapter_*.md")
                        if int(p.stem.split("_")[1]) == canon_n).read_text(
                            encoding="utf-8")
                except (StopIteration, OSError, IndexError, ValueError):
                    pass
            diverged = cw is None
            if not diverged and branch_text and canon_text:
                # Jaccard word overlap: <0.5 = genuinely different telling
                bset = set(branch_text.lower().split())
                cset = set(canon_text.lower().split())
                union = bset | cset
                overlap = len(bset & cset) / len(union) if union else 0
                diverged = overlap < 0.5
            rows.append({
                "branch_chapter": c["n"],
                "canon_chapter": canon_n,
                "branch_words": c["words"],
                "canon_words": cw,
                "text_overlap": round(overlap, 2) if cw is not None and
                branch_text and canon_text else None,
                "diverged": diverged,
            })
        first_div = next((r["branch_chapter"] for r in rows
                          if r["diverged"]), None)
        return {"ok": True, "branch": self.branch_name, "forked_from": fork,
                "chapters": rows, "first_divergence": first_div,
                "choices": meta.get("choices", [])}

    def abandon(self) -> dict[str, Any]:
        if not self.exists:
            return {"ok": False, "reason": "branch does not exist"}
        shutil.rmtree(self._dir)
        return {"ok": True, "branch": self.branch_name, "abandoned": True}


def list_branches(story_slug: str) -> list[dict[str, Any]]:
    """All branches for a story."""
    bdir = _books_root() / story_slug / "branches"
    if not bdir.exists():
        return []
    out = []
    for child in sorted(bdir.iterdir()):
        meta_f = child / "branch.json"
        if meta_f.exists():
            try:
                m = json.loads(meta_f.read_text())
                out.append({
                    "branch": m["branch"],
                    "forked_from": m["forked_from_chapter"],
                    "premise": m["premise"][:120],
                    "chapters": len(m["chapters"]),
                    "choices": len(m.get("choices", [])),
                    "status": m["status"],
                })
            except Exception:
                continue
    return out


def tree(story_slug: str) -> dict[str, Any]:
    """The whole story as a tree: canon trunk + branch limbs + choice nodes.

    AI-Dungeon story-tree style: every fork point, branch, and pending
    decision in one structure, renderable with ``styles.render_tree``.
    """
    story_dir = _books_root() / story_slug
    canon: list[int] = []
    for child in sorted(story_dir.glob("chapter_*.md")):
        try:
            canon.append(int(child.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    root: dict[str, Any] = {
        "name": f"canon ({story_slug})",
        "info": f"{len(canon)} chapters" if canon else "no chapters yet",
        "children": [],
    }
    for b in list_branches(story_slug):
        node: dict[str, Any] = {
            "name": f"⑂ {b['branch']}",
            "info": (f"fork@{b['forked_from']} · {b['chapters']} ch · "
                     f"{b['choices']} choices · {b['status']}"),
            "children": [],
        }
        try:
            meta = json.loads(
                (story_dir / "branches" / b["branch"] / "branch.json")
                .read_text())
            for ch in meta.get("choices", []):
                pick = f" → picked: {ch['pick']}" if ch.get("pick") else \
                    " (undecided)"
                node["children"].append({
                    "name": f"◇ choice #{ch['id']}: {ch['prompt'][:60]}",
                    "info": f"{len(ch['options'])} options{pick}",
                    "children": [],
                })
        except Exception:  # noqa: BLE001
            pass
        root["children"].append(node)
    return root
