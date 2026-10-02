"""The self-improvement arena core.

One cycle:
  1. pick a topic with the adaptive sampler (interest profile x coverage
     balance x difficulty target — never one already digested, never a
     repeat inside the anti-repeat window)
  2. research it with the search engine (quick mode)
  3. digest the findings into durable knowledge (``arena_knowledge``)
  4. (power mode + arena.build) let the builder sub-agent propose ONE small
     self-contained candidate module, written to a sandbox dir and syntax-
     checked — never applied. It becomes a *proposal* the owner reviews.
  5. score the run on verifiable axes (research usefulness, build
     compiled, latency) into ``arena_scores`` — the sampler's feedback.
  6. every step is appended to ``arena_stream`` — the live stream that
     later feeds the owner's own model.

The ship gate (``arena/ship.py``) promotes a pending build into an
``EvolutionProposal``: fast verification (compile + error_scan), then the
owner approves → the evolution machinery applies it to the repo with the
full test gate, or rejects it with a recorded reason. Nothing merges
without the owner saying so.

Candidate code lifecycle: ``pending`` → owner ``/arena approve <id>``
(staged to ``~/.nomorals/arena/approved/<name>/``) or ``/arena deny <id>``.
Nothing is ever merged into the repo without the owner saying so.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..search.engine import SearchEngine
from . import activity
from .topics import sample_topic, surprise_topic

__all__ = ["Arena", "run_cycle_safe"]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class Arena:
    def __init__(self, context: Any, home: str | Path | None = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.settings = getattr(context, "settings", None)
        cfg_home = getattr(self.settings, "home", "~/.nomorals") if self.settings else "~/.nomorals"
        self.home = Path(os.path.expanduser(str(home or cfg_home)))
        self.build_root = self.home / "arena" / "builds"
        self.approve_root = self.home / "arena" / "approved"
        self._stop = threading.Event()
        self._loop_thread: threading.Thread | None = None

    # ── stream (the live feed / training source) ─────────────────────────────
    def _stream(self, kind: str, payload: dict[str, Any]) -> None:
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "INSERT INTO arena_stream (ts, kind, payload) VALUES (?, ?, ?)",
                    (time.time(), kind, json.dumps(payload, ensure_ascii=False)),
                )
        except Exception:  # noqa: BLE001 - the stream must never break a cycle
            pass

    def stream(self, limit: int = 20) -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT ts, kind, payload FROM arena_stream ORDER BY ts DESC LIMIT ?",
                (max(1, min(int(limit), 500)),),
            )
        except Exception:  # noqa: BLE001
            return []
        out = []
        for row in rows:
            try:
                payload = json.loads(row.get("payload") or "{}")
            except Exception:  # noqa: BLE001
                payload = {}
            out.append({"ts": row.get("ts"), "kind": row.get("kind"), "payload": payload})
        return out

    def export(self, limit: int = 200) -> str:
        """The knowledge base as training JSONL (the exporter's format)."""
        lines: list[str] = []
        if self.db is not None:
            try:
                rows = self.db.query(
                    "SELECT topic, category, digest, sources FROM arena_knowledge "
                    "ORDER BY created_at DESC LIMIT ?", (max(1, min(int(limit), 5000)),),
                )
                for row in rows:
                    sources = row.get("sources") or "[]"
                    try:
                        source_list = json.loads(sources)
                    except Exception:  # noqa: BLE001
                        source_list = []
                    source_line = "\n".join(
                        f"- {s.get('title') or s.get('url')}" for s in source_list[:8]
                        if isinstance(s, dict)
                    )
                    assistant = (row.get("digest") or "").strip()
                    if source_line:
                        assistant += f"\n\nSources:\n{source_line}"
                    record = {
                        "messages": [
                            {"role": "user", "content":
                             f"Research this topic and summarize what you learn, with sources: {row.get('topic', '')}"},
                            {"role": "assistant", "content": assistant},
                        ],
                        "weight": 1.0,
                        "source": "arena",
                    }
                    lines.append(json.dumps(record, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(lines)

    # ── knowledge ────────────────────────────────────────────────────────────
    def _save_knowledge(self, topic: str, category: str, digest: str, sources: list[dict]) -> str:
        kid = _new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        """INSERT INTO arena_knowledge
                           (id, topic, category, digest, sources, created_at)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (kid, topic, category, digest, json.dumps(sources, ensure_ascii=False), time.time()),
                    )
            except Exception:  # noqa: BLE001
                pass
        return kid

    # ── research ─────────────────────────────────────────────────────────────
    def _research(self, topic: str) -> dict[str, Any]:
        pages = 3
        if self.settings is not None:
            pages = max(1, min(int(getattr(self.settings.arena, "research_pages", 3)), 8))
        engine = SearchEngine(self.context)
        return engine.run(topic, mode="quick", pages=pages)

    # ── builder sub-agent (power-gated) ──────────────────────────────────────
    _BUILD_SYSTEM = (
        "You are the builder sub-agent of the NoMorals project. "
        "Given a research digest, propose ONE small, genuinely useful, "
        "self-contained Python module that extends the project. "
        "Hard rules: stdlib imports ONLY; new files only (never edits to "
        "existing project files); at most {n} files; each file under 4000 "
        "characters; every file must be syntactically valid Python; "
        "include a module docstring saying what it does and why. "
        "Reply with ONLY a JSON object: "
        '{{"name": "snake_case_name", "purpose": "one sentence", '
        '"files": [{{"path": "relative/path.py", "content": "..."}}]}}'
    )

    def _build(self, topic: str, digest: str) -> dict[str, Any] | None:
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        from ...llm.base import Message

        max_files = 3
        if self.settings is not None:
            max_files = max(1, min(int(getattr(self.settings.arena, "max_build_files", 3)), 8))
        try:
            response = router.chat(
                [
                    Message(role="system", content=self._BUILD_SYSTEM.format(n=max_files)),
                    Message(role="user", content=f"Topic: {topic}\n\nResearch digest:\n{digest[:4000]}"),
                ]
            )
        except Exception:  # noqa: BLE001
            return None
        text = (getattr(response, "text", "") or "").strip()
        if getattr(response, "error", None) or not text:
            return None
        proposal = self._parse_json(text)
        if not proposal or not proposal.get("files"):
            return None
        name = re.sub(r"[^a-z0-9_]", "_", str(proposal.get("name", "")).lower())[:40] or "module"
        build_id = _new_id()
        target = self.build_root / build_id / name
        files: list[dict[str, str]] = []
        try:
            for spec in proposal["files"][:max_files]:
                rel = str(spec.get("path", "")).lstrip("/")
                content = str(spec.get("content", ""))
                if not rel or not content or len(content) > 4000:
                    continue
                dest = (target / rel).resolve()
                if not str(dest).startswith(str(target.resolve())):
                    continue  # path escape attempt
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, "utf-8")
                files.append({"path": rel, "chars": len(content)})
            if not files:
                return None
            errors = self._syntax_check(target)
            record = {
                "id": build_id,
                "name": name,
                "purpose": str(proposal.get("purpose", ""))[:300],
                "topic": topic,
                "files": files,
                "dir": str(target),
                "syntax": "ok" if not errors else "; ".join(errors[:3]),
                "created_at": time.time(),
            }
            if self.db is not None:
                try:
                    with self.db.transaction():
                        self.db.execute(
                            """INSERT INTO arena_builds
                               (id, name, purpose, topic, files, dir, status, created_at)
                               VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                            (build_id, name, record["purpose"], topic,
                             json.dumps(files), str(target), time.time()),
                        )
                except Exception:  # noqa: BLE001
                    pass
            self._stream("build", record)
            return record
        except Exception as exc:  # noqa: BLE001
            self._stream("build", {"error": f"build failed: {exc}", "topic": topic})
            return None

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Extract the first JSON object from a model reply (fence-tolerant)."""
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "")
        start = cleaned.find("{")
        while start != -1:
            depth = 0
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cleaned[start:i + 1])
                            if isinstance(obj, dict):
                                return obj
                        except Exception:  # noqa: BLE001
                            break
            start = cleaned.find("{", start + 1)
        return None

    def _syntax_check(self, directory: Path) -> list[str]:
        errors: list[str] = []
        for path in sorted(directory.rglob("*.py")):
            proc = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode != 0:
                errors.append(f"{path.name}: {proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'syntax error'}")
        return errors

    # ── review (owner decides) ───────────────────────────────────────────────
    def _get_build(self, build_id: str) -> dict | None:
        if self.db is None:
            return None
        try:
            row = self.db.query_one("SELECT * FROM arena_builds WHERE id = ?", (build_id,))
        except Exception:  # noqa: BLE001
            return None
        if row is None:
            return None
        try:
            row["files"] = json.loads(row.get("files") or "[]")
        except Exception:  # noqa: BLE001
            row["files"] = []
        return row

    def _set_build_status(self, build_id: str, status: str) -> None:
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE arena_builds SET status = ?, decided_at = ? WHERE id = ?",
                    (status, time.time(), build_id),
                )
        except Exception:  # noqa: BLE001
            pass

    def approve(self, build_id: str) -> str:
        row = self._get_build(build_id)
        if row is None:
            return f"no arena build {build_id!r}"
        if row.get("status") != "pending":
            return f"build {build_id} is already {row.get('status')}"
        src = Path(row.get("dir", ""))
        dest = self.approve_root / str(row.get("name", build_id))
        try:
            if dest.exists():
                shutil.rmtree(dest)
            if src.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest)
            else:
                return f"build {build_id} files are missing on disk ({src})"
        except Exception as exc:  # noqa: BLE001
            return f"approve failed: {exc}"
        self._set_build_status(build_id, "approved")
        self._stream("decision", {"id": build_id, "decision": "approved", "staged_at": str(dest)})
        return (
            f"approved {build_id} ({row.get('name')}) — staged at {dest}. "
            "review the files, drop the ones you like into the repo, and I can commit on request."
        )

    def deny(self, build_id: str) -> str:
        row = self._get_build(build_id)
        if row is None:
            return f"no arena build {build_id!r}"
        if row.get("status") != "pending":
            return f"build {build_id} is already {row.get('status')}"
        self._set_build_status(build_id, "denied")
        self._stream("decision", {"id": build_id, "decision": "denied", "name": row.get("name")})
        return f"denied {build_id} ({row.get('name')}). it stays in the stream for the training data."

    def builds(self, status: str = "pending") -> list[dict]:
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM arena_builds WHERE status = ? ORDER BY created_at DESC LIMIT 20",
                (status,),
            )
        except Exception:  # noqa: BLE001
            return []

    def all_builds(self, limit: int = 20) -> list[dict]:
        if self.db is None:
            return []
        try:
            return self.db.query(
                "SELECT * FROM arena_builds ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 100)),),
            )
        except Exception:  # noqa: BLE001
            return []

    # ── stats ───────────────────────────────────────────────────────────────
    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cycles": 0, "knowledge": 0, "builds": {}, "last_cycle": None,
        }
        if self.db is None:
            return out
        try:
            row = self.db.query_one(
                "SELECT COUNT(*) AS n FROM arena_stream WHERE kind = 'topic'"
            )
            out["cycles"] = int(row.get("n", 0)) if row else 0
            row = self.db.query_one("SELECT COUNT(*) AS n FROM arena_knowledge")
            out["knowledge"] = int(row.get("n", 0)) if row else 0
            for status in ("pending", "approved", "denied"):
                row = self.db.query_one(
                    "SELECT COUNT(*) AS n FROM arena_builds WHERE status = ?", (status,)
                )
                out["builds"][status] = int(row.get("n", 0)) if row else 0
            row = self.db.query_one("SELECT MAX(ts) AS ts FROM arena_stream WHERE kind = 'topic'")
            out["last_cycle"] = row.get("ts") if row else None
        except Exception:  # noqa: BLE001
            pass
        return out

    # ── digests ──────────────────────────────────────────────────────────────
    def digests(self, limit: int = 1) -> list[dict]:
        if self.db is None:
            return []
        try:
            rows = self.db.query(
                "SELECT topic, category, digest, sources, created_at FROM arena_knowledge "
                "ORDER BY created_at DESC LIMIT ?",
                (max(1, min(int(limit), 10)),),
            )
        except Exception:  # noqa: BLE001
            return []
        return [dict(r) for r in rows]

    # ── loop scheduling ──────────────────────────────────────────────────────
    def interval_hours(self) -> float | None:
        """Live interval override (kv), falling back to config; None = unset."""
        if self.db is None:
            return None
        try:
            row = self.db.query_one(
                "SELECT value FROM kv_store WHERE key = 'arena.interval_hours'"
            )
            if row:
                return float(json.loads(row["value"]).get("hours", 0.0)) or None
        except Exception:  # noqa: BLE001
            pass
        return None

    def set_interval(self, hours: float) -> bool:
        hours = max(0.05, min(float(hours), 24.0))
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO kv_store (key, value, kind, updated_at)
                       VALUES (?, ?, 'json', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                          updated_at = excluded.updated_at""",
                    ("arena.interval_hours", json.dumps({"hours": hours}), time.time()),
                )
        except Exception:  # noqa: BLE001
            return False
        return True

    # ── custom topics ────────────────────────────────────────────────────────
    def custom_topics(self) -> list[str]:
        if self.db is None:
            return []
        try:
            row = self.db.query_one(
                "SELECT value FROM kv_store WHERE key = 'arena.custom_topics'"
            )
            if row:
                data = json.loads(row["value"]).get("topics", [])
                return [str(t) for t in data if str(t).strip()]
        except Exception:  # noqa: BLE001
            pass
        return []

    def add_topic(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return "give me a topic to remember"
        topics = self.custom_topics()
        if text.lower() in [t.lower() for t in topics]:
            return f"already in the topic bank: {text}"
        topics.append(text)
        try:
            with self.db.transaction():
                self.db.execute(
                    """INSERT INTO kv_store (key, value, kind, updated_at)
                       VALUES (?, ?, 'json', ?)
                       ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                                          updated_at = excluded.updated_at""",
                    ("arena.custom_topics", json.dumps({"topics": topics}), time.time()),
                )
        except Exception:  # noqa: BLE001
            return "could not store the topic (db busy?)"
        return f"topic bank now has {len(topics)} custom topic(s). the loop may pick it up."

    def _pick_custom_topic(self) -> tuple[str, str] | None:
        """A random custom topic not yet digested; None when the bank is dry."""
        topics = self.custom_topics()
        if not topics or self.db is None:
            return None
        try:
            used = {r["topic"] for r in self.db.query(
                "SELECT topic FROM arena_knowledge")}
        except Exception:  # noqa: BLE001
            used = set()
        free = [t for t in topics if t not in used]
        if not free:
            return None
        return random.choice(free), "custom"

    # ── the cycle ────────────────────────────────────────────────────────────
    def _interest_profile(self) -> dict[str, float]:
        """What the owner does most → category weights for the sampler."""
        queries: list[str] = []
        goals: list[str] = []
        try:
            engine = SearchEngine(self.context)
            queries = [str(r.get("query", "")) for r in engine.history(60)]
        except Exception:  # noqa: BLE001
            pass
        if self.db is not None:
            try:
                rows = self.db.query("SELECT title FROM agent_goals LIMIT 40")
                goals = [str(r.get("title", "")) for r in rows]
            except Exception:  # noqa: BLE001
                pass
        try:
            return activity.interest_profile(
                self.db, search_queries=queries, goal_texts=goals)
        except Exception:  # noqa: BLE001
            from .topics import CATEGORIES

            return {c: 1.0 for c in CATEGORIES}

    def interest_profile(self) -> dict[str, float]:
        """Public: the current personalization weights (for /arena topics)."""
        return self._interest_profile()

    def _anti_repeat(self) -> int | None:
        """Topic anti-repeat window: settings → None (topics.py resolves
        kv override → default 10)."""
        arena_cfg = getattr(self.settings, "arena", None)
        raw = getattr(arena_cfg, "anti_repeat_window", None)
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            return None

    def run_cycle(self, topic: str | None = None, category: str | None = None,
                  notify: Callable[[str], None] | None = None,
                  surprise: bool = False,
                  seed: int | None = None) -> dict[str, Any]:
        started = time.time()
        anti_repeat = self._anti_repeat()
        kind = "code"
        verify = ""
        if topic:
            cat = category or "general"
        else:
            # Owner-supplied topics get priority over the bank (unless the
            # owner asked for a surprise).
            custom = None if surprise else self._pick_custom_topic()
            if custom is not None:
                topic, cat = custom
            elif surprise:
                cat, topic = surprise_topic(self.db, seed=seed,
                                            anti_repeat=anti_repeat)
            else:
                # Adaptive sampler: interest profile x coverage boost x
                # difficulty target, composed over the classic sampler
                # (anti-repeat window and knowledge skipping intact).
                from . import sampling as _sampling
                cat, topic, entry = _sampling.sample_challenge(
                    self.db, category=category,
                    profile=self._interest_profile(),
                    anti_repeat=anti_repeat)
                kind = str(entry.get("kind") or "code")
                verify = str(entry.get("verify") or "")
        self._stream("topic", {"topic": topic, "category": cat,
                                 "kind": kind, "verify": verify[:200]})
        try:
            report = self._research(topic)
        except Exception as exc:  # noqa: BLE001
            self._stream("research", {"topic": topic, "error": str(exc)})
            return {"ok": False, "topic": topic, "category": cat, "error": f"research failed: {exc}"}
        digest = str(report.get("summary") or "").strip()
        sources = report.get("results") or []
        # SearchEngine reports pages_read as a LIST of urls (an int in the
        # test fixture) — count it, don't coerce it.
        pages_read = report.get("pages_read")
        pages_n = pages_read if isinstance(pages_read, int) else len(pages_read or [])
        if not digest or pages_n == 0:
            # An honest "no readable text" summary is NOT knowledge — don't
            # pollute the training stream with empty digests.
            self._stream("research", {"topic": topic, "error": "no readable pages"})
            return {"ok": False, "topic": topic, "category": cat,
                    "error": "research returned nothing readable (no network? try again later)"}
        self._save_knowledge(topic, cat, digest, sources)
        activity.record(self.db, "arena_run", f"{cat}: {topic}")
        self._stream("digest", {"topic": topic, "category": cat, "digest": digest[:2000],
                                 "sources": sources[:8]})
        result: dict[str, Any] = {
            "ok": True, "topic": topic, "category": cat,
            "kind": kind, "verify": verify,
            "digest": digest, "pages_read": report.get("pages_read", 0),
            "seconds": round(time.time() - started, 1),
        }
        # Builder sub-agent: explicit config + power mode. Research is always
        # allowed; writing code is the power-mode part.
        build = None
        if self.settings is not None and getattr(self.settings.arena, "build", False):
            from ..power import power_mode_for

            if power_mode_for(self.context).active:
                build = self._build(topic, digest)
        if build:
            result["build"] = build
            packet = self._review_packet(build)
            self._stream("review", {"id": build["id"], "name": build["name"], "sent": notify is not None})
            if notify is not None:
                try:
                    notify(packet)
                except Exception:  # noqa: BLE001
                    pass
            elif not notify:
                result["review"] = packet
        # Score the run on verifiable axes — the adaptive sampler reads
        # these back to aim difficulty and balance coverage. Scoring is
        # observability, never a gate: it must not break a cycle.
        try:
            from . import scoring as _scoring
            _scoring.record_score(
                self.db, topic=topic, category=cat, kind=kind,
                research_usefulness=min(1.0, 0.3 + 0.7 * min(pages_n, 5) / 5),
                build_compiled=(1 if (build and build.get("syntax") == "ok")
                                else 0 if build else None),
                latency_s=float(result.get("seconds", 0) or 0),
                notes=verify[:200])
        except Exception:  # noqa: BLE001 - scoring must never break a cycle
            pass
        return result

    @staticmethod
    def _review_packet(build: dict[str, Any]) -> str:
        files = ", ".join(f["path"] for f in build.get("files", [])) or "?"
        return (
            "🧪 arena build awaiting your review\n"
            f"id: {build['id']}\n"
            f"name: {build['name']}\n"
            f"purpose: {build.get('purpose', '')}\n"
            f"topic: {build.get('topic', '')}\n"
            f"files: {files}\n"
            f"syntax: {build.get('syntax', 'unchecked')}\n"
            f"at: {build.get('dir', '')}\n"
            f"→ /arena approve {build['id']}  |  /arena deny {build['id']}"
        )

    # ── background loop (power + flag gated) ─────────────────────────────────
    def start_loop(self, notify: Callable[[str], None] | None = None) -> bool:
        if self._loop_thread is not None and self._loop_thread.is_alive():
            return False
        self._stop.clear()

        def _loop() -> None:
            from ..features import feature_enabled
            from ..power import power_mode_for

            while not self._stop.is_set():
                try:
                    if not feature_enabled(self.context, "arena"):
                        self._stop.wait(300.0)
                        continue
                    if not power_mode_for(self.context).active:
                        self._stop.wait(300.0)
                        continue
                    self.run_cycle(notify=notify)
                except Exception as exc:  # noqa: BLE001 - the loop must never die
                    self._stream("error", {"error": f"cycle crashed: {exc}"})
                hours = 6.0
                if self.settings is not None:
                    hours = max(0.05, float(getattr(self.settings.arena, "interval_hours", 6.0)))
                live = self.interval_hours()
                if live:
                    hours = max(0.05, live)
                self._stop.wait(hours * 3600.0)

        self._loop_thread = threading.Thread(target=_loop, name="arena-loop", daemon=True)
        self._loop_thread.start()
        return True

    def stop_loop(self) -> None:
        self._stop.set()

    def loop_running(self) -> bool:
        return self._loop_thread is not None and self._loop_thread.is_alive()


def run_cycle_safe(arena: Arena, topic: str | None = None,
                   surprise: bool = False,
                   seed: int | None = None) -> dict[str, Any]:
    """Wrapper used by CLI/tests: never raises, always returns a report."""
    try:
        return arena.run_cycle(topic=topic, surprise=surprise, seed=seed)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
