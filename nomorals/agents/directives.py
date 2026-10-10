"""Directives: direct instructions to the agent core.

The control commands steer the *social* companion. Directives are a different
plane: standing or one-off instructions addressed to the AI itself — "research
X", "download Y", "write a script that Z". Each directive is persisted
(``directives``), executed by a router that dispatches to the right subsystem
(search engine, downloader, coding bot, or the model directly), journaled
with its result, and reported back to the owner through the notifier.

    /task add research the state of passkeys in 2026
    /task run                 # execute the next pending one
    nm task add "download https://example.com/report.pdf"
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from ..core.ids import new_id

__all__ = ["DirectivesAgent"]

_URL = re.compile(r"https?://[^\s\"'<>]+")


class DirectivesAgent:
    def __init__(self, context: Any, notifier: Any = None) -> None:
        self.context = context
        self.db = getattr(context, "db", None)
        self.notifier = notifier
        self.router = getattr(context, "router", None)

    # ── persistence ──────────────────────────────────────────────────────────
    def add(self, text: str) -> dict[str, Any]:
        did = new_id()
        if self.db is not None:
            try:
                with self.db.transaction():
                    self.db.execute(
                        "INSERT INTO directives (id, text, status, created_at) VALUES (?, ?, 'pending', ?)",
                        (did, text.strip(), time.time()),
                    )
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"could not store directive: {exc}"}
        return {"ok": True, "id": did, "text": text.strip()}

    def _get(self, did: str) -> dict[str, Any] | None:
        if self.db is None:
            return None
        try:
            return self.db.query_one("SELECT * FROM directives WHERE id = ?", (did,))
        except Exception:  # noqa: BLE001
            return None

    def _set(self, did: str, status: str, result: str = "", error: str = "") -> None:
        if self.db is None:
            return
        try:
            with self.db.transaction():
                self.db.execute(
                    "UPDATE directives SET status = ?, result = ?, error = ?, finished_at = ? WHERE id = ?",
                    (status, result[:8000], error[:2000], time.time(), did),
                )
        except Exception:  # noqa: BLE001
            pass

    def list(self, limit: int = 15, status: str = "") -> list[dict[str, Any]]:
        if self.db is None:
            return []
        try:
            if status:
                return self.db.query(
                    "SELECT * FROM directives WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                )
            return self.db.query(
                "SELECT * FROM directives ORDER BY created_at DESC LIMIT ?", (limit,)
            )
        except Exception:  # noqa: BLE001
            return []

    # ── execution ────────────────────────────────────────────────────────────
    def run(self, did: str = "") -> dict[str, Any]:
        row = None
        if did:
            row = self._get(did)
            if row is None:
                return {"ok": False, "error": f"no directive {did!r}"}
        elif self.db is not None:
            try:
                row = self.db.query_one(
                    "SELECT * FROM directives WHERE status = 'pending' ORDER BY created_at ASC LIMIT 1"
                )
            except Exception:  # noqa: BLE001
                row = None
        if row is None:
            return {"ok": False, "error": "no pending directives (add one: /task add <instruction>)"}
        self._set(row["id"], "running")
        try:
            result = self._execute(row["text"])
            self._set(row["id"], "done", result=result)
            self._notify(f"task done: {row['text'][:70]}", result[:1500])
            return {"ok": True, "id": row["id"], "result": result}
        except Exception as exc:  # noqa: BLE001
            self._set(row["id"], "failed", error=str(exc))
            self._notify(f"task failed: {row['text'][:70]}", str(exc)[:500])
            return {"ok": False, "id": row["id"], "error": str(exc)}

    def _execute(self, text: str) -> str:
        lowered = text.lower()
        # 1) download
        m = _URL.search(text)
        if m and any(w in lowered for w in ("download", "get ", "fetch", "grab", "save")):
            return self._download(m.group(0))
        if m:
            url = m.group(0)
            rest = (text[:m.start()] + text[m.end():]).strip(" .,!?")
            if any(w in lowered for w in ("look", "check", "summar", "read", "research")):
                return self._research(f"what is this about: {url}") + f"\n(source: {url})"
            return self._download(url)
        # 2) research
        if any(w in lowered for w in ("research", "find out", "look into", "what is",
                                      "what's", "explain", "why ", "how does", "summarize")):
            return self._research(text)
        # 3) code
        if any(w in lowered for w in ("write a", "write an", "code", "script",
                                      "program", "function that", "python file")):
            return self._code(text)
        # 4) model handles it directly
        return self._model(text)

    def _download(self, url: str) -> str:
        outcome = self.context.tools.call("web_download", url=url)
        if not outcome.ok:
            raise RuntimeError(f"download failed: {outcome.error.message if outcome.error else 'unknown'}")
        data = outcome.value
        return f"downloaded {data.get('url')}: {data.get('path')} ({data.get('bytes')} bytes)"

    def _research(self, query: str) -> str:
        from .search.engine import SearchEngine

        report = SearchEngine(self.context).run(query, mode="quick", pages=3)
        lines = [f"research: {query}", ""]
        lines.append(str(report.get("summary") or "(nothing readable)"))
        sources = report.get("results") or []
        if sources:
            lines.append("")
            lines.append("sources:")
            for s in sources[:5]:
                lines.append(f"  • {s.get('title') or s.get('url')}\n    {s.get('url')}")
        return "\n".join(lines)

    def _code(self, task: str) -> str:
        from .coding import CodingAgent

        agent = CodingAgent(self.context)
        result = agent.run(task, max_iterations=3, timeout=90.0)
        if result.ok:
            files = ", ".join(getattr(result, "files", []) or []) or "output"
            return (
                f"code done in {result.iterations} iteration(s): {files}\n"
                f"{str(getattr(result, 'output', ''))[:400]}"
            )
        return f"coding bot gave up after {result.iterations} iteration(s): {getattr(result, 'error', '')}"

    def _model(self, instruction: str) -> str:
        if self.router is None:
            raise RuntimeError("no model available to execute this directive")
        from ..llm.base import Message
        from ..llm.brain import Brain

        brain = self.router if isinstance(self.router, Brain) else Brain(router=self.router)
        response = brain.chat(
            [Message(role="user", content=instruction)],
            task_kind="chat",
        )
        text = (getattr(response, "text", "") or "").strip()
        if getattr(response, "error", None) and not text:
            raise RuntimeError(f"model failed: {response.error}")
        return text or "(model returned no output)"

    def _notify(self, title: str, body: str) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.publish("task", title, body)
        except Exception:  # noqa: BLE001
            pass

    # ── formatting for chat ──────────────────────────────────────────────────
    def format_list(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "no directives yet (add one: /task add <instruction>)"
        lines = [f"directives ({len(rows)}):"]
        for row in rows:
            extra = ""
            if row.get("status") == "done":
                extra = f" → {str(row.get('result', ''))[:50]}"
            elif row.get("status") == "failed":
                extra = f" ✗ {str(row.get('error', ''))[:50]}"
            lines.append(f"  {row['id'][:8]} [{row.get('status')}] {str(row.get('text', ''))[:56]}{extra}")
        return "\n".join(lines)
