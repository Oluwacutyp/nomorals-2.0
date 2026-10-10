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
from ..core.jsonutil import extract_json as _extract_json
from ..core.logging_setup import get_logger

__all__ = ["DirectivesAgent"]

_log = get_logger(__name__)

_URL = re.compile(r"https?://[^\s\"'<>]+")


def _as_brain(router: Any) -> Any:
    """Wrap ``router`` in a Brain unless it already is one.

    Tolerates Brain being unpatchable/unimportable in odd runtimes —
    isinstance against a non-type raises TypeError, which we treat as
    "not a Brain".
    """
    from ..llm.brain import Brain

    try:
        if isinstance(router, Brain):
            return router
    except TypeError:  # noqa: BLE001 - Brain mocked as non-type in tests
        pass
    return Brain(router=router)

#: Subsystems a directive can route to. The *brain* picks one per
#: directive (see _route) — there are deliberately no keyword/regex
#: intent shortcuts here: intent classification is the model's job.
_DIRECTIVE_SUBSYSTEMS = (
    ("download", "fetch a file from a URL to local storage"),
    ("research", "investigate a topic on the web and report back with sources"),
    ("code", "write and run a program or script that accomplishes the task"),
    ("answer", "handle directly with the model (questions, explanations, "
               "judgment calls, anything conversational)"),
)


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
        """Route the directive to a subsystem and run it.

        Routing is dynamic: the brain reads the directive and picks the
        subsystem (download / research / code / answer) plus its argument
        as JSON. There are no keyword or regex intent shortcuts — with no
        model available the directive cannot be understood, and that is
        reported honestly instead of guessed.
        """
        route = self._route(text)
        subsystem = route.get("subsystem", "answer")
        argument = (route.get("argument") or "").strip()
        if subsystem == "download":
            url = argument or self._first_url(text)
            if not url:
                raise RuntimeError(
                    "routed to download but no URL was found in the directive")
            return self._download(url)
        if subsystem == "research":
            return self._research(argument or text)
        if subsystem == "code":
            return self._code(argument or text)
        # "answer" and anything unrecognized: the model handles it directly.
        return self._model(text)

    def _route(self, text: str) -> dict[str, Any]:
        """Ask the brain which subsystem should handle this directive.

        Returns ``{"subsystem": ..., "argument": ...}``. When the router
        is unreachable or its answer is unusable, the safe fallback is
        ``answer`` — the model just handles the text — except when there
        is no model at all, which raises honestly: intent cannot be
        classified without a brain.
        """
        if self.router is None:
            raise RuntimeError(
                "no model available to understand this directive "
                "(dynamic routing needs the brain)")
        from ..llm.base import Message, SamplingParams
        from ..llm.brain import Brain

        catalog = "\n".join(f"- {name}: {desc}"
                            for name, desc in _DIRECTIVE_SUBSYSTEMS)
        brain = _as_brain(self.router)
        try:
            response = brain.chat(
                [Message(
                    role="system",
                    content=(
                        "You route a user directive to exactly one subsystem.\n"
                        f"{catalog}\n\n"
                        "Reply with ONLY a JSON object: "
                        '{"subsystem": "<name>", "argument": "<url for '
                        "download | search query for research | task "
                        'description for code | empty for answer>"}')),
                 Message(role="user", content=text)],
                SamplingParams(temperature=0.0, max_tokens=200),
                task_kind="chat",
            )
        except Exception as exc:  # noqa: BLE001 - routing must not crash run()
            _log.warning("directive routing model call failed: %s", exc)
            return {"subsystem": "answer", "argument": ""}
        data = _extract_json((getattr(response, "text", "") or ""))
        if isinstance(data, dict):
            name = str(data.get("subsystem") or "").strip().lower()
            if name in {n for n, _ in _DIRECTIVE_SUBSYSTEMS}:
                return {"subsystem": name,
                        "argument": str(data.get("argument") or "")}
            _log.debug("directive router named unknown subsystem %r", name)
        elif getattr(response, "error", None):
            _log.warning("directive routing failed: %s",
                         getattr(response, "error", ""))
        return {"subsystem": "answer", "argument": ""}

    @staticmethod
    def _first_url(text: str) -> str:
        m = _URL.search(text or "")
        return m.group(0) if m else ""

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

        brain = _as_brain(self.router)
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
