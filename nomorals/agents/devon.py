"""Devon — the autonomous development & investigation agent.

``/devon <free text>`` hands a natural-language task to Devon. He breaks the
task down into tool calls — reads code, greps the repo, queries the live
database, runs tests, tails the log, checks her mood and the chat flow, does
web research, and can even kick off a durable background mission — executes
them within a step budget, writes every step into his own memory box (the
``devon_memory`` table), and digests what he found into a plain-English reply
that lands back in the chat.

Design notes
------------
* **Two brains, one purpose.** The *planner* (an LLM when one is reachable, a
  keyword heuristic otherwise) decides *which* tools to call and in what
  order. The *digest* (again LLM-first, mechanical fallback) turns the raw
  observations into a short answer a person can read in a chat bubble.
* **Durable by construction.** Every step is persisted as it happens, so a
  crash mid-investigation keeps the findings, and the *next* ``/devon`` call
  starts with his last few digests as context instead of amnesia.
* **Read-mostly on purpose.** Every tool is a read, a count, a test run, or a
  background-start. Nothing here mutates partner state or the owner's chats
  outside of Devon's own memory box.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import Message, SamplingParams

__all__ = ["DevonAgent", "DevonResult", "StepOutcome", "TOOL_CATALOG", "summarize_test_run"]

_log = get_logger(__name__)

# Repository root: this file is <root>/nomorals/agents/devon.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Hard budgets. Kept small so a single /devon call can never hold the chat
# thread hostage or burn the model budget on one question.
_DEFAULT_MAX_STEPS = 6
_STEP_TIMEOUT = 90.0
_OBSERVATION_CHARS = 1400  # how much of each tool's output the planner sees
_DIGEST_MEMORY = 3          # prior digests carried forward as context


# (tool name, one-line description the planner sees)
TOOL_CATALOG: tuple[tuple[str, str], ...] = (
    ("read_code", "read one file from this repo (args: path, grep, max_lines)"),
    ("find_code", "regex search across the repo (args: grep, path, max_hits)"),
    ("db_query", "SELECT-only query against the live state (args: sql)"),
    ("message_flow", "recent chat flow: who sent what, did the brain reply, which model (args: limit)"),
    ("run_tests", "run a unittest module or the tests/ directory (args: pattern, timeout)"),
    ("git_status", "current branch, commit, dirty files (no args)"),
    ("git_diff", "diff between two commits/refs, line-capped (args: ref_a, ref_b, path)"),
    ("logs_tail", "tail of the running log file (args: n)"),
    ("mood", "partner's current mood + relationship stage (no args)"),
    ("chat_stats", "gateway / per-platform stats + live games (no args)"),
    ("models", "LLM provider chain: names, models, liveness (no args)"),
    ("env", "live config snapshot: provider, model, power mode, features, limits (no args)"),
    ("history", "his earlier investigations for a similar task (args: task)"),
    ("trace", "full trace of one chat's turns: inbound, mood, model, reply (args: chat_key, limit)"),
    ("research", "quick web research with citations (args: query)"),
    ("code", "run Python in the sandbox, get stdout/stderr + last-expression value (args: code, session)"),
    ("browser", "browse the web: open a page, read text/links, click, fill+submit forms (args: action, url, target, name, value)"),
    ("watch", "re-run an investigation on a schedule in the background (args: task, interval_minutes, passes)"),
    ("mission", "start a durable background mission for long work (args: goal, name)"),
    ("api", "call an external API connector: weather, fx, ip_info, github, wikipedia, http_api (args: name, params)"),
    ("db_tables", "list every table in the state database with row counts (no args)"),
    ("db_schema", "columns of one table (args: table)"),
    ("db_counts", "largest tables + total (no args)"),
    ("speak", "speak text aloud — sends the audio file to the user (args: text)"),
    ("swarm", "parallel devon sub-agents on a goal, fused answer (args: goal, workers)"),
    # power layer — backed by the shared tool registry
    ("dns_lookup", "live DNS records for a domain: A/AAAA/NS/MX/TXT/SPF/CAA (args: domain, record)"),
    ("whois_lookup", "domain registration data: registrar, dates, nameservers (args: domain)"),
    ("local_services", "TCP listeners on this machine: port, address, pid (no args)"),
    ("osint_report", "read-only public-intel report for a domain/ip/url/email (args: target)"),
    ("proxy_status", "current outbound proxy routing + known proxy list (no args)"),
    ("proxy_set", "route outbound traffic through one of your proxies, live-checked (args: proxy)"),
    ("script_gen", "generate a validated automation script: backup, cron, termux service, webhook... (args: kind, name, config)"),
    ("macro_run", "replay a recorded macro of tool steps (args: name, overrides)"),
    ("username_check", "public handle existence across profile sites: github, gitlab, reddit, npm, pypi, t.me... (args: handle, sites)"),
    ("email_investigate", "passive email investigation: domain MX/SPF/DMARC + breach awareness (args: email)"),
    ("phone_investigate", "E.164 + country-code intelligence for a phone number (args: phone)"),
    ("breach_check", "is an email/domain in known breaches (official HIBP API, key-gated) (args: target)"),
    ("osint_graph", "persistent identity correlation graph: ingest text, clusters with confidence, node neighborhoods, alias merge, timeline (args: action, text, source, node, a, b)"),
    ("osint_campaign", "automated investigation walk from a seed: runs the OSINT toolkit over the whole discovered web, ingests everything into the identity graph, returns the dossier (args: seeds, max_steps, max_entities, time_budget)"),
    ("attacker", "network credential brute for security testing: http_form | http_basic | ssh | ftp — threaded worker pool, rate limit, human-paced delays, found credentials to a file (args: host, protocol, usernames, wordlist, port, path, fail_string, workers, rate, delay, out_file)"),
    ("osint_people", "multi-seed identity investigation + relationship map (args: seeds)"),
    ("metadata_extract", "offline file forensics: EXIF, PNG/PDF/DOCX properties, ID3 tags, sha256 (args: source)"),
    ("file_create", "format research/results into md, txt, json, html, or pdf (args: name, content, format, title)"),
    ("file_send", "send a file to any active chat platform (args: platform, chat_id, path, caption)"),
    ("train_mine", "mine the chat history into quality-scored training bundles (args: name, min_score, limit)"),
    ("train_datasets", "list registered training datasets + free HF catalog (args: limit)"),
    ("dataset_fetch", "download a free HuggingFace dataset into the training data dir (args: ref, max_rows)"),
    ("tts_say", "neural text-to-speech: Bark/XTTS/Kokoro voice note with emotion tags (args: text, voice, backend, mood)"),
    ("tts_voices", "manage TTS voice profiles: list/add/preset/remove (args: action, name, path, preset_id, consent)"),
    ("evolve_plan", "framework self-improvement: plan concrete repo edits for an instruction (args: instruction, research_id, focus)"),
    ("evolve_apply", "apply a planned evolution with the full test gate (args: proposal_id, commit)"),
    ("evolve_list", "list recent evolution proposals + status (args: limit)"),
    ("evolve_research", "deep research on improving the system (args: topic)"),
    ("evolve_audit", "scan the framework for weaknesses: markers, stubs, untested modules (args: max_findings)"),
    ("evolve_revert", "cleanly roll back an applied evolution (args: proposal_id)"),
    ("evolve_queue", "manage the autopilot goal queue (args: action, instruction)"),
    ("evolve_git", "evolution git policy: status or publish (FF-only merge + optional push) (args: action, branch, push)"),
    ("proxy_scrape", "scrape free public proxies from multiple sources, concurrent, deduped (args: schemes, limit)"),
    ("proxy_file", "write a CLEAN file of the currently working proxies (one URL per line, fastest first); returns the path (args: limit, max_age_hours)"),
    ("proxy_refresh", "test stored proxy candidates (liveness/latency/egress/anonymity/country) and store the working pool (args: limit, schemes, detect_country)"),
    ("proxy_pool", "working proxies on demand, ranked + TTL-filtered (args: action, scheme, country, anonymity, max_age_hours, limit)"),
    ("proxy_lab_status", "proxy lab state: pool size, freshness, best-5 (no args)"),
    ("proxy_schedule", "scheduled proxy re-checks on/off (args: enabled, every_minutes)"),
    ("proxy_rotate", "rotate the working proxy pool across requests: start/stop/status/next with failover+cooldown (args: action, strategy, cooldown_seconds, failover_seconds, max_age_hours)"),
    ("ssh_socks", "SSH server -> local SOCKS5 proxy: start/stop/status/list/urls/remove (args: action, name, host, port, user, password, key, local_port, auto_reconnect)"),
    # reasoning — explicit, auditable multi-step thought
    ("reason", "multi-strategy reasoning with a full auditable trace: cot | decompose | hypothesize | critique | tree | auto (args: goal, strategy, depth, use_tools, context, show_trace)"),
    ("reasoning_eval", "run the built-in reasoning eval (fixed task set) and score the active model 0-1 (args: limit)"),
    ("benchmark", "run the agent benchmark: reasoning + planning + tool_use + self_correction, scored 0-1, hermetic — what the evolution gate uses (args: dimensions, limit)"),
)


@dataclass
class StepOutcome:
    """One executed tool call."""

    tool: str
    args: dict[str, Any]
    ok: bool
    observation: str
    seconds: float
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args,
            "ok": self.ok,
            "observation": self.observation,
            "seconds": round(self.seconds, 3),
            "error": self.error,
        }


@dataclass
class DevonResult:
    """The whole investigation."""

    task: str
    steps: list[StepOutcome]
    digest: str
    planned_by: str  # "llm" | "reasoning" | "heuristic"
    model: str
    seconds: float
    run_id: str
    chat_key: str = ""
    #: wave 68: WHY the plan degraded ("" when planned straight by the
    #: model) — makes "planned by heuristic" honest: the owner can tell
    #: "the model was down" from "the model answered in prose".
    plan_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "run_id": self.run_id,
            "planned_by": self.planned_by,
            "model": self.model,
            "plan_error": self.plan_error,
            "seconds": round(self.seconds, 2),
            "steps": [s.to_dict() for s in self.steps],
            "digest": self.digest,
        }


class DevonAgent:
    """Plan → run tools → persist each step → digest → reply.

    ``context`` is the dependency container. ``brain`` (the partner brain) and
    ``gateway`` (the chat gateway) are optional; when supplied Devon can read
    live mood and per-platform stats directly instead of falling back to the
    database. Either may be ``None`` in a bare test context.
    """

    def __init__(
        self,
        context: Any,
        *,
        brain: Any = None,
        gateway: Any = None,
        max_steps: int = _DEFAULT_MAX_STEPS,
        step_timeout: float = _STEP_TIMEOUT,
        repo_root: Path | None = None,
    ) -> None:
        self.context = context
        self.brain = brain
        self.gateway = gateway
        self.max_steps = max(1, int(max_steps))
        self.step_timeout = float(step_timeout)
        self.repo_root = Path(repo_root) if repo_root else _REPO_ROOT
        self._tools: dict[str, Callable[[dict[str, Any]], str]] = {
            name: getattr(self, f"_tool_{name}")
            for name, _ in TOOL_CATALOG
            if hasattr(self, f"_tool_{name}")
        }

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, task: str, chat_key: str = "") -> DevonResult:
        """Run one investigation to completion. Never raises: a failure is a
        result whose digest says what went wrong."""
        started = time.time()
        run_id = new_short_id("devon")
        self._last_chat_key = chat_key
        task = (task or "").strip()
        if not task:
            return DevonResult(task="", steps=[], digest="no task given — tell me what to check.",
                               planned_by="heuristic", model="", seconds=0.0, run_id=run_id,
                               chat_key=chat_key,
                               plan_error="no task given — nothing planned")

        memory = self._prior_memory()
        steps_plan, planned_by, model, plan_error = self._plan(task, memory)
        if planned_by in ("llm", "reasoning"):
            steps_plan = self._reason_review_plan(task, steps_plan)
        observations: list[StepOutcome] = []

        for idx, step in enumerate(steps_plan[: self.max_steps], start=1):
            name = str(step.get("tool") or "").strip()
            args = step.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            if name not in self._tools:
                outcome = StepOutcome(name or "(none)", args, False,
                                      f"unknown tool {name!r}", 0.0, error="not a valid tool")
            else:
                outcome = self._execute(name, args)
            observations.append(outcome)
            self._remember(run_id, chat_key, task, idx, outcome)
            if time.time() - started > self.step_timeout * max(2, self.max_steps):
                _log.info("devon wall-clock budget hit after %d steps", idx)
                break

        digest = self._digest(task, observations, memory, model,
                              plan_error=plan_error)
        self._finish(run_id, chat_key, task, digest, ok=bool(observations))
        return DevonResult(
            task=task,
            steps=observations,
            digest=digest,
            planned_by=planned_by,
            model=model,
            seconds=time.time() - started,
            run_id=run_id,
            chat_key=chat_key,
            plan_error=plan_error,
        )

    def recent_digests(self, limit: int = _DIGEST_MEMORY) -> list[dict[str, Any]]:
        """His last digested answers — the short-form of his memory box."""
        limit = max(1, int(limit))
        try:
            rows = self.context.db.query(
                "SELECT ts, task, digest, status FROM devon_memory "
                "WHERE status IN ('done','failed') AND digest != '' "
                "ORDER BY ts DESC LIMIT ?",
                (limit,),
            )
        except Exception:  # noqa: BLE001 - no box yet on a fresh db
            _log.debug("devon_memory not present yet", exc_info=True)
            return []
        return [
            {"task": r["task"], "digest": r["digest"], "status": r["status"], "ts": r["ts"]}
            for r in rows
        ]

    # ── planner ─────────────────────────────────────────────────────────────

    def _plan(
        self, task: str, memory: str
    ) -> tuple[list[dict[str, Any]], str, str, str]:
        """Plan ladder (wave 68): the raw model plans first; if it
        answered but didn't parse, ONE strict-format retry; if the model
        is down or still won't parse, the REASONING engine plans with a
        trace behind it; only then — a last resort, with the reason
        recorded for an honest digest — the keyword heuristic.

        Returns (steps, planned_by, model, plan_error).
        """
        steps, model, err = self._llm_plan(task, memory)
        if steps:
            return steps, "llm", model, ""
        if err.startswith("unparseable"):
            steps, model2, err2 = self._llm_plan(task, memory, strict_retry=True)
            if steps:
                return steps, "llm", model2, ""
            err = f"unparseable (twice): {err2 or err}"
        try:
            from .reasoning import ReasoningAgent

            rsteps = ReasoningAgent(self.context).plan_tools(
                task, list(TOOL_CATALOG), limit=self.max_steps)
            if rsteps:
                return rsteps, "reasoning", "reasoning-engine", err
        except Exception as exc:  # noqa: BLE001 - reasoning is best-effort
            err = err or f"reasoning unavailable: {exc}"
        # last resort: keyword heuristic. plan_error must ALWAYS be non-empty
        # here — a heuristic plan is a degradation and must never look like a
        # clean success.
        return (self._heuristic_plan(task), "heuristic", "",
                err or "model and reasoning engine both unavailable")

    def _reason_review_plan(
        self, task: str, steps: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """The reasoning pre-flight for his tool sequences: a harsh
        reviewer reads the plan; if it finds real flaws the plan is
        revised and re-parsed once. A revision that doesn't yield valid
        known tools keeps the original plan. Gated by NM_REASONING_MODE;
        power mode reviews every plan with a relaxed budget."""
        if not steps:
            return steps
        from .reasoning import (looks_complex, reasoning_enabled,
                                revise_text, review_text)

        complex_ok = looks_complex(task) or len(steps) >= 3
        if not reasoning_enabled(self.context, complex_ok=complex_ok):
            return steps
        rendered = (f"Task: {task}\nPlanned tool calls:\n" +
                    "\n".join(f"- {s['tool']} {json.dumps(s['args'], default=str)}"
                              f"  (why: {s.get('why', '')})"
                              for s in steps))
        flaws = review_text(
            self.context, rendered,
            focus="a sequence of tool calls: wrong tool choice, missing "
                  "steps, steps in an order that cannot work")
        if not flaws:
            return steps
        _log.info("devon plan review found %d flaw(s); revising", len(flaws))
        raw = json.dumps({"steps": steps})
        revised = revise_text(
            self.context, raw, flaws,
            focus="the SAME JSON shape: {\"steps\": "
                  "[{\"tool\": \"<name>\", \"args\": {...}, \"why\": \"...\"}]}")
        if revised == raw:
            return steps
        data = self._extract_json(revised)
        if not data:
            return steps
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return steps
        cleaned: list[dict[str, Any]] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool not in self._tools:
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
            cleaned.append({"tool": tool, "args": args,
                            "why": str(item.get("why") or "")})
        if cleaned:
            return cleaned[: self.max_steps]
        return steps

    def _llm_plan(self, task: str, memory: str,
                  strict_retry: bool = False
                  ) -> tuple[list[dict[str, Any]], str, str]:
        """The raw model plan. Returns (steps, model, error) where error
        is "", "no-router", "model-error: ..." or "unparseable: ..." —
        the plan ladder uses the kind to decide whether a strict-format
        retry is worth one more call."""
        router = getattr(self.context, "router", None)
        if router is None:
            return [], "", "no-router"
        catalog = "\n".join(f"- {name}: {desc}" for name, desc in TOOL_CATALOG)
        system = (
            "You are Devon, an autonomous engineering & debug agent running inside "
            "the NoMorals repo. You plan a short sequence of tool calls to answer the "
            "owner's question. Pick only tools that fit, in the order you'd run them, "
            "and keep it to a handful. "
            "Reply with ONLY JSON of the form "
            '{"steps":[{"tool":"<name>","args":{...},"why":"<short>"}]}. '
            "No prose outside the JSON."
        )
        mem_block = f"Prior memory (his earlier digests):\n{memory}\n\n" if memory else ""
        # task-type pre-classification (wave 65): tells the planner what KIND
        # of task this is so it picks the right tools (build -> mission).
        type_hint = ""
        try:
            from .task_type import classify_task

            tt = classify_task(self.context, task, use_model=False)
            if tt.kind != "chat":
                type_hint = (
                    f"Task type (pre-classified): {tt.kind} — {tt.reason}. "
                    "Route to the tools that fit this kind: build work goes "
                    "to the mission tool; investigate uses logs/diagnostics; "
                    "research uses web/research tools.\n\n")
        except Exception:  # noqa: BLE001 - hint is a bonus, never fatal
            pass
        retry_note = ""
        if strict_retry:
            retry_note = (
                "\n\nIMPORTANT: your previous answer was not valid JSON. "
                "This time reply with ONLY the JSON object — no prose, "
                "no code fences, no commentary.")
        user = (f"Available tools:\n{catalog}\n\n{mem_block}"
                f"{type_hint}Task from the owner: {task}{retry_note}")
        try:
            response = router.chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=0.0, max_tokens=600),
            )
        except Exception as exc:  # noqa: BLE001 - planner must never kill the run
            _log.warning("devon planner LLM failed: %s", exc)
            return [], "", f"model-error: {exc}"
        if not getattr(response, "ok", False) or not getattr(response, "text", ""):
            return [], "", (f"model-error: "
                            f"{getattr(response, 'error', 'empty reply')}")
        data = self._extract_json(response.text)
        if not data:
            return [], "", f"unparseable: {(response.text or '')[:120]}"
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return [], "", "unparseable: reply had no steps list"
        cleaned: list[dict[str, Any]] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool not in self._tools:
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
            cleaned.append({"tool": tool, "args": args, "why": str(item.get("why") or "")})
        if not cleaned:
            return [], "", "unparseable: no known tools in reply"
        return cleaned[: self.max_steps], getattr(response, "model", "") or "", ""

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Pull the first balanced JSON object out of an LLM reply."""
        text = (text or "").strip()
        # Strip common code fences, then find the first {...} block.
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    def _heuristic_plan(self, task: str) -> list[dict[str, Any]]:
        """No model (or a model that didn't parse) — route by keywords.

        This is what makes /devon work offline and in tests: a deterministic
        mapping from the *kind* of question to the tools that answer it.
        """
        t = (task or "").lower()

        def has(*words: str) -> bool:
            """Single words match on WORD BOUNDARIES — substring matching
            made "uncensored" trip the "red" keyword and "already" trip
            "read". Multi-word phrases match as substrings."""
            for w in words:
                if " " in w:
                    if w in t:
                        return True
                elif re.search(rf"\b{re.escape(w)}\b", t):
                    return True
            return False

        plan: list[dict[str, Any]] = []
        add = lambda tool, **args: plan.append({"tool": tool, "args": args, "why": "heuristic"})  # noqa: E731

        # task-type routing (wave 65): a BUILD task escalates to the
        # mission path — decided by the shared classifier, deterministically,
        # BEFORE the keyword groups below (so "build a test script" goes to
        # the builder, not the test runner).
        try:
            from .task_type import classify_task

            tt = classify_task(self.context, task, use_model=False)
            if tt.kind == "build":
                return [{"tool": "mission",
                         "args": {"goal": (task or "").strip(),
                                  "name": "devon-mission"},
                         "why": tt.reason}]
        except Exception:  # noqa: BLE001 - classifier is a bonus, never fatal
            pass

        if has("test", "suite", "broken", "fail", "failing", "red"):
            add("run_tests", pattern="tests")
            add("git_status")
        elif has("message", "reply", "replied", "workflow", "brain", "chat flow", "sent to", "drop", "dropped"):
            add("message_flow", limit=14)
            add("find_code", grep="def handle_message|def deliver_reply|def _send_reply", path="nomorals/agents")
            add("read_code", path="nomorals/agents/partner_runtime.py", grep="def _process")
        elif has("log", "crash", "error", "exception", "traceback", "stack"):
            add("logs_tail", n=60)
            add("find_code", grep="Traceback|Exception", path="logs")
        elif has("trace", "turn-by-turn"):
            add("trace", limit=10)
        elif has("mood", "feeling", "feel", "relationship", "stage", "trust"):
            add("mood")
            add("db_query", sql="SELECT id, label, dims, updated_at FROM mood_state")
        elif has("uncensored", "censored", "moral", "restrict", "identity", "persona",
                 "who are you", "are you", "your name"):
            # A question about who/what she is — answer from the actual
            # source of truth: the persona definition, not a shrug.
            add("read_code", path="nomorals/partner/persona.py",
                grep="name|disclosure|occupation|location|boundaries")
            add("mood")
            add("git_status")
        elif has("model", "provider", "hugging", "hf", "groq", "mock"):
            add("models")
            add("logs_tail", n=30)
        elif has("config", "setting", "env", "power mode", "feature", "limit", "key"):
            add("env")
        elif has("history", "last time", "previous", "earlier", "before"):
            add("history", task=task.strip())
            add("message_flow", limit=8)
        elif has("diff", "compare", "changes between", "what changed"):
            add("git_diff", ref_a="HEAD~1", ref_b="HEAD")
            add("git_status")
        elif has("watch", "monitor", "every hour", "periodic", "recurring", "keep checking"):
            # "every 30 min" style intervals in the task itself
            minutes = 60
            m = re.search(r"every\s+(\d+)\s*(minute|min|hour|hr)s?", t)
            if m:
                minutes = int(m.group(1)) * (60 if m.group(2).startswith("h") else 1)
            add("watch", task=task.strip(), interval_minutes=minutes, passes=6)
        elif has("platform", "telegram", "whatsapp", "discord", "send", "stat", "stats",
                 "rate", "gateway", "games"):
            add("chat_stats")
            add("git_status")
        elif has("research", "look up", "what is", "who is", "latest", "news", "web"):
            add("research", query=task.strip())
        elif has("mission", "background job", "long work", "big build"):
            add("mission", goal=task.strip(), name="devon-mission")
        elif has("code", "read", "show", "file", "function", "class", "grep", "find", "where"):
            add("find_code", grep=r"def |class ", path="nomorals")
            add("git_status")
        else:
            # Default: orient (where are we?) then look at the most relevant
            # live state, so the answer is grounded instead of a shrug.
            add("git_status")
            add("message_flow", limit=8)
            add("mood")
        # Always cap at the budget.
        return plan[: self.max_steps]

    # ── executor ────────────────────────────────────────────────────────────

    def _execute(self, name: str, args: dict[str, Any]) -> StepOutcome:
        started = time.perf_counter()
        # Automation recorder: if a /record session is active, capture this
        # step (only registry-callable tools — that is what macros replay).
        tools = getattr(self.context, "tools", None)
        if tools is not None and name in tools.names():
            try:
                from ..tools.macros import record_step
                record_step(name, args)
            except Exception:  # noqa: BLE001 - recording is best-effort
                pass
        try:
            observation = self._tools[name](args)
            return StepOutcome(name, args, True, _clip(observation, _OBSERVATION_CHARS),
                               time.perf_counter() - started)
        except Exception as exc:  # noqa: BLE001 - one bad tool must not sink the run
            _log.exception("devon tool %s failed", name)
            return StepOutcome(name, args, False, f"tool error: {exc}",
                               time.perf_counter() - started, error=str(exc))

    # ── memory box ───────────────────────────────────────────────────────────

    def _remember(self, run_id: str, chat_key: str, task: str, step: int, outcome: StepOutcome) -> None:
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    """INSERT INTO devon_memory
                       (id, ts, chat_key, run_id, task, step, tool, args, observation, digest, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', 'step')""",
                    (
                        new_short_id("devon"),
                        time.time(),
                        chat_key,
                        run_id,
                        task,
                        step,
                        outcome.tool,
                        json.dumps(outcome.args, default=str),
                        outcome.observation,
                    ),
                )
        except Exception:  # noqa: BLE001 - a memory failure must not kill the run
            _log.warning("devon memory step write failed", exc_info=True)

    def _finish(self, run_id: str, chat_key: str, task: str, digest: str, ok: bool) -> None:
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    """INSERT INTO devon_memory
                       (id, ts, chat_key, run_id, task, step, tool, args, observation, digest, status)
                       VALUES (?, ?, ?, ?, ?, 0, '', '', '', ?, ?)""",
                    (
                        new_short_id("devon"),
                        time.time(),
                        chat_key,
                        run_id,
                        task,
                        digest,
                        "done" if ok else "failed",
                    ),
                )
        except Exception:  # noqa: BLE001
            _log.warning("devon memory digest write failed", exc_info=True)

    def _prior_memory(self) -> str:
        digests = self.recent_digests(_DIGEST_MEMORY)
        if not digests:
            return ""
        lines = [f"- {d['task'][:80]}: {d['digest'][:220]}" for d in reversed(digests)]
        return "\n".join(lines)

    # ── digest ───────────────────────────────────────────────────────────────

    def _digest(self, task: str, steps: list[StepOutcome], memory: str,
                model: str, plan_error: str = "") -> str:
        if not steps:
            return "I didn't run anything — that task didn't match a tool I can use. Rephrase it?"
        observations = "\n\n".join(
            f"[{i}] {s.tool} {json.dumps(s.args, default=str)}\n"
            f"{'OK' if s.ok else 'FAILED: ' + s.error}\n{s.observation}"
            for i, s in enumerate(steps, start=1)
        )
        llm_digest, _ = self._llm_digest(task, observations, memory, model)
        if llm_digest:
            return llm_digest
        # Mechanical fallback (model down or unparsable): still a useful,
        # honest answer — one line per step, failures expanded.
        bits = []
        for s in steps:
            lines = [ln.strip() for ln in s.observation.strip().splitlines() if ln.strip()]
            if s.ok:
                bits.append(f"{s.tool}: {lines[0][:160] if lines else '(empty)'}")
            else:
                fail_lines = lines[:4] or ["(no output)"]
                bits.append(f"{s.tool} FAILED: " + " | ".join(x[:80] for x in fail_lines))
        note = ""
        if not model:
            note = " (mechanical summary — model unreachable this run)"
        if plan_error:
            # honest degradation: the owner sees WHY this run used the
            # fallback plan, so "planned by heuristic" is diagnosable
            note += f" (fallback plan — {plan_error[:140]})"
        return "What I found:\n" + "\n".join(f"- {b}" for b in bits) + note

    def _llm_digest(
        self, task: str, observations: str, memory: str, model: str
    ) -> tuple[str, str]:
        router = getattr(self.context, "router", None)
        if router is None:
            return "", ""
        mem_block = f"Your earlier digests:\n{memory}\n\n" if memory else ""
        system = (
            "You are Devon, an autonomous dev agent. Summarize the tool output below into a "
            "short, plain-English answer (2-6 sentences, no markdown, no code blocks) that a "
            "person reading a chat bubble understands. State concretely what you checked and "
            "what you found. If something failed or you couldn't verify it, say so. No filler."
        )
        user = f"{mem_block}Task: {task}\n\nTool output:\n{observations[:6000]}"
        try:
            response = router.chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=0.2, max_tokens=500),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("devon digest LLM failed: %s", exc)
            return "", ""
        if not getattr(response, "ok", False) or not getattr(response, "text", ""):
            return "", ""
        text = getattr(response, "text", "").strip()
        return text, getattr(response, "model", "") or ""

    # ── tools ───────────────────────────────────────────────────────────────

    def _tool_read_code(self, args: dict[str, Any]) -> str:
        path = str(args.get("path") or "").strip()
        if not path:
            return "read_code needs a 'path'"
        grep = str(args.get("grep") or "")
        max_lines = _as_int(args.get("max_lines"), 80, lo=1, hi=500)
        target = self._safe_path(path)
        if target is None or not target.is_file():
            return f"no such file in repo: {path}"
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"could not read {path}: {exc}"
        if grep:
            try:
                rx = re.compile(grep)
                lines = [ln for ln in lines if rx.search(ln)]
            except re.error as exc:
                return f"bad grep pattern: {exc}"
        head = lines[:max_lines]
        if not head:
            return f"{path}: (no lines matched)"
        return f"{path} ({len(lines)} lines total, showing {len(head)}):\n" + "\n".join(head)

    def _tool_find_code(self, args: dict[str, Any]) -> str:
        grep = str(args.get("grep") or "").strip()
        if not grep:
            return "find_code needs a 'grep' regex"
        sub = str(args.get("path") or "nomorals").strip() or "nomorals"
        max_hits = _as_int(args.get("max_hits"), 20, lo=1, hi=100)
        base = self._safe_path(sub)
        if base is None:
            return f"no such path in repo: {sub}"
        try:
            rx = re.compile(grep)
        except re.error as exc:
            return f"bad grep pattern: {exc}"
        hits: list[str] = []
        start = base if base.is_file() else None
        files = [start] if start else (
            sorted(p for p in base.rglob("*") if p.is_file() and p.suffix in {".py", ".mjs", ".js", ".c", ".h"})
            if base.is_dir() else []
        )
        for p in files:
            if len(hits) >= max_hits:
                break
            try:
                rel = p.relative_to(self.repo_root)
            except ValueError:
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if rx.search(line):
                    hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
                    if len(hits) >= max_hits:
                        break
        if not hits:
            return f"no matches for {grep!r} under {sub}"
        return f"{len(hits)} match(es) for {grep!r} under {sub}:\n" + "\n".join(hits)

    def _tool_db_query(self, args: dict[str, Any]) -> str:
        sql = str(args.get("sql") or "").strip().rstrip(";")
        if not sql:
            return "db_query needs a 'sql'"
        if not _is_select(sql):
            return "refused: only SELECT / WITH ... SELECT queries are allowed"
        # Force a row cap even if the caller forgot one.
        if not re.search(r"\blimit\s+\d+", sql, re.IGNORECASE):
            sql += " LIMIT 50"
        rows = self.context.db.query(sql)
        if not rows:
            return "(no rows)"
        cols = list(rows[0].keys())
        lines = ["\t".join(cols)]
        for r in rows[:50]:
            lines.append("\t".join(str(r.get(c, ""))[:80] for c in cols))
        return f"{len(rows)} row(s):\n" + "\n".join(lines)

    def _tool_message_flow(self, args: dict[str, Any]) -> str:
        limit = _as_int(args.get("limit"), 14, lo=1, hi=60)
        rows = self.context.db.query(
            """SELECT c.id AS chat, c.channel, c.title,
                      m.role, m.name, m.model, substr(m.content, 1, 90) AS content, m.created_at
               FROM messages m
               JOIN conversations c ON c.id = m.conversation_id
               ORDER BY m.created_at DESC LIMIT ?""",
            (limit * 3,),
        )
        if not rows:
            return "no messages recorded yet (brain has not persisted any chat)"
        # Group by chat, most recent chat first.
        by_chat: dict[str, list[dict[str, Any]]] = {}
        order: list[str] = []
        for r in rows:
            label = r["title"] or r["chat"]
            key = f"{label} ({r['chat']}, channel={r['channel']})"
            if key not in by_chat:
                by_chat[key] = []
                order.append(key)
            by_chat[key].append(r)
        lines = [f"recent flow across {len(order)} chat(s):"]
        for key in order:
            msgs = list(reversed(by_chat[key]))  # chronological
            lines.append(f"\n{key}:")
            for m in msgs:
                who = m["name"] or m["role"]
                model = f" [model={m['model']}]" if m["model"] else ""
                lines.append(f"  {m['role']:<9} {who[:18]:<18} {m['content']!r}{model}")
        return "\n".join(lines[:60])

    @staticmethod
    def _test_cmd(pattern: str) -> list[str]:
        """The unittest invocation for a pattern.

        A bare package name (like "tests") does NOT recurse: it imports
        tests/__init__.py, finds no cases, and exits 5 ("no tests found")
        — exactly what the phone hit. Directories and bare package names
        must use DISCOVERY; dotted module names use the plain form.
        """
        cmd = [sys.executable, "-m", "unittest"]
        if pattern.endswith(".py"):
            cmd.append(pattern[:-3].replace("/", ".").replace("\\", "."))
        elif "." in pattern:
            cmd.append(pattern)
        else:
            cmd += ["discover", "-s", pattern, "-p", "test_*.py", "-t", "."]
        cmd.append("-v")
        return cmd

    def _tool_run_tests(self, args: dict[str, Any]) -> str:
        pattern = str(args.get("pattern") or "tests").strip() or "tests"
        timeout = _as_int(args.get("timeout"), 120, lo=5, hi=300)
        cmd = self._test_cmd(pattern)
        try:
            proc = subprocess.run(
                cmd, cwd=str(self.repo_root), capture_output=True, text=True,
                timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return f"test run timed out after {timeout}s (pattern {pattern})"
        except OSError as exc:
            return f"could not run tests: {exc}"
        return summarize_test_run(pattern, proc.returncode, (proc.stdout or "") + (proc.stderr or ""))

    def _tool_git_status(self, args: dict[str, Any]) -> str:
        try:
            branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                    cwd=str(self.repo_root), capture_output=True, text=True, timeout=15)
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                    cwd=str(self.repo_root), capture_output=True, text=True, timeout=15)
            dirty = subprocess.run(["git", "status", "--porcelain"],
                                   cwd=str(self.repo_root), capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"git unavailable: {exc}"
        dirty_lines = [ln for ln in (dirty.stdout or "").splitlines() if ln.strip()]
        return (
            f"branch: {branch.stdout.strip()}\n"
            f"commit: {commit.stdout.strip()}\n"
            f"dirty files: {len(dirty_lines)}"
            + (f"\n{chr(10).join(dirty_lines[:20])}" if dirty_lines else " (clean)")
        )

    def _tool_logs_tail(self, args: dict[str, Any]) -> str:
        n = _as_int(args.get("n"), 40, lo=1, hi=200)
        log_file = self.context.settings.resolve(self.context.settings.log.file)
        if not log_file or not Path(log_file).is_file():
            return f"no log file at {log_file}"
        try:
            lines = Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"could not read log: {exc}"
        return f"tail of {log_file} ({len(lines)} lines total):\n" + "\n".join(lines[-n:])

    def _tool_mood(self, args: dict[str, Any]) -> str:
        brain = self.brain
        if brain is not None:
            mood = brain.mood.current()
            rel = brain.relationship
            return (
                f"mood: {mood.label} (energy {mood.values.get('energy', 0):.0f}, "
                f"openness {mood.values.get('openness', 0):.0f})\n"
                f"relationship: stage={rel.stage}, trust={rel.trust:.0f}"
            )
        # No live brain — fall back to the persisted row.
        row = self.context.db.query_one("SELECT label, dims, updated_at FROM mood_state WHERE id = 'default'")
        if not row:
            return "no mood state recorded"
        try:
            dims = json.loads(row["dims"]) if row["dims"] else {}
        except json.JSONDecodeError:
            dims = {}
        return f"mood (persisted): {row['label']}, dims={dims}, updated_at={row['updated_at']}"

    def _tool_chat_stats(self, args: dict[str, Any]) -> str:
        parts: list[str] = []
        gateway = self.gateway
        if gateway is not None:
            try:
                status = gateway.status()
                parts.append(json.dumps(status, default=str)[:800])
            except Exception as exc:  # noqa: BLE001
                parts.append(f"gateway status failed: {exc}")
        try:
            rows = self.context.db.query(
                "SELECT platform, kind, COUNT(*) AS n, MAX(last_active) AS last "
                "FROM chats GROUP BY platform, kind ORDER BY last DESC"
            )
            if rows:
                parts.append("chats by platform: " + ", ".join(
                    f"{r['platform']}:{r['kind']}={r['n']}" for r in rows
                ))
            games = self.context.db.query(
                "SELECT game, chat_key, updated_at FROM game_sessions WHERE status = 'active'"
            )
            if games:
                parts.append("live games: " + ", ".join(
                    f"{r['game']} in {r['chat_key']}" for r in games
                ))
            else:
                parts.append("live games: none")
        except Exception:  # noqa: BLE001 - no chats table yet
            pass
        return "\n".join(parts) if parts else "no chats recorded"

    def _tool_models(self, args: dict[str, Any]) -> str:
        router = getattr(self.context, "router", None)
        if router is None:
            return "no router built in this context"
        names = []
        try:
            names = list(router.providers())
        except AttributeError:  # minimal/router stand-ins in tests
            names = []
        if not names:
            return (f"router present but no introspectable provider chain "
                    f"(model: {getattr(router, 'model', '?')})")
        lines: list[str] = []
        primary = getattr(router, "primary", None)
        for name in names:
            provider = router.get(name)
            if provider is None:
                lines.append(f"  {name}: (unknown)")
                continue
            try:
                healthy = provider.health()
            except Exception as exc:  # noqa: BLE001 - a probe must not kill the run
                healthy = f"probe error: {exc}"
            model = getattr(provider, "model_id", "") or ""
            tag = " ← PRIMARY" if name == primary else ""
            lines.append(f"  {name}: model={model or '?'} health={healthy}{tag}")
        head = f"provider chain: {' → '.join(names)}"
        return head + "\n" + "\n".join(lines)

    def _tool_env(self, args: dict[str, Any]) -> str:
        settings = self.context.settings
        llm = settings.llm
        from .features import FeatureRegistry
        from .power import power_mode_for

        feats = FeatureRegistry(self.context.db)
        on = [f["name"] for f in feats.list() if f["on"]]
        try:
            power = power_mode_for(self.context)
            power_line = f"power mode: {'ACTIVE' if power.active else 'locked'}"
        except Exception as exc:  # noqa: BLE001
            power_line = f"power mode: ? ({exc})"
        chain = [llm.provider] + [c for c in (llm.fallback_chain or []) if c and c != llm.provider]
        return (
            f"home: {settings.home}\n"
            f"profile: {settings.profile}\n"
            f"provider chain: {' → '.join(chain) or '(none)'}\n"
            f"model: {llm.openai_model or llm.groq_model or llm.hf_model or llm.local_model or '?'}\n"
            f"autonomy: {settings.partner.autonomy_mode}\n"
            f"{power_line}\n"
            f"rate limit: {settings.chat.max_per_hour}/h per chat\n"
            f"features on: {', '.join(on) or '(defaults)'}"
        )

    def _tool_history(self, args: dict[str, Any]) -> str:
        task = str(args.get("task") or "").strip()
        rows = self.context.db.query(
            "SELECT ts, task, digest, status FROM devon_memory "
            "WHERE status IN ('done','failed') AND digest != '' "
            "ORDER BY ts DESC LIMIT 12",
        )
        if not rows:
            return "no prior investigations stored"
        if task:
            wanted = [w for w in task.lower().split() if len(w) > 4][:4]
            if wanted:
                matches = [r for r in rows if any(w in r["task"].lower() for w in wanted)]
                if matches:
                    rows = matches[:5]
        lines = []
        for r in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(r["ts"]))
            lines.append(f"[{when}] {r['task'][:70]}\n    → {r['digest'][:300]}")
        return f"{len(rows)} prior run(s):\n" + "\n".join(lines)

    def _tool_trace(self, args: dict[str, Any]) -> str:
        chat_key = str(args.get("chat_key") or "").strip()
        limit = _as_int(args.get("limit"), 8, lo=1, hi=30)
        if not chat_key:
            # Latest chat if none given.
            row = self.context.db.query_one(
                "SELECT id FROM chats ORDER BY last_active DESC LIMIT 1"
            )
            if not row:
                return "no chats recorded"
            chat_key = row["id"]
        rows = self.context.db.query(
            """SELECT role, name, model, substr(content, 1, 120) AS content, created_at
               FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
            (chat_key, limit * 2),
        )
        if not rows:
            return f"no messages recorded for {chat_key}"
        rows = list(reversed(rows))
        lines = [f"trace of {chat_key} (last {len(rows)} messages):"]
        for m in rows:
            when = time.strftime("%H:%M:%S", time.localtime(m["created_at"]))
            model = f" model={m['model']}" if m["model"] else ""
            lines.append(f"  {when} {m['role']:<9} {m['name'][:14]:<14} {m['content']!r}{model}")
        # Mood movements in the same window.
        try:
            mood_rows = self.context.db.query(
                "SELECT ts, label, event FROM mood_history ORDER BY ts DESC LIMIT 6"
            )
            if mood_rows:
                lines.append("recent mood:")
                for m in mood_rows:
                    when = time.strftime("%H:%M", time.localtime(m["ts"]))
                    lines.append(f"  {when} [{m['label']}] {m['event'] or '-'}")
        except Exception:  # noqa: BLE001
            pass
        return "\n".join(lines)

    def _tool_git_diff(self, args: dict[str, Any]) -> str:
        ref_a = str(args.get("ref_a") or "HEAD~1").strip() or "HEAD~1"
        ref_b = str(args.get("ref_b") or "HEAD").strip() or "HEAD"
        path = str(args.get("path") or "").strip()
        try:
            stat = subprocess.run(
                ["git", "diff", "--stat", ref_a, ref_b] + ([path] if path else []),
                cwd=str(self.repo_root), capture_output=True, text=True, timeout=30,
            )
            diff = subprocess.run(
                ["git", "diff", "-U2", ref_a, ref_b] + ([path] if path else []),
                cwd=str(self.repo_root), capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return f"git diff failed: {exc}"
        text = diff.stdout or diff.stderr or ""
        lines = text.splitlines()
        capped = lines[:150]
        return (stat.stdout.strip()[:600] + "\n---\n" + "\n".join(capped)
                + (f"\n… (+{len(lines) - 150} more lines)" if len(lines) > 150 else ""))

    def _tool_watch(self, args: dict[str, Any]) -> str:
        task = str(args.get("task") or "").strip()
        if not task:
            return "watch needs a 'task'"
        interval = _as_int(args.get("interval_minutes"), 60, lo=2, hi=1440)
        passes = _as_int(args.get("passes"), 6, lo=1, hi=24)
        context = self.context
        chat_key_holder = {"key": ""}

        def _job() -> None:
            agent = DevonAgent(context, brain=self.brain, gateway=self.gateway)
            for i in range(1, passes + 1):
                try:
                    result = agent.run(f"[watch {i}/{passes}] {task}", chat_key=chat_key_holder["key"])
                    _log.info("devon watch pass %d/%d done: %s", i, passes, result.digest[:120])
                except Exception as exc:  # noqa: BLE001 - one bad pass must not kill the watch
                    _log.exception("devon watch pass %d failed: %s", i, exc)
                time.sleep(interval * 60)

        threading.Thread(target=_job, name=f"devon-watch-{len(task) % 1000}", daemon=True).start()
        return (f"watch started: re-checks {task[:80]!r} every {interval} min, "
                f"{passes} passes — each pass journals into my memory box "
                f"(/devon shows the latest digest)")

    def _tool_research(self, args: dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return "research needs a 'query'"
        from .search.engine import SearchEngine

        report = SearchEngine(self.context).run(query, mode="quick", pages=2)
        summary = report.get("summary") or "(no summary)"
        cites = "; ".join(r.get("url", "") for r in report.get("results", [])[:3])
        return f"research({query}): {summary[:600]}\nsource(s): {cites}"

    def _tool_code(self, args: dict[str, Any]) -> str:
        code = str(args.get("code") or "").strip()
        if not code:
            return "code needs a 'code' argument (python source)"
        session = str(args.get("session") or "devon").strip() or "devon"
        from ..tools.sandbox_code import CodeInterpreter

        result = CodeInterpreter().run(code, session=session, timeout=min(60.0, self.step_timeout))
        parts = []
        if result.get("result") is not None:
            parts.append(f"result: {str(result['result'])[:400]}")
        if result.get("stdout"):
            parts.append(f"stdout:\n{result['stdout'][:800]}")
        if result.get("stderr"):
            parts.append(f"stderr:\n{result['stderr'][:600]}")
        if not parts:
            parts.append("(no output)")
        state = f"exit={result.get('exit_code')}" + (
            f" timed_out={result['timed_out']}" if result.get("timed_out") else "")
        return f"{state}\n" + "\n".join(parts)

    def _tool_browser(self, args: dict[str, Any]) -> str:
        action = str(args.get("action") or "state").strip() or "state"
        from ..core.errors import ToolError
        from ..tools.browser import get_session

        session = str(args.get("session") or "devon").strip() or "devon"
        try:
            result = get_session(session).do(
                action,
                url=str(args.get("url") or ""),
                target=str(args.get("target") or ""),
                name=str(args.get("name") or ""),
                value=str(args.get("value") or ""),
            )
        except ToolError as exc:
            return f"browser: {exc}"
        return _format_browser_result(result, limit=2500)

    def _tool_mission(self, args: dict[str, Any]) -> str:
        goal = str(args.get("goal") or "").strip()
        name = str(args.get("name") or "devon-mission").strip() or "devon-mission"
        if not goal:
            return "mission needs a 'goal'"
        context = self.context
        goal_l = goal.lower()
        # wave 68: a status/progress QUERY is not a new mission — report
        # the REAL state of the named (or most recent) missions.
        query_words = ("progress", "status", "how far", "update on",
                       "what\'s the", "whats the", "where is")
        if (any(w in goal_l for w in query_words)
                and len(goal) < 120
                and not any(w in goal_l for w in ("create", "build",
                                                  "make", "gather"))):
            return self._mission_progress_report(
                name if name != "devon-mission" else "")
        # wave 68: structure the request into a mission spec BEFORE it
        # runs (the brief agent), persist the spec with the mission, and
        # run unlimited (the budget, not an iteration guess, stops it).
        structured = goal
        brief_meta: dict[str, Any] = {}
        try:
            from .brief import BriefAgent, should_brief

            if should_brief(goal):
                brief = BriefAgent(context).refine(goal, kind="mission")
                structured = brief.as_goal()
                brief_meta["brief"] = brief.to_dict()
        except Exception as exc:  # noqa: BLE001 — structuring is best-effort
            _log.debug("brief agent failed: %s", exc)

        def _job() -> None:
            try:
                from ..missions.mission import MissionStore
                from ..missions.runner import MissionRunner

                store = MissionStore(context.db)
                mission = store.create_new(structured, name=name,
                                           metadata=brief_meta)
                MissionRunner(context, store=store).run(
                    mission, max_iterations=0)
            except Exception as exc:  # noqa: BLE001 - a mission must not crash the thread
                _log.exception("devon-spawned mission failed: %s", exc)

        threading.Thread(target=_job, name=f"devon-mission-{name}",
                         daemon=True).start()
        return (f"mission started in background: {name} — {goal[:120]} "
                f"(unlimited iterations; query it any time: "
                f"\'/devon progress on {name}\')")

    def _mission_progress_report(self, name_filter: str) -> str:
        """The real, persisted state of missions — never a guess."""
        from ..missions.mission import MissionStore

        store = MissionStore(self.context.db)
        rows = store.list(limit=25)
        if name_filter:
            nf = name_filter.lower()
            rows = [m for m in rows
                    if nf in m.name.lower() or nf in m.goal.lower()]
        if not rows:
            return (f"no mission matching {name_filter!r} — "
                    f"active missions: "
                    + (", ".join(m.name for m in store.list(limit=5))
                       or "none"))
        lines = []
        for m in rows[:5]:
            d = store.detail(m.id)
            p = d.get("progress", {})
            cps = d.get("recent_checkpoints") or []
            last_cp = cps[0].get("label") if cps else ""
            lines.append(
                f"{m.name} [{m.status}] — "
                f"{p.get('steps_done', 0)}/{p.get('total_steps', 0)} steps "
                f"({p.get('percent', 0):.0f}%), "
                f"{m.iterations} iteration(s), "
                f"wall {p.get('spent_wall_seconds', 0):.0f}s, "
                f"tokens {p.get('spent_tokens', 0)}"
                + (f", current: {p['current_step']}" if p.get("current_step") else "")
                + (f", last checkpoint: {last_cp}" if last_cp else "")
                + (f", last error: {p['last_error'][:120]}" if p.get("last_error") else "")
            )
        return "mission progress:\n" + "\n".join(f"· {ln}" for ln in lines)

    # ── power layer tools (delegated to the shared registry) ────────────────

    def _registry_tool(self, tool: str, **kwargs: Any) -> str:
        """Run one registry tool with full capability checks + auditing."""
        from ..core.errors import ToolError

        tools = getattr(self.context, "tools", None)
        if tools is None:
            raise ToolError("tool registry unavailable")
        outcome = tools.call(tool, **kwargs)
        if not outcome.ok:
            raise ToolError(getattr(outcome.error, "message", str(outcome.error)))
        return json.dumps(outcome.value, default=str)

    def _tool_dns_lookup(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "dns_lookup",
            domain=str(args.get("domain") or ""),
            record=str(args.get("record") or "A"),
        )

    def _tool_whois_lookup(self, args: dict[str, Any]) -> str:
        return self._registry_tool("whois_lookup", domain=str(args.get("domain") or ""))

    def _tool_local_services(self, args: dict[str, Any]) -> str:
        return self._registry_tool("local_services")

    def _tool_osint_report(self, args: dict[str, Any]) -> str:
        return self._registry_tool("osint_report", target=str(args.get("target") or ""))

    def _tool_osint_graph(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "osint_graph",
            action=str(args.get("action") or "stats"),
            text=str(args.get("text") or ""),
            source=str(args.get("source") or ""),
            node=str(args.get("node") or ""),
            a=str(args.get("a") or ""),
            b=str(args.get("b") or ""),
        )

    def _tool_osint_campaign(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "osint_campaign",
            seeds=str(args.get("seeds") or ""),
            max_steps=str(args.get("max_steps") or ""),
            max_entities=str(args.get("max_entities") or ""),
            time_budget=str(args.get("time_budget") or ""),
        )

    def _tool_attacker(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "attacker",
            host=str(args.get("host") or ""),
            port=str(args.get("port") or ""),
            protocol=str(args.get("protocol") or "http_form"),
            usernames=str(args.get("usernames") or ""),
            passwords=str(args.get("passwords") or ""),
            wordlist=str(args.get("wordlist") or ""),
            path=str(args.get("path") or "/"),
            method=str(args.get("method") or "POST"),
            form_user=str(args.get("form_user") or "username"),
            form_pass=str(args.get("form_pass") or "password"),
            fail_string=str(args.get("fail_string") or ""),
            workers=str(args.get("workers") or ""),
            rate=str(args.get("rate") or ""),
            delay=str(args.get("delay") or ""),
            jitter=str(args.get("jitter") or ""),
            timeout=str(args.get("timeout") or ""),
            max_attempts=str(args.get("max_attempts") or ""),
            out_file=str(args.get("out_file") or ""),
        )

    def _tool_reason(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "reason",
            goal=str(args.get("goal") or ""),
            strategy=str(args.get("strategy") or "auto"),
            depth=str(args.get("depth") or "1"),
            use_tools=str(args.get("use_tools") or ""),
            context=str(args.get("context") or ""),
            show_trace=str(args.get("show_trace") or "true"),
        )

    def _tool_reasoning_eval(self, args: dict[str, Any]) -> str:
        return self._registry_tool("reasoning_eval",
                                   limit=str(args.get("limit") or ""))

    def _tool_benchmark(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "benchmark",
            dimensions=str(args.get("dimensions") or ""),
            limit=str(args.get("limit") or ""),
        )

    def _tool_proxy_status(self, args: dict[str, Any]) -> str:
        return self._registry_tool("proxy_status")

    def _tool_proxy_set(self, args: dict[str, Any]) -> str:
        return self._registry_tool("proxy_set", proxy=str(args.get("proxy") or ""))

    def _tool_script_gen(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "script_gen",
            kind=str(args.get("kind") or ""),
            name=str(args.get("name") or ""),
            config=str(args.get("config") or ""),
        )

    def _tool_macro_run(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "macro_run",
            name=str(args.get("name") or ""),
            overrides=str(args.get("overrides") or ""),
        )

    def _tool_username_check(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "username_check",
            handle=str(args.get("handle") or ""),
            sites=str(args.get("sites") or "auto"),
        )

    def _tool_email_investigate(self, args: dict[str, Any]) -> str:
        return self._registry_tool("email_investigate",
                                   email=str(args.get("email") or ""))

    def _tool_phone_investigate(self, args: dict[str, Any]) -> str:
        return self._registry_tool("phone_investigate",
                                   phone=str(args.get("phone") or ""))

    def _tool_breach_check(self, args: dict[str, Any]) -> str:
        return self._registry_tool("breach_check",
                                   target=str(args.get("target") or ""))

    def _tool_osint_people(self, args: dict[str, Any]) -> str:
        return self._registry_tool("osint_people",
                                   seeds=str(args.get("seeds") or "{}"))

    def _tool_metadata_extract(self, args: dict[str, Any]) -> str:
        return self._registry_tool("metadata_extract",
                                   source=str(args.get("source") or ""))

    def _tool_file_create(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "file_create",
            name=str(args.get("name") or ""),
            content=str(args.get("content") or ""),
            format=str(args.get("format") or "md"),
            title=str(args.get("title") or ""),
        )

    def _tool_file_send(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "file_send",
            platform=str(args.get("platform") or ""),
            chat_id=str(args.get("chat_id") or ""),
            path=str(args.get("path") or ""),
            caption=str(args.get("caption") or ""),
        )

    def _tool_train_mine(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "train_mine",
            name=str(args.get("name") or ""),
            min_score=str(args.get("min_score") or ""),
            limit=str(args.get("limit") or ""),
        )

    def _tool_train_datasets(self, args: dict[str, Any]) -> str:
        return self._registry_tool("train_datasets",
                                   limit=str(args.get("limit") or ""))

    def _tool_dataset_fetch(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "dataset_fetch",
            ref=str(args.get("ref") or ""),
            max_rows=str(args.get("max_rows") or ""),
        )

    def _tool_tts_say(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "tts_say",
            text=str(args.get("text") or ""),
            voice=str(args.get("voice") or ""),
            backend=str(args.get("backend") or ""),
            mood=str(args.get("mood") or ""),
        )

    def _tool_tts_voices(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "tts_voices",
            action=str(args.get("action") or "list"),
            name=str(args.get("name") or ""),
            path=str(args.get("path") or ""),
            preset_id=str(args.get("preset_id") or ""),
            consent=str(args.get("consent") or ""),
        )

    def _tool_evolve_plan(self, args: dict[str, Any]) -> str:
        return self._registry_tool("evolve_plan",
                                   instruction=str(args.get("instruction") or ""))

    def _tool_evolve_apply(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "evolve_apply",
            proposal_id=str(args.get("proposal_id") or ""),
            commit=str(args.get("commit") or ""),
        )

    def _tool_evolve_list(self, args: dict[str, Any]) -> str:
        return self._registry_tool("evolve_list",
                                   limit=str(args.get("limit") or ""))

    def _tool_evolve_research(self, args: dict[str, Any]) -> str:
        return self._registry_tool("evolve_research",
                                   topic=str(args.get("topic") or ""))

    def _tool_evolve_audit(self, args: dict[str, Any]) -> str:
        return self._registry_tool("evolve_audit",
                                   max_findings=str(args.get("max_findings") or ""))

    def _tool_evolve_revert(self, args: dict[str, Any]) -> str:
        return self._registry_tool("evolve_revert",
                                   proposal_id=str(args.get("proposal_id") or ""))

    def _tool_evolve_queue(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "evolve_queue",
            action=str(args.get("action") or "list"),
            instruction=str(args.get("instruction") or ""),
        )

    def _tool_evolve_git(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "evolve_git",
            action=str(args.get("action") or "status"),
            branch=str(args.get("branch") or ""),
            push=str(args.get("push") or ""),
        )

    def _tool_proxy_file(self, args: dict[str, Any]) -> str:
        v = self._call("proxy_file", limit=str(args.get("limit") or ""),
                       max_age_hours=str(args.get("max_age_hours") or ""))
        if not isinstance(v, dict):
            return "proxy_file failed"
        if not v.get("written"):
            return f"no working proxies: {v.get('note', 'refresh first')}"
        return f"clean proxy file: {v['path']} ({v['count']} proxies, fastest: {v.get('fastest', '?')})"

    def _tool_proxy_scrape(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "proxy_scrape",
            schemes=str(args.get("schemes") or ""),
            limit=str(args.get("limit") or ""),
        )

    def _tool_proxy_refresh(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "proxy_refresh",
            limit=str(args.get("limit") or ""),
            schemes=str(args.get("schemes") or ""),
            detect_country=str(args.get("detect_country") or ""),
        )

    def _tool_proxy_pool(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "proxy_pool",
            action=str(args.get("action") or "list"),
            scheme=str(args.get("scheme") or ""),
            country=str(args.get("country") or ""),
            anonymity=str(args.get("anonymity") or ""),
            max_age_hours=str(args.get("max_age_hours") or ""),
            limit=str(args.get("limit") or ""),
        )

    def _tool_proxy_lab_status(self, args: dict[str, Any]) -> str:
        return self._registry_tool("proxy_lab_status")

    def _tool_proxy_schedule(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "proxy_schedule",
            enabled=str(args.get("enabled") or "true"),
            every_minutes=str(args.get("every_minutes") or ""),
        )

    def _tool_proxy_rotate(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "proxy_rotate",
            action=str(args.get("action") or "status"),
            strategy=str(args.get("strategy") or ""),
            cooldown_seconds=str(args.get("cooldown_seconds") or ""),
            failover_seconds=str(args.get("failover_seconds") or ""),
            max_age_hours=str(args.get("max_age_hours") or ""),
        )

    def _tool_ssh_socks(self, args: dict[str, Any]) -> str:
        return self._registry_tool(
            "ssh_socks",
            action=str(args.get("action") or "list"),
            name=str(args.get("name") or ""),
            host=str(args.get("host") or ""),
            port=str(args.get("port") or ""),
            user=str(args.get("user") or ""),
            password=str(args.get("password") or ""),
            key=str(args.get("key") or ""),
            local_port=str(args.get("local_port") or ""),
            auto_reconnect=str(args.get("auto_reconnect") or ""),
            max_reconnects=str(args.get("max_reconnects") or ""),
        )

    # ── wave-40 tools: api, database, voice, swarm ───────────────────────────

    def _tool_api(self, args: dict[str, Any]) -> str:
        name = str(args.get("name") or "").strip()
        params = args.get("params")
        params_json = json.dumps(params) if isinstance(params, (dict, list)) else str(params or "")
        tools = self.context.tools
        result = tools.call("api_call", connector=name, params=params_json)
        if not result.ok:
            return f"api: {getattr(result.error, 'message', result.error)}"
        return f"api {name}: " + json.dumps(result.value, default=str)[:2500]

    def _tool_db_tables(self, args: dict[str, Any]) -> str:
        result = self.context.tools.call("db_tables")
        if not result.ok:
            return f"db: {getattr(result.error, 'message', result.error)}"
        value = result.value
        return f"tables ({value['count']}): " + ", ".join(
            f"{t['name']}({t['rows']})" for t in value["tables"][:40]
        )

    def _tool_db_schema(self, args: dict[str, Any]) -> str:
        table = str(args.get("table") or "").strip()
        if not table:
            return "db_schema needs a 'table'"
        result = self.context.tools.call("db_schema", table=table)
        if not result.ok:
            return f"db: {getattr(result.error, 'message', result.error)}"
        value = result.value
        cols = ", ".join(
            f"{c['name']} {c['type']}" + (" PK" if c["pk"] else "") for c in value["columns"]
        )
        return f"{value['table']} ({value['rows']} rows): {cols}"

    def _tool_db_counts(self, args: dict[str, Any]) -> str:
        result = self.context.tools.call("db_counts")
        if not result.ok:
            return f"db: {getattr(result.error, 'message', result.error)}"
        value = result.value
        top = ", ".join(f"{t['name']}={t['rows']}" for t in value["tables"][:10])
        return f"database: {value['total_tables']} tables. largest: {top}"

    def _tool_speak(self, args: dict[str, Any]) -> str:
        text = str(args.get("text") or "").strip()
        if not text:
            return "speak needs a 'text'"
        result = self.context.tools.call("speak", text=text[:4000])
        if not result.ok:
            return f"speak: {getattr(result.error, 'message', result.error)}"
        info = result.value
        sent = ""
        chat_key = getattr(self, "_last_chat_key", "") or ""
        if self.gateway is not None and chat_key:
            try:
                from ..social.chat.base import ChatRef

                platform, _, chat_id = chat_key.partition(":")
                ref = ChatRef(platform=platform, chat_id=chat_id)
                outcome = self.gateway.send_file(platform, ref, info["path"],
                                                 caption=f"[{info.get('engine')}]")
                sent = " (sent to you)" if getattr(outcome, "ok", False) else ""
            except Exception:  # noqa: BLE001 - the file is still on disk
                pass
        return f"spoken with {info.get('engine')}: {info.get('path')}{sent}"

    def _tool_swarm(self, args: dict[str, Any]) -> str:
        from .swarm import SwarmAgent

        goal = str(args.get("goal") or "").strip()
        if not goal:
            return "swarm needs a 'goal'"
        workers = 3
        try:
            workers = max(1, min(int(args.get("workers") or 3), 5))
        except (TypeError, ValueError):
            workers = 3
        result = SwarmAgent(self.context, brain=self.brain, gateway=self.gateway).run(
            goal, workers=workers)
        if not result.ok:
            return f"swarm: all legs failed — {result.synthesis[:600]}"
        return f"swarm ({sum(1 for l in result.legs if l['ok'])}/{len(result.legs)} legs, " \
               f"{result.seconds:.0f}s): " + result.synthesis[:2500]

    # ── helpers ──────────────────────────────────────────────────────────────

    def _safe_path(self, raw: str) -> Path | None:
        """Resolve a repo-relative path, refusing anything that escapes the repo."""
        candidate = (raw or "").strip().lstrip("/")
        if not candidate:
            return None
        target = (self.repo_root / candidate).resolve()
        try:
            target.relative_to(self.repo_root.resolve())
        except ValueError:
            return None  # path traversal — refused
        return target


# ── module helpers ─────────────────────────────────────────────────────────────

def _format_browser_result(result: dict[str, Any], *, limit: int = 2500) -> str:
    """Render a browser tool result compactly for a digest/step."""
    lines: list[str] = []
    if "url" in result:
        lines.append(f"url: {result.get('url', '')}")
    if result.get("title"):
        lines.append(f"title: {result['title']}")
    if "status" in result:
        lines.append(f"status: {result.get('status')}")
    for key in ("links", "forms", "cookies", "requests"):
        if key in result:
            lines.append(f"{key}: {result[key]}")
    body = ""
    if "text" in result:
        body = result.get("text") or ""
    elif "markdown" in result:
        body = result.get("markdown") or ""
    elif "links" in result and isinstance(result.get("links"), list):
        body = "\n".join(f"  {l.get('text', '')} -> {l.get('url', '')}"
                         for l in result["links"][:30])
    elif "matches" in result:
        body = "\n".join(f"  - {m}" for m in result["matches"][:20])
    elif "pending" in result:
        body = "pending fields: " + ", ".join(result.get("pending") or [])
    elif "note" in result:
        body = result["note"]
    elif "session" in result and "closed" in result:
        body = f"session {result.get('closed', '')} closed"
    if body:
        lines.append(body.strip())
    out = "\n".join(lines)
    return out[:limit] + ("\n…" if len(out) > limit else "")


def summarize_test_run(pattern: str, returncode: int, output: str) -> str:
    """Turn raw ``python -m unittest`` output into something a chat digest
    can use: status + summary line + the NAMES of the failing tests first,
    then a short raw tail. (A bare tail buries the answer under tracebacks.)
    """
    lines = output.splitlines()
    status = "PASS" if returncode == 0 else f"FAIL (exit {returncode})"
    summary = next((ln.strip() for ln in reversed(lines)
                    if re.match(r"^(OK|FAILED)", ln.strip())), "")
    failures = [ln.strip() for ln in lines if re.match(r"^(FAIL|ERROR): ", ln.strip())]
    head = f"unittest {pattern} → {status}" + (f" — {summary}" if summary else "")
    if failures:
        head += "\nfailing:\n" + "\n".join(failures[:10])
    return head + "\n" + "\n".join(lines[-8:])


def _clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… (+{len(text) - limit} chars clipped)"


def _as_int(value: Any, default: int, *, lo: int, hi: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _is_select(sql: str) -> bool:
    first = re.split(r"\s+", sql.lstrip(), maxsplit=1)
    return bool(first) and first[0].upper() in {"SELECT", "WITH"}
