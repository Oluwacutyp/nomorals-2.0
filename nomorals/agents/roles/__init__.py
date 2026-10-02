"""Specialized agents.

Each role is small on purpose: the intelligence lives in the tools and the model,
the role declares *what it is allowed to do* and *how it turns a task into tool
calls*. Capability narrowing is the interesting part — a critic agent genuinely
cannot write to disk, which is what makes its verdict trustworthy.
"""

from __future__ import annotations

from typing import Any

from ..base import Agent
from ...core.tasks import Task
from ...core.policy import Capability
# Prompt 02: the role specs live in ..role_specs (this name is a package, so
# the spec module could not be called roles.py); re-exported here so the
# whole swarm API is importable from nomorals.agents.roles.
from ..role_specs import (
    RoleRegistry,
    RoleSpec,
    SwarmAgent,
    check_spec_call,
    default_registry,
    record_tool_denial,
)


class RoleAgent(Agent):
    """Base for tool-driven roles.

    A Prompt-02 :class:`RoleSpec` may be bound via ``role_spec=`` (or
    :func:`build_agent(..., role_spec=...)`).  When bound, :meth:`_call_tool`
    enforces the spec's allowlist, read-only flag, and path guards *before*
    the registry call — the same rule the swarm layer applies, so the
    existing agents are held to the locked contract on every path.
    Unbound agents behave exactly as before.
    """

    def __init__(self, *, role_spec: RoleSpec | None = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.role_spec = role_spec

    def _tools(self) -> Any:
        return getattr(self.context, "tools", None) if self.context else None

    def _call_tool(self, name: str, **kwargs: Any) -> Any:
        tools = self._tools()
        if tools is None:
            raise RuntimeError(f"agent {self.name} has no tool registry")
        spec = getattr(self, "role_spec", None)
        if spec is not None:
            get_spec = getattr(tools, "get", None)
            tool_spec = get_spec(name) if callable(get_spec) else None
            denied = check_spec_call(
                spec, name, kwargs,
                getattr(tool_spec, "capability", "") or "")
            if denied is not None:
                record_tool_denial(self.context, spec.name, name,
                                   denied.reason, str(denied))
                raise denied
        outcome = tools.call(name, actor=self.name, capabilities=self.capabilities, **kwargs)
        if not outcome.ok:
            raise RuntimeError(f"tool {name} failed: {outcome.error}")
        return outcome.value

    def _task_goal(self, task_input: Any) -> str:
        if isinstance(task_input, Task):
            return str(task_input.payload.get("goal") or task_input.name)
        if isinstance(task_input, dict):
            return str(task_input.get("goal") or task_input.get("query") or "")
        return str(task_input or "")


class ResearchAgent(RoleAgent):
    role = "research"
    required_capabilities = (
        Capability.NET_OUT, Capability.NET_BROWSER, Capability.NET_DOWNLOAD,
        Capability.FS_READ, Capability.FS_WRITE, Capability.MEM_READ,
        Capability.MEM_WRITE, Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        goal = self._task_goal(task_input)
        self.check_cancelled()
        search = self._call_tool("web_search", query=goal, max_results=6)
        pages = []
        for hit in (search.get("results") or [])[:4]:
            self.check_cancelled()
            url = hit.get("url")
            if not url:
                continue
            fetched = self._call_tool("web_fetch", url=url)
            if fetched.get("text"):
                pages.append({"url": url, "title": fetched.get("title", ""), "text": fetched["text"][:4000]})
        if not pages:
            return {"goal": goal, "sources": [], "summary": "no sources retrieved"}
        summary = self._summarize(goal, pages)
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is not None:
            for page in pages:
                memory.remember(
                    f"Source for {goal[:60]!r}: {page['title'] or page['url']}",
                    kind="episode", importance=0.4, source="research",
                    metadata={"url": page["url"]},
                )
        return {"goal": goal, "sources": [{"url": p["url"], "title": p["title"]} for p in pages], "summary": summary}

    def _summarize(self, goal: str, pages: list[dict[str, Any]]) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            return "\n".join(f"- {p['title'] or p['url']}" for p in pages)
        from ...llm.base import Message, SamplingParams

        corpus = "\n\n".join(f"[{p['url']}]\n{p['text'][:2500]}" for p in pages)
        response = router.chat(
            [
                Message.system("You synthesize research. Cite the source URL inline."),
                Message.user(f"Question: {goal}\n\nSources:\n{corpus}"),
            ],
            SamplingParams(temperature=0.3, max_tokens=1500),
        )
        if not response.ok:
            return "\n".join(f"- {p['url']}" for p in pages)
        self.budget.charge_tokens(response.usage.total_tokens)
        return response.text


class CodingAgent(RoleAgent):
    role = "coding"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_CODE,
        Capability.EXEC_SHELL, Capability.MEM_READ, Capability.MEM_WRITE,
        Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        goal = self._task_goal(task_input)
        self.check_cancelled()
        spec = task_input.payload if isinstance(task_input, Task) else {}
        language = spec.get("language", "python")
        code = self._generate(goal, language)
        if not spec.get("execute", True):
            return {"goal": goal, "language": language, "code": code, "executed": False}

        path = spec.get("path") or f"generated/{_slug(goal)}.{_extension(language)}"
        self._call_tool("fs_write", path=path, content=code)
        executed = self._call_tool("shell_run", command=f"python {path}", timeout=60)
        return {
            "goal": goal, "language": language, "path": path, "code": code,
            "executed": True, "exit_code": executed.get("exit_code"),
            "stdout": (executed.get("stdout") or "")[:2000],
            "stderr": (executed.get("stderr") or "")[:1000],
        }

    def _generate(self, goal: str, language: str) -> str:
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            return f"# No model available to generate code for: {goal}\n"
        from ...llm.base import Message, SamplingParams

        response = router.chat(
            [
                Message.system(f"Write complete, runnable {language}. No placeholders. No commentary outside the code."),
                Message.user(goal),
            ],
            SamplingParams(temperature=0.2, max_tokens=4096),
        )
        self.budget.charge_tokens(response.usage.total_tokens)
        return _strip_code_fence(response.text) if response.ok else ""


class VisionAgent(RoleAgent):
    role = "vision"
    required_capabilities = (Capability.FS_READ, Capability.NET_DOWNLOAD, Capability.MODEL_CALL, Capability.MEM_WRITE)

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {"path": task_input}
        self.check_cancelled()
        return self._call_tool(
            "vision_describe",
            path=spec.get("path", ""),
            url=spec.get("url", ""),
            prompt=spec.get("prompt", ""),
        )


class DataCollectionAgent(RoleAgent):
    role = "data_collection"
    required_capabilities = (
        Capability.NET_OUT, Capability.NET_DOWNLOAD, Capability.FS_READ,
        Capability.FS_WRITE, Capability.MODEL_CALL, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        goal = self._task_goal(task_input)
        self.check_cancelled()
        return self._call_tool(
            "data_harvest",
            topic=goal,
            prompts=spec.get("prompts") or [goal],
            limit=int(spec.get("limit", 10)),
        )


class TrainingAgent(RoleAgent):
    role = "training"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_SHELL,
        Capability.TRAIN_RUN, Capability.MODEL_DOWNLOAD, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        self.check_cancelled()
        return self._call_tool(
            "training_run",
            dataset=spec.get("dataset", ""),
            base_model=spec.get("base_model", ""),
            backend=spec.get("backend", "native"),
            epochs=int(spec.get("epochs", 1)),
        )


class SocialAgent(RoleAgent):
    role = "social"
    required_capabilities = (
        Capability.SOCIAL_POST, Capability.SOCIAL_READ, Capability.NET_OUT,
        Capability.FS_READ, Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        self.check_cancelled()
        return self._call_tool(
            "social_post",
            platform=spec.get("platform", ""),
            content=spec.get("content", ""),
            scheduled_at=spec.get("scheduled_at"),
        )


class ExecutionAgent(RoleAgent):
    role = "execution"
    required_capabilities = (
        Capability.FS_READ, Capability.FS_WRITE, Capability.EXEC_CODE,
        Capability.EXEC_SHELL, Capability.NET_OUT, Capability.DB_WRITE,
    )

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {}
        command = spec.get("command")
        if command:
            self.check_cancelled()
            return self._call_tool("shell_run", command=command, timeout=float(spec.get("timeout", 60)))
        return {"goal": self._task_goal(task_input), "status": "no action specified"}


class MemoryAgent(RoleAgent):
    role = "memory"
    required_capabilities = (
        Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_WRITE, Capability.MODEL_CALL,
    )

    def work(self, task_input: Any) -> Any:
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is None:
            return {"error": "no memory subsystem"}
        spec = task_input.payload if isinstance(task_input, Task) else {}
        action = spec.get("action", "consolidate")
        self.check_cancelled()
        if action == "recall":
            return {"results": memory.recall(spec.get("query", ""), limit=int(spec.get("limit", 8)))}
        if action == "remember":
            return {"id": memory.remember(spec.get("content", ""), kind=spec.get("kind", "episode"))}
        return memory.consolidate()


class CriticAgent(RoleAgent):
    """Verifies other agents' work. Read-only on purpose.

    A critic that could edit files would just fix what it was supposed to be
    judging, and the verdict would be worthless.
    """

    role = "critic"
    required_capabilities = (Capability.MEM_READ, Capability.DB_READ, Capability.FS_READ, Capability.MODEL_CALL)

    def work(self, task_input: Any) -> Any:
        spec = task_input.payload if isinstance(task_input, Task) else {"output": task_input}
        goal = spec.get("goal", "")
        output = spec.get("output", "")
        router = getattr(self.context, "router", None) if self.context else None
        if router is None:
            return {"score": 0.5, "verdict": "no model available to critique", "issues": []}
        from ...llm.base import Message, SamplingParams

        response = router.chat(
            [
                Message.system(
                    "You are a strict reviewer. List concrete defects. "
                    'Reply as JSON: {"score": 0.0-1.0, "issues": ["..."], "verdict": "..."}'
                ),
                Message.user(f"Goal: {goal}\n\nDeliverable:\n{str(output)[:6000]}"),
            ],
            SamplingParams(temperature=0.1, max_tokens=1024, json_mode=True),
        )
        self.budget.charge_tokens(response.usage.total_tokens)
        if not response.ok:
            return {"score": 0.5, "verdict": "critique unavailable", "issues": []}
        from .orchestrator_helpers import parse_json_loose

        data = parse_json_loose(response.text) or {}
        return {
            "score": float(data.get("score", 0.5)),
            "issues": list(data.get("issues") or [])[:8],
            "verdict": str(data.get("verdict", "")),
        }


class ReflectionAgent(RoleAgent):
    role = "reflection"
    required_capabilities = (Capability.MEM_READ, Capability.MEM_WRITE, Capability.DB_READ, Capability.MODEL_CALL)

    def work(self, task_input: Any) -> Any:
        memory = getattr(self.context, "memory", None) if self.context else None
        if memory is None:
            return {"lessons": []}
        self.check_cancelled()
        return memory.consolidate()


#: role name → class, for dynamic spawning.
ROLES: dict[str, type[Agent]] = {
    cls.role: cls
    for cls in (
        ResearchAgent, CodingAgent, VisionAgent, DataCollectionAgent, TrainingAgent,
        SocialAgent, ExecutionAgent, MemoryAgent, CriticAgent, ReflectionAgent,
    )
}


def build_agent(role: str, **kwargs: Any) -> Agent:
    """Instantiate a role by name."""
    try:
        cls = ROLES[role]
    except KeyError:
        return ExecutionAgent(**kwargs)
    return cls(**kwargs)


def _slug(text: str, limit: int = 32) -> str:
    from ...core.text import slugify

    # canonical: nomorals.core.text.slugify (old impl crashed on None;
    # the shared one returns the fallback instead)
    return slugify(text, limit=limit, fallback="task")


def _extension(language: str) -> str:
    return {
        "python": "py", "javascript": "js", "typescript": "ts", "bash": "sh",
        "shell": "sh", "rust": "rs", "go": "go", "c": "c", "cpp": "cpp",
    }.get(language.lower(), "txt")


def _strip_code_fence(text: str) -> str:
    import re

    text = text.strip()
    match = re.match(r"^```[a-zA-Z0-9_+-]*\n(.*?)\n```$", text, re.DOTALL)
    return match.group(1) if match else text


__all__ = [
    "ROLES",
    "CodingAgent",
    "CriticAgent",
    "DataCollectionAgent",
    "ExecutionAgent",
    "MemoryAgent",
    "ReflectionAgent",
    "ResearchAgent",
    "RoleAgent",
    "SocialAgent",
    "TrainingAgent",
    "VisionAgent",
    "build_agent",
]
