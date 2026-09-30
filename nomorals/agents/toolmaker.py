"""Tool creator — design, write, test, and register new tools when the
existing ones are insufficient.

The flow is a real pipeline, not a text suggestion:

  1. ``design(task)``   — the model produces a tool *spec*: name, purpose,
     parameters, and behavior, as JSON.
  2. ``generate(spec)`` — the model writes a complete Python module exposing
     ``register(registry)`` that wires the tool into the registry the same
     way every built-in tool does.
  3. ``test(code)``     — the generated module is executed in a throwaway
     subprocess: it must import cleanly, expose a callable ``register``, and
     actually register at least one tool against a probe registry.  A tool
     that fails the test is never installed.
  4. ``install(name, code, ...)`` — only a *tested* module is written to
     ``nomorals/tools/custom/<name>.py`` and registered into the live tool
     registry (custom tools auto-register on every build).

Custom tools live in their own package so they are isolated, listed, and
removable.  This is how the system grows its own toolset under the same
test-before-ship discipline the evolution gate uses.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .reasoning import _extract_json

_log = get_logger(__name__)

__all__ = ["ToolMaker", "ToolSpec", "register", "register_custom"]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CUSTOM_DIR = Path(__file__).resolve().parent.parent / "tools" / "custom"

# The contract every generated module must satisfy, shown to the model.
_MODULE_TEMPLATE = '''"""Custom tool: {name}. {description}"""
from __future__ import annotations
import json
from typing import Any

def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "{name}",
        description="{description}",
        capability="io",
        parameters={{PARAMS}},
    )
    def {name}({args}):
        {body}
'''

# A probe-registry script used to smoke-test a generated module in a
# subprocess (so a broken generated file can never crash the main process).
_PROBE_SCRIPT = r'''
import importlib.util, sys, json
path = sys.argv[1]
try:
    spec = importlib.util.spec_from_file_location("custom_tool_probe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"import: {exc}"})); sys.exit(0)
register = getattr(mod, "register", None)
if not callable(register):
    print(json.dumps({"ok": False, "error": "no callable register()"}))
    sys.exit(0)
class _Ctx:
    db = None
    settings = None
    router = None
    tools = None
    extras = {}
class _Reg:
    def __init__(self):
        self.context = _Ctx()
        self.tools = {}
    def register(self, name, fn=None, **kw):
        def dec(f):
            self.tools[name] = f
            return f
        if fn is None:
            return dec
        self.tools[name] = fn
        return fn
reg = _Reg()
try:
    register(reg)
except Exception as exc:
    print(json.dumps({"ok": False, "error": f"register: {exc}"}))
    sys.exit(0)
# smoke-invoke: every registered tool must survive a no-arg call —
# either it returns (great) or it raises a *missing-argument* TypeError
# (fine: the tool requires parameters). Anything else is a real bug and
# fails the probe, so a broken tool is never installed.
smoke = {}
for tname, fn in reg.tools.items():
    try:
        fn()
        smoke[tname] = "ok"
    except TypeError as te:
        msg = str(te).lower()
        smoke[tname] = ("needs-args"
                        if ("missing" in msg or "argument" in msg
                            or "required" in msg)
                        else f"error: {te}")
    except Exception as exc:
        smoke[tname] = f"error: {type(exc).__name__}: {exc}"
broken = {k: v for k, v in smoke.items() if v.startswith("error")}
print(json.dumps({"ok": bool(reg.tools) and not broken,
                  "tools": list(reg.tools.keys()), "smoke": smoke,
                  **({"error": f"smoke-invoke failed: {broken}"}
                     if broken else {})}))
'''


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, str]
    behavior: str = ""
    example: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": self.parameters, "behavior": self.behavior,
                "example": self.example}

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> Optional["ToolSpec"]:
        name = str(data.get("name") or "").strip()
        if not name or not data.get("description"):
            return None
        params = data.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        return cls(name=name,
                   description=str(data.get("description"))[:400],
                   parameters={str(k): str(v)[:200]
                               for k, v in params.items()},
                   behavior=str(data.get("behavior") or "")[:800],
                   example=str(data.get("example") or "")[:400])


class ToolMaker:
    def __init__(self, context: Any) -> None:
        self.context = context
        _CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        (_CUSTOM_DIR / "__init__.py").touch(exist_ok=True)

    # ── 1. design ──────────────────────────────────────────────────────────
    def design(self, task: str) -> Optional[ToolSpec]:
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        from ..llm.base import Message, SamplingParams

        prompt = (
            "Design a NEW tool that would help accomplish the task below. "
            "It must be a single, focused, safe tool that fits the NoMorals "
            "tool registry (a Python module exposing register(registry)). "
            "Prefer read-only or local operations. "
            "Respond with JSON ONLY: {\"name\": \"snake_case_tool\", "
            "\"description\": \"<one line>\", "
            "\"parameters\": {\"<param>\": \"<type — <meaning>\"}, "
            "\"behavior\": \"<what it does, step by step>\", "
            "\"example\": \"<one example call>\"}\n\n"
            f"TASK: {task}")
        try:
            resp = router.chat([Message.user(prompt)],
                               SamplingParams(temperature=0.2, max_tokens=600))
            if not getattr(resp, "ok", False):
                return None
            data = _extract_json(resp.text or "")
            spec = ToolSpec.from_data(data) if isinstance(data, dict) else None
            if spec:
                spec.name = _safe_name(spec.name)
            return spec
        except Exception:  # noqa: BLE001
            _log.debug("tool design failed", exc_info=True)
            return None

    # ── 2. generate ────────────────────────────────────────────────────────
    def generate(self, spec: ToolSpec) -> str:
        """Write the Python module for a spec. Returns the code string."""
        router = getattr(self.context, "router", None)
        if router is None:
            return ""
        from ..llm.base import Message, SamplingParams

        param_lines = ", ".join(f'"{k}": "{v}"'
                                for k, v in spec.parameters.items())
        args = ", ".join(list(spec.parameters.keys()) or ["task: str"])
        body = ("        # implement the behavior\n"
                "        result = {\"ok\": True}\n"
                "        return result\n"
                if not spec.behavior else
                "        # implement the behavior\n"
                + "\n".join(f"        # {line}" for line in
                            spec.behavior.splitlines()[:8])
                + "\n        return {\"ok\": True}\n")
        template = _MODULE_TEMPLATE.format(
            name=spec.name, description=spec.description.replace('"', "'"),
            PARAMS=param_lines, args=args, body=body)
        prompt = (
            "Write a COMPLETE, runnable Python module implementing this tool "
            "spec. It must define exactly one function "
            "``register(registry)`` that registers the tool using "
            "``@registry.register(<name>, description=..., capability=..., "
            "parameters={...})``. The tool body must be REAL working code "
            "(no placeholders, no TODOs), defensive about bad input, and "
            "return a JSON-serializable dict. Respond with EXACTLY ONE "
            "fenced ```python code block containing the whole module.\n\n"
            "SPEC:\n" + json.dumps(spec.to_dict(), indent=2) + "\n\n"
            "STARTING POINT (improve it; keep the register() contract):\n"
            + template)
        try:
            resp = router.chat([Message.user(prompt)],
                               SamplingParams(temperature=0.2, max_tokens=2000))
            if not getattr(resp, "ok", False):
                return ""
            from .coding import extract_code_block
            return extract_code_block(resp.text or "")
        except Exception:  # noqa: BLE001
            _log.debug("tool generate failed", exc_info=True)
            return ""

    # ── 3. test ────────────────────────────────────────────────────────────
    def test(self, code: str) -> dict[str, Any]:
        """Run the generated module in a subprocess probe. Returns
        {ok, tools?, error?}."""
        if not code or "def register" not in code:
            return {"ok": False, "error": "no register() in generated code"}
        import tempfile
        tmp = Path(tempfile.mkdtemp(prefix="nm-tool-"))
        target = tmp / "probe_tool.py"
        try:
            target.write_text(code, encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, "-B", "-c", _PROBE_SCRIPT, str(target)],
                cwd=str(_REPO_ROOT), capture_output=True, text=True,
                timeout=60, env={**__import__("os").environ,
                                 "PYTHONPATH": str(_REPO_ROOT)},
            )
            out = (proc.stdout or "").strip().splitlines()
            data = json.loads(out[-1]) if out else {}
            return data
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            try:
                target.unlink(missing_ok=True)
                tmp.rmdir()
            except OSError:
                pass

    # ── 4. install ─────────────────────────────────────────────────────────
    #: Static boundary for generated code: a custom tool may import the
    #: standard library and the registry API, but must not dynamically
    #: load code or hand off to a shell — those paths are how a
    #: "read-only" tool becomes arbitrary execution.
    _FORBIDDEN_TOKENS = ("__import__", "importlib", "os.system",
                         "os.popen", "subprocess", "ctypes",
                         "marshal.", "exec(", "eval(")

    def _validate_imports(self, code: str) -> str | None:
        """Return an error string when the code crosses the boundary,
        else None."""
        low = (code or "")
        for token in self._FORBIDDEN_TOKENS:
            if token in low:
                return (f"generated code contains forbidden construct "
                        f"{token!r} — refusing to install")
        return None

    def install(self, name: str, code: str, *, description: str = "",
                tested: bool = False) -> dict[str, Any]:
        """Write a *tested* module into the custom package and register it
        into the live registry. Refuses to install untested code or code
        that crosses the import boundary."""
        problem = self._validate_imports(code)
        if problem:
            return {"ok": False, "error": problem}
        if not tested:
            probe = self.test(code)
            if not probe.get("ok"):
                return {"ok": False,
                        "error": "refusing to install untested tool: "
                                 + probe.get("error", "test failed")}
        # always-on mid-task reasoning before a new capability lands
        check: dict[str, Any] = {}
        try:
            from .reasoning import ReasoningAgent
            check = ReasoningAgent(self.context).mid_task_check(
                f"install custom tool {name}",
                details=description[:200])
        except Exception:  # noqa: BLE001
            check = {}
        name = _safe_name(name)
        path = _CUSTOM_DIR / f"{name}.py"
        header = (f'"""Custom tool {name} — {description or "user-created"}'
                  f' (generated by the tool creator, {time.strftime("%Y-%m-%d")})."""\n\n')
        path.write_text(header + code, encoding="utf-8")
        # register it into the live registry now (deferred to the next build
        # if no live registry is attached in this context)
        registered = register_custom_module(path, getattr(self.context,
                                                          "tools", None))
        return {"ok": True, "path": str(path), "name": name,
                "registered": registered,
                "mid_task_check": check or None,
                "note": "registered live" if registered is True
                        else ("will register on next build"
                              if registered is None else "failed to register")}

    def list_custom(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(_CUSTOM_DIR.glob("*.py")):
            if path.name == "__init__.py":
                continue
            first = ""
            try:
                text = path.read_text(encoding="utf-8")
                for line in text.splitlines():
                    line = line.strip()
                    if line and not line.startswith(("#", '"""', "from ",
                                                      "import ", "def ")):
                        first = line[:80]
                        break
                out.append({"name": path.stem, "path": str(path),
                            "first_line": first})
            except OSError:
                out.append({"name": path.stem, "path": str(path)})
        return out

    def remove(self, name: str) -> bool:
        name = _safe_name(name)
        path = _CUSTOM_DIR / f"{name}.py"
        registry = getattr(self.context, "tools", None)
        if registry is not None:
            try:
                registry.unregister(name)
            except Exception:  # noqa: BLE001
                pass
        if path.exists():
            path.unlink()
            return True
        return False

    # ── 5. suggest: what tools would the system's own history ask for? ─────
    def suggest(self, *, limit: int = 5) -> list[dict[str, Any]]:
        """Mine the system's own history for new-tool candidates.
        Deterministic, no model:

        * repeated failure families in the failures ledger (>= 2 hits in
          the last 7 days) → a tool that would have prevented/caught them;
        * tools in the call ledger with a high error rate (>= 30% over
          >= 5 calls) → a wrapper/fix candidate;
        * error families seen in the coding log with no matching tool.

        Each candidate: ``{name, why, evidence, priority}`` — ready to
        feed to ``design()``.  Empty list when there is no signal.
        """
        out: list[dict[str, Any]] = []
        db = getattr(self.context, "db", None)
        if db is None:
            return out
        week_ago = time.time() - 7 * 86400.0
        # 1. repeated failure families
        try:
            rows = db.query(
                "SELECT family, COUNT(*) AS n, "
                "MAX(substr(summary, 1, 120)) AS sample FROM failures "
                "WHERE ts > ? GROUP BY family HAVING n >= 2 "
                "ORDER BY n DESC LIMIT ?", (week_ago, limit))
            for r in rows:
                fam = r["family"] or "other"
                out.append({
                    "name": f"guard_{fam.replace('-', '_')}",
                    "why": (f"the system has hit {r['n']} {fam} failures "
                            f"in 7 days — a tool to detect/prevent this "
                            f"family earlier would pay for itself"),
                    "evidence": r["sample"] or "",
                    "priority": min(3, 1 + r["n"] // 3),
                })
        except Exception:  # noqa: BLE001 - pre-migration ledger
            pass
        # 2. flaky tools in the call ledger
        try:
            rows = db.query(
                "SELECT tool, COUNT(*) AS total, "
                "SUM(CASE WHEN status='error' THEN 1 ELSE 0 END) AS errs "
                "FROM tool_calls WHERE created_at > ? GROUP BY tool "
                "HAVING total >= 5 ORDER BY errs*1.0/total DESC LIMIT ?",
                (week_ago, max(1, limit)))
            for r in rows:
                rate = (r["errs"] or 0) * 1.0 / max(1, r["total"] or 1)
                if rate < 0.3:
                    continue
                out.append({
                    "name": f"harden_{r['tool']}",
                    "why": (f"{r['tool']} fails {int(rate * 100)}% of the "
                            f"time ({r['errs']}/{r['total']} calls in 7 "
                            f"days) — a preflight/validation tool for it "
                            f"would cut the noise"),
                    "evidence": f"error rate {rate:.0%} over 7 days",
                    "priority": 2,
                })
        except Exception:  # noqa: BLE001
            pass
        out.sort(key=lambda c: -c["priority"])
        return out[:max(1, limit)]

    # ── one-shot: design -> generate -> test -> install ────────────────────
    def create(self, task: str) -> dict[str, Any]:
        """Full pipeline: design, generate, test, install (only if the test
        passes). Never installs a tool that fails the probe."""
        spec = self.design(task)
        if spec is None:
            return {"ok": False, "error": "could not design a tool "
                    "(no usable model reply)"}
        code = self.generate(spec)
        probe = self.test(code)
        if not probe.get("ok"):
            return {"ok": False, "spec": spec.to_dict(), "code": code,
                    "test": probe, "error": "generated tool failed its test; "
                    "nothing was installed"}
        result = self.install(spec.name, code, description=spec.description,
                              tested=True)
        return {"ok": bool(result.get("ok")), "spec": spec.to_dict(),
                "installed": result.get("path"), "test": probe}


def _safe_name(name: str) -> str:
    import re
    name = re.sub(r"[^a-z0-9_]+", "_", (name or "").lower()).strip("_")
    return (name[:48] or "custom_tool")


# ── custom auto-registration ────────────────────────────────────────────────


def register_custom_module(path: Path, registry: Any) -> bool | None:
    """Import one custom module and call its register(registry). Returns
    True if it registered, False if it failed, None if no registry was
    available to register into."""
    if registry is None:
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"nm_custom_{path.stem}", str(path))
        if spec is None or spec.loader is None:
            return False
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        register = getattr(mod, "register", None)
        if not callable(register):
            return False
        register(registry)
        return True
    except Exception as exc:  # noqa: BLE001 — a bad custom tool must not
        _log.warning("custom tool %s failed to register: %s", path.name, exc)
        return False


def register_custom(registry: Any) -> int:
    """Register every custom tool module into the registry. Called from
    register_builtins so custom tools are available on every build."""
    if not _CUSTOM_DIR.is_dir():
        return 0
    count = 0
    for path in sorted(_CUSTOM_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        if register_custom_module(path, registry):
            count += 1
    return count


# ── registry ────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "tool_create",
        description=(
            "Tool creator: design, write, test, and register a NEW tool when "
            "existing ones are insufficient. action=create (full pipeline) | "
            "suggest (mine the system's own failures for new-tool ideas) | "
            "design | generate | test | install | list | remove. Never "
            "installs a tool that fails its test or crosses the import "
            "boundary."
        ),
        capability="model.call",
        parameters={
            "action": "str — create|suggest|design|generate|test|install|list|remove",
            "task": "str — what the new tool should do (create/design)",
            "name": "str — tool name (install/remove)",
            "code": "str — module code (test/install)",
            "description": "str — for install",
            "spec_json": "str — a ToolSpec JSON (generate)",
            "limit": "int — candidates for suggest",
        },
    )
    def tool_create(
        action: str = "list", *, task: str = "", name: str = "", code: str = "",
        description: str = "", spec_json: str = "", limit: str = "5",
    ) -> dict[str, Any]:
        maker = ToolMaker(context)
        action = (action or "list").strip().lower()
        if action == "create":
            return maker.create(task)
        if action == "suggest":
            try:
                n = max(1, int(limit or 5))
            except ValueError:
                n = 5
            return {"ok": True, "suggestions": maker.suggest(limit=n)}
        if action == "design":
            spec = maker.design(task)
            return {"ok": spec is not None,
                    "spec": spec.to_dict() if spec else None}
        if action == "generate":
            try:
                spec = ToolSpec.from_data(json.loads(spec_json))
            except Exception:  # noqa: BLE001
                spec = None
            if spec is None:
                return {"ok": False, "error": "invalid spec_json"}
            return {"ok": True, "code": maker.generate(spec)}
        if action == "test":
            return maker.test(code)
        if action == "install":
            return maker.install(name, code, description=description)
        if action == "remove":
            return {"ok": maker.remove(name)}
        return {"tools": maker.list_custom()}
