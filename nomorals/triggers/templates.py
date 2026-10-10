"""Trigger templates (blueprints): reusable trigger recipes.

A template is a named, parameterized trigger spec — the Home Assistant
blueprint idea: the shape is authored once, the owner fills in the
inputs.  Placeholders use single braces (``{symbol}``) so they never
collide with the double-brace ``{{evidence}}`` action templates rendered
at fire time.

Use :func:`render_template` to fill a template, then pass the spec to
:meth:`TriggerEngine.add_from_template` (or ``TriggerEngine.add``).
"""

from __future__ import annotations

import re
from typing import Any

from .models import TriggerError

#: single-brace blueprint placeholders — the lookarounds keep
#: double-brace {{evidence}} action templates untouched.
_PLACEHOLDER_RE = re.compile(
    r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})")

TEMPLATES: dict[str, dict[str, Any]] = {
    "price_alert": {
        "description": "Notify when a symbol crosses a price threshold. "
                       "Body includes the live price via evidence templates.",
        "inputs": ["symbol", "direction", "value"],
        "defaults": {"market": "crypto", "direction": "below"},
        "spec": {
            "name": "Price alert: {symbol} {direction} {value}",
            "source": "price",
            "condition": {"symbol": "{symbol}", "market": "{market}",
                          "op": "{_op}", "value": "{value}"},
            "action": "notify",
            "action_params": {
                "title": "{symbol} price alert",
                "body": "{symbol} is now {{price}} (threshold {value})",
            },
            "cooldown_s": 3600,
        },
    },
    "file_watch": {
        "description": "Message a chat when a file changes.",
        "inputs": ["path", "chat"],
        "defaults": {"on": "change"},
        "spec": {
            "name": "File changed: {path}",
            "source": "file",
            "condition": {"path": "{path}", "on": "{on}"},
            "action": "message",
            "action_params": {
                "chat": "{chat}",
                "text": "📝 {path} changed ({{event}})",
            },
        },
    },
    "url_watch": {
        "description": "Notify when a web page changes "
                       "(changedetection.io-style).",
        "inputs": ["url"],
        "defaults": {"poll_s": 900},
        "spec": {
            "name": "Page changed: {url}",
            "source": "url",
            "condition": {"url": "{url}", "text_only": True},
            "action": "notify",
            "action_params": {
                "title": "Page changed",
                "body": "{{url}} changed — {{snippet}}",
            },
            "poll_s": "{poll_s}",
        },
    },
    "morning_briefing": {
        "description": "Run a briefing mission every morning.",
        "inputs": [],
        "defaults": {"daily": "07:00", "goal": "Morning briefing"},
        "spec": {
            "name": "Morning briefing",
            "source": "schedule",
            "condition": {"daily": "{daily}"},
            "action": "mission",
            "action_params": {"goal": "{goal}", "max_iterations": 6},
        },
    },
    "webhook_relay": {
        "description": "Accept a signed webhook and forward it as a chat "
                       "message.",
        "inputs": ["chat"],
        "defaults": {"scheme": "plain"},
        "spec": {
            "name": "Webhook relay",
            "source": "webhook",
            "condition": {"scheme": "{scheme}"},
            "action": "message",
            "action_params": {
                "chat": "{chat}",
                "text": "🔔 webhook: {{payload}}",
            },
        },
    },
    "message_keyword": {
        "description": "React when a chat message matches a pattern.",
        "inputs": ["pattern", "chat"],
        "defaults": {},
        "spec": {
            "name": "Keyword: {pattern}",
            "source": "message",
            "condition": {"pattern": "{pattern}", "chat": "{chat}"},
            "action": "notify",
            "action_params": {
                "title": "Keyword hit",
                "body": "{{sender}} in {{chat}}: {{matched}}",
            },
            "cooldown_s": 300,
        },
    },
}


def list_templates() -> list[dict[str, Any]]:
    """Summaries of every built-in template."""
    return [{"name": name,
             "description": t["description"],
             "inputs": list(t["inputs"]),
             "defaults": dict(t["defaults"])}
            for name, t in TEMPLATES.items()]


def _fill(value: Any, params: dict[str, Any]) -> Any:
    if isinstance(value, str):
        def _sub(m: "re.Match[str]") -> str:
            key = m.group(1)
            if key == "_op":
                direction = str(params.get("direction", "below")).lower()
                return {"below": "lt", "above": "gt"}.get(direction, "lt")
            if key not in params:
                raise TriggerError(
                    f"template input {key!r} is required")
            return str(params[key])
        return _PLACEHOLDER_RE.sub(_sub, value)
    if isinstance(value, dict):
        return {k: _fill(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill(v, params) for v in value]
    return value


def render_template(name: str, params: dict[str, Any] | None = None
                    ) -> dict[str, Any]:
    """Fill a template's ``{inputs}`` and return the trigger spec dict.

    ``params`` supplies the template inputs; unspecified inputs fall back
    to the template's ``defaults``.  Missing required inputs raise
    :class:`TriggerError` — never a half-filled spec.
    """
    tmpl = TEMPLATES.get(name)
    if tmpl is None:
        raise TriggerError(
            f"unknown trigger template {name!r}; expected one of "
            f"{sorted(TEMPLATES)}")
    merged = dict(tmpl["defaults"])
    merged.update(params or {})
    spec = _fill(tmpl["spec"], merged)
    # numeric-ish leftovers stay numeric where the schema wants numbers
    for key in ("cooldown_s", "poll_s"):
        if key in spec and isinstance(spec[key], str):
            try:
                spec[key] = float(spec[key])
            except ValueError:
                pass
    return spec
