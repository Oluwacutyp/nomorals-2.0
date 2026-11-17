"""WolframAlpha API client — the keyed rung of the computational-answer
organ (52-websites site #13).

The v2/query API answers in structured pods (Input interpretation,
Result, Step-by-step solution, ...), not a flat text blob — the
client keeps that structure: every pod, every assumption, every
did-you-mean survives into the :class:`Computation` the organ
returns, so Devon can show the derivation, not just the answer.

Keyed: the AppID comes from ``NM_WOLFRAM_APPID`` (``wolfram.appid``
in config) — the client never invents one, and without one
:meth:`WolframClient.query` refuses in words. Never raises: a dead
network, a bad key, an unparseable query all come back as
``{"ok": False, "error": ...}``.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from collections.abc import Callable
from typing import Any

from ..core.http import HttpClient

__all__ = ["WolframClient", "WOLFRAM_API_URL", "format_wolfram_reply"]

_log = logging.getLogger(__name__)

WOLFRAM_API_URL = "https://api.wolframalpha.com/v2/query"
_HTTP_TIMEOUT = 20.0
_USER_AGENT = "Devon/1.0 (WolframAlpha client)"

#: A fetcher takes a URL and returns the response body as text.
Fetcher = Callable[[str], str]


def _default_fetch(url: str) -> str:
    try:
        client = HttpClient(timeout=_HTTP_TIMEOUT, user_agent=_USER_AGENT)
        resp = client.get(url)
        if not resp.ok:
            return ""
        return resp.text
    except Exception:  # noqa: BLE001 — fetch failure is an empty body
        _log.debug("wolfram fetch failed", exc_info=True)
        return ""


def _pod(subpods: Any) -> dict[str, Any]:
    texts = []
    for sub in subpods if isinstance(subpods, list) else []:
        if not isinstance(sub, dict):
            continue
        text = (sub.get("plaintext") or "").strip()
        if text:
            texts.append({"title": (sub.get("title") or "").strip(),
                          "plaintext": text})
    return texts


class WolframClient:
    """One AppID, one query at a time, over the v2/query JSON API."""

    def __init__(self, appid: str = "", fetcher: Fetcher | None = None):
        self.appid = (appid or "").strip()
        self._fetcher = fetcher or _default_fetch

    @property
    def configured(self) -> bool:
        """True when an AppID is set — the only gate on the rung."""
        return bool(self.appid)

    def query(self, question: str) -> dict[str, Any]:
        """Ask WolframAlpha. Never raises — see the module docstring."""
        question = (question or "").strip()
        if not question:
            return {"ok": False, "pods": [], "assumptions": [],
                    "didyoumeans": [],
                    "error": "WolframAlpha needs a question"}
        if not self.configured:
            return {"ok": False, "pods": [], "assumptions": [],
                    "didyoumeans": [],
                    "error": ("WolframAlpha needs an AppID — set "
                              "NM_WOLFRAM_APPID (wolfram.appid in config) "
                              "and ask again")}
        params = urllib.parse.urlencode({
            "input": question,
            "format": "plaintext",
            "output": "json",
            "appid": self.appid,
        })
        try:
            body = self._fetcher(f"{WOLFRAM_API_URL}?{params}")
            payload = json.loads(body) if body else {}
        except Exception as exc:  # noqa: BLE001 — bad body is a failed call
            return {"ok": False, "pods": [], "assumptions": [],
                    "didyoumeans": [],
                    "error": f"WolframAlpha could not be reached: {exc}"}
        result = payload.get("queryresult") or {}
        if not isinstance(result, dict):
            return {"ok": False, "pods": [], "assumptions": [],
                    "didyoumeans": [],
                    "error": "WolframAlpha returned an unreadable answer"}
        pods = []
        for pod in result.get("pods") or []:
            if not isinstance(pod, dict):
                continue
            subpods = _pod(pod.get("subpods"))
            if subpods:
                pods.append({"title": (pod.get("title") or "").strip(),
                             "id": (pod.get("id") or "").strip(),
                             "subpods": subpods})
        assumptions = []
        for assumption in result.get("assumptions") or []:
            if not isinstance(assumption, dict):
                continue
            for value in assumption.get("values") or []:
                if isinstance(value, dict) and value.get("desc"):
                    assumptions.append(str(value["desc"]))
        didyoumeans = []
        for mean in result.get("didyoumeans") or []:
            if isinstance(mean, dict) and mean.get("val"):
                didyoumeans.append(str(mean["val"]))
            elif isinstance(mean, str) and mean:
                didyoumeans.append(mean)
        if result.get("success") is not True or not pods:
            error = "WolframAlpha could not compute that"
            if didyoumeans:
                error += f" — did you mean: {', '.join(didyoumeans)}?"
            return {"ok": False, "pods": pods, "assumptions": assumptions,
                    "didyoumeans": didyoumeans, "error": error}
        return {"ok": True, "pods": pods, "assumptions": assumptions,
                "didyoumeans": didyoumeans, "error": None,
                "input": (result.get("inputstring") or question)}


def format_wolfram_reply(question: str, res: dict[str, Any]) -> str:
    """One pod per section, assumptions named, never a flat blob."""
    if not res.get("ok"):
        return res.get("error") or "WolframAlpha could not compute that"
    lines = [f"WolframAlpha — {question}"]
    for pod in res.get("pods") or []:
        title = pod.get("title") or "Result"
        for sub in pod.get("subpods") or []:
            text = (sub.get("plaintext") or "").strip()
            if not text:
                continue
            lines.append(f"{title}: {text}")
    if res.get("assumptions"):
        lines.append("Assuming: " + "; ".join(res["assumptions"]))
    return "\n".join(lines)
