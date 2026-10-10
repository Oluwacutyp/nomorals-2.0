"""``nm brain`` — the Brain facade on the command line.

``nm brain ask "..."``     one chat call through the Brain (task-kind
                           threading, per-call timeout, failure taxonomy,
                           context fitting).
``nm brain judge "..."``   fan out to several providers and let a judge
                           model rank the drafts (Brain.best_of).
``nm brain status``        Brain.diagnose(): provider health, failure
                           classes, cooldowns — the same view the chat
                           layer sees.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...llm.base import Message, SamplingParams
from ...llm.brain import brain_for, explain_failure
from ..emit import _emit


def _cmd_brain(args: argparse.Namespace, context: Any) -> int:
    action = getattr(args, "brain_action", "") or ""
    if action == "status":
        return _brain_status(args, context)
    if action == "judge":
        return _brain_judge(args, context)
    if action == "ask":
        return _brain_ask(args, context)
    print("usage: nm brain {ask|judge|status} ...")
    return 2


def _brain_status(args: argparse.Namespace, context: Any) -> int:
    brain = brain_for(context)
    if getattr(args, "json", False):
        print(json.dumps(brain.status(), indent=2, default=str))
        return 0
    print(brain.diagnose())
    return 0


def _brain_ask(args: argparse.Namespace, context: Any) -> int:
    prompt = str(getattr(args, "prompt", "") or "").strip()
    if not prompt:
        print("nm brain ask: empty prompt — nothing asked.")
        return 2
    brain = brain_for(context)
    params = SamplingParams(
        temperature=float(getattr(args, "temperature", 0.7) or 0.7),
        max_tokens=int(getattr(args, "max_tokens", 1024) or 1024),
    )
    response = brain.chat(
        [Message.user(prompt)],
        params,
        task_kind=str(getattr(args, "kind", "") or "chat"),
        timeout_s=float(getattr(args, "timeout", 90.0) or 90.0),
    )
    if response.ok and (response.text or "").strip():
        if getattr(args, "json", False):
            _emit(args, response.to_dict(), response.text.strip())
        else:
            print(response.text.strip())
        return 0
    hint = explain_failure(response)
    payload = {"ok": False, "error": response.error,
               "failure_class": response.failure_class,
               "provider": response.provider, "hint": hint}
    _emit(args, payload,
          f"brain ask failed ({response.failure_class or 'unknown'}): "
          f"{response.error or 'no error detail'}\n{hint}")
    return 1


def _brain_judge(args: argparse.Namespace, context: Any) -> int:
    prompt = str(getattr(args, "prompt", "") or "").strip()
    if not prompt:
        print("nm brain judge: empty prompt — nothing judged.")
        return 2
    brain = brain_for(context)
    params = SamplingParams(
        temperature=float(getattr(args, "temperature", 0.9) or 0.9),
        max_tokens=int(getattr(args, "max_tokens", 1024) or 1024),
    )
    text, judgment = brain.best_of(
        prompt,
        n=int(getattr(args, "fanout", 3) or 3),
        task_kind=str(getattr(args, "kind", "") or "creative"),
        params=params,
        timeout_s=float(getattr(args, "timeout", 120.0) or 120.0),
    )
    payload = {
        "winner": text,
        "reason": getattr(judgment, "reason", ""),
        "scores": getattr(judgment, "scores", {}),
    }
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return 0
    reason = getattr(judgment, "reason", "") or ""
    print(text)
    if reason:
        print(f"\n— judge: {reason}")
    return 0 if (text or "").strip() else 1
