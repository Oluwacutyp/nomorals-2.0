"""Account dossier — one subject, every surface, one score.

Composes `social_enum`, `leaks` and `seed_sources` into a single report:
where the handle exists, what the profile leaks, what has been breached,
and how exposed the subject is overall. Scoring is a table, not a feeling.

The dossier is what you hand over or act on; the components stay
callable on their own.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.policy import Capability
from ..core.logging_setup import get_logger
from . import leaks, seed_sources, social_enum

_log = get_logger(__name__)
__all__ = ["Dossier", "build_dossier", "render", "register"]

_EXPOSURE_POINTS = {
    "stealer-log": 30, "breach": 18, "proxynova": 12, "dehashed": 18,
    "hibp": 15,
}


@dataclass
class Dossier:
    subject: str
    kind: str = "username"          # username | email
    platforms: list[dict[str, Any]] = field(default_factory=list)
    exposure: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    score: int = 0
    tier: str = "low"
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {"subject": self.subject, "kind": self.kind,
                "platforms": self.platforms, "exposure": self.exposure,
                "profile": self.profile, "score": self.score,
                "tier": self.tier, "notes": self.notes}


def _score(d: Dossier) -> None:
    score = 0
    taken = [p for p in d.platforms if p.get("state") == "taken"]
    score += min(20, len(taken) * 2)
    for source in d.exposure.get("sources") or []:
        score += _EXPOSURE_POINTS.get(source, 0)
    if d.exposure.get("passwords"):
        score += min(25, len(d.exposure["passwords"]) * 3)
    if d.exposure.get("sessions"):
        score += min(15, len(d.exposure["sessions"]) * 4)
    if d.profile.get("accounts"):
        score += min(10, len(d.profile["accounts"]))
    d.score = max(0, min(100, score))
    d.tier = ("critical" if d.score >= 70 else "high" if d.score >= 45
              else "medium" if d.score >= 22 else "low")


def build_dossier(subject: str, *, kind: str = "username",
                  workers: int = 10, timeout: float = 20.0) -> Dossier:
    d = Dossier(subject=subject.strip(), kind=kind)
    if not d.subject:
        return d
    if kind == "email":
        d.platforms = [h.as_dict() for h in
                       social_enum.enum_email(d.subject, timeout=timeout)]
        d.profile = next((h.as_dict() for h in
                          seed_sources.harvest(email=d.subject,
                                               only=["gravatar"]).get(
                                                   "profile", [])), {})
        d.exposure = leaks.harvest_exposure(email=d.subject, timeout=timeout)
    else:
        d.platforms = [h.as_dict() for h in
                       social_enum.enum_username(d.subject, workers=workers,
                                                 timeout=timeout)]
        d.exposure = leaks.harvest_exposure(username=d.subject,
                                            timeout=timeout)
    if d.exposure.get("sources_skipped_no_key"):
        d.notes.append("keyed sources not used: "
                       + ", ".join(d.exposure["sources_skipped_no_key"]))
    taken = sum(1 for p in d.platforms if p.get("state") == "taken")
    if taken:
        d.notes.append(f"handle live on {taken} platform(s)")
    _score(d)
    return d


def render(d: Dossier) -> str:
    lines = [f"# Dossier — {d.subject}", "",
             f"- kind: {d.kind}",
             f"- exposure tier: **{d.tier}** ({d.score}/100)", ""]
    if d.notes:
        lines += [f"- {n}" for n in d.notes] + [""]
    taken = [p for p in d.platforms if p.get("state") == "taken"]
    if taken:
        lines += ["## Live accounts", ""]
        lines += [f"- {p['site']}" for p in taken] + [""]
    free = [p["site"] for p in d.platforms if p.get("state") == "free"]
    if free:
        lines += ["## Handle free on", "", ", ".join(free), ""]
    if d.exposure.get("identities"):
        lines += ["## Identities seen in leaks", ""]
        lines += [f"- `{i}`" for i in d.exposure["identities"][:20]] + [""]
    if d.exposure.get("passwords"):
        lines += ["## Passwords seen", ""]
        lines += [f"- `{p}`" for p in d.exposure["passwords"][:20]] + [""]
    if d.exposure.get("platforms"):
        lines += ["## Breached services", "",
                  ", ".join(d.exposure["platforms"][:30]), ""]
    if d.profile:
        lines += ["## Profile", "", "```json",
                  json.dumps(d.profile, indent=1)[:1500], "```", ""]
    return "\n".join(lines)


def register(registry: Any) -> None:
    context = registry.context
    root = Path(getattr(context, "home", ".") or ".") / "data" / "dossiers"

    @registry.register(
        "dossier_build",
        description="Full account dossier: platform presence, profile, leak "
                    "exposure, scored. Username or email.",
        capability=Capability.NET_OUT,
        parameters={"subject": "str", "kind": "str (optional, username|email)",
                    "save": "bool (optional)"},
    )
    def dossier_build(subject: str, *, kind: str = "username",
                      save: str = "false") -> dict[str, Any]:
        d = build_dossier(subject, kind=kind or "username")
        payload = d.as_dict()
        if (save or "").lower() in {"1", "true", "yes"}:
            root.mkdir(parents=True, exist_ok=True)
            slug = re.sub(r"[^a-z0-9._\-]+", "-", subject.lower())
            (root / f"{slug}.md").write_text(render(d), encoding="utf-8")
            (root / f"{slug}.json").write_text(json.dumps(payload, indent=1),
                                               encoding="utf-8")
            payload["saved"] = str(root / f"{slug}.md")
        return payload
