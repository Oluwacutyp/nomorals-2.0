"""``nm arena`` / ``trial`` / ``skill`` / ``simulate`` — games."""

from __future__ import annotations

import argparse
import sys
from typing import Any
from pathlib import Path
from ..emit import _emit



def _cmd_arena(args: argparse.Namespace, context: Any) -> int:
    """`nm arena status|approve <build_id>` — the self-improvement arena CLI."""
    from ...agents.arena import Arena

    arena = Arena(context)
    cmd = getattr(args, "arena_command", "") or "status"

    if cmd == "status":
        stats = arena.stats()
        pending = arena.builds("pending")
        payload = {"stats": stats,
                   "pending_builds": [{"id": b.get("id"), "name": b.get("name"),
                                       "created_at": b.get("created_at")}
                                      for b in pending]}
        lines = ["arena status:",
                 f"  cycles: {stats.get('cycles')} · knowledge entries: {stats.get('knowledge')}"]
        builds = stats.get("builds") or {}
        lines.append("  builds: " + ", ".join(
            f"{k}={builds.get(k, 0)}" for k in ("pending", "approved", "denied")))
        for b in pending:
            lines.append(f"  pending: {b.get('id')} — {b.get('name')}")
        if not pending:
            lines.append("  no pending builds")
        _emit(args, payload, "\n".join(lines))
        return 0

    if cmd == "approve":
        build_id = (getattr(args, "build_id", "") or "").strip()
        if not build_id:
            print("arena approve needs a build id — nm arena approve <build_id>",
                  file=sys.stderr)
            return 2
        msg = arena.approve(build_id)
        ok = msg.startswith("approved")
        _emit(args, {"build_id": build_id, "approved": ok, "message": msg}, msg)
        return 0 if ok else 1

    print(f"arena: unknown command {cmd}", file=sys.stderr)
    return 2


def _cmd_trial(args: argparse.Namespace, context: Any) -> int:
    """`nm trial save|list` — the single-account trial flow CLI.

    Saved credentials go straight into the encrypted vault; they are
    never printed back by save (list shows platforms + logins only).
    """
    from ...agents.trial import TrialFlow

    flow = TrialFlow(context)
    cmd = getattr(args, "trial_command", "") or "list"

    if cmd == "list":
        text = flow.list()
        _emit(args, {"accounts": text}, text)
        return 0

    if cmd == "save":
        platform = (getattr(args, "platform", "") or "").strip()
        login = (getattr(args, "login", "") or "").strip()
        password = getattr(args, "password", "") or ""
        note = getattr(args, "note", "") or ""
        if not (platform and login and password):
            print("trial save needs platform, login and password — "
                  "nm trial save <platform> <login> <password> [--note ...]",
                  file=sys.stderr)
            return 2
        try:
            result = flow.save(platform, login, password, note=note)
        except Exception as exc:  # noqa: BLE001
            print(f"trial save: {exc}", file=sys.stderr)
            return 1
        _emit(args, result,
              f"saved trial account for {result.get('platform')} "
              f"(login: {result.get('login')}) — stored encrypted in the vault")
        return 0

    print(f"trial: unknown command {cmd}", file=sys.stderr)
    return 2


def _cmd_skill(args: argparse.Namespace, context: Any) -> int:
    """`nm skill <list|create|show|delete|prune|restore|stats>` — the
    reusable-skills library CLI, over the real ``SkillLibrary``
    (``nomorals/agents/skills.py``) that the reasoning engine, the
    failure ledger, and ``nm improve skill`` all read from.

    NOTE: the stub advertised list/create/run/delete, but ``run`` was
    fictional — library skills are strategy/playbook texts, not
    executables. The verbs here are the library's real ones.
    """
    from ...agents.skills import SkillLibrary

    lib = SkillLibrary(context.db)
    action = getattr(args, "action", "list") or "list"

    if action == "list":
        skills = lib.list(limit=100,
                          include_pruned=bool(getattr(args, "pruned", False)))
        payload = [s.to_dict() for s in skills]
        lines = [f"  {s['name']:<28} [{s['kind']:<10}] "
                 f"uses={s['uses']} ok={s['success_rate']:.0%} "
                 f"{s['description'][:50]}"
                 + (" (pruned)" if s["pruned"] else "")
                 for s in payload]
        _emit(args, {"skills": payload},
              "skills:\n" + "\n".join(lines) if lines else "no skills yet — nm skill create <name>")
        return 0

    if action == "stats":
        stats = lib.stats()
        _emit(args, stats,
              f"skills: {stats['active']} active, {stats['pruned']} pruned, "
              f"{stats['total']} total")
        return 0

    if action == "create":
        name = (getattr(args, "name", "") or "").strip()
        if not name:
            print("skill create needs a name — nm skill create <name> --body \"...\"",
                  file=sys.stderr)
            return 2
        body = getattr(args, "body", "") or ""
        if body.startswith("@"):
            path = body[1:]
            try:
                body = Path(path).read_text(encoding="utf-8")
            except OSError as exc:
                print(f"skill create: cannot read {path}: {exc}", file=sys.stderr)
                return 1
        tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",")
                if t.strip()]
        skill = lib.save(name,
                         kind=getattr(args, "kind", "strategy") or "strategy",
                         body=body,
                         description=getattr(args, "description", "") or "",
                         tags=tags, source="cli")
        _emit(args, skill.to_dict(),
              f"saved skill {skill.name} ({skill.id})")
        return 0

    if action == "prune":
        result = lib.prune()
        _emit(args, result,
              f"pruned {result['count']} skills: " + ", ".join(result["pruned"])
              if result["pruned"] else "prune pass: nothing met the quarantine criteria")
        return 0

    # show / delete / restore need a skill
    name = (getattr(args, "name", "") or "").strip()
    if not name:
        print(f"skill {action} needs a name — nm skill {action} <name>",
              file=sys.stderr)
        return 2
    skill = lib.get_by_name(name)
    if skill is None:
        print(f"skill: no skill named {name!r}", file=sys.stderr)
        return 1

    if action == "show":
        _emit(args, skill.to_dict(),
              f"{skill.name} [{skill.kind}] v{skill.version} "
              f"(uses={skill.uses}, ok={skill.success_rate:.0%}"
              f"{', pruned' if skill.pruned else ''})\n"
              f"{skill.description}\n---\n{skill.body[:2000]}")
        return 0

    if action == "delete":
        ok = lib.delete(skill.id)
        _emit(args, {"name": name, "deleted": ok},
              f"deleted skill {name}" if ok else f"skill {name}: delete failed")
        return 0 if ok else 1

    if action == "restore":
        restored = lib.restore(skill.id)
        _emit(args, {"name": name, "restored": restored is not None},
              f"restored skill {name}" if restored else f"skill {name}: restore failed")
        return 0 if restored else 1

    print(f"skill: unknown action {action}", file=sys.stderr)
    return 2


def _cmd_simulate(args: argparse.Namespace, context: Any) -> int:
    """`nm simulate <dry-run|run|compare|risk>` — the sandbox simulator CLI."""
    from ...agents.simulation import SandboxSimulator, classify_risk

    sim = SandboxSimulator(context)
    action = getattr(args, "simulate_action", "") or ""

    if action == "dry-run":
        command = getattr(args, "cmd", "") or ""
        result = sim.dry_run(command)
        risk = result.get("risk") or {}
        _emit(args, result,
              f"dry-run: {command}\n"
              f"  risk: {risk.get('level')} — {'; '.join(risk.get('reasons', [])) or 'no flags'}\n"
              f"  would run in: {result.get('would_run_in')}"
              + ("\n  ⚠️ high risk — would require --confirm to execute"
                 if result.get("would_gate") else ""))
        return 0

    if action == "run":
        command = getattr(args, "cmd", "") or ""
        if not command.strip():
            print("simulate run needs a command", file=sys.stderr)
            return 2
        result = sim.run(command,
                         confirm=bool(getattr(args, "confirm", False)),
                         timeout=float(getattr(args, "timeout", 120.0) or 120.0))
        payload = result.to_dict()
        out = (result.stdout or "") + (
            f"\n[stderr]\n{result.stderr}" if result.stderr else "")
        _emit(args, payload,
              f"exit {result.exit_code} in {result.seconds:.1f}s "
              f"(risk: {result.risk}, backend: {result.backend})\n{out[:4000]}")
        return 0 if result.ok else 1

    if action == "compare":
        result = sim.compare(getattr(args, "command_a", "") or "",
                             getattr(args, "command_b", "") or "")
        _emit(args, result, f"verdict: {result.get('verdict')}")
        return 0

    if action == "risk":
        report = classify_risk(getattr(args, "cmd", "") or "")
        payload = report.to_dict()
        _emit(args, payload,
              f"risk: {payload['level']} — "
              f"{'; '.join(payload['reasons']) or 'no flags'}")
        return 0

    print(f"simulate: unknown action {action}", file=sys.stderr)
    return 2
