"""Cross-platform relay with attribution (parity census §6 #2).

"Post this everywhere": one command forwards one existing message
to WhatsApp groups + Telegram channels + her channels at once.

Two modes, decided per target by platform, never by guess:

- **forward** — the target lives on the source's platform, so the
  native forward verb runs (WhatsApp ``forward_message``,
  Telegram ``forward_post``).  The platform's own forwarded mark
  rides along; nothing is retyped.
- **repost** — the target lives on another platform, where no
  native forward exists.  The message is re-sent with an
  attribution header naming the source platform and chat, text
  verbatim, media carried when the source row has it on local
  disk.  A repost is labelled a repost in every verdict — it is
  never dressed up as a forward.

The organ takes data, never the runtime (the roster-diff
discipline, same as ``group_broadcast``): adapters, targets and
stops arrive as arguments, so tests ride fakes and the layering
test stays green.  The handler owns arming, journaling and the
command grammar; this module owns planning, preview and the run.

Controls, same class as /obroadcast and the §14 mass tools:
preview-first, dry-run by default, arm-gated live runs, a 3s
pacing floor between targets the owner can raise but never lower,
a per-run cap, every run journaled, one dead target never kills
the rest.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .chat.base import ChatKind, ChatRef, MediaRef

#: Pacing floor between two targets in one run (seconds).  The
#: same floor /obroadcast carries between rooms (Telegram groups
#: allow 20 messages/minute); the owner can raise it, never lower.
PACE_FLOOR_S = 3.0

#: Default pace between targets when the owner never set one.
DEFAULT_PACE_S = 3.0

#: Hard cap on targets in one run, whatever the target list holds
#: — the plan_crosspost discipline.  Split bigger relays into
#: batches instead of blocking one chat turn for ten minutes.
MAX_TARGETS_PER_RUN = 50

#: The warn-first text (owner's standing rule): shown when the
#: tool is armed, in its status, and in every refusal.  Plain
#: words, once, as state — not a lecture stapled to every run.
RELAY_WARN_TEXT = (
    "⚠️ Before this runs live, the risk in plain words: one message "
    "to all your groups and channels at once is exactly what "
    "platform anti-abuse systems look for. Telegram or WhatsApp "
    "may restrict the account it goes out from, or members may "
    "read a bulk post as spam — pacing lowers the temperature, "
    "it does not make it safe. It forwards as you, keeping each "
    "platform's forwarded mark where the platform matches, and "
    "re-posts with attribution where it doesn't. Every run is "
    "journaled with the rooms it reached.")

#: Platform labels for the attribution header.
_PLATFORM_LABELS = {"whatsapp": "WhatsApp", "telegram": "Telegram",
                    "telegram-bot": "Telegram"}


class RelayRefused(RuntimeError):
    """The relay will not run as asked — no source, no targets,
    pacing under the floor, or every target held.  The refusal
    names the reason; nothing was sent."""


@dataclass(frozen=True)
class RelaySource:
    """The message being relayed, resolved by the handler."""

    platform: str
    chat_key: str
    chat_title: str = ""
    chat_kind: str = ChatKind.GROUP
    message_id: str = ""
    text: str = ""
    media: tuple[MediaRef, ...] = ()

    @property
    def ref(self) -> ChatRef:
        return ChatRef.parse(self.chat_key, kind=self.chat_kind)


@dataclass(frozen=True)
class RelayTarget:
    """One destination room, resolved by the handler."""

    platform: str
    key: str
    title: str = ""
    kind: str = "group"

    @property
    def ref(self) -> ChatRef:
        return ChatRef.parse(
            self.key,
            kind=(ChatKind.CHANNEL if self.kind == "channel"
                  else ChatKind.GROUP))


@dataclass
class RelayPlan:
    """A validated relay, ready to preview or run."""

    ok: bool = True
    reason: str = ""
    source: RelaySource | None = None
    targets: list[RelayTarget] = field(default_factory=list)
    modes: dict[str, str] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    media_carried: tuple[MediaRef, ...] = ()
    media_note: str = ""


def attribution_header(source: RelaySource) -> str:
    """The repost header: source platform + chat, nothing else."""
    label = _PLATFORM_LABELS.get(source.platform, source.platform)
    title = (source.chat_title or "").strip()
    return f"↪ forwarded from {label} · {title}" if title \
        else f"↪ forwarded from {label}"


def _verdict_ok(verdict: Any) -> bool:
    if isinstance(verdict, dict):
        return bool(verdict.get("ok"))
    return bool(getattr(verdict, "ok", False))


def _verdict_error(verdict: Any) -> str:
    if isinstance(verdict, dict):
        return str(verdict.get("error") or "")
    return str(getattr(verdict, "error", "") or "")


def _verdict_message_id(verdict: Any) -> str:
    if isinstance(verdict, dict):
        return str(verdict.get("message_id") or "")
    return str(getattr(verdict, "message_id", "") or "")


def plan_relay(source: RelaySource, targets: list[RelayTarget], *,
               pace_seconds: float = DEFAULT_PACE_S,
               stopped_platforms: Any = ()) -> RelayPlan:
    """Validate a relay into a plan.  Never sends.

    Hard refusals raise :class:`RelayRefused` (nothing to plan);
    per-target problems land in ``skipped`` and the plan stays
    runnable for the targets that remain.
    """
    if not (source.message_id or "").strip():
        raise RelayRefused(
            "relay needs a message to forward — reply to it, or say "
            "from <chat> <message_id>.")
    if not targets:
        raise RelayRefused(
            "no target rooms — name one, or say all / groups / "
            "channels. Nothing was sent.")
    if float(pace_seconds) < PACE_FLOOR_S:
        raise RelayRefused(
            f"pacing floor is {PACE_FLOOR_S:.0f}s between rooms — "
            f"{pace_seconds:g}s is below it. The floor is the safety "
            "control, not a suggestion; raise it freely, it cannot "
            "go under.")
    if len(targets) > MAX_TARGETS_PER_RUN:
        raise RelayRefused(
            f"{len(targets)} rooms exceeds the per-run cap "
            f"({MAX_TARGETS_PER_RUN}) — split into batches.")
    stopped = {(p or "").strip().lower()
               for p in (stopped_platforms or ())}
    seen: set[str] = set()
    kept: list[RelayTarget] = []
    modes: dict[str, str] = {}
    skipped: list[dict[str, Any]] = []
    for target in targets:
        if target.key in seen:
            continue
        seen.add(target.key)
        if (target.platform or "").strip().lower() in stopped:
            skipped.append({"key": target.key, "title": target.title,
                            "platform": target.platform,
                            "reason": "halted by your standing stop"})
            continue
        if target.platform != source.platform \
                and not (source.text or "").strip():
            skipped.append({"key": target.key, "title": target.title,
                            "platform": target.platform,
                            "reason": "couldn't read the message text — "
                            "a repost needs it; the native forwards "
                            "still run"})
            continue
        modes[target.key] = ("forward" if target.platform
                             == source.platform else "repost")
        kept.append(target)
    if not kept:
        reasons = "; ".join(
            f"{row['key']}: {row['reason']}" for row in skipped)
        raise RelayRefused(
            f"every target is held — {reasons}. Nothing was sent.")
    carried: list[MediaRef] = []
    dropped: list[str] = []
    for ref in (source.media or ()):
        path = str(getattr(ref, "path", "") or "")
        if path and os.path.exists(path):
            carried.append(ref)
        else:
            dropped.append(str(getattr(ref, "name", "") or path
                               or "media"))
    media_note = ""
    if dropped:
        media_note = ("media not carried (no local copy): "
                      + ", ".join(dropped))
    return RelayPlan(ok=True, source=source, targets=kept, modes=modes,
                     skipped=skipped, media_carried=tuple(carried),
                     media_note=media_note)


def preview_relay(plan: RelayPlan) -> str:
    """Exactly what a live run would do — the approval surface."""
    source = plan.source
    lines = ["📣 relay dry-run — nothing sent.", "",
             f"source: {source.chat_title or source.chat_key} "
             f"({source.platform}, {source.message_id})",
             f"would reach {len(plan.targets)} room(s):"]
    for target in plan.targets:
        mode = plan.modes.get(target.key, "?")
        lines.append(f"  · {target.title or target.key} "
                     f"({target.key}) — {mode}")
    for row in plan.skipped:
        lines.append(f"  · {row.get('title') or row.get('key')} — held: "
                     f"{row.get('reason')}")
    if plan.media_note:
        lines.append(plan.media_note)
    lines.append("say “| live” to send it — arm first: /relay arm")
    return "\n".join(lines)


def run_relay(*, plan: RelayPlan, adapters: dict[str, Any],
              sleep: Callable[[float], None] = time.sleep,
              pace_seconds: float = DEFAULT_PACE_S) -> dict[str, Any]:
    """Fan the message out across the planned targets.

    One delivery per target, paced; one dead target never kills
    the rest — every target is its own result row.  Returns the
    run record, ready to journal and to render as the run report.
    """
    if float(pace_seconds) < PACE_FLOOR_S:
        raise RelayRefused(
            f"pacing floor is {PACE_FLOOR_S:.0f}s between rooms — "
            f"{pace_seconds:g}s is below it.")
    source = plan.source
    header = attribution_header(source)
    results: list[dict[str, Any]] = []
    sent = failed = 0
    for index, target in enumerate(plan.targets):
        if index:
            sleep(float(pace_seconds))
        row: dict[str, Any] = {"target": target.key,
                               "title": target.title,
                               "platform": target.platform,
                               "mode": plan.modes.get(target.key, "?"),
                               "ok": False, "error": "",
                               "message_id": "", "media_sent": 0,
                               "media_error": ""}
        adapter = (adapters or {}).get(target.platform)
        if adapter is None:
            row["error"] = (f"no live {target.platform} surface here")
            results.append(row)
            failed += 1
            continue
        try:
            if row["mode"] == "forward":
                if target.platform == "whatsapp":
                    fn = getattr(adapter, "forward_message", None)
                else:
                    fn = getattr(adapter, "forward_post", None)
                if not callable(fn):
                    row["error"] = (f"no forward verb on "
                                    f"{target.platform}")
                    results.append(row)
                    failed += 1
                    continue
                verdict = fn(source.ref, source.message_id,
                             target.ref)
                row["ok"] = _verdict_ok(verdict)
                row["error"] = _verdict_error(verdict)
                row["message_id"] = _verdict_message_id(verdict)
            else:
                fn = getattr(adapter, "send", None)
                if not callable(fn):
                    row["error"] = (f"no send verb on "
                                    f"{target.platform}")
                    results.append(row)
                    failed += 1
                    continue
                text = f"{header}\n\n{source.text}"
                verdict = fn(target.ref, text)
                row["ok"] = _verdict_ok(verdict)
                row["error"] = _verdict_error(verdict)
                row["message_id"] = _verdict_message_id(verdict)
                if row["ok"]:
                    media_fn = getattr(adapter, "send_media", None)
                    for ref in plan.media_carried:
                        if not callable(media_fn):
                            row["media_error"] = (
                                f"no send_media on {target.platform}")
                            break
                        mverdict = media_fn(target.ref, ref,
                                            caption=header)
                        if _verdict_ok(mverdict):
                            row["media_sent"] += 1
                        else:
                            row["media_error"] = (
                                _verdict_error(mverdict)
                                or "the surface refused the media")
                            break
        except Exception as exc:  # noqa: BLE001 - one target's
            # failure is a row, never the run's death.
            row["ok"] = False
            row["error"] = str(exc)
        results.append(row)
        if row["ok"]:
            sent += 1
        else:
            failed += 1
    return {"tool": "relay", "live": True,
            "source": {"platform": source.platform,
                       "chat_key": source.chat_key,
                       "chat_title": source.chat_title,
                       "message_id": source.message_id,
                       "text": source.text},
            "targets": [t.key for t in plan.targets],
            "skipped": list(plan.skipped),
            "sent": sent, "failed": failed, "aborted": False,
            "results": results}
