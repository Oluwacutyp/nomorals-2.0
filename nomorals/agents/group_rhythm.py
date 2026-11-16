"""Group rhythm on its own clock (owner live reports 2026-11-16).

Defect A: the richest-balance record, the Market Mile district
event and the weekly gift proposal were evaluated inside
``handle_message`` only — in a quiet room they piled up and all
landed at once on the next command. This module is the clock:
one durable scheduler job (``__group_rhythm_tick__``, every 5m,
same shape as the roster-diff tick) that runs the rhythm +
estate cycle across claimed groups and sends what is due,
without waiting for a message. The per-turn drain stays as the
fast path for rooms with traffic; the job is the clock for
quiet rooms.

Defect B lives here too, at the tick level: the tick marker
(``system:rhythm-tick:<group>:<bucket>`` on the shared claims
ledger) is claimed bot-first — an owner-surface runtime defers
and observes the bot's marker before running a group, so group
system output is posted by the BOT identity whenever the bot is
there, with the userbot as fallback when the bot is absent.
The same defer-and-observe rule is applied to the turn path in
``runtime_group._claim_system_send``.

Safety notes, all deliberate:
- The tick evaluates a group ONLY when it can deliver there
  (an adapter for the group's platform is attached and a send
  callable is available). A notice the cycles surface is never
  consumed without a send — the record would be baselined and
  the announcement lost. When delivery is unavailable the group
  is left for the turn path.
- The cycles are self-gating (rhythm 60s throttle, estate
  day/anchor gates), so a tick and a turn evaluating the same
  group cannot double-fire an event; the shared notice claim
  (identical key to the turn path) dedupes the post itself.
- Silent hours hold into the morning digest exactly like the
  turn path: notices are held, never dropped, and released by
  the next cycle once the quiet ends.
- The proposal feed is drained by the tick (the gift proposal
  is one of the three named announcements). Its render is
  byte-identical to the turn's render (same Card, same
  ``surface._card_render``, same button spec — the runtime's
  ``actionable`` wrapper leaves the text unchanged and the
  dialect of a group-key prefix equals the dialect of the
  chat platform), so the shared ``proposal`` claim dedupes the
  two paths. Pinned by test_rhythm_autonomy_wiring.
- The conflict feed and the board cycle stay turn-only (named
  remainder): drops and board publishes are game/scheduled
  posts with their own timing semantics, not the three named
  announcements.
"""
from __future__ import annotations

import contextlib
import hashlib
import time
from typing import Any

RHYTHM_JOB = "group rhythm tick"
RHYTHM_SPEC = "every 5m"
RHYTHM_TOOL = "__group_rhythm_tick__"

#: How long an owner-surface runtime waits for the bot's claim
#: (tick marker, or a system-send claim) before sending itself.
#: The bot claims within ~1s of the same event in the live shape;
#: the window covers realistic skew, and the fallback still sends.
BOT_WAIT_S = 5.0
#: After one full wait with no bot claim, an owner surface stops
#: waiting for the rest of this pass — the bot is down, not slow.
BOT_ABSENT_COOLDOWN_S = 120.0
#: Tick-marker bucket width (seconds): two runtimes ticking the
#: same group inside one bucket arbitrate on the marker.
TICK_BUCKET_S = 300

_OWN_ID_ATTRS = ("_bot_id", "_me_id", "user_jid", "user_lid")


def _own_ids(adapter: Any) -> list[str]:
    ids: list[str] = []
    if adapter is None:
        return ids
    for attr in _OWN_ID_ATTRS:
        value = getattr(adapter, attr, None)
        if value and str(value) not in ids:
            ids.append(str(value))
    return ids


def _run_identity(adapters_by_platform: Any,
                  registry: Any) -> tuple[str, bool]:
    """(claim_surface, is_owner) for THIS runtime: identity is a
    property of the runtime, not the room. The claim surface is
    the first adapter's own id (``platform:<id>``); it only has
    to be stable per runtime and distinct across runtimes — the
    notice/marker keys already carry the room and the text.
    ``is_owner`` is False when any adapter proves bot-ness (a
    ``_bot_id``) or the registry proves none of our ids is the
    owner's; otherwise True (fail to defer — a slow owner never
    double-sends, and an unseeded registry must never mistake
    the owner for the bot)."""
    own_surface = ""
    all_ids: list[tuple[str, str]] = []
    for platform, adapter in (adapters_by_platform or {}).items():
        ids = _own_ids(adapter)
        for item in ids:
            all_ids.append((str(platform), item))
        if ids and not own_surface:
            own_surface = f"{platform}:{ids[0]}"
        if getattr(adapter, "_bot_id", None):
            return own_surface, False
    if registry is not None:
        for platform, item in all_ids:
            try:
                known = registry.identities(platform) or []
            except Exception:  # noqa: BLE001 - unproven: keep
                known = []  # deferring
            if not known:
                continue
            try:
                if not registry.is_owner(platform, item):
                    return own_surface, False
            except Exception:  # noqa: BLE001 - unproven: defer
                continue
    return own_surface, True


def _open_claim_ledger() -> Any:
    """The shared claims ledger (same file the runtimes write),
    or None when it cannot be opened — the caller fails open."""
    try:
        from .partner.command_claims import (
            DurableConversationClaimLedger,
            open_shared_claims_db,
        )

        db = open_shared_claims_db()
        if db is None:
            return None
        return DurableConversationClaimLedger(db)
    except Exception:  # noqa: BLE001 - no ledger: fail open
        return None


def _claimed_by(ledger: Any, key: str) -> str:
    """The surface holding a live claim on ``key``, or \"\" —
    getattr-guarded for foreign ledgers without the probe."""
    probe = getattr(ledger, "claimed_by", None)
    if probe is None:
        return ""
    try:
        return str(probe(key) or "")
    except Exception:  # noqa: BLE001 - a probe never fails a tick
        return ""


def _claim(ledger: Any, key: str, surface: str) -> bool:
    try:
        return bool(ledger.claim(key, surface))
    except Exception:  # noqa: BLE001 - a gate fault fails open
        return True


def _notice_key(chat_key: str, what: str, text: str) -> str:
    """The exact claim key the turn path uses for a system send,
    so the two paths dedupe each other."""
    return ("system:" + str(chat_key or "") + ":" + str(what or "")
            + ":" + hashlib.sha1(str(text or "").encode("utf-8")
                                 ).hexdigest()[:16])


def _wait_for_bot_claim(ledger: Any, key: str, own: str, *,
                        wait_s: float) -> bool:
    """Defer-and-observe: poll for a sibling's live claim on
    ``key`` for up to ``wait_s``. True when the bot claimed it
    (this runtime stands down) or this runtime already holds it;
    False when the window expired with no claim (fallback)."""
    deadline = time.monotonic() + max(0.0, float(wait_s))
    while True:
        holder = _claimed_by(ledger, key)
        if holder and holder != own:
            return True
        if holder == own:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def render_proposal_announcement(surface: Any, group_key: str,
                                 item: dict[str, Any]
                                 ) -> tuple[str, Any, str]:
    """Render one queued gift-proposal announcement: (text,
    buttons, parse_mode). Byte-identical to the turn's
    ``_proposal_feed_render`` for the same item — same Card, same
    ``surface._card_render``, same button spec. The runtime's
    ``actionable`` wrapper leaves the text unchanged, and the
    dialect of a group-key prefix equals the dialect of the chat
    platform, so the shared ``proposal`` claim dedupes the two
    paths instead of double-posting."""
    from ..social.chat.expression import buttons_from_spec
    from ..social.chat.typography import Card, sparks

    settings = surface.store.settings(group_key)
    card = Card("announcement", title="Gift proposal",
                icon="🎁")
    card.line(f"**{sparks(int(item.get('amount') or 0))}** to "
              f"{item.get('label') or 'a member'} — the week's "
              f"top reviewed contributor "
              f"({int(item.get('count') or 0)} counted).")
    card.note("admins decide — a tap settles it in this room")
    buttons = buttons_from_spec(
        [[("approve", "Approve",
           f"approve {item.get('proposal')}"),
          ("deny", "Deny", f"deny {item.get('proposal')}")]],
        settings=settings)
    text = surface._card_render(group_key, card)
    platform = str(group_key).split(":", 1)[0]
    parse_mode = ("HTML" if platform.startswith("telegram")
                  and "<" in text else "")
    return text, buttons, parse_mode


def ensure_group_rhythm_job(scheduler: Any, *,
                            spec: str = RHYTHM_SPEC) -> Any:
    """Register the durable group-rhythm tick job. Idempotent —
    mirrors ``ensure_group_roster_diff_job``: an existing job at
    the requested cadence is returned as-is, a stale one is
    replaced. Best-effort: never raises."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == RHYTHM_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(RHYTHM_JOB, spec, "tool",
                             {"tool": RHYTHM_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def run_group_rhythm_tick_once(*, context: Any = None,
                               surface: Any = None,
                               workspace_dir: Any = None,
                               now: float | None = None,
                               send: Any = None,
                               adapters: Any = None,
                               claim: Any = None,
                               bot_wait_s: float = BOT_WAIT_S,
                               is_owner_surface: Any = None
                               ) -> dict[str, Any]:
    """One scheduled rhythm pass: every CLAIMED group whose cycle
    is due gets its rhythm + estate cycle run and its due
    announcements posted — no message required. Routine and
    silent; one bad group never kills the pass.

    ``send`` is ``(platform, chat_ref, text, *, buttons,
    parse_mode) -> result`` (a bool, or a result with ``ok`` /
    ``message_id``); when absent it is built from the context's
    gateway. ``adapters`` is the gateway's adapter list
    (``(platform, adapter)`` pairs, as the roster-diff tick
    receives them) — when given, a group is evaluated only when
    its platform has an attached adapter. ``claim`` is the shared
    claims ledger (``claim`` + ``claimed_by``); when absent it is
    opened from the shared file, and when that fails the tick
    fails open (sends unclaimed, like the turn path).
    ``is_owner_surface`` is a bool; when None it is derived from
    the adapters + owner registry.
    """
    from ..social.chat.base import ChatRef

    now = now if now is not None else time.time()
    if surface is None:
        ws = workspace_dir
        if ws is None and context is not None:
            ws = getattr(getattr(context, "settings", None),
                         "workspace_dir", ".") or "."
        from ..social.chat.group_surface import default_surface

        surface = default_surface(ws or ".")
    if send is None:
        send = _default_sender(context)
    if claim is None:
        claim = _open_claim_ledger()
    registry = None
    ws = workspace_dir
    if ws is None and context is not None:
        ws = getattr(getattr(context, "settings", None),
                     "workspace_dir", ".") or "."
    if ws is not None:
        with contextlib.suppress(Exception):
            from ..social.chat.owner_system import default_registry

            registry = default_registry(ws)
    adapters_by_platform = dict(adapters or [])
    # Identity is run-level, not per-room: the claim surface is
    # this runtime's own id, and the deferral is this runtime
    # being the owner's surface. An explicit ``is_owner_surface``
    # overrides the derived deferral only; the claim surface
    # still comes from the adapters.
    own, derived_owner = _run_identity(adapters_by_platform,
                                       registry)
    owner_side = (is_owner_surface
                  if isinstance(is_owner_surface, bool)
                  else derived_owner)
    summary: dict[str, Any] = {"ok": True, "groups": 0,
                               "posted": 0, "held": 0,
                               "skipped": 0, "failed": []}
    bot_absent_until = 0.0

    for key in sorted(surface.store.data.keys()):
        try:
            if surface.is_channel(key):
                continue
        except Exception:  # noqa: BLE001 - unreadable: skip
            continue
        try:
            if not surface.claim_state(key)["claimed"]:
                summary["skipped"] += 1
                continue
        except Exception as exc:  # noqa: BLE001 - one bad group
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
            continue
        platform = str(key).split(":", 1)[0]
        adapter = adapters_by_platform.get(platform)
        if send is None:
            # Cannot deliver here: leave the cycle for the turn
            # path, which always has a live adapter for the room.
            summary["skipped"] += 1
            continue
        if adapters is not None and adapter is None:
            # The caller constrained delivery to these adapters
            # and this room's platform has none attached: leave
            # the cycle for the turn path.
            summary["skipped"] += 1
            continue
        # Tick marker (bot-first): the owner surface defers and
        # observes the bot's marker; the bot claims at once.
        if claim is not None:
            bucket = int(now // TICK_BUCKET_S)
            marker = (f"system:rhythm-tick:{key}:{bucket}")
            holder = _claimed_by(claim, marker)
            if holder and holder != own:
                summary["skipped"] += 1
                continue
            if holder == own:
                pass  # own live marker: re-run is a no-op
            elif owner_side and time.monotonic() < bot_absent_until:
                pass  # bot already proven down this pass
            elif owner_side:
                if _wait_for_bot_claim(claim, marker, own,
                                      wait_s=bot_wait_s):
                    summary["skipped"] += 1
                    continue
                bot_absent_until = (time.monotonic()
                                    + BOT_ABSENT_COOLDOWN_S)
            if not _claim(claim, marker, own):
                summary["skipped"] += 1
                continue
        try:
            summary["groups"] += 1
            notices: list[str] = []
            notices.extend(surface.estate_cycle(key, now=now))
            notices.extend(surface.rhythm_cycle(key, now=now))
            if notices and surface.rhythm_silent(key, now=now):
                for held_text in notices:
                    surface.rhythm_hold(key, "cycle", held_text,
                                        now=now)
                summary["held"] += len(notices)
                notices = []
            chat = ChatRef.parse(key, kind="group")
            for text in notices:
                if claim is not None and not _claim(
                        claim, _notice_key(key, "notice", text), own):
                    continue
                if _sent_ok(send(
                        platform, chat, text,
                        parse_mode=("HTML" if platform.startswith(
                            "telegram") and "<" in text else ""))):
                    summary["posted"] += 1
            _drain_proposal_feed(surface, key, platform, chat,
                                 send, claim, own, now, summary)
            surface.store.save()
        except Exception as exc:  # noqa: BLE001 - one bad group
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
    return summary


def _default_sender(context: Any) -> Any:
    """The gateway sender for the tick, or None when no gateway
    is bound — the tick then evaluates nothing (a notice must
    never be consumed without a send)."""
    if context is None:
        return None
    try:
        from ..agents.notifier import resolve_gateway

        gateway = resolve_gateway(context)
        if gateway is None:
            return None

        def _send(platform: str, chat: Any, text: str, *,
                  buttons: Any = None, parse_mode: str = "") -> Any:
            return gateway.send(platform, chat, str(text),
                                buttons=buttons,
                                parse_mode=parse_mode)

        return _send
    except Exception:  # noqa: BLE001 - no gateway: no sends
        return None


def _sent_ok(result: Any) -> bool:
    """True when a send result means delivered: a bare bool as
    given, else the result's ``ok`` flag. A result object is
    never trusted for truthiness on its own."""
    if isinstance(result, bool):
        return result
    return bool(getattr(result, "ok", False))


def _drain_proposal_feed(surface: Any, group_key: str,
                         platform: str, chat: Any, send: Any,
                         claim: Any, own: str, now: float,
                         summary: dict[str, Any]) -> None:
    """Drain the queued governance announcements (the weekly gift
    proposal) for one group — the turn path's
    ``_post_proposal_feed`` with the same gates: silent hours and
    §33 keep the queue; the proposal itself is state either way.
    A posted card's message id is recorded (``proposal_msgs``)
    so the decision can later edit it in place, as the turn
    path does."""
    extras = surface._extras(group_key)
    if not (extras.get("proposal_feed") or []):
        return
    if surface.rhythm_silent(group_key, now=now):
        return
    if not surface.system_output_allowed(
            group_key, via_platform=platform):
        return
    items = surface.proposal_feed_take(group_key)
    for item in items:
        try:
            text, buttons, parse_mode = render_proposal_announcement(
                surface, group_key, item)
            if claim is not None and not _claim(
                    claim, _notice_key(group_key, "proposal", text),
                    own):
                continue
            result = send(platform, chat, text, buttons=buttons,
                          parse_mode=parse_mode)
            if not _sent_ok(result):
                continue
            summary["posted"] += 1
            mid = getattr(result, "message_id", "") or ""
            if mid:
                msgs = dict(extras.get("proposal_msgs") or {})
                msgs[str(item.get("proposal"))] = str(mid)
                extras["proposal_msgs"] = msgs
        except Exception:  # noqa: BLE001 - one bad item never
            continue  # breaks the tick
