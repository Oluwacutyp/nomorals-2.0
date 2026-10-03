"""``nm wisdom`` — WisdomKeeper: esoteric corpus, history, practice."""
from __future__ import annotations

import json
import sys
from typing import Any


def _keeper(context: Any):
    from ...wisdom import WisdomKeeper
    return WisdomKeeper(context)


def _cmd_wisdom(args: Any, context: Any) -> int:
    """Route ``nm wisdom <verb>``."""
    words = list(getattr(args, "task", None) or [])
    if not words:
        print("usage: nm wisdom status [--json]\n"
              "       nm wisdom ask <query> [--limit N] [--json]\n"
              "       nm wisdom ingest <slug> | --all\n"
              "       nm wisdom seed\n"
              "       nm wisdom search <query> [--limit N] [--json]\n"
              "       nm wisdom timeline [--tradition T] [--start Y] [--end Y] [--json]\n"
              "       nm wisdom compare <topic> [--json]\n"
              "       nm wisdom practice list [--json]\n"
              "       nm wisdom practice <session-id> [--rounds N] [--chat]",
              file=sys.stderr)
        return 2
    verb = words[0]
    if verb == "status":
        return _wisdom_status(args, context)
    if verb == "ask":
        return _wisdom_ask(args, context, words[1:])
    if verb == "ingest":
        return _wisdom_ingest(args, context, words[1:])
    if verb == "seed":
        return _wisdom_seed(args, context)
    if verb == "search":
        return _wisdom_search(args, context, words[1:])
    if verb == "timeline":
        return _wisdom_timeline(args, context, words[1:])
    if verb == "compare":
        return _wisdom_compare(args, context, words[1:])
    if verb == "practice":
        return _wisdom_practice(args, context, words[1:])
    print(f"unknown wisdom verb: {verb}", file=sys.stderr)
    return 2


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _wisdom_status(args: Any, context: Any) -> int:
    k = _keeper(context)
    st = k.status()
    if _as_json(args):
        print(json.dumps(st, indent=2, default=str))
        return 0
    c = st["corpus"]
    print(f"corpus: {c['ingested']}/{c['texts']} texts ingested")
    for trad, n in sorted(c["by_tradition"].items()):
        print(f"  {trad}: {n}")
    return 0


def _wisdom_ask(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm wisdom ask <query> [--limit N] [--json]",
              file=sys.stderr)
        return 2
    k = _keeper(context)
    limit = getattr(args, "limit", 0) or 5
    try:
        ans = k.ask(" ".join(rest), top=limit)
    except Exception as exc:  # noqa: BLE001 - fail fast with the real error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(ans.to_dict(), indent=2, default=str))
        return 0
    print(ans.synthesis)
    print()
    for p in ans.passages:
        print(f"[{p.work} — {p.section}]")
        print(f"  {p.snippet.strip()}")
        src = p.url or "(no url)"
        print(f"  source: {src} [{p.canon_status}]")
        print()
    return 0


def _wisdom_practice_chat(args: Any, context: Any, session_id: str) -> int:
    """Deliver a practice session to chat instead of pacing it in the
    terminal.

    The pacer thread runs here and every message goes out through the
    Notifier on the live gateway; the CLI blocks until the session
    closes so delivery completes. The journal prompt goes out at the
    end and the journal-await marker is persisted, so a reply in chat
    is journaled by the live runtime.

    Fail fast: with no live gateway there is nothing to deliver to —
    the owner should start the session from their DMs instead.
    """
    from ...agents.notifier import resolve_gateway
    from ...wisdom import PracticeError
    from ...wisdom import chat_session as _chat_session

    if resolve_gateway(context) is None:
        print(f"error: no live chat gateway — start this from your DMs "
              f"with /wisdom practice {session_id}", file=sys.stderr)
        return 1
    settings = getattr(context, "settings", None)
    partner = getattr(settings, "partner", None) if settings else None
    owner_chats = [c.strip() for c in
                   str(getattr(partner, "owner_chats", "") or "").split(",")
                   if c.strip()]
    if not owner_chats:
        print("error: no owner chat configured "
              "(settings.partner.owner_chats) — nothing to deliver to",
              file=sys.stderr)
        return 1
    chat_key = owner_chats[0]
    platform = chat_key.partition(":")[0]
    mgr = _chat_session.WisdomChatManager(context)
    try:
        ack = mgr.start_session(chat_key, session_id, platform=platform)
    except PracticeError as exc:
        # unknown session id — the message lists what's available
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(ack)
    session = mgr.active_session(chat_key)
    if session is not None:
        # block so the pacer thread (daemon) finishes delivery; the
        # journal prompt is sent by the session itself before it ends.
        session.join(timeout=session.estimated_seconds() + 120)
    print("session delivered to chat — reply there and it will be journaled.")
    return 0


def _wisdom_seed(args: Any, context: Any) -> int:
    k = _keeper(context)
    try:
        n = k.corpus.seed()
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"seeded {n} texts into the manifest")
    return 0


def _wisdom_ingest(args: Any, context: Any, rest: list[str]) -> int:
    from ...wisdom import ArchiveIngestor
    ing = ArchiveIngestor(context)
    if rest and rest[0] == "--all":
        entries = ing.corpus.list()
        if not entries:
            print("manifest is empty — nothing to ingest", file=sys.stderr)
            return 1
        ok, failed = 0, 0
        for e in entries:
            try:
                ing.ingest_entry(e)
                print(f"  ok: {e.slug}")
                ok += 1
            except Exception as exc:  # noqa: BLE001 - report, continue
                print(f"  FAIL: {e.slug}: {exc}", file=sys.stderr)
                failed += 1
        print(f"ingested {ok}, failed {failed}")
        return 1 if failed else 0
    if not rest:
        print("usage: nm wisdom ingest <slug> | --all", file=sys.stderr)
        return 2
    try:
        entry = ing.ingest_entry(ing.corpus.get(rest[0]))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"ingested: {entry.slug} ({entry.title})")
    return 0


def _wisdom_search(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm wisdom search <query> [--limit N] [--json]",
              file=sys.stderr)
        return 2
    from ...wisdom import ArchiveIngestor
    ing = ArchiveIngestor(context)
    limit = getattr(args, "limit", 0) or 10
    try:
        results = ing.search(" ".join(rest), max_results=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(results, indent=2, default=str))
        return 0
    for r in results:
        print(f"- {r.get('title', r.get('identifier', '?'))}")
        print(f"  {r.get('url', '')}")
    return 0


def _wisdom_timeline(args: Any, context: Any, rest: list[str]) -> int:
    k = _keeper(context)
    tradition = getattr(args, "tradition", "") or ""
    start = getattr(args, "start", -3000)
    end = getattr(args, "end", 2100)
    try:
        events = k.history.timeline(tradition, start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(events, indent=2, default=str))
        return 0
    for e in events:
        yr = f"{e['start']}" if e['start'] == e['end'] else \
            f"{e['start']}…{e['end']}"
        print(f"{yr}: {e['title']} [{e['tradition']}]")
    return 0


def _wisdom_compare(args: Any, context: Any, rest: list[str]) -> int:
    if not rest:
        print("usage: nm wisdom compare <topic> [--json]", file=sys.stderr)
        return 2
    k = _keeper(context)
    try:
        out = k.history.compare(" ".join(rest), corpus=k)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(out, indent=2, default=str))
        return 0
    print(f"topic: {out['topic']}")
    for p in out["passages"]:
        print(f"[{p.get('work', '?')} — {p.get('canon_status', '')}]")
        print(f"  {p.get('snippet', '')[:200]}")
    print("-- timeline context --")
    for e in out["timeline_context"][:10]:
        print(f"  {e['start']}: {e['title']}")
    return 0


def _wisdom_practice(args: Any, context: Any, rest: list[str]) -> int:
    k = _keeper(context)
    if not rest or rest[0] == "list":
        sessions = k.practice.list_sessions()
        if _as_json(args):
            print(json.dumps(sessions, indent=2, default=str))
            return 0
        for s in sessions:
            tag = "beginner" if s["beginner"] else "advanced"
            print(f"{s['id']}: {s['name']} ({s['total_seconds']}s, {tag})")
        return 0
    session_id = rest[0]
    if getattr(args, "chat", False):
        return _wisdom_practice_chat(args, context, session_id)
    rounds = getattr(args, "rounds", 0) or None
    if _as_json(args):
        print(json.dumps({"session_id": session_id,
                          "note": "interactive pacing needs a terminal; "
                                  "use phases_for_chat via the API"},
                         indent=2))
        return 0
    try:
        result = k.practice.run(session_id, rounds=rounds)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"done: {result['phases_done']} phases, "
          f"completed={result['completed']}")
    return 0
