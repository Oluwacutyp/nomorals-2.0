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
              "       nm wisdom read <work> [chapter] [page]\n"
              "       nm wisdom traditions [tradition]\n"
              "       nm wisdom seeds\n"
              "       nm wisdom ingest <slug> | --all\n"
              "       nm wisdom seed\n"
              "       nm wisdom search <query> [--limit N] [--json]\n"
              "       nm wisdom timeline [tradition] [n] [--start Y] [--end Y] [--json]\n"
              "       nm wisdom compare <topic> [--json]\n"
              "       nm wisdom practice list [--json]\n"
              "       nm wisdom practice <session-id> [--rounds N] [--chat]\n"
              "       nm wisdom <tradition>  (bare tradition lists its works)",
              file=sys.stderr)
        return 2
    verb = words[0]
    if verb == "status":
        return _wisdom_status(args, context)
    if verb == "ask":
        return _wisdom_ask(args, context, words[1:])
    if verb == "read":
        return _wisdom_read(args, context, words[1:])
    if verb == "traditions":
        return _wisdom_traditions(args, context, words[1:])
    if verb == "seeds":
        return _wisdom_seeds(args, context)
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
    # Bare tradition: `/wisdom christian-gnostic` lists that tradition's
    # works instead of "unknown verb" (case-insensitive). Only when the
    # whole tail is the tradition name — anything longer is a question.
    if len(words) == 1 and _is_tradition(context, verb):
        return _wisdom_traditions(args, context, words)
    print(f"unknown wisdom verb: {verb}", file=sys.stderr)
    return 2


def _is_tradition(context: Any, word: str) -> bool:
    from ...wisdom.reading import normalize_tradition
    try:
        keeper = _keeper(context)
        known = {normalize_tradition(e.tradition)
                 for e in keeper.corpus.list()}
        known |= {normalize_tradition(t)
                  for t in keeper.history.traditions()}
    except Exception:  # noqa: BLE001 - fall back to the static table
        from ...wisdom.reading import TRADITION_INFO
        known = set(TRADITION_INFO)
    return normalize_tradition(word) in known


def _as_json(args: Any) -> bool:
    return bool(getattr(args, "json", False))


def _wisdom_status(args: Any, context: Any) -> int:
    k = _keeper(context)
    full = k.status()
    if _as_json(args):
        print(json.dumps(full, indent=2, default=str))
        return 0
    c = full["corpus"]
    print(f"corpus: {c['ingested']}/{c['texts']} texts ingested")
    print("A \"tradition\" is a family of texts + teachings that grew "
          "together around shared scriptures and practices.")
    for trad in k.corpus.traditions_detail():
        print(f"\n{trad['label']} ({trad['tradition']}): "
              f"{trad['ingested']}/{trad['total']} ingested")
        print(f"  {trad['description']}")
        for w in trad["works"]:
            state = (f"ingested {w['ingested_date']}, "
                     f"{w['chars']:,} chars, {w['pages']} pages"
                     if w["ingested"] else "PENDING — not ingested yet")
            print(f"  {'✓' if w['ingested'] else '○'} {w['title']} "
                  f"({w['slug']}) — seed: {w['seed']}")
            print(f"      source: {w['source_url']}")
            print(f"      {state} — read: {w['read_command']}")
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
        if p.slug:
            ch = f" {p.chapter_number}" if p.chapter_number else ""
            print(f"  read full: /wisdom read {p.slug}{ch}")
        print()
    return 0


def _wisdom_read(args: Any, context: Any, rest: list[str]) -> int:
    """``nm wisdom read <work> [chapter] [page]`` — read the actual text."""
    from ...wisdom.reading import parse_read_args
    if not rest:
        print("usage: nm wisdom read <work> [chapter] [page]\n"
              "       e.g. nm wisdom read gospel-of-thomas\n"
              "            nm wisdom read gospel-of-thomas 2  (page 2)\n"
              "            nm wisdom read gospel-of-thomas 2 1  (chapter 2, page 1)",
              file=sys.stderr)
        return 2
    k = _keeper(context)
    # The work query may be several words ("Gospel of Thomas"); resolve
    # the longest leading prefix that names a work, the rest is
    # chapter/page args.
    for split in range(len(rest), 0, -1):
        work_q = " ".join(rest[:split])
        tail = rest[split:]
        try:
            chapter, page = parse_read_args(tail)
            got = k.corpus.read_work(work_q, chapter=chapter, page=page)
            break
        except Exception as exc:  # noqa: BLE001 - try a shorter prefix
            last_exc = exc
            if split == 1:
                print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
                return 1
            continue
    if _as_json(args):
        print(json.dumps(got, indent=2, default=str))
        return 0
    head = f"{got['title']}"
    if got["chapter"]:
        head += f" — chapter {got['chapter']}"
        if got["chapter_title"]:
            head += f": {got['chapter_title']}"
    print(f"📖 {head} (page {got['page']}/{got['total_pages']}, "
          f"{got['total_chars']:,} chars total)")
    print(f"   source: {got['source_url']}  ·  seed: {got['seed']}")
    print("─" * 60)
    print(got["text"])
    print("─" * 60)
    if got["has_prev"]:
        print(f"← prev: {got['prev_command']}")
    if got["has_next"]:
        print(f"next → {got['next_command']}")
    return 0


def _wisdom_traditions(args: Any, context: Any, rest: list[str]) -> int:
    k = _keeper(context)
    wanted = " ".join(rest) if rest else ""
    try:
        detail = k.corpus.traditions_detail(wanted)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(detail, indent=2, default=str))
        return 0
    if not detail:
        print("no traditions registered yet — run /wisdom seed")
        return 0
    print("A \"tradition\" is a family of texts + teachings that grew "
          "together around shared scriptures, teachers, and practices. "
          "Each work below names its seed (where it came from) and its "
          "source URL, and opens with /wisdom read <slug>.")
    for trad in detail:
        print(f"\n{trad['label']} ({trad['tradition']}): "
              f"{trad['ingested']}/{trad['total']} ingested")
        print(f"  {trad['description']}")
        for w in trad["works"]:
            state = (f"ingested, {w['chars']:,} chars, {w['pages']} pages"
                     if w["ingested"] else "PENDING")
            print(f"  {'✓' if w['ingested'] else '○'} {w['title']} "
                  f"({w['slug']}) — seed: {w['seed']} — {state}")
            print(f"      {w['read_command']}  ·  {w['source_url']}")
    return 0


def _wisdom_seeds(args: Any, context: Any) -> int:
    k = _keeper(context)
    seeds = k.corpus.seeds()
    if _as_json(args):
        print(json.dumps(seeds, indent=2, default=str))
        return 0
    print("Seeds — every source WisdomKeeper knowledge can come from. "
          "A seed is a starting source list / discovery channel, not "
          "the texts themselves. Each work records which seed it came "
          "from (see /wisdom status).")
    for s in seeds:
        print(f"\n{s['id']} — {s['name']} ({s['works']} work(s))")
        print(f"  {s['what']}")
        print(f"  Add: {s['how_to_add']}")
        if s["work_slugs"]:
            print(f"  works: {', '.join(s['work_slugs'][:12])}"
                  + (" …" if len(s["work_slugs"]) > 12 else ""))
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
    # --all is a real argparse flag; it also arrives inside task words on
    # some dispatch paths, so accept both.
    if getattr(args, "all", False) or (rest and rest[0] == "--all"):
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
    from ...wisdom.reading import parse_timeline_args
    k = _keeper(context)
    # Positional form: `timeline <tradition> [n]` — a trailing integer
    # is the event count, never part of the tradition name.
    tradition, limit = parse_timeline_args(rest)
    flag_tradition = getattr(args, "tradition", "") or ""
    if flag_tradition:
        tradition = flag_tradition
    start = getattr(args, "start", -3000)
    end = getattr(args, "end", 2100)
    try:
        events = k.history.timeline(tradition, start, end, limit=limit)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(events, indent=2, default=str))
        return 0
    if not events:
        print("no timeline events found.")
        return 0
    head = (f"📜 timeline — {tradition}" if tradition else "📜 timeline")
    print(f"{head} ({len(events)} events):")
    for e in events:
        yr = f"{e['start']}" if e['start'] == e['end'] else \
            f"{e['start']}…{e['end']}"
        # Full summary + sources — not a one-line stub: the owner
        # should be able to read what happened, not just its title.
        print(f"\n  {yr}: {e['title']} [{e['tradition']}] — {e['region']}")
        print(f"    {e['summary']}")
        print(f"    sources: {', '.join(e['sources'])}")
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
        print(f"  {p.get('snippet', '')}")
        if p.get("slug"):
            ch = f" {p['chapter_number']}" if p.get("chapter_number") else ""
            print(f"  read full: /wisdom read {p['slug']}{ch}")
    print("-- timeline context --")
    for e in out["timeline_context"][:10]:
        print(f"  {e['start']}: {e['title']} — {e.get('summary', '')}")
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
