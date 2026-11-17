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
              "       nm wisdom lineage <name> [--json]\n"
              "       nm wisdom practice list [--json]\n"
              "       nm wisdom practice <session-id> [--rounds N] [--chat]\n"
              "       nm wisdom practice recall add <prompt> | <answer> | <tradition> | <work>\n"
              "       nm wisdom practice recall due | review <id> <0-5> | stats [--json]\n"
              "       nm wisdom oral submit <id> --file <transcript> [--language L] [--pack P]\n"
              "                        [--restriction R] [--granted-by NAME --scope S]\n"
              "       nm wisdom oral verify <id> [--notes ...] [--reject]\n"
              "       nm wisdom oral ingest <id>\n"
              "       nm wisdom oral list [--status S] [--json]\n"
              "       nm wisdom oral read <id> [--json]\n"
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
    if verb == "lineage":
        return _wisdom_lineage(args, context, words[1:])
    if verb == "practice":
        return _wisdom_practice(args, context, words[1:])
    if verb == "oral":
        return _wisdom_oral(args, context, words[1:])
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
    oral = full.get("oral", {})
    print(f"oral tradition: {oral.get('ingested', 0)} ingested draft(s)")
    comp = full.get("completeness", {}) or {}
    if comp:
        print(f"completeness: {comp.get('done', '?')}/"
              f"{comp.get('total', '?')} checks done")
        for item in comp.get("items", []):
            if item.get("status") != "done":
                print(f"  missing: {item.get('item')} — "
                      f"{item.get('evidence', '')}")
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
    # Mapped positions, when the topic is one the disagreement map
    # knows — same rule as /wisdom compare in chat.
    known = {t.lower(): t for t in k.disagreement_topics()}
    mapped = known.get(" ".join(rest).strip().lower())
    if mapped:
        try:
            out["mapped_positions"] = k.compare_positions(mapped).to_dict()
        except Exception as exc:  # noqa: BLE001 - map failure ≠ compare failure
            out["mapped_positions_error"] = str(exc)
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
    if "mapped_positions" in out:
        print("-- mapped positions (never averaged) --")
        for p in out["mapped_positions"]["positions"]:
            print(f"  {p['tradition']}: {p['claim']}")
    return 0


def _wisdom_lineage(args: Any, context: Any, rest: list[str]) -> int:
    k = _keeper(context)
    try:
        if not rest:
            nodes = k.lineage_nodes()
            if _as_json(args):
                print(json.dumps(nodes, indent=2, default=str))
                return 0
            print(f"lineage — {len(nodes)} nodes from the timeline:")
            for n in nodes:
                print(f"  {n['name']} [{n['kind']}"
                      + (f", {n['tradition']}" if n.get("tradition")
                         else "") + "]")
            return 0
        found = k.lineage(" ".join(rest))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(found, indent=2, default=str))
        return 0
    if not found:
        print(f"no lineage nodes match {' '.join(rest)!r}")
        return 0
    print(f"lineage — {' '.join(rest)} ({len(found)} node(s)):")
    for entry in found:
        node = entry["node"]
        head = f"\n{node['name']} [{node['kind']}"
        if node.get("tradition"):
            head += f", {node['tradition']}"
        if node.get("dates"):
            head += f", {node['dates']}"
        print(head + "]")
        for nb in entry["neighbours"]:
            edge = nb["edge"]
            print(f"  {edge['relation']}: {nb['other']['name']} "
                  f"(source: {edge['source']})")
    return 0


def _wisdom_practice_recall(args: Any, context: Any,
                            rest: list[str]) -> int:
    """``nm wisdom practice recall <add|due|review|stats>``."""
    k = _keeper(context)
    mem = k.practice_memory
    if not rest:
        print("usage: nm wisdom practice recall add <prompt> | <answer> | "
              "<tradition> | <work>\n"
              "       nm wisdom practice recall due | review <id> <0-5> | "
              "stats [--json]", file=sys.stderr)
        return 2
    action = rest[0].lower()
    if action == "add":
        fields = [f.strip() for f in " ".join(rest[1:]).split("|")]
        if len(fields) != 4 or not all(fields):
            print("usage: nm wisdom practice recall add <prompt> | "
                  "<answer> | <tradition> | <work>", file=sys.stderr)
            return 2
        try:
            item = mem.add_item(
                fields[0], fields[1],
                source={"tradition": fields[2], "work": fields[3]})
        except Exception as exc:  # noqa: BLE001
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"recall added: {item.id} — {item.prompt}")
        return 0
    if action == "due":
        due = mem.due_items()
        if _as_json(args):
            print(json.dumps([i.to_dict() for i in due], indent=2,
                             default=str))
            return 0
        if not due:
            print("nothing due for review.")
            return 0
        print(f"due for review ({len(due)}):")
        for item in due:
            print(f"  {item.id} — {item.prompt} "
                  f"[{item.source.get('tradition')} / "
                  f"{item.source.get('work')}]")
        return 0
    if action == "review":
        if len(rest) != 3:
            print("usage: nm wisdom practice recall review <id> "
                  "<quality 0-5>", file=sys.stderr)
            return 2
        try:
            item = mem.review(rest[1], int(rest[2]))
        except Exception as exc:  # noqa: BLE001
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        print(f"reviewed {item.id}: quality {rest[2]} — "
              f"due in {item.interval_days}d")
        return 0
    if action == "stats":
        st = mem.stats()
        if _as_json(args):
            print(json.dumps(st, indent=2, default=str))
            return 0
        print(f"recall: {st['items']} item(s), {st['due']} due, "
              f"{st['reviewed']} reviewed, {st['lapses']} lapse(s), "
              f"avg easiness {st['avg_easiness']}")
        return 0
    print(f"unknown recall verb: {action}", file=sys.stderr)
    return 2


def _wisdom_practice(args: Any, context: Any, rest: list[str]) -> int:
    k = _keeper(context)
    if rest and rest[0].lower() == "recall":
        return _wisdom_practice_recall(args, context, rest[1:])
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

# ── oral tradition ──────────────────────────────────────────────────
# The pipeline itself lives in nomorals.wisdom.oral (draft → owner
# verifies → ingest); these verbs are its CLI surface. Drafts are
# durable on disk, so submit / verify / ingest may run as separate
# one-shot invocations.

_ORAL_USAGE = (
    "usage: nm wisdom oral submit <id> --file <transcript> "
    "[--language L] [--pack P]\n"
    "                        [--restriction R] "
    "[--granted-by NAME --scope S]\n"
    "       nm wisdom oral verify <id> [--notes ...] [--reject]\n"
    "       nm wisdom oral ingest <id>\n"
    "       nm wisdom oral list [--status S] [--json]\n"
    "       nm wisdom oral read <id> [--json]")


def _wisdom_oral(args: Any, context: Any, rest: list[str]) -> int:
    """Route ``nm wisdom oral <subverb>``.

    Flags arrive via argparse (registered on the wisdom subparser);
    ``rest`` carries the subverb + positionals only.
    """
    if not rest:
        print(_ORAL_USAGE, file=sys.stderr)
        return 2
    sub = rest[0].lower()
    flags: dict[str, Any] = {
        "file": getattr(args, "file", "") or "",
        "language": getattr(args, "language", "") or "",
        "lang": getattr(args, "lang", "") or "",
        "pack": getattr(args, "pack", "") or "",
        "restriction": getattr(args, "restriction", "") or "",
        "granted-by": getattr(args, "granted_by", "") or "",
        "scope": getattr(args, "scope", "") or "",
        "contact-ref": getattr(args, "contact_ref", "") or "",
        "audio-ref": getattr(args, "audio_ref", "") or "",
        "contributor": getattr(args, "contributor", "") or "",
        "notes": getattr(args, "notes", "") or "",
        "status": getattr(args, "status", "") or "",
        "reject": bool(getattr(args, "reject", False)),
    }
    pos = rest[1:]
    if sub == "submit":
        return _oral_submit(args, context, flags, pos)
    if sub == "verify":
        return _oral_verify(args, context, flags, pos)
    if sub == "ingest":
        return _oral_ingest(args, context, flags, pos)
    if sub == "list":
        return _oral_list(args, context, flags, pos)
    if sub == "read":
        return _oral_read(args, context, flags, pos)
    print(f"unknown oral verb: {sub}\n{_ORAL_USAGE}", file=sys.stderr)
    return 2


def _oral_permission(flags: dict[str, Any]):
    """Build a PermissionRecord from flags, or None.

    A permission is only recorded when the custodian is named; a bare
    --scope without --granted-by is a named error, not a silent drop.
    """
    granted_by = str(flags.get("granted-by", "") or "")
    scope = str(flags.get("scope", "") or "")
    contact_ref = str(flags.get("contact-ref", "") or "")
    if not granted_by and (scope or contact_ref):
        raise ValueError(
            "--scope / --contact-ref need --granted-by (who granted it)")
    if not granted_by:
        return None
    from ...wisdom import PermissionRecord
    return PermissionRecord(granted_by=granted_by, scope=scope,
                            contact_ref=contact_ref)


def _oral_submit(args: Any, context: Any, flags: dict[str, Any],
                 pos: list[str]) -> int:
    if not pos:
        print(_ORAL_USAGE, file=sys.stderr)
        return 2
    draft_id = pos[0]
    path = str(flags.get("file", "") or "")
    if not path:
        print("oral submit needs --file <transcript> (plain text, "
              "\"SPEAKER: text\" lines)", file=sys.stderr)
        return 2
    try:
        with open(path, encoding="utf-8") as handle:
            transcript = handle.read()
    except OSError as exc:
        print(f"error: cannot read transcript {path}: {exc}",
              file=sys.stderr)
        return 1
    try:
        permission = _oral_permission(flags)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    k = _keeper(context)
    try:
        draft = k.oral.submit_draft(
            draft_id,
            audio_ref=str(flags.get("audio-ref", "") or ""),
            transcript=transcript,
            language=str(flags.get("language", "")
                         or flags.get("lang", "") or ""),
            tradition_pack_id=str(flags.get("pack", "") or ""),
            restriction=str(flags.get("restriction", "open") or "open"),
            permission=permission,
            contributor=str(flags.get("contributor", "") or ""))
    except Exception as exc:  # noqa: BLE001 - fail fast, honestly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"draft {draft.draft_id} submitted: {len(draft.turns)} turn(s), "
          f"language={draft.language or '-'}, "
          f"pack={draft.tradition_pack_id or '-'}, "
          f"restriction={draft.restriction} — verify with: "
          f"nm wisdom oral verify {draft.draft_id}")
    return 0


def _oral_verify(args: Any, context: Any, flags: dict[str, Any],
                 pos: list[str]) -> int:
    if not pos:
        print(_ORAL_USAGE, file=sys.stderr)
        return 2
    k = _keeper(context)
    try:
        draft = k.oral.verify(
            pos[0], owner="owner",
            approve=not bool(flags.get("reject", False)),
            notes=str(flags.get("notes", "") or ""))
    except Exception as exc:  # noqa: BLE001 - fail fast, honestly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if draft.status == "verified":
        print(f"draft {draft.draft_id} verified by owner — ingest with: "
              f"nm wisdom oral ingest {draft.draft_id}")
    else:
        print(f"draft {draft.draft_id} rejected — it can never ingest; "
              f"submit a fresh draft to try again")
    return 0


def _oral_ingest(args: Any, context: Any, flags: dict[str, Any],
                 pos: list[str]) -> int:
    if not pos:
        print(_ORAL_USAGE, file=sys.stderr)
        return 2
    k = _keeper(context)
    try:
        out = k.oral.ingest(pos[0])
    except Exception as exc:  # noqa: BLE001 - fail fast, honestly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"ingested: {out['draft_id']} ({out['turns']} turn(s), "
          f"{out['words']} words) → {out['path']}")
    return 0


def _oral_list(args: Any, context: Any, flags: dict[str, Any],
               pos: list[str]) -> int:
    k = _keeper(context)
    status = str(flags.get("status", "") or "")
    drafts = k.oral.list_drafts(status=status)
    ingested = k.oral.list_ingested()
    if _as_json(args):
        print(json.dumps({"drafts": drafts, "ingested": ingested},
                         indent=2, default=str))
        return 0
    if not drafts and not ingested:
        print("no oral drafts yet — submit one with: "
              "nm wisdom oral submit <id> --file <transcript>")
        return 0
    for d in drafts:
        perm = "permission on file" if d.get("permission") else "no permission"
        print(f"{d['draft_id']}: {d['status']} — "
              f"{len(d.get('turns', []))} turn(s), "
              f"{d.get('language') or '-'} / "
              f"{d.get('tradition_pack_id') or '-'}, "
              f"{d.get('restriction')}, {perm}")
    for e in ingested:
        print(f"{e['draft_id']}: ingested — {e['turns']} turn(s), "
              f"{e['words']} words, {e.get('language') or '-'} / "
              f"{e.get('tradition_pack_id') or '-'}, {e['restriction']}")
    return 0


def _oral_read(args: Any, context: Any, flags: dict[str, Any],
               pos: list[str]) -> int:
    if not pos:
        print(_ORAL_USAGE, file=sys.stderr)
        return 2
    k = _keeper(context)
    try:
        rec = k.read_oral(pos[0])
    except Exception as exc:  # noqa: BLE001 - fail fast, honestly
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if _as_json(args):
        print(json.dumps(rec, indent=2, default=str))
        return 0
    prov = rec.get("provenance", {}) or {}
    perm = rec.get("permission") or {}
    print(f"🎙️ oral draft {rec['draft_id']} — "
          f"{rec.get('language') or '-'} / "
          f"{rec.get('tradition_pack_id') or '-'} "
          f"({rec.get('restriction', 'open')})")
    print(f"   contributor: {prov.get('contributor') or '-'} · "
          f"permission: {perm.get('granted_by') or '-'} "
          f"({perm.get('scope') or '-'})")
    print("─" * 60)
    for t in rec.get("turns", []):
        print(f"{t.get('speaker', 'unknown')}: {t.get('text', '')}")
    print("─" * 60)
    return 0
