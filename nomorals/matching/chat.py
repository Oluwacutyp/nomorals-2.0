"""Chat control for /match — daily batches + stable matching (#83).

Never raises. Not owner-gated: matching is a shared-surface mechanic
(gigs, communities, mentoring) — anyone in the chat drives it.

Usage:
    /match daily [gig|community|learning]
    /match add <id> | <title> | tag1,tag2 | attr=val,attr2=val2
    /match questionnaire [gig|community|learning]
    /match answer <question_id> <1-5> [value]
    /match dealbreaker <name> <attr> <eq|ne|le|ge|in|contains> <value>
    /match rank <proposer|reviewer> <who> <id1,id2,id3...>
    /match run
"""

from __future__ import annotations

from ..core.logging_setup import get_logger
from .batches import Candidate, DailyBatchStore
from .questionnaire import Questionnaire
from .stable import stable_match

_log = get_logger(__name__)

__all__ = ["control_match"]


def _usage() -> str:
    return (
        "🤝 /match — curated picks + two-sided matching\n"
        "  /match daily [gig|community|learning] — today's batch\n"
        "  /match add <id> | <title> | tag1,tag2 | attr=val,...\n"
        "  /match questionnaire [surface] — start / see unanswered\n"
        "  /match answer <question_id> <1-5> [value]\n"
        "  /match dealbreaker <name> <attr> <eq|ne|le|ge|in|contains> <value>\n"
        "  /match rank <proposer|reviewer> <who> <id1,id2,id3>\n"
        "  /match run — Gale-Shapley stable matching"
    )


def _store(context) -> tuple[DailyBatchStore, Questionnaire, str]:
    batch_store = getattr(context, "matching_batch_store", None)
    if not isinstance(batch_store, DailyBatchStore):
        batch_store = DailyBatchStore()
    owner = getattr(context, "matching_owner", None) or "owner"
    q = getattr(context, "matching_questionnaire", None)
    if not isinstance(q, Questionnaire):
        q = Questionnaire(owner=owner)
    return batch_store, q, owner


def _parse_attrs(text: str) -> dict:
    attrs: dict = {}
    for part in (text or "").split(","):
        part = part.strip()
        if "=" not in part:
            continue
        k, _, v = part.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            continue
        try:
            attrs[k] = int(v)
        except ValueError:
            try:
                attrs[k] = float(v)
            except ValueError:
                low = v.lower()
                attrs[k] = {"true": True, "false": False}.get(low, v)
    return attrs


def format_batch(batch: list[Candidate], surface: str) -> str:
    if not batch:
        return ("📭 no candidates registered for '%s' yet.\n"
                "Add some: /match add <id> | <title> | tag1,tag2"
                % (surface or "gig"))
    lines = ["🎯 today's %s picks (%d):" % (surface or "gig", len(batch))]
    for i, c in enumerate(batch, 1):
        tags = (" [" + ", ".join(c.tags) + "]") if c.tags else ""
        lines.append("%d. %s%s" % (i, c.title or c.candidate_id, tags))
        if c.summary:
            lines.append("   %s" % c.summary[:120])
    return "\n".join(lines)


def control_match(tail: str, context=None, chat=None,
                  sender_id: str = "", sender: str = "") -> str:
    """/match — daily batches + stable matching. Never raises."""
    try:
        rest = (tail or "").strip()
        batch_store, q, owner = _store(context)
        if not rest or rest.split()[0] in ("help", "?"):
            return _usage()
        parts = rest.split(None, 1)
        action, args = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if action == "daily":
            surface = (args.split() or ["gig"])[0].lower()
            weights = q.weights()
            pool = batch_store.candidates(surface)
            # Deal-breakers filter first (front-loaded, Feeld pattern).
            filtered = q.filter_dealbreakers(
                [{"candidate_id": c.candidate_id, "attributes": c.attributes}
                 for c in pool])
            keep = {d["candidate_id"] for d in filtered}
            pool = [c for c in pool if c.candidate_id in keep]
            batch = batch_store.today(surface, owner, weights, pool=pool)
            return format_batch(batch, surface)

        if action == "add":
            segs = [s.strip() for s in args.split("|")]
            if len(segs) < 2:
                return "usage: /match add <id> | <title> | tag1,tag2 | attr=val,..."
            cid, title = segs[0], segs[1]
            tags = tuple(t.strip() for t in (segs[2] if len(segs) > 2 else "").split(",") if t.strip())
            attrs = _parse_attrs(segs[3] if len(segs) > 3 else "")
            surface = getattr(context, "matching_surface", None) or "gig"
            ok = batch_store.add_candidate(
                Candidate(candidate_id=cid, title=title, tags=tags,
                          attributes=attrs),
                surface=surface)
            return ("✅ '%s' registered for %s picks." % (title or cid, surface)
                    if ok else "couldn't register that candidate — try again.")

        if action == "questionnaire":
            surface = (args.split() or ["gig"])[0].lower()
            q.ensure_starters(surface)
            todo = q.unanswered()
            if not todo:
                return "✅ all questions answered — more answers still sharpen results."
            lines = ["❓ answer more → better matches (%d left):" % len(todo)]
            for item in todo[:8]:
                lines.append("  • %s — /match answer %s <1-5>"
                             % (item.text, item.question_id))
            return "\n".join(lines)

        if action == "answer":
            segs = args.split(None, 2)
            if len(segs) < 2:
                return "usage: /match answer <question_id> <1-5> [value]"
            try:
                importance = int(segs[1])
            except ValueError:
                return "importance must be 1–5."
            ok = q.answer(segs[0], importance, segs[2] if len(segs) > 2 else "")
            left = len(q.unanswered())
            return ("✅ recorded (%d-%s). %d question(s) left — answer more → better matches."
                    % (importance, "★" * importance, left)
                    if ok else "unknown question — /match questionnaire to see the list.")

        if action == "dealbreaker":
            segs = args.split(None, 3)
            if len(segs) < 4 or segs[2] not in ("eq", "ne", "le", "ge", "in", "contains"):
                return ("usage: /match dealbreaker <name> <attr> "
                        "<eq|ne|le|ge|in|contains> <value>")
            name, attr, pred, raw = segs
            value: object = raw
            if pred in ("le", "ge"):
                try:
                    value = float(raw)
                except ValueError:
                    return "le/ge need a number."
            if pred == "in":
                value = [v.strip() for v in raw.split(",")]
            ok = q.set_dealbreaker(name, attr, pred, value)
            return ("🚧 deal-breaker '%s' set — candidates that fail it "
                    "never reach matching." % name if ok
                    else "couldn't set that deal-breaker.")

        if action == "rank":
            segs = args.split(None, 2)
            if len(segs) < 3 or segs[0] not in ("proposer", "reviewer"):
                return ("usage: /match rank <proposer|reviewer> <who> "
                        "<id1,id2,id3> (most-preferred first)")
            ok = q.set_ranking(segs[1], segs[0],
                               [r.strip() for r in segs[2].split(",")])
            return ("✅ %s's %s ranking saved (%d ranked)."
                    % (segs[1], segs[0], len(segs[2].split(",")))
                    if ok else "couldn't save that ranking.")

        if action == "run":
            prop_ranks = q.rankings("proposer")
            rev_ranks = q.rankings("reviewer")
            if not prop_ranks or not rev_ranks:
                return ("need rankings on both sides first:\n"
                        "/match rank proposer <who> <id1,id2>\n"
                        "/match rank reviewer <who> <id1,id2>")
            result = stable_match(list(prop_ranks), list(rev_ranks),
                                  prop_ranks, rev_ranks)
            return result.summary()

        return _usage()
    except Exception:  # noqa: BLE001
        _log.warning("matching.chat: control_match failed", exc_info=True)
        return "matching hiccup — try again."
