"""Chat control for /match — daily batches + stable matching (#83).

Never raises. Not owner-gated: matching is a shared-surface mechanic
(gigs, communities, mentoring) — anyone in the chat drives it.

Usage:
    /match daily [gig|community|learning] [n]
    /match add <id> | <title> | tag1,tag2 | attr=val,attr2=val2
    /match remove <id>
    /match like <id> | /match pass <id>     (trains future batches)
    /match why <id>                          (why was this picked?)
    /match questionnaire [gig|community|learning]
    /match answer <question_id> <1-5> [value]
    /match acceptable <question_id> <v1,v2>  (answers I'd accept)
    /match dealbreaker <name> <attr> <eq|ne|le|ge|in|contains|between> <value>
    /match rank <proposer|reviewer> <who> <id1,id2,id3...>
    /match capacity <reviewer> <n>            (mentor's seats)
    /match run                                (stable matching; capacity-aware)
    /match roommates <id1,id2,...>            (single-pool Irving pairing)
    /match assign                             (max-total-score assignment)
    /match ttc <a1>=<i1> ... -- <a1>:<i2,i1> ...(Top Trading Cycles)
    /match stats                              (feedback + batch stats)
"""

from __future__ import annotations

from ..core.logging_setup import get_logger
from .batches import Candidate, DailyBatchStore, explain_pick
from .questionnaire import Questionnaire
from .stable import (
    optimal_assignment,
    roommate_match,
    serial_dictatorship,
    stable_match,
    stable_match_capacities,
    top_trading_cycles,
    verify_roommate_stable,
    verify_stable,
)

_log = get_logger(__name__)

__all__ = ["control_match", "format_batch"]


def _usage() -> str:
    return (
        "🤝 /match — curated picks + two-sided matching\n"
        "  /match daily [surface] [n] — today's batch (stable all day)\n"
        "  /match add <id> | <title> | tag1,tag2 | attr=val,...\n"
        "  /match remove <id> — drop a candidate\n"
        "  /match like|pass <id> — train future batches\n"
        "  /match why <id> — why was this picked?\n"
        "  /match questionnaire [surface] — start / see unanswered\n"
        "  /match answer <question_id> <1-5> [value]\n"
        "  /match acceptable <question_id> <v1,v2> — answers I'd accept\n"
        "  /match dealbreaker <name> <attr> <pred> <value>\n"
        "      pred: eq|ne|le|ge|in|contains|between\n"
        "  /match rank <proposer|reviewer> <who> <id1,id2,id3>\n"
        "  /match capacity <reviewer> <n> — reviewer takes n proposers\n"
        "  /match run — stable matching (no blocking pairs, verified)\n"
        "  /match roommates <id1,id2,...> — single-pool pairing\n"
        "  /match assign — max total score (efficient, may be unstable)\n"
        "  /match ttc <a>=<item> ... -- <a>:<i1,i2> ... — trade cycles\n"
        "  /match stats — feedback + batch history"
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


def _surface(context) -> str:
    return getattr(context, "matching_surface", None) or "gig"


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
    lines = ["🎯 today's %s picks (%d) — same all day, fresh tomorrow:"
             % (surface or "gig", len(batch))]
    for i, c in enumerate(batch, 1):
        tags = (" [" + ", ".join(c.tags) + "]") if c.tags else ""
        lines.append("%d. %s%s" % (i, c.title or c.candidate_id, tags))
        if c.summary:
            lines.append("   %s" % c.summary[:120])
    lines.append("👍 /match like <id> · 👎 /match pass <id> · ❓ /match why <id>")
    return "\n".join(lines)


def _borda_scores(rankings: dict[str, list[str]]) -> dict[str, dict[str, float]]:
    """Borda scores from rank lists: top rank = len(list) points."""
    out: dict[str, dict[str, float]] = {}
    for who, ranking in rankings.items():
        n = len(ranking)
        out[who] = {other: float(n - i) for i, other in enumerate(ranking)}
    return out


def control_match(tail: str, context=None, chat=None,
                  sender_id: str = "", sender: str = "") -> str:
    """/match — daily batches + stable matching. Never raises."""
    try:
        rest = (tail or "").strip()
        batch_store, q, owner = _store(context)
        surface = _surface(context)
        if not rest or rest.split()[0] in ("help", "?"):
            return _usage()
        parts = rest.split(None, 1)
        action, args = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

        if action == "daily":
            segs = args.split()
            surf = (segs[0] if segs else surface).lower()
            try:
                n = int(segs[1]) if len(segs) > 1 else 5
            except ValueError:
                n = 5
            weights = q.weights()
            pool = batch_store.candidates(surf)
            # Deal-breakers filter first (front-loaded, Feeld pattern).
            filtered = q.filter_dealbreakers(
                [{"candidate_id": c.candidate_id, "attributes": c.attributes}
                 for c in pool])
            keep = {d["candidate_id"] for d in filtered}
            pool = [c for c in pool if c.candidate_id in keep]
            batch = batch_store.today(surf, owner, weights, n=n, pool=pool)
            return format_batch(batch, surf)

        if action == "add":
            segs = [s.strip() for s in args.split("|")]
            if len(segs) < 2:
                return "usage: /match add <id> | <title> | tag1,tag2 | attr=val,..."
            cid, title = segs[0], segs[1]
            if not cid:
                return "give the candidate an id: /match add <id> | <title> | ..."
            tags = tuple(t.strip() for t in (segs[2] if len(segs) > 2 else "").split(",") if t.strip())
            attrs = _parse_attrs(segs[3] if len(segs) > 3 else "")
            ok = batch_store.add_candidate(
                Candidate(candidate_id=cid, title=title, tags=tags,
                          attributes=attrs),
                surface=surface)
            return ("✅ '%s' registered for %s picks." % (title or cid, surface)
                    if ok else "couldn't register that candidate — try again.")

        if action == "remove":
            cid = args.strip()
            if not cid:
                return "usage: /match remove <id>"
            ok = batch_store.remove_candidate(cid, surface=surface)
            return ("🗑 '%s' removed." % cid if ok
                    else "no candidate '%s' on %s." % (cid, surface))

        if action in ("like", "pass"):
            cid = args.strip()
            if not cid:
                return "usage: /match %s <id>" % action
            ok = batch_store.record_feedback(surface, owner, cid,
                                             liked=(action == "like"))
            if not ok:
                return "couldn't record that — try again."
            stats = batch_store.feedback_stats(surface, owner).get(cid, (0, 0))
            return ("👍 liked '%s' — future batches learn from this %s."
                    % (cid, "(%d likes / %d views)" % stats)
                    if action == "like" else
                    "👎 passed on '%s' — future batches learn from this %s."
                    % (cid, "(%d likes / %d views)" % stats))

        if action == "why":
            cid = args.strip()
            if not cid:
                return "usage: /match why <id>"
            cand = next((c for c in batch_store.candidates(surface)
                         if c.candidate_id == cid), None)
            if cand is None:
                return "no candidate '%s' on %s." % (cid, surface)
            lines = [explain_pick(cand, q.weights())]
            rows = q.match_breakdown(cand.attributes)
            if rows:
                lines.append("   questionnaire:")
                for r in rows[:8]:
                    mark = "✅" if r["matched"] else "❌"
                    lines.append("   %s ×%d %s — %s"
                                 % (mark, r["weight"],
                                    (r["text"] or r["question_id"])[:60],
                                    r["detail"]))
            return "\n".join(lines)

        if action == "questionnaire":
            surf = (args.split() or [surface])[0].lower()
            q.ensure_starters(surf)
            todo = q.unanswered()
            if not todo:
                return "✅ all questions answered — more answers still sharpen results."
            lines = ["❓ answer more → better matches (%d left):" % len(todo)]
            for item in todo[:8]:
                lines.append("  • %s — /match answer %s <1-5> [your answer]"
                             % (item.text, item.question_id))
            lines.append("💡 /match acceptable %s <v1,v2> — which answers you'd accept"
                         % todo[0].question_id)
            return "\n".join(lines)

        if action == "answer":
            segs = args.split(None, 2)
            if len(segs) < 2:
                return "usage: /match answer <question_id> <1-5> [your answer]"
            try:
                importance = int(segs[1])
            except ValueError:
                return "importance must be 1–5."
            ok = q.answer(segs[0], importance, segs[2] if len(segs) > 2 else "")
            if not ok:
                return "unknown question — /match questionnaire to see the list."
            left = len(q.unanswered())
            got = q.get(segs[0])
            hint = ""
            if got is not None and not got.acceptable:
                hint = (" — now set /match acceptable %s <v1,v2> "
                        "for the full OkCupid pattern." % segs[0])
            return ("✅ recorded (%d %s). %d question(s) left — answer more → better matches.%s"
                    % (importance, "★" * importance, left, hint))

        if action == "acceptable":
            segs = args.split(None, 1)
            if len(segs) < 2:
                return "usage: /match acceptable <question_id> <v1,v2,...>"
            vals = [v.strip() for v in segs[1].split(",") if v.strip()]
            ok = q.set_acceptable(segs[0], vals)
            return ("✅ acceptable answers set for '%s': %s."
                    % (segs[0], ", ".join(vals)) if ok
                    else "unknown question — /match questionnaire to see the list.")

        if action == "dealbreaker":
            segs = args.split(None, 3)
            valid = ("eq", "ne", "le", "ge", "in", "contains", "between")
            if len(segs) < 4 or segs[2] not in valid:
                return ("usage: /match dealbreaker <name> <attr> "
                        "<eq|ne|le|ge|in|contains|between> <value>")
            name, attr, pred, raw = segs
            value: object = raw
            if pred in ("le", "ge"):
                try:
                    value = float(raw)
                except ValueError:
                    return "le/ge need a number."
            if pred == "between":
                try:
                    lo, hi = [float(x) for x in raw.split(",", 1)]
                    value = [lo, hi]
                except ValueError:
                    return "between needs <lo>,<hi>."
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

        if action == "capacity":
            segs = args.split()
            if len(segs) < 2:
                return "usage: /match capacity <reviewer> <n>"
            try:
                n = int(segs[1])
            except ValueError:
                return "capacity must be a number."
            ok = q.set_capacity(segs[0], n)
            return ("✅ '%s' now takes up to %d." % (segs[0], max(1, n))
                    if ok else "couldn't set that capacity.")

        if action == "run":
            prop_ranks = q.rankings("proposer")
            rev_ranks = q.rankings("reviewer")
            if not prop_ranks or not rev_ranks:
                return ("need rankings on both sides first:\n"
                        "/match rank proposer <who> <id1,id2>\n"
                        "/match rank reviewer <who> <id1,id2>")
            caps = q.capacities()
            if any(c > 1 for c in caps.values()):
                result = stable_match_capacities(
                    list(prop_ranks), list(rev_ranks),
                    prop_ranks, rev_ranks, caps)
                note = " (capacity-aware Hospital–Resident)"
            else:
                result = stable_match(list(prop_ranks), list(rev_ranks),
                                      prop_ranks, rev_ranks)
                note = ""
            ok, blockers = verify_stable(result, prop_ranks, rev_ranks)
            out = result.summary() + note
            out += ("\n✅ verified: no blocking pairs."
                    if ok else "\n⚠️ %d blocking pair(s): %s"
                    % (len(blockers), blockers[:5]))
            return out

        if action == "roommates":
            ids = [i.strip() for i in args.split(",") if i.strip()]
            if len(ids) < 2:
                return "usage: /match roommates <id1,id2,...> (even count)"
            ranks = q.rankings("proposer")
            missing = [i for i in ids if i not in ranks]
            if missing:
                return ("need proposer rankings for: %s\n"
                        "/match rank proposer <who> <id1,id2,...>"
                        % ", ".join(missing))
            idset = set(ids)
            prefs = {i: [x for x in ranks[i] if x in idset and x != i]
                     for i in ids}
            result = roommate_match(ids, prefs)
            if not result.stable:
                return result.summary()
            ok, blockers = verify_roommate_stable(
                {p: q_ for p, q_ in result.items()}, prefs)
            out = result.summary()
            out += ("\n✅ verified: no blocking pairs."
                    if ok else "\n⚠️ blocking pairs: %s" % (blockers[:5],))
            return out

        if action == "assign":
            prop_ranks = q.rankings("proposer")
            rev_ranks = q.rankings("reviewer")
            if not prop_ranks or not rev_ranks:
                return ("need rankings on both sides first:\n"
                        "/match rank proposer <who> <id1,id2>\n"
                        "/match rank reviewer <who> <id1,id2>")
            pb = _borda_scores(prop_ranks)
            rb = _borda_scores(rev_ranks)
            scores: dict[str, dict[str, float]] = {}
            for p in prop_ranks:
                scores[p] = {}
                for r in rev_ranks:
                    scores[p][r] = pb.get(p, {}).get(r, 0.0) + rb.get(
                        r, {}).get(p, 0.0)
            result = optimal_assignment(scores)
            total = sum(scores[p][r] for p, r in result.items() if r)
            lines = ["🎯 max-total assignment (score %.0f) — efficient, "
                     "not guaranteed stable:" % total]
            for p, r in result.items():
                lines.append("  %s → %s" % (p, r or "—"))
            lines.append("💡 /match run for the stable (no-blocking-pair) version.")
            return "\n".join(lines)

        if action == "ttc":
            if "--" not in args:
                return ("usage: /match ttc <agent>=<item> ... -- "
                        "<agent>:<item1,item2> ...")
            left, right = args.split("--", 1)
            endow: dict[str, str] = {}
            for tok in left.split():
                if "=" in tok:
                    a, _, i = tok.partition("=")
                    if a.strip() and i.strip():
                        endow[a.strip()] = i.strip()
            prefs: dict[str, list[str]] = {}
            for tok in right.split():
                if ":" in tok:
                    a, _, items = tok.partition(":")
                    prefs[a.strip()] = [x.strip() for x in items.split(",")
                                        if x.strip()]
            if not endow:
                return "no endowments parsed — /match ttc <a>=<i> ... -- <a>:<i1,i2>"
            agents = [a for a in endow if a in prefs] or list(endow)
            result = top_trading_cycles(agents, prefs, endow)
            lines = ["🔄 top trading cycles (core allocation):"]
            for a, item in result.items():
                mine = " (kept own)" if endow.get(a) == item else ""
                lines.append("  %s → %s%s" % (a, item or "—", mine))
            return "\n".join(lines)

        if action == "draft":
            # Serial dictatorship pick-order draft (RSD with seed).
            segs = args.split()
            agents = [s for s in segs if not s.startswith("seed=")]
            seed = None
            for s in segs:
                if s.startswith("seed="):
                    try:
                        seed = int(s.split("=", 1)[1])
                    except ValueError:
                        seed = None
            ranks = q.rankings("proposer")
            items = sorted({i for r in ranks.values() for i in r})
            if not agents or not items:
                return ("usage: /match draft <agent1> <agent2> ... "
                        "[seed=N] — needs proposer rankings as pick lists")
            prefs = {a: ranks.get(a, []) for a in agents}
            result = serial_dictatorship(agents, prefs, items, seed=seed)
            lines = ["🎲 serial-dictatorship draft:"]
            for a in agents:
                lines.append("  %s → %s" % (a, result.get(a) or "—"))
            return "\n".join(lines)

        if action == "stats":
            stats = batch_store.feedback_stats(surface, owner)
            hist = batch_store.batch_history(surface, owner, 5)
            lines = ["📊 %s matching stats:" % surface]
            if stats:
                ranked = sorted(stats.items(),
                                key=lambda kv: (-kv[1][0], kv[1][1]))[:8]
                lines.append("  feedback (likes/views):")
                for cid, (likes, views) in ranked:
                    lines.append("    %s — %d/%d" % (cid, likes, views))
            else:
                lines.append("  no feedback yet — /match like|pass <id>")
            if hist:
                lines.append("  recent batches:")
                for day, ids in hist:
                    lines.append("    %s: %s" % (day, ", ".join(ids[:6])))
            brk = q.dealbreakers()
            if brk:
                lines.append("  deal-breakers: %s"
                             % ", ".join(b.name for b in brk))
            caps = q.capacities()
            if caps:
                lines.append("  capacities: %s"
                             % ", ".join("%s×%d" % kv for kv in caps.items()))
            return "\n".join(lines)

        return _usage()
    except Exception:  # noqa: BLE001
        _log.warning("matching.chat: control_match failed", exc_info=True)
        return "matching hiccup — try again."
