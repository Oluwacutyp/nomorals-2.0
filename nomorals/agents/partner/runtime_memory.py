"""RuntimeMemoryMixin: PartnerRuntime command group (memory)."""

from __future__ import annotations

import re
import time

class RuntimeMemoryMixin:
    """RuntimeMemoryMixin for :class:`PartnerRuntime`."""


    # ── long-term memory: /remember /recall /forget ──────────────────────────
    def _control_remember(self, tail: str, *, chat_key: str = "") -> str:
        """Explicitly store something: /remember <text> [kind] [tags:a,b]."""
        from ...memory.base import ALL_KINDS

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /remember <what to remember> [kind] [tags:a,b]\n"
                    "kinds: fact preference decision relationship episode skill lesson")
        tokens = raw.split()
        kind = "episode"
        for i, tok in enumerate(tokens):
            if tok.lower() in ALL_KINDS:
                kind = tok.lower()
                tokens = tokens[:i] + tokens[i + 1:]
                break
        content = " ".join(tokens)
        tags = ""
        if " " in content and content.rsplit(" ", 1)[-1].startswith("tags:"):
            content, tagpart = content.rsplit(" ", 1)
            tags = tagpart[len("tags:"):]
        content = content.strip()
        if not content:
            return "usage: /remember <what to remember> [kind] [tags:a,b]"
        record_id = self.context.memory.remember(
            content, kind=kind, importance=0.9,
            source="user:command", origin=f"chat:{chat_key or 'command'}",
            tags=tags,
        )
        return f"remembered it. [{kind}] {content[:120]} (id {record_id[:8]})"

    def _control_recall(self, tail: str) -> str:
        """Show the top memories matching the query (or the freshest, if none)."""
        query = (tail or "").strip()
        if self.context.memory is None:
            return "memory is off in this session."
        result = self.context.memory.recall(query, limit=5)
        if not result.records:
            return "nothing in memory matches that yet — try /remember."
        lines = ["from memory:"]
        now = time.time()
        for r in result.records:
            days = (now - r.created_at) / 86400.0
            age = f"today" if days < 0.05 else f"{days:.0f}d ago" if days < 60 else f"{days / 30.0:.0f}mo ago"
            score = f" {r.score:.2f}" if r.score else ""
            tags = f" #{r.tags}" if r.tags else ""
            lines.append(f"  [{r.kind}]{tags} {r.content[:130]}  ({age}, id {r.id[:6]})")
        return "\n".join(lines)

    def _control_forget(self, tail: str) -> str:
        """Forget by id, or by the best matching description."""
        target = (tail or "").strip()
        if not target:
            return "usage: /forget <id or description>"
        if self.context.memory is None:
            return "memory is off in this session."
        if re.fullmatch(r"[0-9a-f]{12}", target):
            removed = self.context.memory.forget(target)
            return ("forgotten." if removed
        else f"no memory with id {target} — /recall <query> to search instead")
        record = self.context.memory.find_one(target)
        if record is None:
            return "didn't find that in memory — give me the id from /recall, or better words."
        self.context.memory.forget(record.id)
        return f"forgotten: [{record.kind}] {record.content[:110]}"

    # ── core mind (wave 87) ──────────────────────────────────────────────────
    def _control_mind(self, tail: str, chat_key: str) -> str:
        """/mind — the manual override over the Core Mind."""
        tail = (tail or "").strip()
        if tail in ("", "status"):
            return self.mind.status()
        if tail == "clear":
            n = self.mind.clear(chat_key)
            return f"mind: {n} pending clarification(s) cleared."
        if tail == "pending":
            pend = self.mind.pending()
            if not pend:
                return "no open clarifications."
            lines = []
            for key, p in pend.items():
                lines.append(f"  {key}: “{p.get('question', '')[:80]}”")
            return "open clarifications:\n" + "\n".join(lines)
        # /mind <goal> — force the routing and show the decision
        intent = self.mind.decide(tail, live_game=self.mind._live_game(chat_key))
        if intent.kind == "chat":
            return (f"the mind reads no goal in “{tail[:60]}” — “{tail[:30]}” "
                    f"stays conversation. Try a verb: research/build/browse/"
                    f"download/mission: …")
        if intent.action == "ask":
            question = self.mind._question_for(intent)
            self.mind._set_pending(chat_key, intent, question)
            return f"decision: {intent.kind} (ask) — {intent.why}\n{question}"
        reply = self.mind._dispatch(intent, chat_key, self._fake_message(chat_key))
        return f"decision: {intent.kind} (route {intent.route}) — {intent.why}\n" + \
            (reply or "(no reply)")
