"""The triage engine (autocraft port): sync, duplicates, classify,
priority, decide, apply.

Stages ported: Sync (mirror issues of tracked repos), Duplicates
(candidate pairs + model verdict + groups + decisions), Classify
(cheap hint + model second opinion + referee, settled per the
upstream rules), Priority (P0–P4 + reach + work rank), Decide (scope
and duplicate questions the owner answers), Label/Act (apply an
answered decision: stored comment texts + closes, or the type label).

Stages NOT ported: Review, Fix, Fix-up, Verify, Merge queue,
Post-merge review — they need the Claude CLI + bubblewrap + Podman
sandbox workers, and Devon has her own code-fix machinery. PRs are
not mirrored in this version either (issues only); the GitHub
connector's issue list filters PRs out, and PR triage is the
non-portable half. The model is the brain through an injected
``suggest`` callable (``(prompt) -> text``); with no model the code
rung runs and model verdicts are recorded as unavailable, never
fabricated. Every GitHub write is recorded in the store's write
audit right after it happens, so a retry never repeats it.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ..core.logging_setup import get_logger
from . import classify as _classify
from . import similarity as _similarity
from . import urgency as _urgency
from .store import TriageStore

__all__ = ["TriageEngine", "DUPLICATE_CLOSE_TEXT"]

_log = get_logger(__name__)

#: The fixed closing line autocraft writes on a duplicate member.
DUPLICATE_CLOSE_TEXT = "Closing as a duplicate of #{keep}."
#: Model calls allowed per triage run (autocraft caps every step).
DEFAULT_MODEL_BUDGET = 50


class TriageEngine:
    """Sync + triage for one GitHub client over one store."""

    def __init__(self, client: Any, store: TriageStore, *,
                 suggest: Callable[[str], str] | None = None,
                 now: datetime | None = None,
                 model_budget_per_run: int = DEFAULT_MODEL_BUDGET
                 ) -> None:
        self.client = client
        self.store = store
        self.suggest = suggest
        self.now = now or datetime.now(UTC)
        self.model_budget_per_run = model_budget_per_run
        self.model_calls = 0
        self.model_budget_exhausted = False

    # ── model ──────────────────────────────────────────────────────
    def _ask(self, prompt: str) -> dict[str, Any] | None:
        """One model call, JSON-parsed, never raising.

        Returns None when there is no model, the budget is spent, or
        the answer is not usable JSON — the caller falls back to the
        code rung and records that.
        """
        if self.suggest is None:
            return None
        if self.model_calls >= self.model_budget_per_run:
            self.model_budget_exhausted = True
            return None
        self.model_calls += 1
        try:
            raw = self.suggest(prompt) or ""
        except Exception:  # noqa: BLE001 - a dead model is "no model"
            _log.debug("triage model call failed", exc_info=True)
            return None
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        try:
            parsed = json.loads(text)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # ── sync ───────────────────────────────────────────────────────
    def sync(self, repo: str, *, limit: int = 100) -> dict[str, Any]:
        """Mirror a repo's issues (open + closed) into the store."""
        counts = {"repo": repo, "new": 0, "updated": 0, "unchanged": 0,
                  "error": ""}
        try:
            seen = self.client.list_issues(repo, state="open", limit=limit)
            seen += self.client.list_issues(repo, state="closed",
                                            limit=limit)
        except Exception as exc:  # noqa: BLE001 - a failed sync reports
            counts["error"] = str(exc)
            return counts
        for raw in seen:
            item = {
                "repo": repo,
                "number": int(raw.get("number", 0)),
                "kind": "pr" if raw.get("is_pull_request") else "issue",
                "title": str(raw.get("title", "")),
                "body": str(raw.get("body", "")),
                "state": str(raw.get("state", "open")),
                "labels": list(raw.get("labels") or []),
                "author": str(raw.get("author", "")),
                "author_association": str(
                    raw.get("author_association", "")),
                "reactions": int(raw.get("reactions") or 0),
                "comments": int(raw.get("comments") or 0),
                "created_at": str(raw.get("created_at", "")),
                "updated_at": str(raw.get("updated_at", "")),
                "url": str(raw.get("url", "")),
            }
            if item["kind"] == "pr":
                continue
            counts[self.store.upsert_item(item)] += 1
        return counts

    # ── duplicates ─────────────────────────────────────────────────
    def _record_candidate_once(self, key: str, other_key: str,
                               sim: float) -> None:
        """Record an awaiting-model pair once, never twice."""
        for cand in self.store.candidates(status="awaiting-model"):
            if ({cand.get("a"), cand.get("b")}
                    == {key, other_key}):
                return
        self.store.record_candidates([{
            "a": key, "b": other_key,
            "similarity": round(sim, 3),
            "status": "awaiting-model"}])

    def _known_pairs(self) -> set[tuple[str, str]]:
        known = set(self.store.ruled_out_pairs())
        for group in self.store.groups():
            members = list(group.get("members", []))
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    known.add((a, b))
                    known.add((b, a))
        return known

    def find_duplicates(self, repo: str) -> list[dict[str, Any]]:
        """Find duplicate groups among a repo's open issues.

        Candidate pairs come from token similarity; a pair is grouped
        only on a model duplicate verdict. With no model the pairs are
        recorded as awaiting-model candidates and no group is formed.
        """
        items = self.store.items_for(repo, state="open")
        known = self._known_pairs()
        ruled = self.store.ruled_out_pairs()
        parent: dict[str, str] = {}

        def find(key: str) -> str:
            while parent.get(key, key) != key:
                key = parent[key]
            return parent.setdefault(key, key)

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        verdicts: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            key = item["key"]
            cands = _similarity.duplicate_candidates(
                items, item, ruled_out=ruled, now=self.now)
            for other, sim in cands:
                other_key = other["key"]
                pair = (key, other_key)
                if pair in known or (other_key, key) in known:
                    continue
                known.add(pair)
                known.add((other_key, key))
                if self.suggest is None:
                    self._record_candidate_once(key, other_key, sim)
                    continue
                prompt = (
                    "You judge whether two GitHub issues describe the "
                    "same bug or request (a duplicate pair).\n\n"
                    f"Issue A: #{item['number']} {item['title']}\n"
                    f"{item.get('body', '')}\n\n"
                    f"Issue B: #{other['number']} {other['title']}\n"
                    f"{other.get('body', '')}\n\n"
                    "Answer with JSON only: {\"verdict\": \"duplicate\" "
                    "or \"distinct\", \"confidence\": 0.0-1.0, \"reason\": "
                    "\"one sentence\", \"adds\": [\"what B adds\"]}.")
                verdict = self._ask(prompt)
                if verdict is None:
                    self._record_candidate_once(key, other_key, sim)
                    continue
                verdicts[pair] = verdict
                is_dup = (str(verdict.get("verdict", "")).lower()
                          == "duplicate")
                conf = float(verdict.get("confidence") or 0.0)
                if is_dup and conf >= 0.5:
                    union(key, other_key)
                elif not is_dup and conf >= 0.9:
                    self.store.rule_out(key, other_key)

        groups: dict[str, list[str]] = {}
        for item in items:
            root = find(item["key"])
            groups.setdefault(root, []).append(item["key"])
        made: list[dict[str, Any]] = []
        for members in groups.values():
            if len(members) < 2:
                continue
            members = sorted(members,
                             key=lambda k: int(k.rsplit("#", 1)[1]))
            keep = members[0]
            keep_n = keep.rsplit("#", 1)[1]
            reasons = []
            texts: dict[str, str] = {}
            for member in members[1:]:
                verdict = (verdicts.get((keep, member))
                           or verdicts.get((member, keep)) or {})
                reason = str(verdict.get("reason", ""))
                adds = [str(a) for a in verdict.get("adds", []) or []]
                if reason:
                    reasons.append(reason)
                text = DUPLICATE_CLOSE_TEXT.format(keep=keep_n)
                if adds:
                    carry = "; ".join(adds[:2])
                    text += (f"\nParts to carry over to #{keep_n}: "
                             f"{carry}")
                texts[member] = text
            kept_text = "Details from the closed duplicates:"
            bullets = []
            for member in members[1:]:
                verdict = (verdicts.get((keep, member))
                           or verdicts.get((member, keep)) or {})
                for add in [str(a) for a in verdict.get("adds", []) or []]:
                    bullets.append(
                        f"- #{member.rsplit('#', 1)[1]}: {add}")
            kept_text += "\n" + "\n".join(bullets[:6]) if bullets else ""
            texts[keep] = kept_text
            options = [f"keep-{keep_n}"]
            for member in members[1:3]:
                options.append(f"keep-{member.rsplit('#', 1)[1]}")
            options.append("not-duplicates")
            summary = ("; ".join(reasons[:2])
                       or "these reports describe the same issue.")
            decision = self.store.add_decision({
                "kind": "duplicate",
                "repo": repo,
                "members": members,
                "keep": keep,
                "question": (
                    f"Keep #{keep_n} and close "
                    + ", ".join(f"#{m.rsplit('#', 1)[1]}"
                                for m in members[1:])
                    + " as its duplicates?"),
                "options": options,
                "summary": summary,
                "texts": texts,
                "texts_hash": hashlib.sha256(
                    json.dumps(texts, sort_keys=True).encode()
                ).hexdigest(),
                "status": "open",
            })
            group = self.store.add_group({
                "repo": repo,
                "members": members,
                "keep": keep,
                "decision_id": decision["id"],
            })
            made.append(group)
        return made

    # ── classify + priority ────────────────────────────────────────
    def classify_item(self, key: str) -> dict[str, Any]:
        """Classify one mirrored item and store the verdict."""
        item = self.store.get_item(key)
        if item is None:
            return {"type": None, "source": "unsettled",
                    "confidence": 0.0, "needs_decision": True,
                    "security": None, "priority": "P2",
                    "reach": 0, "work_rank": 0, "model_used": False}
        hint = _classify.keyword_hint(str(item.get("title", "")),
                                       str(item.get("body", "")))
        model: dict[str, Any] | None = None
        referee: dict[str, Any] | None = None
        model_used = False
        hint_conf = float(hint.get("confidence") or 0.0)
        flags_unsure = (
            0.3 <= float(hint.get("security_confidence") or 0.0) < 0.9
            or 0.3 <= float(hint.get("needs_decision_confidence") or 0.0)
            < 0.8)
        if hint_conf < _classify.SETTLE_CONFIDENCE or flags_unsure:
            prompt = (
                "Classify this GitHub issue. Answer with JSON only: "
                "{\"type\": one of bugfix, performance, missing "
                "feature, small extra, plugin candidate, infra; "
                "\"type_confidence\": 0.0-1.0; \"security\": true or "
                "false; \"needs_decision\": true or false; \"reason\": "
                "\"one sentence\"}.\n\n"
                f"Issue #{item.get('number')} {item.get('title')}\n"
                f"{item.get('body', '')}")
            model = self._ask(prompt)
            model_used = model is not None
            if (model is not None and hint.get("type")
                    and model.get("type") != hint["type"]
                    and hint["type"] not in _classify.SCOPE_LABELS
                    and model.get("type") not in _classify.SCOPE_LABELS):
                referee = self._ask(
                    "Referee this classification disagreement. Answer "
                    "with JSON only: {\"type\": one of bugfix, "
                    "performance, missing feature, small extra, plugin "
                    "candidate, infra; \"confidence\": 0.0-1.0; "
                    "\"sure\": true or false}.\n\n"
                    f"Issue #{item.get('number')} {item.get('title')}\n"
                    f"{item.get('body', '')}\n"
                    f"First pass: {hint.get('type')}. Second opinion: "
                    f"{model.get('type')}.")
        settled = _classify.settle_type(
            {"type": hint.get("type"), "confidence": hint_conf},
            ({"type": model.get("type"),
              "confidence": float(model.get("type_confidence") or 0.0)}
             if model else None),
            referee=referee)
        security = _classify.settle_flag(
            bool(hint.get("security")),
            float(hint.get("security_confidence") or 0.0),
            (model.get("security") if model is not None else None))
        needs_decision = settled["needs_decision"]
        if not needs_decision and model is not None:
            model_nd = _classify.settle_flag(
                bool(hint.get("needs_decision")),
                float(hint.get("needs_decision_confidence") or 0.0),
                model.get("needs_decision"),
                high=_classify.NEEDS_DECISION_SET_CONFIDENCE)
            needs_decision = bool(model_nd) if model_nd is not None \
                else bool(hint.get("needs_decision"))
        duplicate_count = sum(
            1 for group in self.store.groups()
            if key in group.get("members", []))
        priority = _classify.priority_for(
            settled["type"], security=bool(security),
            signals=hint.get("signals") or {})
        reach_value = _classify.reach(
            {**item, "security": bool(security)},
            duplicate_count=duplicate_count)
        rank = _classify.work_rank(priority, settled["type"], "M",
                                    reach_value)
        out = {
            "type": settled["type"],
            "source": settled["source"],
            "confidence": settled["confidence"],
            "needs_decision": needs_decision,
            "security": security,
            "priority": priority,
            "reach": reach_value,
            "work_rank": rank,
            "model_used": model_used,
            "model_budget_exhausted": self.model_budget_exhausted,
        }
        self.store.set_classification(key, out)
        if needs_decision and settled["type"] is None:
            self._ask_scope_decision(item, settled, security)
        return out

    def _ask_scope_decision(self, item: dict[str, Any],
                            settled: dict[str, Any],
                            security: bool | None) -> None:
        """Open a scope decision for an unsettled item, once."""
        key = item["key"]
        for dec in self.store.decisions(status="open"):
            if dec.get("kind") == "scope" and dec.get("item") == key:
                return
        number = item.get("number")
        question = (f"Is #{number} in scope? The classifiers could "
                    "not settle its type.")
        self.store.add_decision({
            "kind": "scope",
            "repo": str(item.get("repo", "")),
            "item": key,
            "question": question,
            "options": ["take", "small-extra", "plugin", "close",
                        "backlog"],
            "summary": (f"\"{item.get('title', '')}\" could not be "
                        "typed with confidence; the owner decides "
                        "whether it belongs in core, as a small extra, "
                        "as a plugin candidate, or out of scope."),
            "status": "open",
        })

    def prioritize_item(self, key: str) -> dict[str, Any]:
        """Recompute priority/reach/rank from the stored mirror."""
        item = self.store.get_item(key)
        if item is None:
            return {"priority": "P2", "reach": 0, "work_rank": 0}
        duplicate_count = sum(
            1 for group in self.store.groups()
            if key in group.get("members", []))
        priority = _classify.priority_for(
            item.get("type"), security=bool(item.get("security")),
            signals=_classify.keyword_hint(
                str(item.get("title", "")),
                str(item.get("body", ""))).get("signals") or {})
        reach_value = _classify.reach(item,
                                       duplicate_count=duplicate_count)
        rank = _classify.work_rank(priority, item.get("type"), "M",
                                    reach_value)
        self.store.set_classification(key, {
            "type": item.get("type"),
            "source": item.get("type_source"),
            "confidence": item.get("type_confidence"),
            "security": item.get("security"),
            "needs_decision": item.get("needs_decision"),
            "priority": priority,
            "reach": reach_value,
            "work_rank": rank,
        })
        return {"priority": priority, "reach": reach_value,
                "work_rank": rank}

    def triage(self, repo: str, *, limit: int = 100) -> dict[str, Any]:
        """One full pass: sync, duplicates, classify, prioritize."""
        self.model_calls = 0
        self.model_budget_exhausted = False
        synced = self.sync(repo, limit=limit)
        groups = self.find_duplicates(repo)
        classified = 0
        for item in self.store.items_for(repo, state="open"):
            key = item["key"]
            if (item.get("type") is None
                    or item.get("classified_hash") != item.get("text_hash")):
                self.classify_item(key)
                classified += 1
                row = self.store.data["items"].get(key)
                if row is not None:
                    row["classified_hash"] = row.get("text_hash")
                    self.store.save()
            self.prioritize_item(key)
        return {"repo": repo, "synced": synced,
                "groups": len(groups), "classified": classified,
                "model_calls": self.model_calls,
                "model_budget_exhausted": self.model_budget_exhausted}

    def queue(self, repo: str) -> list[dict[str, Any]]:
        """Open items in work-rank order, highest first.

        Items with no classification (or whose text changed since
        they were classified) are classified on demand first, so the
        queue never shows an unrated item as P2 by default when the
        code rung can see better.
        """
        items = self.store.items_for(repo, state="open")
        for item in items:
            if (item.get("type") is None
                    or item.get("classified_hash") != item.get("text_hash")):
                self.classify_item(item["key"])
                self.prioritize_item(item["key"])
        items = self.store.items_for(repo, state="open")
        for item in items:
            if item.get("work_rank") is None:
                item["work_rank"] = _classify.work_rank(
                    item.get("priority") or "P2", item.get("type"),
                    "M", int(item.get("reach") or 0))
        items.sort(key=lambda row: (-int(row.get("work_rank") or 0),
                                    int(row.get("number", 0))))
        return items

    def urgent(self, repo: str, *, latest_tag: str | None = None,
               keep: int = 10) -> dict[str, Any]:
        """The repo's open issues ranked by the urgency points table
        (:mod:`nomorals.triage.urgency`): score, critical flag and
        reasons per issue, the repo urgency (sum of the top five
        scores) and the critical count. Store-only — no GitHub call.
        Rows mirrored before the urgency fields landed score without
        reactions / author association, which the next sync fills in.
        """
        items = self.store.items_for(repo, state="open")
        ranked, urgency, critical = _urgency.rank_issues(
            [_urgency.IssueInput.from_dict(row) for row in items],
            self.now, latest_tag, keep)
        return {"repo": repo,
                "ranked": [row.as_dict() for row in ranked],
                "urgency": urgency, "critical": critical,
                "open": len(items), "scored": len(ranked)}

    # ── decisions ──────────────────────────────────────────────────
    def decisions(self, repo: str | None = None, *,
                  status: str = "open") -> list[dict[str, Any]]:
        return self.store.decisions(status=status, repo=repo)

    def answer_decision(self, decision_id: str, option: str, *,
                        note: str = "") -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"no decision {decision_id} on record")
        options = list(decision.get("options", []))
        if option not in options:
            raise ValueError(
                f"{option!r} is not an option here — "
                f"the options are {', '.join(options)}")
        if (decision.get("kind") == "duplicate"
                and option == "not-duplicates"):
            members = list(decision.get("members", []))
            for i, a in enumerate(members):
                for b in members[i + 1:]:
                    self.store.rule_out(a, b)
        updated = self.store.update_decision(
            decision_id, status="answered", answer=option,
            answer_note=note or "")
        if updated is None:  # pragma: no cover - get checked above
            raise ValueError(f"no decision {decision_id} on record")
        return updated

    def apply_decision(self, decision_id: str) -> dict[str, Any]:
        decision = self.store.get_decision(decision_id)
        if decision is None:
            raise ValueError(f"no decision {decision_id} on record")
        if decision.get("status") == "applied":
            return {"status": "already-applied", "id": decision_id}
        if decision.get("status") != "answered":
            return {"status": "not-answered", "id": decision_id,
                    "reason": "answer it first — /triage answer "
                              f"{decision_id} <option>"}
        if decision.get("kind") == "duplicate":
            return self._apply_duplicate(decision)
        if decision.get("kind") == "scope":
            return self._apply_scope(decision)
        return {"status": "applied", "id": decision_id}

    def _apply_duplicate(self, decision: dict[str, Any]
                         ) -> dict[str, Any]:
        decision_id = str(decision["id"])
        members = list(decision.get("members", []))
        keep = str(decision.get("keep", ""))
        repo = str(decision.get("repo", ""))
        texts = dict(decision.get("texts", {}))
        closed = 0
        for member in members:
            if member == keep:
                continue
            number = int(member.rsplit("#", 1)[1])
            write_key = f"dup-close:{decision_id}:{member}"
            if self.store.already_written(write_key):
                continue
            try:
                live = self.client.get_issue(repo, number)
            except Exception:  # noqa: BLE001 - a dead read skips
                continue
            if str(live.get("state", "")) == "closed":
                self.store.record_write(
                    {"key": write_key, "what": "already-closed",
                     "member": member})
                continue
            text = texts.get(member, DUPLICATE_CLOSE_TEXT.format(
                keep=keep.rsplit("#", 1)[1]))
            self.client.comment_issue(repo, number, text)
            self.client.close_issue(repo, number)
            self.store.record_write(
                {"key": write_key, "what": "comment+close",
                 "member": member})
            closed += 1
        self.store.update_decision(decision_id, status="applied")
        return {"status": "applied", "id": decision_id,
                "closed": closed}

    def _apply_scope(self, decision: dict[str, Any]) -> dict[str, Any]:
        decision_id = str(decision["id"])
        item = self.store.get_item(str(decision.get("item", "")))
        if item is None:
            return {"status": "failed", "id": decision_id,
                    "reason": "the item is no longer mirrored"}
        repo = str(decision.get("repo") or item.get("repo") or "")
        answer = str(decision.get("answer", ""))
        number = int(item.get("number", 0))
        label_for_answer = {
            "take": str(decision.get("pick")
                        or item.get("type") or ""),
            "small-extra": "small extra",
            "plugin": "plugin candidate",
        }
        if answer == "backlog":
            self.store.update_decision(decision_id, status="backlog")
            return {"status": "applied", "id": decision_id,
                    "wrote": "nothing — backlogged"}
        if answer == "close":
            write_key = f"scope-close:{decision_id}"
            if not self.store.already_written(write_key):
                self.client.comment_issue(
                    repo, number, "Closing this as out of scope.")
                self.client.close_issue(repo, number)
                self.store.record_write(
                    {"key": write_key, "what": "comment+close"})
            self.store.update_decision(decision_id, status="applied")
            return {"status": "applied", "id": decision_id,
                    "wrote": "closed as out of scope"}
        label = label_for_answer.get(answer, "")
        if not label:
            return {"status": "failed", "id": decision_id,
                    "reason": f"no label for answer {answer!r}"}
        write_key = f"scope-label:{decision_id}:{label}"
        if not self.store.already_written(write_key):
            labels = [str(label) for label in item.get("labels", [])
                      or []]
            if label not in labels:
                labels.append(label)
            self.client.set_issue_labels(repo, number, labels)
            self.store.record_write(
                {"key": write_key, "what": "set-labels",
                 "labels": labels})
        self.store.update_decision(decision_id, status="applied")
        return {"status": "applied", "id": decision_id,
                "wrote": f"label {label}"}
