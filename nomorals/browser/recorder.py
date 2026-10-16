"""Action recorder: record → replay → diff.

A :class:`Recording` is a list of actions plus DOM snapshot hashes. The
hashes are the contract: :meth:`Recording.diff` recomputes the hash of
each observed selector's current text and reports *every* mismatch —
changed text, missing selectors, and newly appeared selectors the caller
asked about. A diff is never silently ignored: :meth:`Recording.diff`
returns the full :class:`DiffReport`, and :meth:`Recording.assert_no_diff`
/ :func:`replay_or_raise` raise :class:`RecordingDiffError` carrying the
whole report.

Replay is explicit: :meth:`Recording.replay` re-runs the actions through
an executor (any callable taking an action dict) and then re-reads the
DOM through a reader (any callable taking a selector and returning the
current text). The recorder never touches the network itself.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "RecordingDiffError",
    "RecordedAction",
    "DomSnapshot",
    "DiffEntry",
    "DiffReport",
    "Recording",
    "replay_or_raise",
]

_log = get_logger(__name__)


class RecordingDiffError(AssertionError):
    """Replay found DOM differences. ``.report`` is the full DiffReport —
    the caller must handle it; it is never swallowed."""

    def __init__(self, message: str, report: "DiffReport") -> None:
        super().__init__(message)
        self.report = report


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


@dataclass
class RecordedAction:
    """One recorded browser action."""

    kind: str = ""      # navigate | click | fill | select | submit | ...
    target: str = ""    # selector / URL / field
    value: Any = None
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RecordedAction":
        raw = dict(raw or {})
        return cls(
            kind=str(raw.get("kind") or ""),
            target=str(raw.get("target") or ""),
            value=raw.get("value"),
            ts=float(raw.get("ts") or 0.0),
        )


@dataclass
class DomSnapshot:
    """A hashed DOM observation: selector + sha256 of its text at record
    time. The text itself is NOT stored — a hash proves equality without
    keeping a copy of page content that could go stale silently."""

    selector: str = ""
    text_hash: str = ""
    text_len: int = 0
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DomSnapshot":
        raw = dict(raw or {})
        return cls(
            selector=str(raw.get("selector") or ""),
            text_hash=str(raw.get("text_hash") or ""),
            text_len=int(raw.get("text_len") or 0),
            ts=float(raw.get("ts") or 0.0),
        )


@dataclass
class DiffEntry:
    """One observed difference. ``kind`` is ``changed`` (same selector,
    different text), ``missing`` (selector gone), or ``added`` (selector
    appeared that was absent at record time)."""

    selector: str = ""
    kind: str = ""  # changed | missing | added
    expected_hash: str = ""
    actual_hash: str = ""
    expected_len: int = 0
    actual_len: int = 0
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DiffReport:
    """The complete result of a diff: every entry, plus the counts. An
    empty ``entries`` list is the ONLY clean result — there is no
    'warnings-only' bucket for a diff to hide in."""

    entries: list[DiffEntry] = field(default_factory=list)
    checked: int = 0
    replayed_actions: int = 0

    @property
    def has_diffs(self) -> bool:
        return bool(self.entries)

    @property
    def changed(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "changed"]

    @property
    def missing(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "missing"]

    @property
    def added(self) -> list[DiffEntry]:
        return [e for e in self.entries if e.kind == "added"]

    def summarize(self) -> str:
        if not self.entries:
            return f"no diffs ({self.checked} selectors checked, {self.replayed_actions} actions replayed)"
        lines = [f"{len(self.entries)} diff(s) across {self.checked} selectors:"]
        for e in self.entries:
            lines.append(
                f"  - {e.kind} {e.selector}: "
                f"expected sha256 {e.expected_hash[:12] or '(none)'} "
                f"(len {e.expected_len}) vs actual "
                f"{e.actual_hash[:12] or '(none)'} (len {e.actual_len})"
                + (f" — {e.detail}" if e.detail else ""))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self.entries],
            "checked": self.checked,
            "replayed_actions": self.replayed_actions,
            "has_diffs": self.has_diffs,
        }


class Recording:
    """A record → replay → diff session."""

    def __init__(self, name: str = "") -> None:
        self.name = (name or "").strip() or "recording"
        self.actions: list[RecordedAction] = []
        self.snapshots: list[DomSnapshot] = []
        self.started_at: float = time.time()

    # -- recording -----------------------------------------------------------
    def record_action(self, kind: str, target: str = "",
                      value: Any = None) -> RecordedAction:
        """Append one action. Empty ``kind`` is refused — an action with
        no kind cannot be replayed."""
        kind = (kind or "").strip()
        if not kind:
            raise ValueError("record_action needs a kind")
        action = RecordedAction(kind=kind, target=target or "", value=value,
                                ts=time.time())
        self.actions.append(action)
        return action

    def snapshot(self, selector: str, text: str) -> DomSnapshot:
        """Hash the current text of ``selector`` and store the snapshot.
        Re-snapshotting a selector replaces its earlier snapshot — the
        latest observation is the contract."""
        selector = (selector or "").strip()
        if not selector:
            raise ValueError("snapshot needs a selector")
        snap = DomSnapshot(selector=selector, text_hash=_sha256_text(text or ""),
                           text_len=len(text or ""), ts=time.time())
        for i, existing in enumerate(self.snapshots):
            if existing.selector == selector:
                self.snapshots[i] = snap
                return snap
        self.snapshots.append(snap)
        return snap

    # -- diff ------------------------------------------------------------------
    def diff(self, current: dict[str, str | None],
             *, extra_selectors: list[str] | None = None) -> DiffReport:
        """Compare ``current`` (selector -> present text, None = absent)
        against the recorded snapshots.

        Every recorded selector is checked. ``extra_selectors`` are
        selectors that were ABSENT at record time: if any of them is
        present now, it is reported as ``added`` — new content is a diff
        too, not background noise. The report is complete; nothing is
        dropped, downgraded, or summarized away.
        """
        entries: list[DiffEntry] = []
        checked = 0
        for snap in self.snapshots:
            checked += 1
            if snap.selector not in current:
                entries.append(DiffEntry(
                    selector=snap.selector, kind="missing",
                    expected_hash=snap.text_hash, actual_hash="",
                    expected_len=snap.text_len, actual_len=0,
                    detail="selector absent at replay time"))
                continue
            text = current[snap.selector]
            if text is None:
                entries.append(DiffEntry(
                    selector=snap.selector, kind="missing",
                    expected_hash=snap.text_hash, actual_hash="",
                    expected_len=snap.text_len, actual_len=0,
                    detail="selector read returned None (absent)"))
                continue
            actual_hash = _sha256_text(text)
            if actual_hash != snap.text_hash:
                entries.append(DiffEntry(
                    selector=snap.selector, kind="changed",
                    expected_hash=snap.text_hash, actual_hash=actual_hash,
                    expected_len=snap.text_len, actual_len=len(text),
                    detail="text differs from recorded snapshot"))
        for selector in (extra_selectors or []):
            checked += 1
            text = current.get(selector)
            if text is not None:
                entries.append(DiffEntry(
                    selector=selector, kind="added",
                    expected_hash="", actual_hash=_sha256_text(text),
                    expected_len=0, actual_len=len(text),
                    detail="selector was absent at record time, present now"))
        return DiffReport(entries=entries, checked=checked,
                          replayed_actions=len(self.actions))

    def assert_no_diff(self, current: dict[str, str | None],
                       *, extra_selectors: list[str] | None = None) -> DiffReport:
        """Diff, and raise :class:`RecordingDiffError` (carrying the full
        report) when anything differs. The only way to 'ignore' a diff
        is to catch this exception and handle the report explicitly."""
        report = self.diff(current, extra_selectors=extra_selectors)
        if report.has_diffs:
            raise RecordingDiffError(
                f"replay diff for {self.name!r}:\n{report.summarize()}", report)
        return report

    # -- replay ------------------------------------------------------------------
    def replay(
        self,
        executor: Callable[[dict[str, Any]], dict[str, Any]] | None,
        dom_reader: Callable[[str], str | None],
        *,
        extra_selectors: list[str] | None = None,
    ) -> DiffReport:
        """Re-run every action through ``executor`` (None = skip actions,
        diff only), re-read every snapshotted selector through
        ``dom_reader``, and return the complete diff report.

        An executor failure does NOT produce a silent pass: the exception
        propagates, and no diff report is returned for a replay that
        never ran. A replay whose actions ran but whose DOM differs
        returns a report with ``has_diffs`` True — the caller must check
        it (or use :func:`replay_or_raise`).
        """
        for action in self.actions:
            if executor is not None:
                executor(action.to_dict())
        current: dict[str, str | None] = {}
        for snap in self.snapshots:
            current[snap.selector] = dom_reader(snap.selector)
        if extra_selectors:
            for selector in extra_selectors:
                if selector not in current:
                    current[selector] = dom_reader(selector)
        report = self.diff(current, extra_selectors=extra_selectors)
        report.replayed_actions = len(self.actions)
        return report

    # -- persistence -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "name": self.name,
            "started_at": self.started_at,
            "actions": [a.to_dict() for a in self.actions],
            "snapshots": [s.to_dict() for s in self.snapshots],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Recording":
        raw = dict(raw or {})
        rec = cls(name=str(raw.get("name") or ""))
        rec.started_at = float(raw.get("started_at") or time.time())
        rec.actions = [RecordedAction.from_dict(a) for a in (raw.get("actions") or [])]
        rec.snapshots = [DomSnapshot.from_dict(s) for s in (raw.get("snapshots") or [])]
        return rec

    def save(self, path: str | Path) -> Path:
        dest = Path(path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(dest)
        return dest

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"no recording at {p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"corrupt recording {p}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"corrupt recording {p}: top level must be an object")
        return cls.from_dict(raw)


def replay_or_raise(recording: Recording,
                    executor: Callable[[dict[str, Any]], dict[str, Any]] | None,
                    dom_reader: Callable[[str], str | None],
                    *, extra_selectors: list[str] | None = None) -> DiffReport:
    """Replay and raise :class:`RecordingDiffError` on any diff. Use this
    when a diff must stop the run — never check-then-ignore."""
    report = recording.replay(executor, dom_reader, extra_selectors=extra_selectors)
    if report.has_diffs:
        raise RecordingDiffError(
            f"replay diff for {recording.name!r}:\n{report.summarize()}", report)
    return report
