"""Account-history export into the training pipeline.

The live-use collector (``nomorals/training/collect.py``) only sees the
conversations the companion herself had.  The owner's own account
history — their Telegram account, read through the saved userbot
session — is a much larger corpus of *how they talk and the world they
live in*.  This module harvests it, PII-scrubs it exactly like the
collector does, and registers it as a durable training dataset so
``python -m nomorals train --run --dataset <id>`` can fine-tune on it.

What the pipeline learns from it: the owner's own replies become the
``assistant`` turns, so the model's voice material shifts toward the
owner's phrasing, rhythms, and references — personalization toward the
people who share the relationship.  That is the intended effect; the
dataset is removable (unregister it or simply stop passing its id), and
nothing exported is ever sent back out to any chat platform.

Privacy rules (same contract as the collector, enforced here too):
  * every turn passes through ``scrub_pii`` (numbers, emails, urls,
    addresses, currency) before it is persisted;
  * near-duplicate conversations are dropped by the simhash dedup;
  * the corpus is written only under the local training directory and
    referenced by dataset id — it is never uploaded anywhere.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .core.logging_setup import get_logger
from .training.collect import _example_simhash, scrub_example
from .training.dataset import Example, Turn, write_jsonl
from .training.preprocess import clean_text, hamming

_log = get_logger(__name__)

__all__ = [
    "ExportResult",
    "ExportStats",
    "export_telegram",
    "history_to_examples",
    "scrub_examples",
]

# Simhash dedup distance: same threshold as TrainingCollector.
_DEDUP_THRESHOLD = 3


@dataclass
class ExportStats:
    """Counts for one export pass, safe to persist (no raw text)."""

    dialogs: int = 0
    messages: int = 0
    examples: int = 0
    pii_scrubbed: int = 0
    duplicates: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "dialogs": self.dialogs,
            "messages": self.messages,
            "examples": self.examples,
            "pii_scrubbed": self.pii_scrubbed,
            "duplicates": self.duplicates,
        }


@dataclass
class ExportResult:
    stats: ExportStats
    path: str = ""
    dataset_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats.to_dict(),
            "path": self.path,
            "dataset_id": self.dataset_id,
        }


def _entries_to_turns(entries: Iterable[tuple[bool, str]]) -> list[Turn]:
    """Merge consecutive messages from the same side into single turns.

    ``entries`` are ``(is_owner, text)`` pairs in chronological order.
    Owner messages become ``assistant`` turns — they are the voice
    material the fine-tune imitates.  Every other sender (the other
    person in a DM, any member in a group, any channel post) becomes a
    ``user`` turn, so each window reads like a normal conversation.
    """
    turns: list[Turn] = []
    for is_owner, text in entries:
        text = text.strip()
        if not text:
            continue
        role = "assistant" if is_owner else "user"
        if turns and turns[-1].role == role:
            turns[-1].content = f"{turns[-1].content}\n{text}"
        else:
            turns.append(Turn(role=role, content=text))
    return turns


def history_to_examples(
    dialogs: Iterable[tuple[str, list[tuple[bool, str]]]],
    *,
    window_turns: int = 12,
    min_turns: int = 2,
) -> list[Example]:
    """Chunk per-dialog turn sequences into fine-tune windows.

    ``dialogs`` is an iterable of ``(title, entries)`` where ``entries``
    are ``(is_owner, text)`` pairs in chronological order.  Each dialog
    is split into non-overlapping windows of ``window_turns``; a window
    is kept only when it contains at least one owner reply *and* at
    least one other-side message — a fine-tune window needs both
    something to respond to and something to imitate.  Windows of pure
    owner text (e.g. Saved Messages notes) and pure other-side text
    (channels nobody replied in) are dropped.
    """
    examples: list[Example] = []
    for title, entries in dialogs:
        turns = _entries_to_turns(entries)
        if len(turns) < min_turns:
            continue
        source = clean_text(title) or "chat"
        for start in range(0, len(turns), window_turns):
            window = turns[start : start + window_turns]
            if len(window) < min_turns:
                continue
            roles = {t.role for t in window}
            if roles != {"user", "assistant"}:
                continue
            examples.append(Example(turns=list(window), source=source))
    return examples


def scrub_examples(examples: list[Example]) -> tuple[list[Example], int, int]:
    """Apply the collector's exact hygiene pass: PII scrub, whitespace
    normalization, then simhash dedup.  Returns ``(kept, pii_count,
    duplicates)``."""
    scrubbed: list[Example] = []
    pii_count = 0
    for example in examples:
        clean = scrub_example(example)
        if clean.to_dict() != example.to_dict():
            pii_count += 1
        for turn in clean.turns:
            turn.content = clean_text(turn.content)
        clean.source = clean_text(clean.source)
        if any(turn.content for turn in clean.turns):
            scrubbed.append(clean)

    kept: list[Example] = []
    seen: list[int] = []
    duplicates = 0
    for example in scrubbed:
        fingerprint = _example_simhash(example)
        if any(hamming(fingerprint, prior) <= _DEDUP_THRESHOLD for prior in seen):
            duplicates += 1
            continue
        seen.append(fingerprint)
        kept.append(example)
    return kept, pii_count, duplicates


def _telegram_credentials(settings: Any) -> tuple[str, int, str]:
    """Session path + api id/hash from Settings, with a clear error when
    the account was never configured or never logged in."""
    chat = settings.chat
    missing = [
        name
        for name, value in (
            ("NM_TELEGRAM_API_ID (telegram_api_id)", chat.telegram_api_id),
            ("NM_TELEGRAM_API_HASH (telegram_api_hash)", chat.telegram_api_hash),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Telegram is not configured: missing " + ", ".join(missing)
            + " in ~/.nomorals/.env (get credentials at https://my.telegram.org)"
        )
    session = str(settings.resolve(chat.telegram_session))
    return session, int(chat.telegram_api_id), chat.telegram_api_hash


def export_telegram(
    settings: Any,
    *,
    db: Any,
    max_dialogs: int = 50,
    max_per_chat: int = 300,
    only_chats: str = "",
    register: bool = True,
) -> ExportResult:
    """Harvest the owner's Telegram account history and register it as a
    training dataset.

    Uses the same authenticated session as the running chat (the one
    created at first login).  Reads only — it never sends anything.
    Caps keep a phone-sized device sane: ``max_dialogs`` dialogs, at
    most ``max_per_chat`` recent messages each, oldest first.

    ``only_chats`` (comma-separated numeric chat ids) restricts the
    harvest; the chat id appears in the userbot log and in the
    ``/status`` command.
    """
    try:
        from telethon import TelegramClient
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "telethon is not installed; run: pip install telethon"
        ) from exc

    session, api_id, api_hash = _telegram_credentials(settings)
    wanted = {c.strip() for c in only_chats.split(",") if c.strip()}

    stats = ExportStats()
    harvested: list[tuple[str, list[tuple[bool, str]]]] = []

    async def _harvest() -> None:
        client = TelegramClient(session, api_id, api_hash)
        # An authenticated session connects silently.  A missing/expired
        # session falls back to the same interactive login the chat uses
        # (phone number + code), so running this in a terminal is safe.
        await client.start()
        try:
            seen = 0
            async for dialog in client.iter_dialogs():
                if seen >= max_dialogs:
                    break
                if wanted and str(dialog.id) not in wanted:
                    continue
                seen += 1
                title = (dialog.title or f"chat-{dialog.id}").strip()
                entries: list[tuple[bool, str]] = []
                # reverse=True => oldest first, which is the order the
                # conversation windowing below expects.
                async for msg in client.iter_messages(dialog.entity, limit=max_per_chat, reverse=True):
                    text = (getattr(msg, "raw_text", None) or "").strip()
                    if not text:
                        continue  # media-only / service messages
                    stats.messages += 1
                    entries.append((bool(getattr(msg, "out", False)), text))
                if entries:
                    harvested.append((title, entries))
                    stats.dialogs += 1
        finally:
            await client.disconnect()

    _log.info("telegram export: harvesting (dialogs<=%d, per-chat<=%d)", max_dialogs, max_per_chat)
    asyncio.run(_harvest())

    examples = history_to_examples(harvested)
    kept, pii_count, duplicates = scrub_examples(examples)
    stats.pii_scrubbed = pii_count
    stats.duplicates = duplicates
    stats.examples = len(kept)
    _log.info(
        "telegram export: %d dialogs, %d messages -> %d examples (%d pii, %d dupes)",
        stats.dialogs,
        stats.messages,
        stats.examples,
        pii_count,
        duplicates,
    )

    if not kept:
        return ExportResult(stats=stats)

    train_dir = Path(str(settings.resolve("data/training")))
    train_dir.mkdir(parents=True, exist_ok=True)
    name = f"exported-{time.strftime('%Y%m%d-%H%M%S')}"
    path = train_dir / f"{name}.jsonl"
    write_jsonl(path, (example.to_dict() for example in kept))

    result = ExportResult(stats=stats, path=str(path))
    if register:
        from .training.dataset import DatasetRegistry

        dataset = DatasetRegistry(db).register_examples(
            name,
            kept,
            str(train_dir),
            kind="chat",
            metadata={
                "source": "telegram-export",
                "collected": True,
                "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "collector_stats": stats.to_dict(),
            },
        )
        result.dataset_id = dataset.id
    return result


def register(registry: Any) -> None:
    """Expose Telegram history export as an agent tool for training datasets."""

    @registry.register(
        "export_training_data",
        description=(
            "Harvest the owner's Telegram chat history (read-only) and register "
            "it as a training dataset for fine-tuning. PII-scrubbed. "
            "Useful for personalizing a model toward the owner's voice."
        ),
        capability="training.export",
        parameters={
            "max_dialogs": "int — max dialogs to harvest (default 50)",
            "max_per_chat": "int — max messages per chat (default 300)",
            "only_chats": "str — comma-separated chat ids to restrict to (optional)",
        },
    )
    def _export_training_data(
        max_dialogs: int = 50,
        max_per_chat: int = 300,
        only_chats: str = "",
    ) -> dict[str, Any]:
        context = registry.context
        settings = getattr(context, "settings", None)
        db = getattr(context, "db", None)
        try:
            result = export_telegram(
                settings,
                db=db,
                max_dialogs=int(max_dialogs),
                max_per_chat=int(max_per_chat),
                only_chats=only_chats or "",
            )
            return {
                "ok": True,
                "dataset_id": getattr(result, "dataset_id", ""),
                "examples": getattr(result, "examples", 0),
                "stats": result.to_dict() if hasattr(result, "to_dict") else str(result),
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
