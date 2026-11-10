"""Personal-surface draft platforms: WhatsApp status + Telegram channel.

These two adapters make the owner's PAIRED PERSONAL accounts into
SocialManager draft platforms, through the same seams the chat
commands use — no bot token, no second posting path:

- ``whatsapp-status`` → the WhatsApp chat adapter's ``send_status``
  (the ``/poststatus`` verb: the bridges' ``status_send``).
- ``telegram`` → the Telegram userbot adapter's ``send`` into the
  channel the social account names (its handle, or
  ``metadata["channel"]``) — a channel post, the counterpart surface.

The personal-surface live gate governs both, in the same order the
commands check it (``runtime_chatops._chatops_personal_account`` +
``_cmd_poststatus``):

1. a live chat surface must exist in this process (the partner
   runtime publishes its gateway on ``context.extras["gateway"]`` —
   draft posting runs there, not in a bare CLI process);
2. the paired-account registry must READ cleanly — pairing truth is
   consulted through a fresh read-only snapshot of the registry file
   (:func:`...pairing.read_registry_snapshot`), never a second live
   ``PairingRegistry`` instance (a rival instance would hold stale
   stop state and its load path quarantines the file);
3. an account must be paired for the platform — else the post fails
   IN WORDS naming that state;
4. the owner's sticky stop governs exactly as it governs the
   commands: a stopped platform refuses in words, nothing is posted.

Journaling: the manager's own ``social_posts`` ledger records every
adapter post (platform, account, content, external id, status) — that
is this path's receipt. The personal vault's ``status_posted`` note
stays the ``/poststatus`` command's own journal: it is written through
the runtime's ``PersonalSurfaces``, which is bound to the runtime's
live registry instance for the same rival-instance reason above.

Metrics: neither surface exposes an engagement read through these
seams, so neither adapter overrides ``metrics()`` — the measurement
spine names them unavailable rather than inventing numbers.
"""

from __future__ import annotations

import time
from typing import Any

from ..base import ERROR_AUTH, ERROR_UNKNOWN, Account, PlatformAdapter, PostResult

__all__ = [
    "WhatsAppStatusAdapter",
    "TelegramChannelAdapter",
    "ContextPersonalSurface",
]


class ContextPersonalSurface:
    """The production surface: gateway + registry snapshot, resolved
    lazily from the manager's context on every post.

    Duck-typed protocol (what the adapters consume, and what tests
    fake): ``chat_adapter(platform)``, ``paired_account(platform)``,
    ``owner_stopped(platform)``, ``registry_error()``.
    """

    def __init__(self, context: Any) -> None:
        self._context = context

    def chat_adapter(self, platform: str) -> Any | None:
        extras = getattr(self._context, "extras", None)
        gateway = extras.get("gateway") if isinstance(extras, dict) else None
        if gateway is None:
            return None
        lookup = getattr(gateway, "_adapter_for", None)
        if callable(lookup):
            try:
                return lookup(platform)
            except Exception:  # noqa: BLE001 - probe, never a crash
                return None
        adapters = getattr(gateway, "adapters", None)
        if isinstance(adapters, dict):
            return adapters.get(platform)
        return None

    def _snapshot(self) -> Any:
        from ..pairing import read_registry_snapshot, registry_path_for

        settings = getattr(self._context, "settings", None)
        workspace = getattr(settings, "workspace_dir", "")
        if not workspace:
            from ..pairing import RegistrySnapshot
            return RegistrySnapshot(
                error="no workspace is configured on this context, so "
                      "the paired-account registry cannot be located")
        return read_registry_snapshot(registry_path_for(workspace))

    def registry_error(self) -> str:
        try:
            return self._snapshot().error
        except Exception as exc:  # noqa: BLE001 - verdict, not crash
            return f"the paired-account registry could not be read ({exc})"

    def paired_account(self, platform: str) -> Any | None:
        from ..pairing import select_account

        snap = self._snapshot()
        if snap.error:
            return None
        return select_account(snap.records(platform))

    def owner_stopped(self, platform: str) -> bool:
        snap = self._snapshot()
        if snap.error:
            return False
        return (platform or "").strip().lower() in snap.owner_stopped


class _PersonalAdapter(PlatformAdapter):
    """Shared gate for the personal-surface platforms."""

    #: The pairing/gateway platform this surface speaks as.
    pairing_platform = ""

    def __init__(self, context: Any = None, surface: Any = None) -> None:
        self._context = context
        self._surface = surface

    def _surface_or_default(self) -> Any:
        if self._surface is not None:
            return self._surface
        return ContextPersonalSurface(self._context)

    def _gate(self, started: float) -> tuple[Any, Any, PostResult | None]:
        """Run the live gate. Returns ``(surface, chat_adapter, None)``
        when the post may proceed, else ``(surface, None, refusal)``."""
        surface = self._surface_or_default()
        platform = self.pairing_platform
        chat = surface.chat_adapter(platform)
        if chat is None:
            return surface, None, PostResult(
                platform=self.name, ok=False,
                error=(f"no live {platform} surface in this process — "
                       f"{self.name} posts from your paired personal "
                       "account, and draft posting runs in the partner "
                       "runtime where that account is attached"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        error = surface.registry_error()
        if error:
            return surface, None, PostResult(
                platform=self.name, ok=False,
                error=(f"{error} — pairing state cannot be verified, "
                       "so nothing was posted"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        rec = surface.paired_account(platform)
        if rec is None:
            return surface, None, PostResult(
                platform=self.name, ok=False,
                error=(f"no paired {platform} account — /pair one "
                       f"first; {self.name} posts from your paired "
                       "personal account, and there isn't one"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        if surface.owner_stopped(platform):
            return surface, None, PostResult(
                platform=self.name, ok=False,
                error=(f"{platform} is under your standing stop — it "
                       "stays halted until you start it again; "
                       "nothing was posted"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        return surface, chat, None

    def engagement_gate(self) -> str:
        """The live gate's verdict for an engagement READ (WS3 P8).

        Returns ``""`` when a read may proceed, else the refusal in
        words — the same ordered checks :meth:`_gate` runs for a
        post (live surface → registry readable → account paired →
        owner's sticky stop). The engagement poll consults this
        BEFORE any read attempt on a personal surface: a stopped
        platform is never even asked. Reads are observation, so the
        degraded-brain observe-only rule does not block them; the
        sticky stop does, exactly as it blocks sends.
        """
        _surface, _chat, refusal = self._gate(time.perf_counter())
        return refusal.error if refusal is not None else ""

    def health(self, account: Account) -> bool:
        try:
            surface = self._surface_or_default()
            return surface.chat_adapter(self.pairing_platform) is not None
        except Exception:  # noqa: BLE001 - health is a probe
            return False


class WhatsAppStatusAdapter(_PersonalAdapter):
    """Queued drafts → WhatsApp status on the paired personal account."""

    name = "whatsapp-status"
    pairing_platform = "whatsapp"
    max_chars = 700  # the tone profile's status budget
    supports_media = True  # send_status takes a media path + caption
    supports_thread = False

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        _surface, chat, refusal = self._gate(started)
        if refusal is not None:
            return refusal
        media_paths = [str(p) for p in (kwargs.get("media_paths") or []) if p]
        media_path = media_paths[0] if media_paths else ""
        try:
            if media_path:
                verdict = chat.send_status(
                    "", media_path=media_path, caption=content)
            else:
                verdict = chat.send_status(content)
        except Exception as exc:  # noqa: BLE001 - bridge failure → words
            return PostResult(
                platform=self.name, ok=False, error=str(exc),
                error_code=ERROR_UNKNOWN,
                seconds=time.perf_counter() - started)
        if not isinstance(verdict, dict) or not verdict.get("ok"):
            reason = ""
            if isinstance(verdict, dict):
                reason = str(verdict.get("error") or "")
            return PostResult(
                platform=self.name, ok=False,
                error=(reason or "the WhatsApp bridge did not confirm "
                                 "the status post"),
                error_code=ERROR_UNKNOWN,
                seconds=time.perf_counter() - started)
        return PostResult(
            platform=self.name, ok=True,
            external_id=str(verdict.get("message_id") or ""),
            seconds=time.perf_counter() - started)


class TelegramChannelAdapter(_PersonalAdapter):
    """Queued drafts → posts in the owner's Telegram channel, sent by
    the paired Telegram userbot (the channel the social account names:
    its handle, or ``metadata["channel"]`` when set)."""

    name = "telegram"
    pairing_platform = "telegram"
    max_chars = 4096
    supports_media = True
    supports_thread = False

    def post(self, account: Account, content: str, **kwargs: Any) -> PostResult:
        started = time.perf_counter()
        _surface, chat, refusal = self._gate(started)
        if refusal is not None:
            return refusal
        metadata = account.metadata if isinstance(account.metadata, dict) else {}
        target = str(metadata.get("channel") or account.handle or "").strip()
        if not (target.startswith("@") or target.startswith("-100")):
            return PostResult(
                platform=self.name, ok=False,
                error=("the telegram draft account names no channel — "
                       "connect the telegram platform with the channel "
                       "(@name or -100… chat id) it should post to"),
                error_code=ERROR_AUTH,
                seconds=time.perf_counter() - started)
        from ..chat.base import ChatKind, ChatRef, MediaRef

        ref = ChatRef(platform="telegram", chat_id=target,
                      kind=ChatKind.CHANNEL)
        media_paths = [str(p) for p in (kwargs.get("media_paths") or []) if p]
        try:
            if media_paths:
                result = chat.send_media(
                    ref, MediaRef(path=media_paths[0]), caption=content)
            else:
                result = chat.send(ref, content)
        except Exception as exc:  # noqa: BLE001 - send failure → words
            return PostResult(
                platform=self.name, ok=False, error=str(exc),
                error_code=ERROR_UNKNOWN,
                seconds=time.perf_counter() - started)
        if not getattr(result, "ok", False):
            return PostResult(
                platform=self.name, ok=False,
                error=(str(getattr(result, "error", "") or "")
                       or "the telegram send did not confirm"),
                error_code=ERROR_UNKNOWN,
                seconds=time.perf_counter() - started)
        message_id = str(getattr(result, "message_id", "") or "")
        url = ""
        if message_id and target.startswith("@"):
            url = f"https://t.me/{target[1:]}/{message_id}"
        return PostResult(
            platform=self.name, ok=True, external_id=message_id, url=url,
            seconds=time.perf_counter() - started)
