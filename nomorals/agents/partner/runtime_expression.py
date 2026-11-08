"""RuntimeExpressionMixin: the media-expression layer (DESIGN
§21 + §22) in the live event path.

The organ itself (``social/chat/expression.py``) is
transport-free; this mixin owns the live wiring:

- **Moments**: the group runtime calls :meth:`_fire_expression`
  after a welcome / celebration notice / ban has actually been
  posted.  Firing respects, in order: the §31 quiet switch
  (defense autonomy off → silent), the §33 system-output gate
  (claimed rooms only, evaluated at send time through the one
  shared helper), and the group's own expression settings.
  Every outcome — sent or not, and why — lands in the group's
  held audit as an ``expression`` entry: an unsent GIF is a
  journaled non-event, never a claimed send.
- **Commands**: /steal (claim a replied sticker/GIF into the
  library — and a sticker also lands in the THIEF'S personal
  pack, §21 via social/chat/sticker_packs.py), /sticker (the
  maker; ``pack`` lists/sends the caller's pack, ``steal``
  is the steal verb under this surface too), /convert (the
  conversion organ), /gif (library + providers), /emoji (the
  full Unicode table), /customemoji (her own set), /react
  (place a reaction).  In groups, their media posts pass the
  same §33 gate — her library does not even GROW from a
  foreign room, and neither does any pack.
- Expression never breaks its host: every path is guarded, and
  a failure degrades to text or silence, never to an exception
  in the message pipeline.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .failures import record_failure

_log = logging.getLogger(__name__)

_CELEBRATIONS = frozenset({
    "quest_complete", "jackpot", "season_podium", "duel_win",
    "estate_milestone", "celebration"})


class RuntimeExpressionMixin:
    """Expression moments + commands for PartnerRuntime."""

    # ---- the service -------------------------------------------
    def _expression_root(self) -> Path:  # type: ignore[no-untyped-def]
        """Where the library lives: beside the group store, so
        expression state travels with group state."""
        try:
            store_path = Path(self._group_surface().store.path)
            if str(store_path):
                return store_path.parent
        except Exception:  # noqa: BLE001 - fall back to the layout
            pass
        ws = getattr(self.context.settings, "workspace_dir", ".")
        return Path(str(ws)) / "data" / "social"

    def _expression_service(self):  # type: ignore[no-untyped-def]
        service = getattr(self, "_expression_service_obj", None)
        if service is not None:
            return service
        from ...social.chat.expression import (
            ExpressionLibrary, ExpressionService, default_providers)

        root = self._expression_root()
        library = ExpressionLibrary(root / "expression_library.json")
        settings = getattr(self.context, "settings", None)
        tenor = str(getattr(settings, "tenor_api_key", "")
                    or os.environ.get("NM_TENOR_API_KEY", "") or "")
        giphy = str(getattr(settings, "giphy_api_key", "")
                    or os.environ.get("NM_GIPHY_API_KEY", "") or "")
        extra = getattr(settings, "gif_providers", None) or ()
        service = ExpressionService(
            library,
            providers=default_providers(tenor_key=tenor,
                                        giphy_key=giphy,
                                        extra=extra),
            media_dir=root / "expression_media")
        self._expression_service_obj = service
        return service

    def _expression_library(self):  # type: ignore[no-untyped-def]
        return self._expression_service().library

    def _combo_library(self):  # type: ignore[no-untyped-def]
        """Her combo library (emoji art organ): one JSON
        beside the expression library, seeded with her own
        drawn pieces."""
        lib = getattr(self, "_combo_library_obj", None)
        if lib is not None:
            return lib
        from ...social.chat.emoji_art import ComboLibrary

        lib = ComboLibrary(self._expression_root()
                           / "combo_library.json")
        self._combo_library_obj = lib
        return lib

    def _emojiart_remember(self, message, text: str) -> None:  # type: ignore[no-untyped-def]
        """The room's last art piece — the remix verbs act on
        it ('invert that')."""
        try:
            keep = getattr(self, "_emojiart_last", None)
            if keep is None:
                keep = self._emojiart_last = {}
            keep[str(message.chat.key)] = str(text)
        except Exception:  # noqa: BLE001 - memory is garnish
            pass

    def _emojiart_last_for(self, message) -> str:  # type: ignore[no-untyped-def]
        keep = getattr(self, "_emojiart_last", None) or {}
        return str(keep.get(str(message.chat.key), "") or "")

    def _sticker_pack_store(self):  # type: ignore[no-untyped-def]
        """The per-user pack store: one JSON beside the
        expression library, so packs travel with it."""
        store = getattr(self, "_sticker_pack_store_obj", None)
        if store is not None:
            return store
        from ...social.chat.sticker_packs import StickerPackStore

        store = StickerPackStore(
            self._expression_root() / "sticker_packs.json")
        self._sticker_pack_store_obj = store
        return store

    def _pack_person(self, message) -> str:  # type: ignore[no-untyped-def]
        """The pack key: the §29 person identity (canonical
        platform + the platform's own id) — the same person
        keeps ONE pack across the bot and userbot hats, and
        across every room."""
        from ...community.group_gangs import person_key

        platform = str(getattr(message.chat, "platform", "")
                       or "")
        user_id = str(getattr(message, "sender_id", "")
                      or getattr(message, "sender", "") or "")
        return person_key(platform, user_id)

    # ---- sending -------------------------------------------------
    def _expression_send(self, message, ref, caption: str = ""):  # type: ignore[no-untyped-def]
        """Send one MediaRef through the room's own adapter
        (native media path — Telegram stickers/animations,
        Discord uploads, the WhatsApp bridge).  Falls back to
        the gateway's file send; reports honestly when neither
        exists on this platform path.  ``caption`` rides the
        media message itself (every adapter's send_media and
        the gateway's send_file carry one) — it is only passed
        when non-empty, so existing callers and adapter shapes
        are untouched."""
        from types import SimpleNamespace

        platform = str(getattr(message.chat, "platform", "") or "")
        adapter = self._group_adapter(platform)
        fn = getattr(adapter, "send_media", None) if adapter else None
        if fn is not None:
            if caption:
                return fn(message.chat, ref, caption=caption)
            return fn(message.chat, ref)
        gateway = getattr(self, "gateway", None)
        send_file = getattr(gateway, "send_file", None)
        if send_file is not None and ref.path:
            return send_file(platform, message.chat, ref.path,
                             caption=caption)
        return SimpleNamespace(
            ok=False, platform=platform, message_id="",
            media_file_id="",
            error="this platform path has no media send")

    # ---- moments ---------------------------------------------------
    def _fire_expression(self, message, moment: str) -> dict:  # type: ignore[no-untyped-def]
        """Fire one expression moment for this message's room.
        Never raises; the outcome dict says what happened."""
        try:
            surface = self._group_surface()
            key = message.chat.key
            settings = surface.store.settings(key)
            if (settings.get("defense") or {}).get(
                    "autonomy", True) is False:
                surface._audit(
                    key, "expression", moment=str(moment),
                    sent=False,
                    reason="defense autonomy off (§31 quiet)")
                surface.store.save()
                return {"sent": False, "reason": "quiet"}
            if not self._system_send_ok(surface, key, message,
                                        "expression"):
                return {"sent": False,
                        "reason": "system output not claimed here"}
            service = self._expression_service()
            out = service.fire(
                str(moment), room=key,
                platform=str(message.chat.platform or ""),
                settings=settings.get("expression") or {},
                send=lambda ref: self._expression_send(message, ref))
            surface._audit(
                key, "expression", moment=str(moment),
                sent=bool(out.get("sent")),
                reason=str(out.get("reason") or ""),
                item_id=str(out.get("item_id") or ""))
            surface.store.save()
            if out.get("sent") and str(moment) in _CELEBRATIONS \
                    and message.message_id:
                self._react_quietly(message, str(moment))
            return out
        except Exception:  # noqa: BLE001 - expression never breaks its host
            _log.debug("expression fire failed", exc_info=True)
            return {"sent": False, "reason": "expression unavailable"}

    def _react_quietly(self, message, context: str) -> None:  # type: ignore[no-untyped-def]
        """A celebration also earns her reaction on the message
        that caused it, where the adapter carries reactions —
        best-effort, unreported (the audit already has the
        moment; a reaction is garnish, not a record)."""
        try:
            from ...social.chat.expression import pick_reaction

            adapter = self._group_adapter(
                str(message.chat.platform or ""))
            fn = getattr(adapter, "react", None) if adapter else None
            if fn is None:
                return
            emoji = pick_reaction(
                context, platform=str(message.chat.platform or ""))
            if emoji:
                fn(message.chat, str(message.message_id), emoji)
        except Exception:  # noqa: BLE001 - garnish only
            _log.debug("celebration reaction failed", exc_info=True)

    # ---- replied-media resolution (steal / convert) ----------------
    def _replied_media_local(self, message):  # type: ignore[no-untyped-def]
        """Resolve the media a command acts on, in order: the
        message's own attachment, the replied message's bytes
        when the surface already downloaded them at inbound
        (``replied_media_local`` meta — the WhatsApp bridges
        download the quoted message's media with the event, and
        the Telegram userbot downloads the replied message on
        arrival; a reply is the only moment either surface can
        reach those bytes), the replied message's Telegram
        handles (downloaded on demand through the adapter — the
        /sham precedent), the replied audio id, and Discord's
        fetch-by-message.  Returns ``(path, kind, file_ids)`` or
        None when this surface cannot yield the bytes."""
        platform = str(getattr(message.chat, "platform", "") or "")
        adapter = self._group_adapter(platform)
        for media in getattr(message, "media", None) or []:
            path = str(getattr(media, "path", "") or "")
            file_id = str(getattr(media, "file_id", "") or "")
            if path or file_id:
                return (path, str(getattr(media, "kind", "") or ""),
                        {platform: file_id} if file_id else {})
        meta = getattr(message, "meta", None) or {}
        local = meta.get("replied_media_local") or {}
        if isinstance(local, dict) and local.get("path"):
            return (str(local["path"]),
                    str(local.get("kind") or ""), {})
        replied = meta.get("replied_media") or {}
        download = getattr(adapter, "download_file", None) \
            if adapter else None
        if isinstance(replied, dict) and replied.get("file_id"):
            file_ids = {platform: str(replied["file_id"])}
            if download is not None:
                path = download(str(replied["file_id"]),
                                str(replied.get("name") or "media"))
                if path:
                    return (str(path),
                            str(replied.get("kind") or ""), file_ids)
            # No bytes, but the handle itself is stealable on
            # Telegram: the library can hold and re-send it.
            return ("", str(replied.get("kind") or ""), file_ids)
        audio_id = str(meta.get("replied_audio_file_id") or "")
        if audio_id and download is not None:
            path = download(audio_id)
            if path:
                return (str(path), "audio", {platform: audio_id})
        fetch = getattr(adapter, "fetch_media", None) if adapter \
            else None
        reply_to = str(getattr(message, "reply_to", "") or "")
        if fetch is not None and reply_to:
            ref = fetch(message.chat, reply_to)
            if ref is not None and getattr(ref, "path", ""):
                return (str(ref.path), str(ref.kind or ""), {})
        return None

    def _group_gate(self, message, what: str) -> bool:  # type: ignore[no-untyped-def]
        """Group gate for expression commands: media posts and
        library growth are bot-system output (§33).  DMs are
        the owner's own room — never gated."""
        from ...social.chat.base import ChatKind

        if getattr(message.chat, "kind", None) != ChatKind.GROUP:
            return True
        surface = self._group_surface()
        return self._system_send_ok(surface, message.chat.key,
                                    message, what)

    # ---- auto-collect (§21: the stealer's passive mode) --------
    def _auto_collect_media(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """Passively collect stickers landing in this room
        into her library — the auto-collect mode §21 names
        ("an auto-collect mode a group admin enables"), run
        as one beat of the group message cycle.

        Scope is the §33 posture applied to collecting:
        groups only (never DMs), the room must be claimed
        for her bot systems (checked quietly through the
        surface's own predicate — a passive beat journals
        no suppression per sticker), and the group's
        expression settings must carry the admin-enabled
        ``auto_collect`` toggle.  Foreign rooms collect
        nothing and hold no state from this.  The kind
        check runs first, so non-sticker traffic costs one
        comparison.  Never raises: a capture failure is
        recorded in the failure ledger and the room's
        message handling proceeds untouched."""
        try:
            from ...social.chat.base import ChatKind

            if getattr(message.chat, "kind", None) != ChatKind.GROUP:
                return
            stickers = [
                m for m in (getattr(message, "media", None) or [])
                if str(getattr(m, "kind", "") or "") == "sticker"]
            if not stickers:
                return
            settings = surface.store.settings(key)
            expr = settings.get("expression") or {}
            if not expr.get("auto_collect"):
                return
            platform = str(getattr(message.chat, "platform", "")
                           or "")
            if not surface.system_output_allowed(
                    key, via_platform=platform):
                return
            from ...social.chat.expression import (
                CaptureError, capture_library_sticker)

            library = self._expression_library()
            media_dir = (self._expression_root()
                         / "expression_media")
            adapter = self._group_adapter(platform)
            download = getattr(adapter, "download_file", None) \
                if adapter else None
            for entry in stickers:
                try:
                    path = str(getattr(entry, "path", "") or "")
                    file_id = str(getattr(entry, "file_id", "")
                                  or "")
                    file_ids = ({platform: file_id}
                                if file_id else {})
                    if not path and file_id \
                            and download is not None:
                        path = str(download(
                            file_id,
                            str(getattr(entry, "name", "")
                                or "sticker")) or "")
                    capture_library_sticker(
                        library, path=path, file_ids=file_ids,
                        media_dir=media_dir, origin=key,
                        name=str(getattr(entry, "name", "")
                                 or ""))
                except CaptureError as exc:
                    # An honest refusal (the room's passive
                    # bound, or nothing holdable): a held
                    # audit line, never a failure — and the
                    # next sticker in the room is unaffected.
                    surface._audit(key, "auto_collect_refused",
                                   reason=str(exc))
                except Exception as exc:  # noqa: BLE001 - collection never breaks the room
                    record_failure("expression.auto_collect",
                                   exc)
                    _log.debug("auto-collect failed",
                               exc_info=True)
        except Exception as exc:  # noqa: BLE001 - the beat never breaks the turn
            record_failure("expression.auto_collect", exc)
            _log.debug("auto-collect beat failed", exc_info=True)

    # ---- /steal ------------------------------------------------------
    def _control_steal(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Claim a replied sticker/GIF/animation into her
        library, tagged by the /steal tail words + its kind."""
        if message is None:
            return "reply to a sticker or GIF with /steal to claim it."
        found = self._replied_media_local(message)
        if found is None:
            return ("reply to a sticker, GIF or video with /steal "
                    "and I'll add it to my library — on this "
                    "platform path I couldn't reach the media "
                    "itself, so there's nothing to claim yet.")
        if not self._group_gate(message, "steal"):
            return ("this room isn't claimed for my bot systems "
                    "(§33), so I don't collect media here.")
        path, kind, file_ids = found
        tags = [w.lower() for w in str(tail or "").split() if w]
        if kind:
            tags.append(kind)
        if str(kind or "") == "sticker":
            # The thief's own copy (§21 packs): a sticker also
            # lands in the caller's personal pack, normalized
            # and pack-owned (or held by reference, honestly,
            # when the bytes never reached us).  A pack refusal
            # is the steal's refusal — nothing is claimed
            # anywhere until both captures can happen.
            from ...social.chat.sticker_packs import (
                PackError, capture_sticker)

            replied = (getattr(message, "meta", None) or {}).get(
                "replied_media") or {}
            try:
                capture_sticker(
                    self._sticker_pack_store(),
                    self._pack_person(message),
                    path=path, kind=kind, file_ids=file_ids,
                    platform=str(message.chat.platform or ""),
                    media_dir=self._expression_root()
                    / "expression_media",
                    origin=message.chat.key,
                    name=str(replied.get("name")
                             or "stolen sticker"))
            except PackError as exc:
                return str(exc)
        library = self._expression_library()
        item = library.add_item(
            kind=kind or "gif", path=path, file_ids=file_ids,
            tags=tags, source="steal", origin=message.chat.key)
        from ...social.chat.base import MediaRef

        ref = MediaRef(path=path, kind=kind or "gif",
                       file_id=str(file_ids.get(
                           str(message.chat.platform or "")) or ""),
                       name="stolen media")
        result = self._expression_send(message, ref)
        if getattr(result, "ok", False):
            captured = str(getattr(result, "media_file_id", "") or "")
            if captured:
                library.note_file_id(
                    str(item.get("id")),
                    str(message.chat.platform or ""), captured)
            library.note_use(str(item.get("id")),
                             room=message.chat.key)
            return ""
        return (f"saved to my library — but the re-send failed "
                f"here: {getattr(result, 'error', '')}")

    # ---- /sticker (the maker) -----------------------------------------
    def _control_sticker(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Make a sticker: text/emoji → the house maker's real
        raster sticker; a replied image → image-to-sticker; a
        replied video/GIF → animated sticker.  The ``pack``
        and ``steal`` heads route to the caller's personal
        pack and the stealer instead of the maker."""
        if message is None:
            return "usage: /sticker <text> — or reply to an image/video."
        from ...social.chat import media_convert
        from ...social.chat.base import MediaRef

        text = str(tail or "").strip()
        words = text.split()
        head = words[0].lower() if words else ""
        if head == "pack":
            return self._control_sticker_pack(words[1:], message)
        if head == "steal":
            return self._control_steal(" ".join(words[1:]),
                                        message)
        out_dir = self._expression_root() / "expression_media"
        produced = ""
        note = ""
        if text:
            made = media_convert.make_text_sticker(
                text, out_dir=out_dir)
            produced = str(made.get("raster") or "")
            note = str(made.get("note") or "")
            if not produced:
                return (f"the SVG sticker is ready "
                        f"({made['spec'].id}) — {note}")
        else:
            found = self._replied_media_local(message)
            if found is None or not found[0]:
                return ("usage: /sticker <text> — or reply to an "
                        "image or a short video and I'll make it "
                        "a sticker.")
            path, kind, _ids = found
            try:
                if kind in ("video", "animation", "gif"):
                    produced = str(media_convert.video_to_sticker(
                        path, out_dir=out_dir))
                else:
                    produced = str(media_convert.image_to_sticker(
                        path, out_dir=out_dir))
            except media_convert.ConversionError as exc:
                return str(exc)
        if not self._group_gate(message, "sticker"):
            return ("the sticker is made, but this room isn't "
                    "claimed for my bot-system posts (§33), so "
                    "I'm not dropping it here.")
        # Made stickers join her media library (provenance
        # journaled) — the maker feeds the shelf she draws
        # moments from.
        try:
            self._expression_library().add_item(
                kind="sticker", path=str(produced),
                tags=("sticker",), source="self:sticker-maker")
        except Exception as exc:  # noqa: BLE001 - shelving is garnish
            record_failure("expression.sticker.shelve", exc)
        ref = MediaRef(path=produced, kind="sticker",
                       name=Path(produced).name)
        result = self._expression_send(message, ref)
        if getattr(result, "ok", False):
            return ""
        return (f"sticker made, but the send failed here: "
                f"{getattr(result, 'error', '')}")

    # ---- /sticker pack --------------------------------------------------
    def _control_sticker_pack(self, args, message=None) -> str:  # type: ignore[no-untyped-def]
        """The caller's personal pack (§21): `/sticker pack`
        lists it, `/sticker pack <n>` sends entry n.  Listing
        and sending are bot-system output in a group — the
        same §33 gate as every expression post; in DM the
        pack is simply the owner's own shelf."""
        if message is None:
            return ("usage: /sticker pack — your stolen "
                    "stickers; /sticker pack <n> sends one.")
        if not self._group_gate(message, "sticker"):
            return ("this room isn't claimed for my bot "
                    "systems (§33), so packs stay shut here "
                    "— your pack works in DM.")
        store = self._sticker_pack_store()
        person = self._pack_person(message)
        items = store.pack(person)
        if not args:
            if not items:
                return ("your sticker pack is empty — reply "
                        "to a sticker with /steal and it "
                        "lands here.")
            lines = [f"your sticker pack ({len(items)}):"]
            for i, item in enumerate(items, 1):
                shape = ("by reference"
                         if item.get("by_reference")
                         else str(item.get("format")
                                  or "sticker"))
                lines.append(
                    f"  {i}. {item.get('name') or 'sticker'}"
                    f" — {shape}")
            lines.append("/sticker pack <n> sends one.")
            return "\n".join(lines)
        if not str(args[0]).isdigit():
            return "usage: /sticker pack [number]"
        item = store.get(person, int(args[0]))
        if item is None:
            return (f"no sticker #{args[0]} in your pack — "
                    f"/sticker pack lists them.")
        platform = str(message.chat.platform or "")
        held = item.get("file_ids") or {}
        if item.get("by_reference") and platform not in held:
            return ("that sticker lives as a reference from "
                    "another platform — I hold its handle "
                    "there, not bytes I can send here.")
        from ...social.chat.base import MediaRef

        ref = MediaRef(
            path=str(item.get("path") or ""), kind="sticker",
            file_id=str(held.get(platform) or ""),
            name=str(item.get("name") or "sticker"))
        if not ref.path and not ref.file_id:
            return ("that pack entry has neither bytes nor a "
                    "handle for this platform — it can't be "
                    "re-sent here.")
        result = self._expression_send(message, ref)
        if getattr(result, "ok", False):
            store.note_use(person, str(item.get("id")))
            return ""
        return (f"couldn't send that pack sticker here: "
                f"{getattr(result, 'error', '')}")

    # ---- /convert -------------------------------------------------------
    def _control_convert(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """One conversion, by name, on the replied media (or the
        message's own).  `/convert` alone lists the verbs."""
        from ...social.chat import media_convert

        words = str(tail or "").strip().lower().split()
        if not words or words[0] in ("list", "help"):
            lines = ["conversions I run (reply to the media, then "
                     "name one):"]
            for name, (_fn, blurb) in sorted(
                    media_convert.CONVERSIONS.items()):
                lines.append(f"  /convert {name} — {blurb}")
            return "\n".join(lines)
        if message is None:
            return "reply to the media and name the conversion."
        name = words[0]
        found = self._replied_media_local(message)
        if found is None or not found[0]:
            return ("I couldn't reach that media's bytes on this "
                    "platform path — nothing to convert.")
        out_dir = self._expression_root() / "expression_media"
        try:
            produced = media_convert.convert_media(
                name, found[0], out_dir=out_dir)
        except media_convert.ConversionError as exc:
            return str(exc)
        if not self._group_gate(message, "convert"):
            return ("converted — but this room isn't claimed for "
                    "my bot-system posts (§33), so I'm not "
                    "dropping the file here.")
        from ...social.chat.base import MediaRef

        kind = {"sticker-to-mp4": "animation",
                "gif-to-mp4": "animation",
                "mp4-to-mp3": "audio",
                "image-to-sticker": "sticker",
                "video-to-sticker": "sticker",
                "mp4-to-gif": "animation"}.get(name, "video")
        ref = MediaRef(path=str(produced), kind=kind,
                       name=Path(str(produced)).name)
        result = self._expression_send(message, ref)
        if getattr(result, "ok", False):
            return ""
        return (f"converted, but the send failed here: "
                f"{getattr(result, 'error', '')}")

    # ---- /gif ---------------------------------------------------------
    def _control_gif(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """A GIF on demand: her library first (it grows from
        steals and sends), then the provider chain."""
        if message is None:
            return "usage: /gif <what you're feeling>"
        if not self._group_gate(message, "gif"):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — my GIFs only play where my bot runs.")
        service = self._expression_service()
        out = service.gif_for(
            str(tail or ""), room=message.chat.key,
            platform=str(message.chat.platform or ""),
            send=lambda ref: self._expression_send(message, ref))
        if out.get("sent"):
            return ""
        return str(out.get("reason") or "no GIF for that yet")

    # ---- per-emotion reactions (/hug, /pat, … — wave C) -------------
    def _control_reaction(self, kind: str, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """One per-emotion reaction GIF, captioned on the
        media message itself — one message total, so success
        returns "" (the media WAS the reply; the kind rides
        MEDIA_DELIVERED_KINDS + REACTION_GIF_COMMANDS on the
        control path).  The target resolves reply-first, then
        @mention/name through the §29 resolver, else the
        reaction is solo and Devon performs it.  A name that
        resolves nowhere earns the resolver's honest line —
        never a silent solo at the wrong person."""
        if message is None:
            return (f"usage: /{kind} — or /{kind} @someone, or "
                    f"reply to their message with /{kind}.")
        if not self._group_gate(message, kind):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — my reaction GIFs only play where my "
                    "bot runs.")
        from ...social.chat.expression import reaction_caption

        target_name = ""
        named = bool(str(tail or "").strip())
        if named or getattr(message, "reply_to", ""):
            try:
                resolution = self._resolver(
                    self._group_surface()).resolve_command(
                        message.chat.key, message, str(tail or ""))
            except Exception as exc:  # noqa: BLE001 - resolution never breaks a fun command
                record_failure("expression.reaction.resolve", exc)
                resolution = None
            if resolution is not None:
                if resolution.member is not None:
                    member = resolution.member
                    target_name = str(member.name or "") or (
                        f"@{member.username}" if member.username
                        else str(member.user_id or ""))
                elif named and not resolution.via_reply \
                        and not resolution.channel_post:
                    return resolution.exhaustion_text()
        actor = str(getattr(message, "sender", "") or "").strip() \
            or "Someone"
        owner = False
        try:
            owner = bool(self._operator_or_false(message))
        except Exception:  # noqa: BLE001 - no authority surface here
            owner = False
        caption = reaction_caption(kind, actor=actor,
                                   target=target_name, owner=owner)
        service = self._expression_service()
        out = service.reaction_gif(
            kind, room=message.chat.key,
            platform=str(message.chat.platform or ""),
            send=lambda ref: self._expression_send(
                message, ref, caption=caption))
        if out.get("sent"):
            return ""
        return str(out.get("reason") or f"no /{kind} GIF yet")

    # ---- /emoji ---------------------------------------------------------
    def _control_emoji(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Three powers.  FIND: the full Unicode table,
        searchable by meaning — any emoji that exists,
        including the ones some phones can't render; given an
        emoji itself, it reports what it is.  CREATE:
        ``/emoji make <words>`` authors a REAL custom emoji —
        her own art at 100×100, registered in her Telegram
        custom-emoji pack where the surface allows, living in
        her /customemoji set with a unicode fallback — and a
        search no single emoji confidently covers ('family
        rocket') is a creation too, never a dump of near-miss
        parts.  STICKER: ``/emoji sticker <words>`` is the
        sticker maker, named for what it makes (the two were
        conflated until the owner's 2026-11-07 correction:
        emoji are not stickers)."""
        from ...social.chat.expression import (
            confident_emoji_match,
            emoji_info,
            search_emoji,
        )

        query = str(tail or "").strip()
        if not query:
            return ("usage: /emoji <words> finds one in the "
                    "whole Unicode table — melting face, thumbs "
                    "up medium, rocket… /emoji make <words> "
                    "creates a real custom emoji of my own. "
                    "/emoji sticker <words> makes a sticker. "
                    "Or give me an emoji and I'll tell you "
                    "what it is.")
        words = query.split(None, 1)
        head = words[0].lower()
        rest = words[1].strip() if len(words) > 1 else ""
        if head in ("make", "create"):
            description = rest
            for prefix in ("an emoji of ", "emoji of ", "of "):
                if description.lower().startswith(prefix):
                    description = description[len(prefix):]
                    break
            if not description:
                return ("usage: /emoji make <what it should "
                        "be> — a laughing cat, a gold rocket…")
            return self._emoji_create_and_send(description, message)
        if head == "sticker":
            if not rest:
                return ("usage: /emoji sticker <what it should "
                        "be> — that makes a sticker (for a real "
                        "emoji, /emoji make <words>).")
            return self._emoji_sticker_and_send(rest, message)
        info = emoji_info(query)
        if info is not None and info["char"] == query:
            kws = ", ".join(info["keywords"][:8])
            return (f"{info['char']} — {info['name']} "
                    f"({info['codepoints']})"
                    + (f" · also: {kws}" if kws else ""))
        hits = search_emoji(query, limit=6)
        if hits and confident_emoji_match(query) is not None:
            faces = " ".join(entry.char for entry in hits)
            names = " · ".join(f"{entry.char} {entry.name}"
                                for entry in hits[:3])
            return f"{faces}\n{names}"
        return self._emoji_create_and_send(query, message)

    @staticmethod
    def _emoji_name_for(description: str, registry) -> str:  # type: ignore[no-untyped-def]
        """A registry name for a created emoji: the described
        words, slug-shaped, uniquified against her set."""
        import re as _re

        slug = _re.sub(r"[^a-z0-9]+", "_",
                       str(description or "").lower()).strip("_")
        slug = slug[:20].strip("_") or "emoji"
        name, n = slug, 2
        while registry.get(name) is not None:
            name = f"{slug}_{n}"
            n += 1
        return name

    def _emoji_create_and_send(self, description: str, message) -> str:  # type: ignore[no-untyped-def]
        """Create one REAL custom emoji: her studio draws the
        design at 100×100, the piece joins her custom set
        (name → unicode fallback + art), and on a Telegram bot
        surface it registers in her own custom-emoji pack —
        the pack hands back a real custom_emoji_id, and the
        emoji is SENT as an emoji (entity-carried), not as a
        sticker.  Surfaces that cannot register (WhatsApp has
        no custom emoji at all — a platform wall) receive the
        art as a sticker with the wall named.  Every failure
        is a plain line; nothing is fabricated."""
        from ...social.chat import emoji_studio

        if message is None:
            return ("no single emoji in the Unicode table "
                    "covers that — in a chat I'd create a real "
                    "custom emoji of my own; here I can only "
                    "search.")
        if not self._group_gate(message, "emoji"):
            return ("I can create that — but this room isn't "
                    "claimed for my bot-system posts (§33), so "
                    "I'm not dropping it here.")
        library = self._expression_library()
        registry = library.custom_emoji
        out_dir = self._expression_root() / "expression_media"
        try:
            spec = emoji_studio.parse_design(description)
            name = self._emoji_name_for(description, registry)
            png = emoji_studio.emoji_png(spec, out_dir, name=name)
        except Exception as exc:  # noqa: BLE001 - her renderer failed
            record_failure("expression.emoji_create.render", exc)
            return (f"no emoji in the table covers that, and "
                    f"my own renderer failed too: {exc}")
        fallback = emoji_studio.fallback_char_for(spec)
        registry.add(name, unicode=fallback, image_path=str(png))
        library.save_custom_emoji(registry)
        platform = str(getattr(message.chat, "platform", "") or "")
        adapter = self._group_adapter(platform)
        custom_id = ""
        wall = ""
        if platform.startswith("telegram") and adapter is not None \
                and hasattr(adapter, "create_or_extend_custom_emoji_pack"):
            owners = sorted(getattr(adapter, "owner_sender_ids", ())
                            or ())
            if owners:
                try:
                    made = adapter.create_or_extend_custom_emoji_pack(
                        owners[0], name, f"Devon — {name}", str(png),
                        fallback)
                    custom_id = str(made.get("custom_emoji_id") or "")
                    if custom_id:
                        registry.add(name, platform=platform,
                                     ref=custom_id)
                        library.save_custom_emoji(registry)
                except Exception as exc:  # noqa: BLE001 - refused is a fact
                    record_failure(
                        "expression.emoji_create.register", exc)
                    wall = f"Telegram refused the pack add ({exc})"
            else:
                wall = ("the bot doesn't know its owner's id yet, "
                        "so the pack can't be registered")
        elif platform.startswith("whatsapp"):
            wall = ("WhatsApp has no custom emoji at all — "
                    "a platform wall, not a missing feature")
        if custom_id and adapter is not None:
            sent = adapter.send_custom_emoji(
                message.chat, fallback, custom_id)
            if getattr(sent, "ok", False):
                return (f":{name}: is live — the one above is "
                        f"mine, a real custom emoji now. Its "
                        f"fallback face is {fallback}; it's in "
                        f"/customemoji.")
            record_failure(
                "expression.emoji_create.send",
                RuntimeError(str(getattr(sent, "error", ""))))
        from ...social.chat.base import MediaRef

        ref = MediaRef(path=str(png), kind="sticker",
                       name=Path(str(png)).name)
        result = self._expression_send(message, ref)
        note = f" :{name}: is created and in my custom set"
        if wall:
            note += f" — {wall}, so here it travels as a sticker"
        note += f". Its fallback face is {fallback}."
        if getattr(result, "ok", False):
            return note.strip()
        return (f"created :{name}:, but the send failed here: "
                f"{getattr(result, 'error', '')}")

    def _emoji_sticker_and_send(self, description: str, message) -> str:  # type: ignore[no-untyped-def]
        """The STICKER maker (formerly mislabeled as emoji
        creation): generates art through the ladder — the
        configured image rung first, her own studio renderer
        when no backend is configured — and sends it as what
        it is: a sticker."""
        from ...social.chat.expression import (
            EmojiCreateUnavailable,
            create_emoji_sticker,
        )

        if message is None:
            return ("in a chat I'd make that sticker; here I "
                    "can only describe it.")
        if not self._group_gate(message, "emoji"):
            return ("I can make that — but this room isn't "
                    "claimed for my bot-system posts (§33), so "
                    "I'm not dropping it here.")
        out_dir = self._expression_root() / "expression_media"
        try:
            produced = create_emoji_sticker(description, out_dir)
        except EmojiCreateUnavailable as exc:
            record_failure("expression.sticker_create.unavailable",
                           exc)
            return (f"I can't make that sticker on this "
                    f"install yet: {exc}")
        # The maker feeds her media library (provenance
        # journaled) — made stickers are hers to reuse.
        try:
            self._expression_library().add_item(
                kind="sticker", path=str(produced),
                tags=("sticker",), source="self:sticker-maker")
        except Exception as exc:  # noqa: BLE001 - shelving is garnish
            record_failure("expression.sticker_create.shelve", exc)
        from ...social.chat.base import MediaRef

        ref = MediaRef(path=str(produced), kind="sticker",
                       name=Path(str(produced)).name)
        result = self._expression_send(message, ref)
        if getattr(result, "ok", False):
            return ""
        return (f"made the sticker, but the send failed here: "
                f"{getattr(result, 'error', '')}")

    # ---- /customemoji -------------------------------------------------
    def _control_customemoji(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Her own set: list / add / ref / show / remove.
        Curated by the owner (DM) or a group admin — it's her
        signature, not a public whiteboard."""
        library = self._expression_library()
        registry = library.custom_emoji
        words = str(tail or "").strip().split()
        head = words[0].lower() if words else "list"
        if head in ("list", ""):
            names = registry.names()
            if not names:
                return ("my custom set is empty so far — "
                        "/customemoji add <name> <emoji> starts it.")
            lines = ["my custom emoji set:"]
            for name in names:
                entry = registry.get(name) or {}
                lines.append(f"  :{name}: {entry.get('unicode') or '—'}"
                             f" (refs: {', '.join((entry.get('refs') or {}).keys()) or 'unicode only'})")
            return "\n".join(lines)
        if head == "show" and len(words) >= 2:
            platform = str(message.chat.platform or "") \
                if message is not None else ""
            rendered = registry.render(words[1], platform)
            if rendered["kind"] == "missing":
                return f"no custom emoji named :{words[1]}: yet."
            if rendered["kind"] == "custom_ref" \
                    and platform.startswith("telegram") \
                    and message is not None:
                # It goes out AS an emoji: the entity channel
                # carries her art inline.  A refusal falls back
                # to the honest description below.
                adapter = self._group_adapter(platform)
                fn = getattr(adapter, "send_custom_emoji", None) \
                    if adapter is not None else None
                entry = registry.get(words[1]) or {}
                if fn is not None:
                    sent = fn(message.chat,
                              str(entry.get("unicode") or "✨"),
                              str(rendered["value"]))
                    if getattr(sent, "ok", False):
                        return (f":{words[1]}: — that's it, "
                                f"live as an emoji.")
            return (f":{words[1]}: on {platform or 'here'} goes "
                    f"out as {rendered['kind']}: {rendered['value']}")
        # mutations are curated
        allowed = False
        if message is not None:
            try:
                allowed = bool(self._operator_or_false(message)) \
                    or bool(self._group_admin_ok(message))
            except Exception:  # noqa: BLE001 - no authority surface
                allowed = False
        if not allowed:
            return ("my custom set is curated by the owner (or a "
                    "group admin here) — /customemoji list shows it.")
        if head == "add" and len(words) >= 3:
            name = words[1].lower()
            registry.add(name, unicode=words[2])
            library.save_custom_emoji(registry)
            return f":{name}: added — {words[2]} is its face everywhere."
        if head == "ref" and len(words) >= 4:
            name = words[1].lower()
            registry.add(name, platform=words[2], ref=words[3])
            library.save_custom_emoji(registry)
            return (f":{name}: now carries a {words[2]} custom "
                    f"ref — that platform renders mine, the rest "
                    f"fall back to the unicode face.")
        if head == "remove" and len(words) >= 2:
            if registry.remove(words[1]):
                library.save_custom_emoji(registry)
                return f":{words[1]}: removed from my set."
            return f"no custom emoji named :{words[1]}:."
        return ("usage: /customemoji list · add <name> <emoji> · "
                "ref <name> <platform> <ref> · show <name> · "
                "remove <name>")

    # ---- /emojiart ------------------------------------------------------
    def _control_emojiart(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """The emoji-art organ (mined from emojicombos, built
        native): pictures become text art — braille dot art
        (padded with U+2800 blanks so chat renderers hold the
        grid), ASCII ramp art, emoji mosaics; she DRAWS
        described subjects; her own combo library serves
        kaomoji/combos/small art; remix verbs reshape the
        room's last piece; the fancy styler restyles words.
        Pieces are returned as the reply text itself — the
        art IS the message."""
        from ...social.chat import emoji_art

        if message is None:
            return ("usage: /emojiart dot|ascii|mosaic (reply "
                    "to a picture, or paste its link) · "
                    "/emojiart draw <subject> · /emojiart combo "
                    "<topic> · /emojiart fancy <style> <words> "
                    "· /emojiart remix invert|mirror|smaller|"
                    "bigger|ascii|braille|mosaic · /emojiart "
                    "save <name>")
        platform = str(getattr(message.chat, "platform", "") or "")
        words = str(tail or "").strip().split()
        head = words[0].lower() if words else ""
        rest = " ".join(words[1:])
        library = self._combo_library()

        if head in ("dot", "ascii", "mosaic"):
            width = emoji_art.width_for_platform(platform)
            for word in words[1:]:
                try:
                    width = int(word)
                except ValueError:
                    continue
            data = self._emojiart_image_bytes(message, tail)
            if data is None:
                return ("reply to a picture (or paste its "
                        "link) and I'll render it — no image "
                        "reached me here.")
            if head == "dot":
                art = emoji_art.image_to_braille(data, width=width)
            elif head == "ascii":
                art = emoji_art.image_to_ascii(data, width=width)
            else:
                art = emoji_art.image_to_mosaic(data, width=width)
            self._emojiart_remember(message, art)
            return art
        if head == "draw":
            if not rest:
                return ("usage: /emojiart draw <subject> — "
                        "a rocket, a sleepy cat, a gold heart…")
            made = emoji_art.describe_art(
                rest, library,
                width=emoji_art.width_for_platform(platform))
            if made is None:
                return "I couldn't draw that one."
            art, how = made
            self._emojiart_remember(message, art)
            return f"{art}\n— {how}"
        if head == "combo":
            hits = library.search(rest) if rest \
                else library.pieces()[:8]
            if not hits:
                return (f"nothing in my combo library for "
                        f"'{rest}' yet — /emojiart draw makes "
                        f"new pieces, and save keeps them.")
            if len(hits) == 1:
                self._emojiart_remember(
                    message, str(hits[0].get("text") or ""))
            lines = [f"my combo library — {len(library)} pieces"
                     f" (showing {len(hits)}):"]
            for piece in hits[:8]:
                lines.append(f"· {piece.get('name')}: "
                             f"{piece.get('text')}")
            return "\n".join(lines)
        if head == "fancy":
            parts = rest.split(None, 1)
            if len(parts) < 2:
                return ("usage: /emojiart fancy <style> "
                        "<words> — styles: "
                        + " · ".join(emoji_art.FANCY_STYLES))
            styled = emoji_art.fancy_text(parts[1], parts[0])
            if styled is None:
                return (f"no style called '{parts[0]}' — "
                        f"styles: "
                        + " · ".join(emoji_art.FANCY_STYLES))
            return styled
        if head == "remix":
            verb = words[1].lower() if len(words) > 1 else ""
            last = self._emojiart_last_for(message)
            if not last:
                return ("nothing to remix yet — make a piece "
                        "first (/emojiart draw or dot).")
            out = None
            if verb == "invert":
                out = emoji_art.invert_piece(last)
            elif verb == "mirror":
                out = emoji_art.mirror_piece(last)
            elif verb == "smaller":
                out = emoji_art.scale_piece(last, 0.5)
            elif verb == "bigger":
                out = emoji_art.scale_piece(last, 2.0)
            elif verb in ("ascii", "braille", "mosaic"):
                out = emoji_art.restyle_piece(last, verb)
            if out is None:
                return (f"can't remix that piece with "
                        f"'{verb or '?'}' — verbs: invert · "
                        f"mirror · smaller · bigger · ascii · "
                        f"braille · mosaic.")
            self._emojiart_remember(message, out)
            return out
        if head == "save":
            if not self._group_gate(message, "emojiart"):
                return ("this room isn't claimed for my bot "
                        "systems (§33) — my library doesn't "
                        "grow from here.")
            last = self._emojiart_last_for(message)
            if not last:
                return "nothing to save yet — make a piece first."
            name = rest or "untitled piece"
            piece = library.add(
                name=name, text=last, kind="dotart",
                tags=name.lower().split(),
                source="chat")
            return (f"saved :{piece['id']}: — '{name}' is in "
                    f"my combo library now.")
        topics = sorted({tag for piece in library.pieces()
                         for tag in (piece.get("tags") or [])})
        return ("emoji art — pictures as text:\n"
                "/emojiart dot|ascii|mosaic — reply to a "
                "picture or paste its link\n"
                "/emojiart draw <subject> — I draw it in dots\n"
                "/emojiart combo <topic> — my library "
                f"({len(library)} pieces; topics: "
                f"{', '.join(topics[:12])})\n"
                "/emojiart fancy <style> <words>\n"
                "/emojiart remix invert|mirror|smaller|bigger"
                "|ascii|braille|mosaic\n"
                "/emojiart save <name> — keep the last piece")

    def _emojiart_image_bytes(self, message, tail: str):  # type: ignore[no-untyped-def]
        """The picture an /emojiart conversion acts on: the
        replied/attached media's local bytes first, else a
        link in the command tail, fetched through the
        expression organ's own HTTP seam.  None when neither
        reaches her."""
        resolved = self._replied_media_local(message)
        if resolved is not None and resolved[0]:
            try:
                return Path(str(resolved[0])).read_bytes()
            except OSError:
                pass
        for word in str(tail or "").split():
            if word.startswith(("http://", "https://")):
                try:
                    from ...social.chat.expression import _http_bytes

                    return _http_bytes(word)
                except Exception as exc:  # noqa: BLE001 - dead link is a fact
                    record_failure("expression.emojiart.fetch", exc)
                    return None
        return None

    # ---- /react ---------------------------------------------------------
    def _control_react(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Place a reaction on the replied message (or this
        one): /react [emoji] — no emoji named means she picks
        one from the moment, through the §22 rules."""
        if message is None:
            return "reply to a message with /react [emoji]."
        from ...social.chat.expression import (
            pick_reaction, reaction_allowed)

        platform = str(message.chat.platform or "")
        words = str(tail or "").strip().split()
        emoji = words[0] if words else ""
        if not emoji:
            context = str((getattr(message, "meta", None) or {})
                          .get("reply_to_text") or "")
            emoji = pick_reaction(context or "hello",
                                  platform=platform)
        if not emoji:
            return "no reaction fits here."
        if not reaction_allowed(platform, emoji):
            return (f"{emoji} isn't placeable here — Telegram "
                    f"bots may only react with the standard set "
                    f"(👍 ❤️ 🔥 🎉 …). Pick one of those and it's on.")
        adapter = self._group_adapter(platform)
        fn = getattr(adapter, "react", None) if adapter else None
        if fn is None:
            return ("this platform path carries no reaction verb "
                    "— on WhatsApp the bridge has no react command "
                    "yet, so I won't pretend I reacted.")
        if not self._group_gate(message, "reaction"):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — I don't react as the bot here.")
        target = str(getattr(message, "reply_to", "") or "") \
            or str(getattr(message, "message_id", "") or "")
        if not target:
            return "nothing to react to here."
        result = fn(message.chat, target, emoji)
        if getattr(result, "ok", False):
            return ""
        return (f"couldn't react here: "
                f"{getattr(result, 'error', '')}")
