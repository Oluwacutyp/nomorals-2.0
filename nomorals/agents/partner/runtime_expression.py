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
        # Her source registry: ONE source list for the whole
        # media family — the settings providers and the
        # runtime-registered sources ride the same chain the
        # pool walks read.  A registry problem sheds the rung,
        # never the service.
        registry = None
        try:
            from ...anime.registered_sources import open_registry

            home_hint = str(
                getattr(settings, "home_path", "") or "") \
                or str(getattr(settings, "workspace_dir", "")
                       or "")
            registry = open_registry(home_hint or None)
        except Exception as exc:  # noqa: BLE001 - a rung sheds
            record_failure("expression.registry", exc)
        # The unified pool, wired for /gif: the same client
        # the picture commands draw from (seeds + registry +
        # her studio), with the studio rendering into her
        # expression media dir.
        pool_client = None
        try:
            from ...anime.sources import ImagePoolClient
            from ...anime.transport import core_http_transport

            media_dir = root / "expression_media"

            def _pool_studio(kind: str) -> str:
                from ...social.chat import emoji_studio

                return str(emoji_studio.mood_animation(
                    kind, media_dir, name=f"pool_{kind}"))

            pool_client = ImagePoolClient(
                core_http_transport(), generator=_pool_studio,
                tenor_key=tenor, giphy_key=giphy,
                registry=registry)
        except Exception as exc:  # noqa: BLE001 - a rung sheds
            record_failure("expression.pool", exc)
        service = ExpressionService(
            library,
            providers=default_providers(tenor_key=tenor,
                                        giphy_key=giphy,
                                        extra=extra,
                                        registry=registry),
            media_dir=root / "expression_media",
            pool_client=pool_client)
        self._expression_service_obj = service
        return service

    def _expression_library(self):  # type: ignore[no-untyped-def]
        return self._expression_service().library

    # ── in-text custom-emoji weaving (F4) ───────────────────
    def _custom_emoji_glyph_map(self):  # type: ignore[no-untyped-def]
        """The weaving provider the telegram bot adapters
        consult per send: her registry as a glyph map, read
        live from the library so a glyph she registers (or
        imports) mid-boot weaves from the next message on."""
        try:
            return self._expression_library().custom_emoji \
                .glyph_map("telegram")
        except Exception as exc:  # noqa: BLE001 - plain text wins
            record_failure("expression.weave.glyph_map", exc)
            return {}

    def _wire_custom_emoji_weaving(self, adapter=None):  # type: ignore[no-untyped-def]
        """Stamp the glyph-map provider onto telegram bot
        adapters (all of the gateway's, or one hot-built
        adapter): the adapters expose the
        ``custom_emoji_weave_source`` attribute; anything
        without it is not a weaving surface and is skipped."""
        if adapter is not None:
            targets = [adapter]
        else:
            gateway = getattr(self, "gateway", None)
            targets = list(
                (getattr(gateway, "adapters", {}) or {}).values())
        for target in targets:
            if hasattr(target, "custom_emoji_weave_source"):
                target.custom_emoji_weave_source = \
                    self._custom_emoji_glyph_map

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
    def _brand_marked_ref(self, ref):  # type: ignore[no-untyped-def]
        """The ref to actually send for a PUBLIC-classified image:
        a sibling file with the designed brand mark burned into
        its corner (social/chat/brandmark.py).  Personal and
        unclassified refs pass through untouched — the owner's own
        media is never marked.  A marking failure degrades
        honestly: the failure is recorded and the original goes
        out; a delivery never dies on its watermark."""
        if (str(getattr(ref, "audience", "") or "") != "public"
                or str(getattr(ref, "kind", "") or "") != "image"
                or not getattr(ref, "path", "")):
            return ref
        try:
            from dataclasses import replace as _replace

            from ...social.chat.brandmark import burn_brand_mark

            src = Path(str(ref.path))
            marked = src.with_name(
                f"{src.stem}.brandmarked{src.suffix}")
            marked.write_bytes(burn_brand_mark(src.read_bytes()))
            return _replace(ref, path=str(marked))
        except Exception as exc:  # noqa: BLE001 - degrade, never block
            record_failure("expression.brandmark", exc)
            return ref

    def _expression_send(self, message, ref, caption: str = "", buttons=None):  # type: ignore[no-untyped-def]
        """Send one MediaRef through the room's own adapter
        (native media path — Telegram stickers/animations,
        Discord uploads, the WhatsApp bridge).  Falls back to
        the gateway's file send; reports honestly when neither
        exists on this platform path.  ``caption`` rides the
        media message itself (every adapter's send_media and
        the gateway's send_file carry one) — it is only passed
        when non-empty, so existing callers and adapter shapes
        are untouched.  The gateway fallback declares the ref's
        own kind and name (delivery-shape law): a picture or a
        GIF that reaches the fallback still lands AS a picture
        or an animation, never as a document wearing a
        content-hash name.  ``buttons`` ride where the media
        verb carries them and degrade silently where it doesn't
        — the media is the deliverable, buttons are garnish."""
        from types import SimpleNamespace

        platform = str(getattr(message.chat, "platform", "") or "")
        adapter = self._group_adapter(platform)
        fn = getattr(adapter, "send_media", None) if adapter else None
        if fn is not None:
            # the adapter path takes the ref itself, so a
            # public-classified image is burned HERE; the gateway
            # fallback below burns at its own seam instead — a ref
            # is marked exactly once, on whichever path sends it.
            ref = self._brand_marked_ref(ref)
            kwargs: dict = {}
            if caption:
                kwargs["caption"] = caption
            if buttons:
                kwargs["buttons"] = buttons
            try:
                return fn(message.chat, ref, **kwargs)
            except TypeError:
                kwargs.pop("buttons", None)
                return fn(message.chat, ref, **kwargs)
        gateway = getattr(self, "gateway", None)
        send_file = getattr(gateway, "send_file", None)
        if send_file is not None and ref.path:
            extra: dict = {}
            if str(getattr(ref, "audience", "") or "") == "public":
                # only a public-classified ref changes the call
                # shape; every other send is byte-historical.
                extra["audience"] = "public"
            return send_file(platform, message.chat, ref.path,
                             caption=caption,
                             kind=str(getattr(ref, "kind", "") or ""),
                             name=str(getattr(ref, "name", "") or ""),
                             buttons=buttons, **extra)
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
                # A celebration reaction lands BIG where the
                # platform carries the big form (Telegram's
                # is_big — the large animated reaction); the
                # adapter degrades a refused big to normal
                # size itself, and an adapter without the big
                # form takes the plain call.
                if str(message.chat.platform or "").startswith(
                        "telegram"):
                    try:
                        fn(message.chat, str(message.message_id),
                           emoji, big=True)
                        return
                    except TypeError:
                        pass
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
            return "usage: /sticker text — or reply to an image/video."
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
                return ("usage: /sticker text — or reply to an "
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
        lists it, `/sticker pack` + an entry number sends that entry.  Listing
        and sending are bot-system output in a group — the
        same §33 gate as every expression post; in DM the
        pack is simply the owner's own shelf."""
        if message is None:
            return ("usage: /sticker pack — your stolen "
                    "stickers; /sticker pack + an entry number sends one.")
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
            lines.append("/sticker pack (entry number) sends one.")
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
            return "usage: /gif what you're feeling"
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
        ``/emoji make words`` authors a REAL custom emoji —
        her own art at 100×100, registered in her Telegram
        custom-emoji pack where the surface allows, living in
        her /customemoji set with a unicode fallback — and a
        search no single emoji confidently covers ('family
        rocket') is a creation too, never a dump of near-miss
        parts.  STICKER: ``/emoji sticker words`` is the
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
            return ("usage: /emoji words finds one in the "
                    "whole Unicode table — melting face, thumbs "
                    "up medium, rocket… /emoji make words "
                    "creates a real custom emoji of my own. "
                    "/emoji sticker words makes a sticker. "
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
                        "emoji, /emoji make words).")
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
            if len(hits) == 1 and message is not None:
                # A single found emoji on Telegram goes out as
                # its OWN message: exactly one emoji and nothing
                # else is the only shape Telegram clients render
                # as the big animated emoji — buried in the text
                # answer it sits small and still (owner,
                # 2026-11-11).  The name line follows as the
                # reply.  Another platform, an adapter without
                # the verb, or a failed lone send keeps the
                # combined answer — the emoji is delivered
                # honestly either way.
                platform = str(getattr(message.chat, "platform",
                                       "") or "")
                if platform.startswith("telegram"):
                    adapter = self._group_adapter(platform)
                    send = getattr(adapter, "send", None) \
                        if adapter is not None else None
                    if send is not None:
                        landed = None
                        try:
                            landed = send(message.chat, hits[0].char)
                        except Exception:  # noqa: BLE001 - garnish
                            _log.debug("lone emoji send failed",
                                       exc_info=True)
                        if getattr(landed, "ok", False):
                            return names
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
        ANIMATED (a .tgs from emoji_anim) wherever the design
        fits Telegram's animated cap, the pack hands back a
        real custom_emoji_id, and the emoji is SENT as an
        emoji (entity-carried), not as a sticker.  Surfaces
        that cannot register (WhatsApp has no custom emoji at
        all — a platform wall) receive the art as a sticker
        with the wall named.  Every failure is a plain line;
        nothing is fabricated."""
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
        # The animated rung (owner, 2026-11-11: "still using
        # still emoji's"): the design becomes a .tgs through
        # emoji_anim and the pack registers it as "animated",
        # so the emoji MOVES inline.  When a design cannot fit
        # Telegram's 64 KB animated cap even at the leanest
        # detail, the static PNG registers as "static" instead
        # — and the registry records whichever format actually
        # landed, never the one that was hoped for.
        art_path, art_format = str(png), "static"
        try:
            from ...social.chat import emoji_anim

            tgs = emoji_anim.animated_emoji_tgs(spec, out_dir,
                                                name=name)
            art_path, art_format = str(tgs), "animated"
        except Exception as exc:  # noqa: BLE001 - static is the floor
            record_failure("expression.emoji_create.animate", exc)
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
                        owners[0], name, f"Devon — {name}", art_path,
                        fallback, art_format)
                    custom_id = str(made.get("custom_emoji_id") or "")
                    if custom_id:
                        registry.add(
                            name, platform=platform, ref=custom_id,
                            format=str(made.get("format")
                                       or art_format))
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
                landed = str((registry.get(name) or {})
                             .get("format") or "")
                motion = (" — and it MOVES: registered animated, "
                          "not a still"
                          if landed == "animated" else "")
                return (f":{name}: is live — the one above is "
                        f"mine, a real custom emoji now{motion}. "
                        f"Its fallback face is {fallback}; it's "
                        f"in /customemoji.")
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
                        "/customemoji add name emoji starts it.")
            lines = ["my custom emoji set:"]
            for name in names:
                entry = registry.get(name) or {}
                fmt = str(entry.get("format") or "")
                lines.append(
                    f"  :{name}: {entry.get('unicode') or '—'}"
                    f" (refs: {', '.join((entry.get('refs') or {}).keys()) or 'unicode only'})"
                    + (f" · {fmt}" if fmt else ""))
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
        if head == "status":
            return self._customemoji_status(message)
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
        if head == "import" and len(words) >= 2:
            return self._customemoji_import(words[1], message)
        return ("usage: /customemoji list · status · "
                "add name emoji · ref name platform ref · "
                "show name · import pack_name · remove name")

    def _customemoji_status(self, message=None) -> str:  # type: ignore[no-untyped-def]
        """The set + the weaving state, plainly (F4): how
        many glyphs she holds, how many carry a Telegram id,
        how many of those MOVE — and per bot adapter whether
        in-text weaving is on or stands refused, with
        Telegram's own refusal words when it does.  OFF and
        BROKEN are never the same sentence."""
        library = self._expression_library()
        registry = library.custom_emoji
        names = registry.names()
        glyph_map = registry.glyph_map("telegram")
        ids = {str(rec.get("id") or "")
               for rec in glyph_map.values()}
        moving = {str(rec.get("id") or "")
                  for rec in glyph_map.values()
                  if rec.get("format") in ("animated", "video")}
        lines = [
            f"my custom set: {len(names)} named · "
            f"{len(ids)} glyphs carry a Telegram id · "
            f"{len(moving)} of them move (animated/video)."]
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        seen = False
        for key in sorted(adapters):
            fn = getattr(adapters[key],
                         "custom_emoji_weaving_status", None)
            if fn is None:
                continue
            seen = True
            state = fn()
            refused = state.get("refused") or {}
            if refused:
                detail = "; ".join(
                    f"{scope}: {refused[scope]}"
                    for scope in sorted(refused))
                lines.append(
                    f"weaving on {key}: REFUSED by Telegram — "
                    f"{detail}. My text goes out plain there "
                    f"until the bot is eligible (owner Premium "
                    f"or a Fragment username).")
            elif state.get("wired"):
                lines.append(
                    f"weaving on {key}: on — registered glyphs "
                    f"in my Telegram text ride as animated "
                    f"custom emoji.")
            else:
                lines.append(
                    f"weaving on {key}: not wired on this bot.")
        if not seen:
            lines.append(
                "weaving: no Telegram bot adapter in this rig "
                "to report on.")
        return "\n".join(lines)

    def _customemoji_import(self, pack_name: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Grow her set from a whole pack (F4): every
        custom-emoji sticker in the named set registers its
        glyph → custom_emoji_id with the format the sticker
        itself reports — this is how the standard animated
        glyphs get in, since packs are readable by any bot
        even though SENDING custom emoji is eligibility
        gated.  Ids are cross-checked through
        getCustomEmojiStickers when the adapter can; an
        unverifiable id is left out when verification ran,
        and the reply says so when it couldn't run."""
        platform = str(message.chat.platform or "") \
            if message is not None else ""
        adapter = self._group_adapter(platform) \
            if platform else None
        fn = getattr(adapter, "get_custom_emoji_pack", None) \
            if adapter is not None else None
        if fn is None:
            return ("pack import needs a Telegram bot surface "
                    "— this chat's adapter can't read sticker "
                    "packs.")
        try:
            stickers = fn(str(pack_name))
        except Exception as exc:  # noqa: BLE001 - refused is a fact
            record_failure("expression.customemoji.import", exc)
            return f"couldn't read a pack named {pack_name}: {exc}"
        if not stickers:
            return (f"no custom emoji in a pack named "
                    f"{pack_name} that I can read — nothing "
                    f"imported.")
        note = ""
        verify = getattr(adapter, "verify_custom_emoji_ids", None)
        if verify is not None:
            confirmed = verify(
                [row["custom_emoji_id"] for row in stickers])
            if confirmed is not None:
                kept = [row for row in stickers
                        if row["custom_emoji_id"] in confirmed]
                if len(kept) < len(stickers):
                    note = (f" ({len(stickers) - len(kept)} ids "
                            f"Telegram wouldn't verify were left "
                            f"out)")
                stickers = kept
                if not stickers:
                    return (f"Telegram wouldn't verify any id "
                            f"in {pack_name} — nothing imported.")
            else:
                note = (" (ids unverified — the verify call "
                        "failed; it's in the journal)")
        library = self._expression_library()
        registry = library.custom_emoji
        result = registry.import_stickers(stickers,
                                          pack=str(pack_name))
        library.save_custom_emoji(registry)
        return (f"imported from {pack_name}: "
                f"{result['added']} new · "
                f"{result['upgraded']} upgraded · "
                f"{result['kept']} already mine{note}. "
                f"Registered glyphs now weave into my Telegram "
                f"text wherever their face appears.")

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
                    "/emojiart draw subject · /emojiart combo "
                    "topic · /emojiart fancy style words "
                    "· /emojiart remix invert|mirror|smaller|"
                    "bigger|ascii|braille|mosaic · /emojiart "
                    "save name")
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
                return ("usage: /emojiart draw subject — "
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
                return ("usage: /emojiart fancy style "
                        "words — styles: "
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
                "/emojiart draw subject — I draw it in dots\n"
                "/emojiart combo topic — my library "
                f"({len(library)} pieces; topics: "
                f"{', '.join(topics[:12])})\n"
                "/emojiart fancy style words\n"
                "/emojiart remix invert|mirror|smaller|bigger"
                "|ascii|braille|mosaic\n"
                "/emojiart save name — keep the last piece")

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
            except OSError as exc:
                record_failure("expression.emojiart.local_read", exc)
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
                    "— so I won't pretend I reacted.")
        # WhatsApp reactions are acts AS THE OWNER on his
        # personal account (the verb is parity G1's): a standing
        # owner stop halts them like every other personal-
        # surface act, and a landed one is journaled in the
        # personal vault — an unjournaled act would be a silent
        # one.  (Other platforms' reactions are the bot's own
        # expression, unchanged.)
        wa_svc = None
        wa_rec = None
        if platform == "whatsapp":
            try:
                wa_svc, wa_rec = self._chatops_personal_account(
                    "whatsapp")
            except Exception:  # noqa: BLE001 - the gate is a guard
                wa_svc, wa_rec = None, None
            if wa_rec is not None and \
                    wa_svc["registry"].owner_stopped("whatsapp"):
                return ("WhatsApp is under your standing stop "
                        "— it stays halted until you start it "
                        "again; nothing was reacted to.")
        if not self._group_gate(message, "reaction"):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — I don't react as the bot here.")
        target = str(getattr(message, "reply_to", "") or "") \
            or str(getattr(message, "message_id", "") or "")
        if not target:
            return "nothing to react to here."
        result = fn(message.chat, target, emoji)
        if getattr(result, "ok", False):
            if platform == "whatsapp" and wa_rec is not None:
                try:
                    wa_svc["personal"].note_message_acted(
                        "whatsapp", kind="message_reacted",
                        account_id=wa_rec.id,
                        chat_id=str(message.chat.chat_id or ""),
                        source_id=target,
                        note=f"reacted {emoji} as the owner "
                             f"via /react")
                except Exception:  # noqa: BLE001 - journal is a record
                    _log.debug("react journal failed",
                               exc_info=True)
            return ""
        return (f"couldn't react here: "
                f"{getattr(result, 'error', '')}")

    # ---- /create: procedural creation (doc 6) -------------------

    def _control_create(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Procedural creation: the brief becomes an EXECUTABLE
        DESCRIPTION — a studio composition, a Scene3D document,
        or a motion explainer spec — her own renderers execute
        it, the pipeline inspects the real output and repairs
        the description (bounded, journaled), and the verified
        artifact is delivered. STRICT: no generation model is
        ever called on this path; an unrenderable description
        fails with its inspection evidence. The creation's
        progress and result revise ONE living card."""
        from ...social.chat.creation import (
            author_motion_spec,
            author_scene_spec,
            author_studio_spec,
            create_procedural,
        )

        if message is None:
            return ("in a chat I'd create that procedurally — "
                    "my own renderers executing a description I "
                    "author, inspect and repair; here I can only "
                    "describe it.")
        text = " ".join(str(tail or "").split()).strip()
        kind, brief = "studio", text
        lowered = text.lower()
        for prefix, k in (("scene ", "scene3d"),
                          ("motion ", "motion"),
                          ("video ", "motion")):
            if lowered.startswith(prefix):
                kind, brief = k, text[len(prefix):].strip()
                break
        if not brief:
            return ("what should I create? /create <brief> "
                    "makes a sticker from a studio composition; "
                    "/create scene <brief> builds a 3D scene; "
                    "/create motion <brief> renders a short "
                    "explainer. Every render is inspected and "
                    "repaired before it reaches you — no "
                    "generation model, ever.")
        if not self._group_gate(message, "emoji"):
            return ("I can create that — but this room isn't "
                    "claimed for my bot-system posts (§33), so "
                    "I'm not dropping it here.")
        author = {"studio": author_studio_spec,
                  "scene3d": author_scene_spec,
                  "motion": author_motion_spec}[kind]
        try:
            spec = author(brief)
        except Exception as exc:  # noqa: BLE001 - authoring is guarded
            record_failure("expression.create.author", exc)
            return (f"I couldn't compose a description from "
                    f"that brief: {exc}")
        # The living card: one message, revised as the
        # creation moves (working → repairing → delivered /
        # failed), never a stream of separate posts.
        card = None
        host = None
        try:
            from ...social.chat.living_cards import (
                FAMILY_CREATION,
                Card,
                new_card_id,
            )
            from . import runtime as _runtime_mod

            host = self._living_card_host()
            card = Card(
                card_id=new_card_id(), family=FAMILY_CREATION,
                chat_key=str(getattr(message.chat, "key", "") or ""),
                owner_key=_runtime_mod._pick_owner_key(message),
                platform=str(getattr(message.chat, "platform", "")
                             or ""),
                chat_id=str(getattr(message.chat, "chat_id", "")
                            or ""),
                title=f"Creating — {brief[:48]}",
                body=(f"authoring a {kind} description and "
                      f"rendering it with my own engines — the "
                      f"render is inspected, and repaired if it "
                      f"misses, before anything is delivered."),
                status="working")
            host.publish(card, message.chat)
        except Exception as exc:  # noqa: BLE001 - card is a surface
            record_failure("expression.create.card", exc)
            card = None
            host = None
        out_dir = self._expression_root() / "procedural"
        try:
            result = create_procedural(
                spec, out_dir,
                db=getattr(self.context, "db", None))
        except Exception as exc:  # noqa: BLE001 - pipeline is guarded
            record_failure("expression.create.pipeline", exc)
            return (f"the creation pipeline itself failed: {exc}")
        if card is not None and host is not None:
            try:
                host.revise(
                    card, message.chat,
                    body=result.describe(),
                    rows=result.summary_rows(),
                    status="delivered" if result.ok else "failed",
                    terminal=True)
            except Exception as exc:  # noqa: BLE001 - surface only
                record_failure("expression.create.card_revise", exc)
        if not result.ok or result.path is None:
            return result.describe()
        # Shelve the verified artifact in her library
        # (provenance journaled), like the sticker maker does.
        media_kind = {"studio": "sticker", "scene3d": "image",
                      "motion": "video"}[result.spec.kind]
        try:
            self._expression_library().add_item(
                kind=media_kind, path=str(result.path),
                tags=("procedural", result.spec.kind),
                source="self:procedural-creation")
        except Exception as exc:  # noqa: BLE001 - shelving is garnish
            record_failure("expression.create.shelve", exc)
        from ...social.chat.base import MediaRef

        ref = MediaRef(path=str(result.path), kind=media_kind,
                       name=result.path.name)
        sent = self._expression_send(message, ref)
        if getattr(sent, "ok", False):
            return ""
        return (f"created and verified it, but the send failed "
                f"here: {getattr(sent, 'error', '')}")

    # ---- /stylepin: owner style pins ----------------------------

    def _control_stylepin(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Owner style pins: set / clear / show. A pin
        (typography intent, font family, theme, preset, traits)
        persists in the one pin store, applies globally or per
        surface family, and outranks semantic defaults at the
        selection seams — the typography service and the chat
        theme resolver both consult core.style_pins."""
        from ...core.style_pins import effective_pin, get_pin_store

        store = get_pin_store()
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        if not parts:
            lines = ["style pins in force:"]
            scopes = store.scopes()
            if not scopes:
                lines.append("  none — semantic defaults rule "
                             "everywhere")
            for scope in scopes:
                pin = store.get(scope)
                label = "global" if scope == "" else \
                    f"family:{scope}"
                bits = []
                if pin.intent:
                    bits.append(f"intent {pin.intent}")
                if pin.font_family:
                    bits.append(f"font {pin.font_family}")
                if pin.theme:
                    bits.append(f"theme {pin.theme}")
                if pin.preset:
                    bits.append(f"preset {pin.preset}")
                for k in sorted(pin.traits):
                    bits.append(f"{k} {pin.traits[k]}")
                lines.append(f"  {label}: {', '.join(bits)}")
            lines.append(
                "set: /stylepin set <global|family> "
                "<intent|font|theme|preset|trait> <value…> · "
                "clear: /stylepin clear <global|family> [field]")
            return "\n".join(lines)
        verb = parts[0].lower()
        if verb == "set" and len(parts) >= 4:
            scope = parts[1]
            field = parts[2].lower()
            value = " ".join(parts[3:]).strip()
            if not value:
                return "a pin needs a value."
            if field == "intent":
                from ...media.typography.selection import INTENTS
                if value not in INTENTS:
                    return (f"{value!r} is not a typography "
                            f"intent — the vocabulary is: "
                            f"{', '.join(INTENTS)}")
                store.set(scope, intent=value)
            elif field in ("font", "font_family", "family"):
                # A font pin names a family in the stocked
                # library (or one the host catalogue holds) —
                # validated NOW, at set time, against the real
                # catalogue: a pin for a family nothing holds
                # is refused by name instead of surfacing as a
                # failed render later.
                stocked = False
                try:
                    from ...media.typography import get_service
                    service = get_service()
                    stocked = service.font_library().stocked(
                        value) or bool(
                            service.catalogue().faces(
                                family=value))
                except Exception:  # noqa: BLE001 - catalogue
                    stocked = False      # unreachable: refuse
                if not stocked:
                    return (f"{value!r} is not a font family "
                            f"the stocked library or this host "
                            f"holds — browse what is stocked "
                            f"with /fonts, then pin a real name.")
                store.set(scope, font_family=value)
            elif field == "theme":
                from ...social.chat.typography import THEMES
                if value.lower() not in THEMES:
                    return (f"{value!r} is not a theme pack — "
                            f"the packs are: "
                            f"{', '.join(sorted(THEMES))}")
                store.set(scope, theme=value.lower())
            elif field == "preset":
                from ...social.chat.typography import PRESETS
                if value.lower() not in PRESETS:
                    return (f"{value!r} is not a preset — the "
                            f"presets are: {', '.join(PRESETS)}")
                store.set(scope, preset=value.lower())
            elif field in ("trait", "traits"):
                name, _, trait_value = value.partition(" ")
                if not name or not trait_value.strip():
                    return ("a trait pin reads: /stylepin set "
                            "<scope> trait <name> <value>")
                store.set(scope, trait={name: trait_value.strip()})
            else:
                return (f"pins carry intent, font, theme, "
                        f"preset or a trait — not {field!r}.")
            eff = effective_pin(store, "" if scope.lower()
                                in ("", "global") else scope)
            label = eff.label if eff is not None else scope
            return (f"pinned ({label}): "
                    f"{eff.describe() if eff is not None else ''}"
                    f" — it outranks the semantic defaults "
                    f"wherever this style is selected, and "
                    f"surfaces under it name it.")
        if verb == "clear" and len(parts) >= 2:
            scope = parts[1]
            field = parts[2].lower() if len(parts) >= 3 else None
            if field in ("font", "family"):
                field = "font_family"
            store.clear(scope, field=field)
            where = "global" if scope.lower() in ("", "global") \
                else f"family:{scope}"
            return (f"pin cleared ({where}"
                    f"{f', field {field}' if field else ''}) — "
                    f"the semantic defaults rule there again.")
        return ("usage: /stylepin — show · /stylepin set "
                "<global|family> <intent|font|theme|preset|"
                "trait> <value…> · /stylepin clear "
                "<global|family> [field]")

    # ---- /pack: experience packs (Living Chat P4) --------------

    def _pack_facts(self, message) -> dict:  # type: ignore[no-untyped-def]
        """The fact snapshot pack homes and verbs compose over:
        this chat's live cards, the expression library, the
        pack store, a writable artifact dir. Every source is
        guarded — a missing organ reads as empty, never as a
        crashed pack."""
        from ...social.chat import packs as _packs
        from . import runtime as _runtime_mod

        facts: dict = {"store": _packs.get_pack_store()}
        chat = getattr(message, "chat", None)
        chat_key = str(getattr(chat, "key", "") or "")
        facts["chat_key"] = chat_key
        facts["actor"] = _runtime_mod._pick_owner_key(message)
        try:
            host = self._living_card_host()
            facts["live_cards"] = [
                (c.family, c.title, c.status)
                for c in host.store.live_for_chat(chat_key)]
        except Exception as exc:  # noqa: BLE001 - facts degrade
            record_failure("packs.facts.live_cards", exc)
            facts["live_cards"] = []
        try:
            library = self._expression_library()
            facts["library_items"] = library.items()
            facts["library_len"] = len(library)
        except Exception as exc:  # noqa: BLE001 - facts degrade
            record_failure("packs.facts.library", exc)
            facts["library_items"] = []
            facts["library_len"] = 0
        try:
            facts["out_dir"] = self._expression_root() / "packs"
        except Exception as exc:  # noqa: BLE001 - facts degrade
            record_failure("packs.facts.out_dir", exc)
        facts["db"] = getattr(self.context, "db", None)
        facts["digest_items"] = []
        return facts

    @staticmethod
    def _pack_card_id(pack, chat_key: str, scope: str) -> str:  # type: ignore[no-untyped-def]
        """One home card per (pack, chat, scope): owner-scoped
        packs get a card per owner, chat-scoped packs (the
        decision room) share one card per room."""
        import hashlib

        raw = f"{pack.id}\x1f{chat_key}\x1f{scope}".encode()
        return "pk" + hashlib.sha256(raw).hexdigest()[:10]

    def _control_pack(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Experience packs (Living Chat P4): /pack lists the
        registry with honest state; /pack open <id> publishes
        the pack's living home card (family = the pack's own
        id, so a style pin targets it through the P3 seam);
        /pack on|off <id> flips the persisted lifecycle;
        /pack act <id> <verb> runs a native verb as a state
        transition over the pack store. A disabled pack's
        surfaces stop resolving — open and act answer
        unavailable in words and publish nothing."""
        from ...social.chat import packs as _packs

        store = _packs.get_pack_store()
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        if not parts or parts[0].lower() == "list":
            from ...social.chat.typography import card_text

            return card_text(_packs.listing_card(store))
        verb = parts[0].lower()
        if verb in ("on", "off"):
            if len(parts) < 2:
                return "which pack? /pack lists them all."
            pack_id = parts[1].lower()
            if pack_id not in _packs.PACK_IDS:
                return (f"no pack called {pack_id!r} — /pack "
                        f"lists them all.")
            store.set_enabled(pack_id, verb == "on")
            pack = _packs.pack_by_id(pack_id)
            if verb == "on":
                return f"{pack.title} ({pack_id}) is on now."
            return (f"{pack.title} ({pack_id}) is off now — its "
                    f"surfaces stop resolving until you switch "
                    f"it back on.")
        if verb == "act":
            if len(parts) < 3:
                return "usage: /pack act <id> <verb> [args…]"
            pack_id = parts[1].lower()
            if pack_id not in _packs.PACK_IDS:
                return (f"no pack called {pack_id!r} — /pack "
                        f"lists them all.")
            pack = _packs.pack_by_id(pack_id)
            if not store.is_enabled(pack_id):
                return (f"{pack.title} ({pack_id}) is off — "
                        f"unavailable until you switch it back "
                        f"on: /pack on {pack_id}")
            if message is None:
                return ("pack verbs run in a chat, where their "
                        "state lives.")
            return self._pack_act(pack, parts[2].lower(),
                                  " ".join(parts[3:]), message)
        pack_id = (parts[1] if verb == "open" and len(parts) > 1
                   else parts[0]).lower()
        if pack_id not in _packs.PACK_IDS:
            return (f"no pack called {pack_id!r} — /pack lists "
                    f"all {len(_packs.PACKS)} of them.")
        pack = _packs.pack_by_id(pack_id)
        if not store.is_enabled(pack_id):
            return (f"{pack.title} ({pack_id}) is off — "
                    f"unavailable until you switch it back on: "
                    f"/pack on {pack_id}")
        if message is None:
            return (f"in a chat I'd open the {pack.title} pack "
                    f"as a living card; here I can only describe "
                    f"it: {pack.tagline}.")
        return self._pack_open(pack, message)

    def _skillbook(self):  # type: ignore[no-untyped-def]
        cached = getattr(self, "_skillbook_cache", None)
        if cached is None:
            from ...skills.filebook import default_library

            cached = default_library()
            self._skillbook_cache = cached
        return cached

    def _delegation_store(self):  # type: ignore[no-untyped-def]
        cached = getattr(self, "_delegation_store_cache", None)
        if cached is None:
            db = getattr(self.context, "db", None)
            if db is None:
                return None
            from ...agents.delegation import DelegationStore

            cached = DelegationStore(db)
            self._delegation_store_cache = cached
        return cached

    def _control_skillbook(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """The flat-file skillbook (SKILLS system): /skills lists the
        routed library, /skills show <id> reads one file, /skills
        search <text> routes a task text, /skills drafts shows
        distilled drafts with their promotion progress, /skills
        promote <id> promotes a draft now. The list also publishes a
        living card (family "skills") whose buttons open skills in
        place — the same Living Chat surface as everything else."""
        library = self._skillbook()
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        verb = parts[0].lower() if parts else "list"
        if verb == "show" and len(parts) >= 2:
            doc = library.load(parts[1])
            if doc is None:
                return (f"no skill called {parts[1]!r} — /skills "
                        "lists the library.")
            body = doc.body.strip()
            if len(body) > 3200:
                body = body[:3200].rstrip() + "\n[truncated — the full " \
                    "file loads into context when a task routes to it]"
            label = " (draft — unproven)" if doc.is_draft else ""
            return f"{doc.title} [{doc.id}]{label}\n\n{body}"
        if verb == "search":
            query = text[len(parts[0]):].strip()
            if not query:
                return "usage: /skills search <task text>"
            routed = library.route(query, limit=5)
            if not routed:
                return (f"no skill routes to {query!r} yet — when a "
                        "mission teaches one, it lands as a draft.")
            lines = [f"skills routed for: {query}"]
            for doc, score in routed:
                mark = " (draft)" if doc.is_draft else ""
                lines.append(f"• {doc.id}{mark} — {doc.title} "
                             f"(score {score:.0f})")
            return "\n".join(lines)
        if verb == "drafts":
            drafts = library.drafts()
            if not drafts:
                return ("no distilled drafts waiting — drafts appear "
                        "when a mission finishes with something to "
                        "teach.")
            lines = ["distilled drafts (promote at 2 successful "
                     "missions, or /skills promote <id>):"]
            for doc in drafts:
                progress = library.draft_progress(doc.id)
                revises = f" revises {doc.revises}" if doc.revises else ""
                lines.append(
                    f"• {doc.id}{revises} — "
                    f"{progress['successful_missions']}/"
                    f"{progress['needed']} successful missions")
            return "\n".join(lines)
        if verb == "promote" and len(parts) >= 2:
            result = library.promote(parts[1])
            if not result.get("ok"):
                return f"cannot promote: {result.get('reason')}"
            return (f"promoted {result['skill_id']} to the live "
                    f"library (v{result['version']}) — it now routes "
                    "as a proven skill.")
        live = library.docs(include_drafts=False)
        drafts = library.drafts()
        lines = [f"skill library — {len(live)} live, "
                 f"{len(drafts)} draft(s):"]
        for doc in sorted(live, key=lambda d: d.id):
            lines.append(f"• {doc.id} — {doc.title}")
        for doc in drafts:
            lines.append(f"• {doc.id} (draft — unproven)")
        if message is not None:
            self._skillbook_card(library, live, message)
            lines.append("the library card is live above — tap a "
                         "skill to open it.")
        return "\n".join(lines)

    def _skillbook_card(self, library, live, message) -> None:  # type: ignore[no-untyped-def]
        """Publish/revise the skills living card for this chat."""
        try:
            from ...social.chat.action_schema import Action
            from ...social.chat.living_cards import Card
            from . import runtime as _runtime_mod

            if not self._group_gate(message, "skills"):
                return
            chat = message.chat
            chat_key = str(getattr(chat, "key", "") or "")
            owner_key = _runtime_mod._pick_owner_key(message)
            rows = tuple(
                (doc.id, doc.title)
                for doc in sorted(live, key=lambda d: d.id)[:8])
            actions = tuple(
                Action(label=doc.id, intent="callback",
                       data=f"skills show {doc.id}")
                for doc in sorted(live, key=lambda d: d.id)[:6])
            card = Card(
                card_id="skills:" + chat_key, family="skills",
                chat_key=chat_key, owner_key=owner_key,
                platform=str(getattr(chat, "platform", "") or ""),
                chat_id=str(getattr(chat, "chat_id", "") or ""),
                title="Skill library",
                body="Flat-file skills the brain routes by. Tap one "
                     "to read it; /skills search <task> shows what a "
                     "task would load.",
                rows=rows, actions=actions, status="open")
            host = self._living_card_host()
            existing = host.store.get(card.card_id)
            if existing is not None and not existing.terminal:
                host.revise(card, chat, title=card.title, body=card.body,
                            rows=rows, actions=actions, status="open")
            else:
                host.publish(card, chat)
        except Exception as exc:  # noqa: BLE001 - card is garnish
            record_failure("skills.card", exc)

    def _control_delegate(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Accountable delegation cockpit (doc 6): /delegate lists
        work contracts with their verification state, /delegate show
        <id> renders one contract with its claim ledger, /delegate
        claims [id] reads the ledger. State renders as text here and
        as a living card (family "delegation") in chat — no second
        control surface."""
        store = self._delegation_store()
        if store is None:
            return ("delegation state needs the database, which this "
                    "runtime doesn't carry — contracts are unavailable "
                    "here, not empty.")
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        verb = parts[0].lower() if parts else "list"
        if verb == "show" and len(parts) >= 2:
            contract = store.get_contract(parts[1])
            if contract is None:
                return f"no contract {parts[1]!r} — /delegate lists them."
            summary = store.summary(contract)
            lines = [
                f"contract {contract.id} — {contract.title}",
                f"status: {contract.status} · generation "
                f"{contract.generation} · worker {contract.worker} · "
                f"verifier {contract.verifier}",
                f"goal: {contract.goal}",
            ]
            if contract.specialist_class:
                lines.append(f"specialist: {contract.specialist_class}")
            if summary["blockers"]:
                lines.append("completion blocked by: "
                             + "; ".join(summary["blockers"]))
            lines.append("claim ledger:")
            for claim in store.claims(contract.id, limit=12):
                lines.append(
                    f"• [{claim.kind}] {claim.claim[:90]} — "
                    f"{claim.verdict} (by {claim.producer or '?'})")
            return "\n".join(lines)
        if verb == "claims":
            contract_id = parts[1] if len(parts) >= 2 else ""
            entries = store.claims(contract_id, limit=15)
            if not entries:
                return "the claim ledger is empty."
            lines = ["claim ledger (latest first):"]
            for claim in entries:
                lines.append(
                    f"• [{claim.kind}] {claim.claim[:80]} — "
                    f"{claim.verdict} · {claim.contract_id or '—'}")
            return "\n".join(lines)
        contracts = store.recent_contracts(limit=12)
        if not contracts:
            return ("no work contracts yet — contracts are cut when a "
                    "mission delegates a step to a specialist.")
        lines = ["work contracts:"]
        for contract in contracts:
            summary = store.summary(contract)
            verdicts = ", ".join(
                f"{k} {v}" for k, v in sorted(
                    summary["verdicts"].items())) or "no claims"
            lines.append(
                f"• {contract.id} [{contract.status}] "
                f"{contract.title[:48]} — {summary['claims']} claims "
                f"({verdicts})")
        if message is not None:
            self._delegation_card(store, contracts, message)
            lines.append("the delegation card is live above.")
        return "\n".join(lines)

    def _control_computer(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """Devon's own computer (owner-only): /computer shows the
        organ's state — availability, sessions, and each session's
        control gate; open starts a persistent desktop session;
        view perceives and sends the live frame as a photo;
        watch starts a live tail of the session's stream (new
        frames delivered to this chat as they are produced — a
        durable job on /runs, stoppable from either surface);
        unwatch stops this chat's watch through the cancellation
        facade; takeover puts the agent's hands off the desktop
        until handback; stop is the owner's STICKY stop (survives
        restarts, cleared only by resume); close ends a session.
        Watching is never gated; acting is — the organ enforces
        it, this surface only speaks it."""
        from ...computer.service import default_service

        svc = default_service(context=getattr(self, "context", None))
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        verb = parts[0].lower() if parts else "status"
        session_id = ""
        watch_id = ""
        for part in parts[1:]:
            if part.startswith("desk-"):
                session_id = part
            elif part.startswith("cw-"):
                watch_id = part
        if verb == "status":
            return svc.status_text()
        if verb == "open":
            result = svc.open(mission=" ".join(parts[1:]))
            if not result.get("ok"):
                return f"the computer would not open: {result.get('error')}"
            rec = result["session"]
            return (f"desktop session {rec['id']} is open — "
                    f"{rec['mode']} display {rec['display']} "
                    f"({rec['width']}x{rec['height']}). It persists "
                    "between tasks; /computer view shows it live.")
        if verb == "view":
            seen = svc.perceive(session_id)
            if not seen.get("ok"):
                return f"nothing to view: {seen.get('error')}"
            lines = [f"desktop {seen['session']} — "
                     f"{seen['width']}x{seen['height']}, "
                     f"{len(seen['windows'])} window(s):"]
            for win in seen["windows"]:
                mark = " *focused*" if win.get("focused") else ""
                lines.append(f"• {win.get('title') or '(untitled)'}"
                             f"{mark}")
            if seen.get("frame_path"):
                lines.append("the live frame is attached.")
                self._computer_send_frame(message, seen["frame_path"])
            return "\n".join(lines)
        if verb == "watch":
            return self._computer_watch_start(svc, session_id, message)
        if verb == "unwatch":
            return self._computer_watch_stop(session_id, watch_id,
                                             message)
        if verb in ("takeover", "handback", "stop", "resume", "close"):
            handler = {"takeover": svc.takeover, "handback": svc.handback,
                       "stop": svc.stop, "resume": svc.resume,
                       "close": svc.close}[verb]
            result = handler(session_id)
            if not result.get("ok"):
                return f"/computer {verb} failed: {result.get('error')}"
            rec = result["session"]
            if verb == "takeover":
                return (f"the desktop is yours — the agent's hands are "
                        f"off {rec['id']} until /computer handback.")
            if verb == "handback":
                return (f"handed back — the agent may drive "
                        f"{rec['id']} again"
                        + (" (the owner stop is still on — /computer "
                           "resume clears it)."
                           if rec.get("owner_stop") else "."))
            if verb == "stop":
                return ("computer stopped. The stop is sticky: it "
                        "survives restarts and reconciles until you "
                        "say /computer resume.")
            if verb == "resume":
                return f"computer resumed — {rec['id']} may be driven."
            return f"session {rec['id']} closed."
        return ("computer verbs: status · open · view · watch · "
                "unwatch · takeover · handback · stop · resume · "
                "close — /computer on its own shows the state.")

    def _computer_send_frame(self, message, frame_path: str) -> None:  # type: ignore[no-untyped-def]
        """Send a desktop frame to the owner's chat as a photo."""
        if message is None:
            return
        try:
            platform = str(getattr(message.chat, "platform", "") or "")
            chat_key = str(getattr(message.chat, "key", "") or "")
            if not platform or not chat_key:
                return
            self.gateway.send_file(
                platform, chat_key, str(frame_path),
                caption="the desktop, live", kind="photo",
                name="desktop.png")
        except Exception as exc:  # noqa: BLE001 — the text reply still lands
            record_failure("computer view send", exc)

    def _computer_watch_manager(self):  # type: ignore[no-untyped-def]
        """The ONE computer-watch manager for this runtime (cached).

        Mirrors _browser_run_jobs: the manager owns the watch
        threads and the durable watch records, so it is built
        once, never fresh per command. Its construction
        reconciles records a previous process left standing
        live — a watch never silently resurrects across a
        restart. The capture/gate functions close over their own
        SessionManager (perception is ungated; the store is the
        shared truth across manager instances)."""
        mgr = getattr(self, "_computer_watch_manager_inst", None)
        if mgr is None:
            from ...computer.sessions import SessionManager
            from ...computer.watch import (
                ComputerWatchManager,
                session_capture_fn,
                session_gate_fn,
            )

            sessions = SessionManager()
            mgr = ComputerWatchManager(
                capture_fn=session_capture_fn(sessions),
                gate_fn=session_gate_fn(sessions),
                db=getattr(getattr(self, "context", None), "db",
                           None))
            self._computer_watch_manager_inst = mgr
        return mgr

    def _computer_watch_senders(self, message):  # type: ignore[no-untyped-def]
        """(send_frame, send_note) bound to the chat that asked.

        Frames ride the gateway's file seam as photos (the
        _computer_send_frame precedent); notes ride its text
        seam. A failed send raises into the watch's delivery
        loop, which counts and ledgers it — the watch itself
        never dies on a send."""
        platform = str(getattr(message.chat, "platform", "") or "")
        chat_key = str(getattr(message.chat, "key", "") or "")
        gateway = self.gateway

        def _check(result, what):  # type: ignore[no-untyped-def]
            if result is not None and not getattr(result, "ok", True):
                raise RuntimeError(
                    f"{what} send failed: "
                    f"{getattr(result, 'error', '') or 'unknown error'}")

        def send_frame(frame, caption):  # type: ignore[no-untyped-def]
            _check(gateway.send_file(
                platform, chat_key, frame.frame_path,
                caption=caption, kind="photo", name="desktop.png"),
                "frame")

        def send_note(text):  # type: ignore[no-untyped-def]
            _check(gateway.send(platform, chat_key, text), "note")

        return send_frame, send_note

    def _computer_watch_start(self, svc, session_id, message) -> str:  # type: ignore[no-untyped-def]
        """``/computer watch [session]`` — start the live tail."""
        if message is None:
            return ("a watch delivers to the chat that starts "
                    "it — there is no chat here to stream to.")
        platform = str(getattr(message.chat, "platform", "") or "")
        chat_key = str(getattr(message.chat, "key", "") or "")
        if not platform or not chat_key:
            return ("this surface has no chat address to stream "
                    "to, so no watch was started.")
        rec = svc.manager.resolve(session_id)
        if rec is None:
            return ("no desktop session exists yet — /computer "
                    "open starts one, then /computer watch "
                    "streams it here.")
        if rec.state != "open":
            return (f"session {rec.id} is {rec.state}"
                    + (f" — {rec.note}" if rec.note else "")
                    + "; a watch needs an open session.")
        mgr = self._computer_watch_manager()
        send_frame, send_note = self._computer_watch_senders(message)
        watch, duplicate = mgr.start(
            rec, platform=platform, chat_key=chat_key,
            send_frame=send_frame, send_note=send_note)
        budget = watch.budget_s
        budget_words = (f"{budget / 60:.0f}m" if budget >= 120
                        else f"{budget:.0f}s")
        if duplicate:
            return (f"already watching {rec.id} — the stream is "
                    f"live in this chat. /computer unwatch stops "
                    f"it; it is also on /runs.")
        return (f"watching {rec.id} live — new frames land here "
                f"as the desktop changes (watch budget "
                f"{budget_words}). /computer unwatch stops it; "
                f"it also shows on /runs.")

    def _computer_watch_stop(self, session_id, watch_id,  # type: ignore[no-untyped-def]
                             message) -> str:
        """``/computer unwatch`` — stop through the ONE facade."""
        mgr = self._computer_watch_manager()
        watch = None
        if watch_id:
            watch = mgr.get(watch_id)
        if watch is None and message is not None:
            chat_key = str(getattr(message.chat, "key", "") or "")
            if chat_key:
                watch = mgr.active_for(chat_key, session_id)
        if watch is None:
            return ("no watch is running for this chat — "
                    "/computer watch starts one.")
        from ...core.cancellation import cancel_with_receipt

        db = getattr(getattr(self, "context", None), "db", None)
        try:
            result = cancel_with_receipt(
                db, f"watch:{watch.watch_id}",
                requested_by="owner",
                reason="stopped from /computer unwatch",
                watch_manager=mgr)
        except Exception as exc:  # noqa: BLE001 - the facade
            # re-raises a classifiable organ refusal unchanged
            # (its doctrine) — speak the organ's words.
            return (f"the watch answered the stop in its own "
                    f"words: {exc}")
        receipt = result.get("receipt")
        if receipt is not None:
            state = receipt.get("receipt_state") or ""
            if state == "requested":
                return ("⏹ stop requested — the watch lands "
                        "cancelled at its next tick and delivery "
                        "ceases. Receipt: requested — it rides "
                        "the watch's /runs ticket.")
            if state == "confirmed":
                return ("⏹ stopped — the watch stands cancelled "
                        "on its own record. Receipt: confirmed.")
            if state == "not_found":
                return ("the watch was already gone — nothing "
                        "to stop. Receipt: not_found.")
            return (f"the stop was not honored (receipt: "
                    f"{state or 'refused'}) — "
                    f"{receipt.get('detail') or 'the watch manager declined'}")
        return (f"⏹ the watch was stopped natively, but no "
                f"receipt could be minted: "
                f"{result.get('receipt_error') or 'unknown reason'}")

    def _delegation_card(self, store, contracts, message) -> None:  # type: ignore[no-untyped-def]
        """Publish/revise the delegation living card for this chat."""
        try:
            from ...social.chat.action_schema import Action
            from ...social.chat.living_cards import Card
            from . import runtime as _runtime_mod

            if not self._group_gate(message, "delegate"):
                return
            chat = message.chat
            chat_key = str(getattr(chat, "key", "") or "")
            owner_key = _runtime_mod._pick_owner_key(message)
            rows = tuple(
                (c.id, f"{c.status} · {c.title[:40]}")
                for c in contracts[:8])
            actions = tuple(
                Action(label=c.id, intent="callback",
                       data=f"delegate show {c.id}")
                for c in contracts[:6])
            card = Card(
                card_id="delegation:" + chat_key, family="delegation",
                chat_key=chat_key, owner_key=owner_key,
                platform=str(getattr(chat, "platform", "") or ""),
                chat_id=str(getattr(chat, "chat_id", "") or ""),
                title="Delegation — work contracts",
                body="Delegated work under contract: evidence, "
                     "independent verification and the claim ledger. "
                     "Tap a contract for its full record.",
                rows=rows, actions=actions, status="open")
            host = self._living_card_host()
            existing = host.store.get(card.card_id)
            if existing is not None and not existing.terminal:
                host.revise(card, chat, title=card.title, body=card.body,
                            rows=rows, actions=actions, status="open")
            else:
                host.publish(card, chat)
        except Exception as exc:  # noqa: BLE001 - card is garnish
            record_failure("delegation.card", exc)

    def _pack_open(self, pack, message) -> str:  # type: ignore[no-untyped-def]
        """Publish (or revise) the pack's home card in this
        chat and record the use on the action bar's memory."""
        from ...social.chat import packs as _packs
        from ...social.chat.living_cards import Card
        from . import runtime as _runtime_mod

        if not self._group_gate(message, "pack"):
            return (f"I can open the {pack.title} pack — but "
                    f"this room isn't claimed for my bot-system "
                    f"posts (§33), so I'm not dropping its card "
                    f"here.")
        store = _packs.get_pack_store()
        chat = message.chat
        chat_key = str(getattr(chat, "key", "") or "")
        owner_key = _runtime_mod._pick_owner_key(message)
        scope = _packs.scope_key(pack, chat_key, owner_key)
        session = store.session(pack.id, scope)
        if not session:
            session = _packs.initial_session(pack.id)
            if session:
                store.save_session(pack.id, scope, session)
        facts = self._pack_facts(message)
        home = _packs.build_home(pack, session=session,
                                 facts=facts)
        host = self._living_card_host()
        card_id = self._pack_card_id(pack, chat_key, scope)
        existing = host.store.get(card_id)
        card = Card(
            card_id=card_id, family=pack.family,
            chat_key=chat_key, owner_key=owner_key,
            platform=str(getattr(chat, "platform", "") or ""),
            chat_id=str(getattr(chat, "chat_id", "") or ""),
            title=home.title, body=home.body, rows=home.rows,
            actions=home.actions, status="open")
        if existing is not None and not existing.terminal:
            host.revise(card, chat, title=home.title,
                        body=home.body, rows=home.rows,
                        actions=home.actions, status="open")
        else:
            host.publish(card, chat)
        store.record_use(pack.id)
        return (f"{pack.icon} {pack.title} — its card is live "
                f"above; it revises as the pack moves.")

    def _pack_act(self, pack, verb: str, args: str, message) -> str:  # type: ignore[no-untyped-def]
        """Run one native verb: a pure transition over the
        pack's session, persisted, with the pack's home card
        revised to the new state when one is live here."""
        from ...social.chat import packs as _packs
        from . import runtime as _runtime_mod

        store = _packs.get_pack_store()
        chat = message.chat
        chat_key = str(getattr(chat, "key", "") or "")
        owner_key = _runtime_mod._pick_owner_key(message)
        scope = _packs.scope_key(pack, chat_key, owner_key)
        session = store.session(pack.id, scope)
        facts = self._pack_facts(message)
        outcome = _packs.apply_verb(pack, verb, args,
                                    session=session, facts=facts)
        if outcome.session is not None:
            store.save_session(pack.id, scope, outcome.session)
            session = outcome.session
        if outcome.dispatch:
            if getattr(self, "_pack_dispatching", False):
                return ("that workflow would loop back into a "
                        "pack dispatch — refused.")
            self._pack_dispatching = True
            try:
                return self.handle_control(
                    "/" + outcome.dispatch, chat_key,
                    message=message)
            finally:
                self._pack_dispatching = False
        # Refresh the home card to the post-verb state.
        try:
            host = self._living_card_host()
            card_id = self._pack_card_id(pack, chat_key, scope)
            existing = host.store.get(card_id)
            if existing is not None and not existing.terminal:
                home = _packs.build_home(pack, session=session,
                                         facts=facts)
                host.revise(existing, chat, title=home.title,
                            body=home.body, rows=home.rows,
                            actions=home.actions)
        except Exception as exc:  # noqa: BLE001 - card is garnish
            record_failure("packs.act.card_revise", exc)
        if outcome.artifact is not None:
            path, media_kind = outcome.artifact
            try:
                from ...social.chat.base import MediaRef

                ref = MediaRef(path=str(path), kind=media_kind,
                               name=Path(path).name)
                sent = self._expression_send(message, ref)
                if not getattr(sent, "ok", False):
                    return (outcome.reply + f"\n(the send failed "
                            f"here: "
                            f"{getattr(sent, 'error', '')})")
            except Exception as exc:  # noqa: BLE001 - named below
                record_failure("packs.act.artifact_send", exc)
                return (outcome.reply + "\n(the artifact send "
                        "itself failed on my end.)")
        return outcome.reply
