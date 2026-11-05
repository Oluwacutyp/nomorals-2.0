"""RuntimeMakersMixin: the makers & tools (FEI gap wave E) in
the live event path.

The organ itself (``social/chat/makers.py``) is transport-free;
this mixin owns the live wiring:

- **Cards** (/qcard + the census skins): the quote-card maker.
  Reply form reads the replied author + text the adapters
  stash in message meta (``reply_to_sender`` /
  ``reply_to_text`` — the Bot API surface provides both);
  explicit form is ``<name> | <text>``.  The rendered PNG
  rides the expression send path, caption on the media
  message — one message, so success returns "" and the kinds
  ride expression.MEDIA_DELIVERED_KINDS.
- **/removebg**: the replied/attached picture's bytes go to
  the segmentation organ (via the makers service, which
  self-provisions the rembg rung); the transparent PNG comes
  back the same media way.
- **/qr**: bare text → a QR PNG attached; ``/qr read`` on a
  replied picture → the vision organ's decode as text.
- **/shorturl /unshorten /bible**: text answers, formatted
  here — the shortener reply always names the full
  destination, the bible reply caps its verses per message
  and attributes its source.
- **/couplepp** (alias /pp): the diptych canvas attached,
  captioned as the one split artwork it is.
- **Gate**: in groups these are bot-system output, so they
  pass the same §33 gate as the fun/expression commands
  beside them (``_group_gate``); DMs are never gated.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .failures import record_failure

_log = logging.getLogger(__name__)

#: Verses shown per /bible reply — a chapter can run past a
#: hundred; the rest is one more command away, named in the
#: closing line.
_BIBLE_VERSE_CAP = 40


class RuntimeMakersMixin:
    """Maker & tool commands for PartnerRuntime."""

    # ---- the service -------------------------------------------
    def _makers_service(self):  # type: ignore[no-untyped-def]
        service = getattr(self, "_makers_service_obj", None)
        if service is not None:
            return service
        from ...social.chat.makers import MakerService

        service = MakerService(
            media_dir=self._expression_root() / "maker_media")
        self._makers_service_obj = service
        return service

    # ---- the one handler ----------------------------------------
    def _control_makers(self, kind: str, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """One maker/tool command.  Media kinds return ""
        after a confirmed send (the picture WAS the reply);
        everything else is text — or one honest line."""
        kind = str(kind or "").strip().lower()
        tail = str(tail or "").strip()
        if message is not None and not self._group_gate(message, kind):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — my makers only run where my bot "
                    "runs.")
        from ...social.chat.makers import CARD_KINDS

        if kind in CARD_KINDS:
            return self._makers_card(kind, tail, message)
        if kind == "removebg":
            return self._makers_removebg(message)
        if kind == "qr":
            return self._makers_qr(tail, message)
        service = self._makers_service()
        if kind == "shorturl":
            out = service.shorten(tail)
            if out.get("ok"):
                return (f"🔗 {out['short']}\n"
                        f"full destination: {out['url']}\n"
                        f"— via {out['source']}")
            return str(out.get("reason") or "no short link "
                                            "right now")
        if kind == "unshorten":
            out = service.unshorten(tail)
            if out.get("ok"):
                if not out.get("hops"):
                    return ("that link doesn't redirect — it "
                            f"already points straight at "
                            f"{out['final']}")
                return (f"that link lands on:\n{out['final']}\n"
                        f"(after {out['hops']} redirect"
                        f"{'s' if out['hops'] != 1 else ''})")
            return str(out.get("reason") or "couldn't follow "
                                            "that link")
        if kind == "bible":
            return self._makers_bible(tail)
        if kind in ("couplepp", "pp"):
            if message is None:
                return ("usage: /couplepp in a chat — the pair "
                        "needs somewhere to land.")
            out = service.couplepp(
                send=lambda ref, caption: self._expression_send(
                    message, ref, caption=caption))
            if out.get("sent"):
                return ""
            return str(out.get("reason") or "no couple pp "
                                            "right now")
        return f"no maker called /{kind}"

    # ---- cards ---------------------------------------------------
    def _makers_reply_context(self, message) -> tuple[str, str]:  # type: ignore[no-untyped-def]
        """(author name, text) of the replied-to message, from
        the meta the adapters stash: the Bot API surface
        provides ``reply_to_sender`` + ``reply_to_text``;
        surfaces that stash neither leave both empty and the
        explicit form is the way in."""
        meta = getattr(message, "meta", None) or {}
        name = ""
        sender = meta.get("reply_to_sender") or {}
        if isinstance(sender, dict):
            name = str(sender.get("name")
                       or sender.get("username") or "")
        if not name:
            chat = meta.get("reply_to_sender_chat") or {}
            if isinstance(chat, dict):
                name = str(chat.get("title") or "")
        text = ""
        for key in ("reply_to_text", "quoted_text", "reply_text"):
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                text = val.strip()
                break
        return name, text

    def _makers_card(self, kind: str, tail: str, message) -> str:  # type: ignore[no-untyped-def]
        if message is None:
            return (f"usage: reply to a message with /{kind}, "
                    f"or /{kind} <name> | <text> — in a chat, so "
                    f"the card has somewhere to land.")
        reply_name, reply_text = self._makers_reply_context(
            message)
        service = self._makers_service()
        out = service.card(
            kind, tail, reply_name=reply_name,
            reply_text=reply_text,
            sender_name=str(getattr(message, "sender", "")
                            or ""),
            send=lambda ref, caption: self._expression_send(
                message, ref, caption=caption))
        if out.get("sent"):
            return ""
        return str(out.get("reason") or f"no /{kind} card "
                                        f"right now")

    # ---- removebg --------------------------------------------------
    def _makers_removebg(self, message) -> str:  # type: ignore[no-untyped-def]
        if message is None:
            return ("usage: reply to a photo with /removebg — "
                    "I'll send it back with the background cut "
                    "out.")
        found = self._replied_media_local(message)
        if not found or not found[0]:
            return ("reply to a photo with /removebg (or send "
                    "the photo with /removebg as its caption) "
                    "and I'll cut its background out.")
        try:
            data = Path(str(found[0])).read_bytes()
        except OSError as exc:
            record_failure("makers.removebg.read", exc)
            return ("I couldn't read that picture's bytes here "
                    "— try sending it again.")
        service = self._makers_service()
        out = service.removebg(
            data,
            send=lambda ref, caption: self._expression_send(
                message, ref, caption=caption))
        if out.get("sent"):
            return ""
        return str(out.get("reason") or "no cutout right now")

    # ---- qr ---------------------------------------------------------
    def _makers_qr(self, tail: str, message) -> str:  # type: ignore[no-untyped-def]
        service = self._makers_service()
        if tail.strip().lower() == "read":
            if message is None:
                return ("usage: reply to a picture with "
                        "/qr read — I'll decode what's in it.")
            found = self._replied_media_local(message)
            if not found or not found[0]:
                return ("reply to a picture with /qr read and "
                        "I'll decode the code in it.")
            try:
                data = Path(str(found[0])).read_bytes()
            except OSError as exc:
                record_failure("makers.qr.read", exc)
                return ("I couldn't read that picture's bytes "
                        "here — try sending it again.")
            out = service.qr_read(data)
            if not out.get("ok"):
                return str(out.get("reason") or "no decode "
                                                "right now")
            codes = out.get("codes") or []
            if not codes:
                return ("no QR code or barcode in that "
                        "picture — I looked with the real "
                        "decoder.")
            lines = ["that picture's code says:"]
            for code in codes:
                lines.append(str(code.get("data") or "")[:500]
                             or "(an empty payload)")
            return "\n".join(lines)
        if not tail:
            return ("usage: /qr <text or a link> — or reply "
                    "to a picture with /qr read to decode one.")
        if message is None:
            return ("usage: /qr in a chat — the code needs "
                    "somewhere to land.")
        out = service.qr_make(
            tail,
            send=lambda ref, caption: self._expression_send(
                message, ref, caption=caption))
        if out.get("sent"):
            return ""
        return str(out.get("reason") or "no QR right now")

    # ---- bible -------------------------------------------------------
    def _makers_bible(self, tail: str) -> str:  # type: ignore[no-untyped-def]
        service = self._makers_service()
        out = service.bible(tail)
        if not out.get("ok"):
            return str(out.get("reason") or "no passage "
                                            "right now")
        if out.get("mode") == "search":
            hits = (out.get("hits") or [])[:8]
            if not hits:
                return (f"no verses matched “{out.get('query')}” "
                        f"in the {out.get('translation')} text "
                        f"I can search.")
            lines = [f"🔎 bible search — “{out.get('query')}” "
                     f"({out.get('translation')} via "
                     f"{out.get('source')})", ""]
            for hit in hits:
                text = str(hit.get("text") or "")
                if len(text) > 160:
                    text = text[:160].rsplit(" ", 1)[0] + "…"
                lines.append(f"• {hit['reference']} — {text}")
            return "\n".join(lines)
        verses = out.get("verses") or []
        shown = verses[:_BIBLE_VERSE_CAP]
        lines = [f"📖 {out.get('reference')} — "
                 f"{out.get('translation')} "
                 f"(via {out.get('source')})", ""]
        for verse in shown:
            lines.append(f"{verse['verse']} {verse['text']}")
        if len(verses) > len(shown):
            left = len(verses) - len(shown)
            lines.append(
                f"…and {left} more verse"
                f"{'s' if left != 1 else ''} in this stretch — "
                f"the chapter is long; /bible with a verse "
                f"range reads the part you name.")
        return "\n".join(lines)
