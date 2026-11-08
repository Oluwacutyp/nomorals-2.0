"""Runtime surface for the anime organ (FEI gap wave B).

One dispatch entry — :meth:`RuntimeAnimeMixin._control_anime` —
serves the whole family; the capability logic lives in
:mod:`nomorals.anime` (sources → service → presentation).  This
mixin owns only the chat-shaped concerns: the §33 group gate
(mirroring the expression media commands), per-user favorite
identity, cover/picture delivery through the gateway's file path,
and the whatanime image resolution (own attachment or the replied
message's media, via the expression mixin's resolver).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from .failures import record_failure

_log = get_logger(__name__)

__all__ = ["RuntimeAnimeMixin", "ANIME_KINDS"]

#: Every kind this mixin serves (kept in step with the registry in
#: social/chat/control.py — the doctor audit cross-checks both).
ANIME_KINDS = frozenset({
    "anime", "manga", "anichar", "charpic", "staff", "studio",
    "episodes", "animechars", "trailer", "streaming", "animereview",
    "genres", "genre", "topanime", "topmanga", "popular", "trending",
    "airing", "upcoming", "season", "topchars", "anischedule",
    "nextep", "recommend", "randomanime", "randommanga", "randomchar",
    "animequote", "favorite", "whatanime",
    "waifu", "neko", "kitsune", "husbando",
})

_IMAGE_POOL_KINDS = {"waifu", "neko", "kitsune", "husbando", "charpic"}

#: Usage lines are bracket-free on purpose: WhatsApp's
#: presentation strips tag-shaped text, so a "<title>"
#: placeholder reached the owner's phone eaten (the names
#: inside the brackets vanished).  Plain words survive every
#: surface.
_USAGE = {
    "anime": "usage: /anime title — e.g. /anime frieren",
    "manga": "usage: /manga title — e.g. /manga berserk",
    "anichar": "usage: /anichar character name",
    "staff": "usage: /staff person name — voice actors, directors",
    "studio": "usage: /studio studio name",
    "episodes": "usage: /episodes anime title",
    "animechars": "usage: /animechars anime title — its cast",
    "trailer": "usage: /trailer anime title",
    "streaming": "usage: /streaming anime title — legal watch links",
    "animereview": "usage: /animereview title — scores + reviews",
    "nextep": "usage: /nextep anime title",
    "recommend": "usage: /recommend title — more like it",
    "genre": "usage: /genre name — /genres lists them",
}


class RuntimeAnimeMixin:
    """Chat surface for the anime organ. Mixed into PartnerRuntime."""

    # ── wiring ────────────────────────────────────────────────────
    def _anime_service(self):  # type: ignore[no-untyped-def]
        """The shared AnimeService (transport injectable for tests
        via context.extras["anime_transport"])."""
        svc = getattr(self, "_anime_service_obj", None)
        if svc is not None:
            return svc
        from ...anime import (
            AniListClient,
            ImagePoolClient,
            JikanClient,
            QuoteClient,
            TraceMoeClient,
        )
        from ...anime.favorites import FavoritesStore
        from ...anime.service import AnimeService
        from ...anime.transport import core_http_transport

        extras = getattr(self.context, "extras", {}) or {}
        transport = extras.get("anime_transport") or core_http_transport()
        db = getattr(self.context, "db", None)
        svc = AnimeService(
            anilist=AniListClient(transport),
            jikan=JikanClient(transport),
            quotes=QuoteClient(transport),
            images=ImagePoolClient(transport),
            trace=TraceMoeClient(transport),
            favorites=FavoritesStore(db),
        )
        self._anime_service_obj = svc
        return svc

    def _anime_user_key(self, message: Any) -> str:
        """The §29 person key: one identity per person, every room."""
        if message is None:
            return "owner"
        from ...community.group_gangs import person_key

        platform = str(getattr(message.chat, "platform", "") or "")
        user_id = str(getattr(message, "sender_id", "")
                      or getattr(message, "sender", "") or "")
        return person_key(platform, user_id)

    def _anime_cache_dir(self) -> Path:
        try:
            home = Path(self.context.settings.home_path)
        except Exception:  # noqa: BLE001 - fall back to the layout
            home = Path.home() / ".nomorals"
        path = home / "anime_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ── image delivery ────────────────────────────────────────────
    def _anime_send_image(self, chat_key: str, outcome: Any, *,
                          buttons: Any = None,
                          label: str = "") -> str:
        """Fetch the outcome's image and send it into the chat
        AS A PHOTO (delivery-shape law, 2026-11-07): the send
        declares kind "image" and carries a human name, so the
        picture lands inline — never as a hash-named document
        attachment (the /husbando defect).  ``buttons`` ride
        the photo on surfaces whose media sends carry them.

        Returns "" when the image went out (or none was offered);
        a plain-words note when it could not be delivered — the
        text answer always stands on its own."""
        url = str(getattr(outcome, "image_url", "") or "")
        if not url or not chat_key:
            return ""
        chat = self._ref_from_key(chat_key)
        if chat is None or getattr(self, "gateway", None) is None:
            return ""
        try:
            from ...anime.transport import core_http_transport

            extras = getattr(self.context, "extras", {}) or {}
            transport = extras.get("anime_transport") \
                or core_http_transport()
            result = transport.request("GET", url, timeout=15.0)
            if not result.ok or not result.content:
                return "(the picture itself wouldn't download)"
            suffix = ".png" if url.lower().split("?")[0].endswith(".png") \
                else ".jpg"
            digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
            path = self._anime_cache_dir() / f"{digest}{suffix}"
            path.write_bytes(result.content)
            raw_label = str(label or getattr(outcome, "title", "")
                            or "").strip().lower()
            # A presented name must be a legal file name on
            # every host (the live install is Windows): strip
            # the path-hostile characters, keep it human.
            friendly = "".join(
                ch for ch in " ".join(raw_label.split())[:40]
                if ch not in '<>:"/\\|?*').strip() or "anime picture"
            sent = self.gateway.send_file(
                chat.platform, chat.key, str(path),
                caption=str(getattr(outcome, "image_caption", "") or ""),
                kind="image", name=f"{friendly}{suffix}",
                buttons=buttons)
            if sent is not None and getattr(sent, "ok", False):
                return ""
            return "(the picture couldn't be sent on this surface)"
        except Exception as exc:  # noqa: BLE001 - garnish never kills
            record_failure("runtime_anime._anime_send_image", exc)
            return "(the picture couldn't be sent on this surface)"

    # ── dispatch ──────────────────────────────────────────────────
    def _control_anime(self, kind: str, tail: str, *,
                       chat_key: str = "",
                       message: Any = None) -> Any:
        """The whole anime family behind one dispatch branch."""
        from ...anime.present import present_outcome

        # §33: in a group, anime is bot-system output — claimed
        # rooms only, exactly like the expression media commands.
        # DMs are the owner's/member's own room — never gated.
        if message is not None and not self._group_gate(message, "anime"):
            return ("this room isn't switched on for my group "
                    "systems yet — anime runs where I'm claimed "
                    "(my bot operating here, or the owner's "
                    "designation: /gsettings system output on).")
        svc = self._anime_service()
        text = (tail or "").strip()
        words = text.split()
        head = words[0].lower() if words else ""
        rest = text[len(words[0]):].strip() if words else ""

        # "/<kind> open <n>" — reopen an entry from the last list.
        # (/favorite has its own open: its list is the saved list,
        # not the result cache, so it is handled in its branch.)
        if kind != "favorite" and head == "open" \
                and len(words) >= 2 and words[1].isdigit():
            return present_outcome(svc.open_result(
                kind, int(words[1]), chat_key=chat_key))

        if kind in ("anime", "manga"):
            if not text:
                return _USAGE[kind]
            return self._anime_reply(
                svc.search_title(kind, text, chat_key=chat_key),
                chat_key)
        if kind == "anichar":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(
                svc.search_character(text, chat_key=chat_key), chat_key)
        if kind == "staff":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(
                svc.search_staff(text, chat_key=chat_key), chat_key)
        if kind == "studio":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(
                svc.search_studio(text, chat_key=chat_key), chat_key)
        if kind == "animechars":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.cast_of(text), chat_key)
        if kind == "episodes":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.episodes(text), chat_key)
        if kind == "trailer":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.trailer(text), chat_key)
        if kind == "streaming":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.streaming(text), chat_key)
        if kind == "animereview":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.review(text), chat_key)
        if kind == "nextep":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(svc.next_episode(text), chat_key)
        if kind == "recommend":
            if not text:
                return _USAGE[kind]
            return self._anime_reply(
                svc.recommend(text, chat_key=chat_key), chat_key)
        if kind == "genres":
            return present_outcome(svc.genres())
        if kind == "genre":
            if not text:
                return _USAGE[kind]
            return present_outcome(
                svc.genre(text, chat_key=chat_key))
        if kind == "topanime":
            return present_outcome(svc.top("anime", chat_key=chat_key))
        if kind == "topmanga":
            return present_outcome(svc.top("manga", chat_key=chat_key))
        if kind in ("popular", "trending"):
            media_kind = "manga" if head == "manga" else "anime"
            fn = svc.popular if kind == "popular" else svc.trending
            return present_outcome(fn(media_kind, chat_key=chat_key))
        if kind == "airing":
            return present_outcome(svc.airing(chat_key=chat_key))
        if kind == "upcoming":
            return present_outcome(svc.upcoming(chat_key=chat_key))
        if kind == "season":
            if not text:
                name, year = svc.current_season()
                return present_outcome(svc.season(
                    name, year, chat_key=chat_key))
            season_word = ""
            year = None
            for word in words:
                if word.isdigit() and len(word) == 4:
                    year = int(word)
                elif word.lower() in ("winter", "spring", "summer",
                                      "fall", "autumn"):
                    season_word = "FALL" if word.lower() == "autumn" \
                        else word.upper()
            if not season_word:
                return ("usage: /season [winter|spring|summer|fall] "
                        "[year] — bare /season shows this season")
            return present_outcome(svc.season(
                season_word, year, chat_key=chat_key))
        if kind == "topchars":
            return present_outcome(svc.top_characters(chat_key=chat_key))
        if kind == "anischedule":
            return present_outcome(svc.schedule())
        if kind == "randomanime":
            return self._anime_reply(svc.random_media("anime"), chat_key)
        if kind == "randommanga":
            return self._anime_reply(svc.random_media("manga"), chat_key)
        if kind == "randomchar":
            return self._anime_reply(svc.random_character(), chat_key)
        if kind == "animequote":
            return present_outcome(svc.quote(text))
        if kind == "favorite":
            return self._anime_favorite(svc, head, rest, text, message,
                                        chat_key)
        if kind == "whatanime":
            return self._anime_whatanime(svc, text, message)
        if kind in _IMAGE_POOL_KINDS:
            outcome = svc.picture(kind)
            if not outcome.ok:
                return present_outcome(outcome)
            # The follow-up choice rides the photo itself
            # (owner consistency rule): one tap draws another.
            note = self._anime_send_image(
                chat_key, outcome, label=kind,
                buttons=[[("🔁 Another one", kind)]])
            if note:
                return f"found one, but {note}."
            return "sent it here. 🖼️"
        return f"/{kind} isn't wired yet."

    def _anime_reply(self, outcome: Any, chat_key: str) -> Any:
        """Present an outcome, delivering its image first if any."""
        from ...anime.present import present_outcome

        if getattr(outcome, "ok", False):
            note = self._anime_send_image(chat_key, outcome)
            reply = present_outcome(outcome)
            if note:
                if hasattr(reply, "text"):
                    reply.text = f"{reply.text}\n{note}"
                else:
                    reply = f"{reply}\n{note}"
            return reply
        return present_outcome(outcome)

    def _anime_favorite(self, svc: Any, head: str, rest: str,
                        text: str, message: Any, chat_key: str) -> Any:
        from ...anime.present import present_outcome

        user_key = self._anime_user_key(message)
        if head in ("add", "save") and rest:
            return self._anime_reply(
                svc.favorites_add(user_key, rest), chat_key)
        if head in ("remove", "rm", "delete") and rest:
            return present_outcome(svc.favorites_remove(user_key, rest))
        if head == "open" and rest.isdigit():
            return self._anime_reply(
                svc.favorites_open(user_key, int(rest)), chat_key)
        if not text or head in ("list", "show"):
            return present_outcome(svc.favorites_list(user_key))
        if head:
            # "/favorite <title>" reads as an add — the natural form.
            return self._anime_reply(
                svc.favorites_add(user_key, text), chat_key)
        return present_outcome(svc.favorites_list(user_key))

    def _anime_whatanime(self, svc: Any, text: str,
                         message: Any) -> Any:
        from ...anime.present import present_outcome

        # A pasted image URL works from anywhere.
        if text.startswith(("http://", "https://")):
            return present_outcome(svc.identify_url(text.split()[0]))
        if message is None:
            return ("reply to an anime screenshot or frame with "
                    "/whatanime, or paste the image's URL after it.")
        found = self._replied_media_local(message)
        if found is None or not found[0]:
            return ("reply to an anime screenshot or frame with "
                    "/whatanime and I'll name the show, the episode "
                    "and the timestamp — on this platform path I "
                    "couldn't reach the picture itself.")
        path, media_kind, _ids = found
        if str(media_kind or "").lower() not in ("image", "sticker",
                                                 "photo", ""):
            return (f"that reply is a {media_kind or 'file'}, not a "
                    "picture — /whatanime reads still frames.")
        suffix = Path(path).suffix.lower()
        mime = {".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif"}.get(suffix, "image/jpeg")
        return self._anime_reply(svc.identify_file(path, mime),
                                 str(message.chat.key or ""))
