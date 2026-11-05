"""Fun one-shots (FEI gap wave D): the one-tap, group-playable
content commands — /joke, /fact, /dog, /pokemon, /nasa, /riddle,
/8ball, /calc and their kin.

The organ is transport-free, like ``expression.py`` beside it:
providers fetch, the service walks health-checked seed chains,
and the runtime mixin (``agents/partner/runtime_fun.py``) owns
the live wiring — the §33 group gate, media delivery through
the expression send path, and /calc, which reuses the games
organ's own arithmetic evaluator (never a raw ``eval``).

Honesty rules, the same as every source organ in the house:

- Sources are SEEDS, never a closed list — each chain is an
  ordered walk, a dead source sheds through the shared
  :class:`ProviderHealth` breaker, and settings can grow the
  chains later without touching this file's logic.
- Response shapes are VALIDATED.  A payload that doesn't match
  the source's real shape is a named failure (:class:`FunError`
  or a shape ``ValueError``), never a crash and never content
  invented to fill the gap.
- All sources down = ONE honest line.  A command never presents
  invented content as fetched.  (The brain can always answer
  conversationally — these commands report their sources.)
- /riddle is the one held-state game here: the question posts,
  the answer is held per room and revealed on ``/riddle
  answer``.  Its floor is a bank of classic riddles shipped in
  this file — the code rung, like the expression library floor
  under the reaction GIFs: real content, never fabricated on
  the spot.

Image one-shots (dog/cat/fox/meme/nasa/pokemon) deliver the
picture ATTACHED through the caller's send callable with the
caption riding the media message — never a bare URL.  Fetched
bytes are sniffed by magic before they can ride: a source that
answers with an error page has not sent a picture.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import random
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .expression import ProviderHealth

__all__ = [
    "FUN_COMMANDS", "FUN_IMAGE_KINDS", "FUN_TEXT_KINDS",
    "FunError", "FunService", "default_fun_chains",
    "EIGHT_BALL_ANSWERS", "eight_ball_answer",
    "split_choices", "choose_pick", "parse_history_date",
    "JokeApiProvider", "OfficialJokeProvider", "DadJokeProvider",
    "ChuckProvider", "UselessFactsProvider", "CatFactProvider",
    "NumbersApiProvider", "AdviceSlipProvider", "DummyQuotesProvider",
    "BoredProvider",
    "WikimediaOnThisDayProvider", "MuffinLabsHistoryProvider",
    "DogCeoProvider", "TheCatApiProvider", "CataasProvider",
    "RandomFoxProvider", "MemeApiProvider", "NasaApodProvider",
    "PokeApiProvider", "RiddlesApiProvider", "LOCAL_RIDDLES",
]


class FunError(Exception):
    """A named source failure: the message is fit to reach the
    user when this source was the only route (a rate limit, a
    Pokédex miss) — it names what actually happened."""


#: Every wave-D kind, in registry order.  /animequote is NOT
#: here: the sibling anime wave (B) owns it — same collision
#: discipline as wave C's kick/kill: a name Devon already has
#: for a working command is never hijacked.
FUN_COMMANDS: tuple[str, ...] = (
    "joke", "dadjoke", "progjoke", "chuck", "fact", "fact2",
    "catfact", "numfact", "advice", "quote",
    "dog", "cat", "fox", "meme", "riddle", "8ball", "choose",
    "activity", "pokemon", "history", "nasa", "calc",
)

#: Kinds answered by attaching a picture (they ride
#: expression.MEDIA_DELIVERED_KINDS: success returns "" from
#: the handler because the media WAS the reply).
FUN_IMAGE_KINDS: frozenset[str] = frozenset(
    {"dog", "cat", "fox", "meme", "nasa", "pokemon"})

#: Kinds answered from a text source chain.
FUN_TEXT_KINDS: frozenset[str] = frozenset({
    "joke", "dadjoke", "progjoke", "chuck", "fact", "fact2",
    "catfact", "numfact", "advice", "quote",
    "activity", "history"})


# ── http (the house urllib shape, fun UA) ────────────────────────

def _http_json(url: str, *, timeout: float = 15.0,
               headers: dict[str, str] | None = None
               ) -> Any:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-fun/1.0",
                      **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_text(url: str, *, timeout: float = 15.0) -> str:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-fun/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def _http_bytes(url: str, *, timeout: float = 30.0,
                cap: int = 20 * 1024 * 1024) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-fun/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(cap + 1)
    if len(data) > cap:
        raise ValueError("media exceeds the 20MB cap")
    return data


def _sniff_image(data: bytes) -> tuple[str, str] | None:
    """(extension, mime) for REAL image bytes — the magic
    decides, never the URL's claim: JPEG, PNG, GIF, WebP."""
    head = bytes(data or b"")[:16]
    if head[:3] == b"\xff\xd8\xff":
        return (".jpg", "image/jpeg")
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return (".png", "image/png")
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return (".gif", "image/gif")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return (".webp", "image/webp")
    return None


def _trim_caption(text: str, cap: int = 1000) -> str:
    """Caption discipline: media captions cap at 1024 on the
    strictest house platform (platform_limits) — stay under it,
    cutting on a word boundary."""
    text = str(text or "").strip()
    if len(text) <= cap:
        return text
    cut = text[:cap].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return cut + "…"


def _need_text(value: Any, what: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"fun source shape: missing {what}")
    return text


# ── jokes ────────────────────────────────────────────────────────

class JokeApiProvider:
    """JokeAPI v2 (v2.jokeapi.dev) — the primary joke seed.
    ``safe-mode`` always: these commands play in groups.
    Single jokes answer ``{"type": "single", "joke": …}``;
    two-parters answer ``{"type": "twopart", "setup": …,
    "delivery": …}``; a failure answers ``{"error": true}``."""

    name = "jokeapi"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def joke(self, category: str = "") -> dict[str, Any]:
        cat = str(category or "Any").strip() or "Any"
        data = self._fetch(
            f"https://v2.jokeapi.dev/joke/{cat}?safe-mode")
        if not isinstance(data, dict) or data.get("error"):
            raise FunError("jokeapi answered an error")
        if data.get("type") == "twopart":
            setup = _need_text(data.get("setup"), "setup")
            delivery = _need_text(data.get("delivery"), "delivery")
            return {"text": f"{setup}\n{delivery}"}
        return {"text": _need_text(data.get("joke"), "joke")}


class OfficialJokeProvider:
    """official-joke-api — the second joke seed:
    ``{"setup": …, "punchline": …, "type": …}``."""

    name = "official-joke-api"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def joke(self, category: str = "") -> dict[str, Any]:
        data = self._fetch(
            "https://official-joke-api.appspot.com/random_joke")
        if not isinstance(data, dict):
            raise FunError("official-joke-api answered a non-object")
        setup = _need_text(data.get("setup"), "setup")
        punchline = _need_text(data.get("punchline"), "punchline")
        return {"text": f"{setup}\n{punchline}"}


class DadJokeProvider:
    """icanhazdadjoke — the dad seed (JSON only with the Accept
    header): ``{"joke": …, "status": 200}``."""

    name = "icanhazdadjoke"

    def __init__(self, *,
                 fetcher: Callable[..., Any] | None = None
                 ) -> None:
        self._fetch = fetcher or self._fetch_json

    @staticmethod
    def _fetch_json(url: str) -> Any:
        return _http_json(url, headers={"Accept": "application/json"})

    def joke(self, category: str = "") -> dict[str, Any]:
        data = self._fetch("https://icanhazdadjoke.com/")
        if not isinstance(data, dict) \
                or int(data.get("status") or 0) != 200:
            raise FunError("icanhazdadjoke answered a non-200 shape")
        return {"text": _need_text(data.get("joke"), "joke")}


class ChuckProvider:
    """api.chucknorris.io — ``{"value": …}``."""

    name = "chucknorris.io"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def fact(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://api.chucknorris.io/jokes/random")
        if not isinstance(data, dict):
            raise FunError("chucknorris.io answered a non-object")
        return {"text": _need_text(data.get("value"), "value")}


# ── facts & short text ───────────────────────────────────────────

class UselessFactsProvider:
    """uselessfacts.jsph.pl v2 — ``{"text": …, "source": …}``.
    ``mode="random"`` backs /fact; ``mode="today"`` (the fact
    of the day endpoint) backs /fact2 — two real endpoints,
    honestly differentiated."""

    name = "uselessfacts"

    def __init__(self, *, mode: str = "random",
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self.mode = "today" if mode == "today" else "random"
        self._fetch = fetcher or _http_json

    def fact(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch(
            "https://uselessfacts.jsph.pl/api/v2/facts/"
            f"{self.mode}?language=en")
        if not isinstance(data, dict):
            raise FunError("uselessfacts answered a non-object")
        return {"text": _need_text(data.get("text"), "text")}


class CatFactProvider:
    """catfact.ninja — ``{"fact": …, "length": n}``."""

    name = "catfact.ninja"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def fact(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://catfact.ninja/fact")
        if not isinstance(data, dict):
            raise FunError("catfact.ninja answered a non-object")
        return {"text": _need_text(data.get("fact"), "fact")}


class NumbersApiProvider:
    """numbersapi.com — plain-text trivia facts: ``/<n>/trivia``
    or ``/random/trivia``.  A text source, so it carries its own
    text fetcher (injectable like every fetcher here)."""

    name = "numbersapi"

    def __init__(self, *,
                 fetcher: Callable[[str], str] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_text

    def fact(self, arg: str = "") -> dict[str, Any]:
        num = str(arg or "").strip()
        path = f"{urllib.parse.quote(num)}/trivia" if num \
            else "random/trivia"
        text = str(self._fetch(f"http://numbersapi.com/{path}")
                   or "").strip()
        if not text or text.lower().startswith("<!doctype") \
                or text.lower().startswith("<html"):
            raise FunError("numbersapi answered a non-fact page")
        return {"text": text}


class AdviceSlipProvider:
    """api.adviceslip.com — ``{"slip": {"advice": …}}``."""

    name = "adviceslip"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def advice(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://api.adviceslip.com/advice")
        slip = (data or {}).get("slip") if isinstance(data, dict) \
            else None
        if not isinstance(slip, dict):
            raise FunError("adviceslip answered a non-slip shape")
        return {"text": _need_text(slip.get("advice"), "advice")}


class DummyQuotesProvider:
    """dummyjson quotes — the quotable-class seed:
    ``{"quote": …, "author": …}``."""

    name = "dummyjson-quotes"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def quote(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://dummyjson.com/quotes/random")
        if not isinstance(data, dict):
            raise FunError("dummyjson answered a non-object")
        quote = _need_text(data.get("quote"), "quote")
        author = str(data.get("author") or "").strip()
        return {"text": f"“{quote}”" + (f"\n— {author}" if author
                                        else "")}


class BoredProvider:
    """The Bored API (App Brewery's rehome is the primary seed,
    the original boredapi.com the second) — a flat activity
    object: ``{"activity": …, "type": …, "participants": n}``."""

    def __init__(self, name: str, url: str, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self.name = name
        self.url = url
        self._fetch = fetcher or _http_json

    def activity(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch(self.url)
        if not isinstance(data, dict):
            raise FunError(f"{self.name} answered a non-object")
        text = _need_text(data.get("activity"), "activity")
        kind = str(data.get("type") or "").strip()
        try:
            people = int(data.get("participants") or 0)
        except (TypeError, ValueError):
            people = 0
        flavor = " · ".join(p for p in (
            kind, f"{people} person" if people == 1 else
            f"{people} people" if people > 1 else "") if p)
        return {"text": text + (f"\n({flavor})" if flavor else "")}


# ── on this day ──────────────────────────────────────────────────

class WikimediaOnThisDayProvider:
    """Wikimedia's on-this-day feed — ``{"events": [{"text":
    …, "year": n, "pages": […]}]}`` for one month/day."""

    name = "wikimedia-onthisday"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def events(self, day: datetime.date) -> dict[str, Any]:
        data = self._fetch(
            "https://api.wikimedia.org/feed/v1/wikipedia/en/"
            f"onthisday/events/{day.month:02d}/{day.day:02d}")
        rows = (data or {}).get("events") if isinstance(data, dict) \
            else None
        if not isinstance(rows, list) or not rows:
            raise FunError("wikimedia onthisday answered no events")
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            try:
                year = int(row.get("year") or 0)
            except (TypeError, ValueError):
                year = 0
            if text:
                out.append({"year": year, "text": text})
        if not out:
            raise FunError("wikimedia onthisday events were shapeless")
        return {"events": out}


class MuffinLabsHistoryProvider:
    """history.muffinlabs.com — ``{"data": {"Events":
    [{"year": "1969", "text": …}]}}`` (years arrive as strings)."""

    name = "muffinlabs-history"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def events(self, day: datetime.date) -> dict[str, Any]:
        data = self._fetch(
            f"https://history.muffinlabs.com/date/"
            f"{day.month}/{day.day}")
        rows = ((data or {}).get("data") or {}).get("Events") \
            if isinstance(data, dict) else None
        if not isinstance(rows, list) or not rows:
            raise FunError("muffinlabs answered no events")
        out = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            text = str(row.get("text") or "").strip()
            try:
                year = int(str(row.get("year") or "0").strip())
            except ValueError:
                year = 0
            if text:
                out.append({"year": year, "text": text})
        if not out:
            raise FunError("muffinlabs events were shapeless")
        return {"events": out}


# ── image one-shots ──────────────────────────────────────────────

class DogCeoProvider:
    """dog.ceo — ``{"message": <image url>, "status":
    "success"}``.  The breed rides the image URL's path
    (``/breeds/hound-afghan/…``) — real data, named in the
    caption."""

    name = "dog.ceo"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def image(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://dog.ceo/api/breeds/image/random")
        if not isinstance(data, dict) \
                or data.get("status") != "success":
            raise FunError("dog.ceo answered a non-success shape")
        url = _need_text(data.get("message"), "message")
        breed = ""
        m = re.search(r"/breeds/([^/]+)/", url)
        if m:
            breed = m.group(1).replace("-", " ")
        caption = "🐶 " + (f"{breed} · " if breed else "") \
            + "via dog.ceo"
        return {"image_url": url, "caption": caption}


class TheCatApiProvider:
    """api.thecatapi.com image search (keyless tier) —
    ``[{"url": …, "width": n, "height": n}]``."""

    name = "thecatapi"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def image(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch(
            "https://api.thecatapi.com/v1/images/search")
        rows = data if isinstance(data, list) else []
        if not rows or not isinstance(rows[0], dict):
            raise FunError("thecatapi answered no image rows")
        url = _need_text(rows[0].get("url"), "url")
        return {"image_url": url, "caption": "🐱 via thecatapi"}


class CataasProvider:
    """cataas.com (cat as a service) — ``{"url": "/cat/<id>"}``
    (relative on older answers, absolute on newer; both are
    real shapes it serves)."""

    name = "cataas"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def image(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://cataas.com/cat?json=true")
        if not isinstance(data, dict):
            raise FunError("cataas answered a non-object")
        url = _need_text(data.get("url"), "url")
        if url.startswith("/"):
            url = "https://cataas.com" + url
        return {"image_url": url, "caption": "🐱 via cataas"}


class RandomFoxProvider:
    """randomfox.ca — ``{"image": <url>, "link": …}``."""

    name = "randomfox"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def image(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://randomfox.ca/floof/")
        if not isinstance(data, dict):
            raise FunError("randomfox answered a non-object")
        url = _need_text(data.get("image"), "image")
        return {"image_url": url, "caption": "🦊 via randomfox"}


class MemeApiProvider:
    """meme-api.com/gimme — ``{"title": …, "subreddit": …,
    "url": …, "nsfw": bool, "spoiler": bool, "ups": n}``.
    The flags ride along: the service skips flagged hits —
    these commands play in groups."""

    name = "meme-api"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def image(self, arg: str = "") -> dict[str, Any]:
        data = self._fetch("https://meme-api.com/gimme")
        if not isinstance(data, dict):
            raise FunError("meme-api answered a non-object")
        url = _need_text(data.get("url"), "url")
        title = str(data.get("title") or "").strip()
        sub = str(data.get("subreddit") or "").strip()
        caption = title
        if sub:
            caption += (f"\n— r/{sub}" if caption else f"r/{sub}")
        caption += " · via meme-api"
        return {"image_url": url, "caption": caption,
                "nsfw": bool(data.get("nsfw")),
                "spoiler": bool(data.get("spoiler"))}


class NasaApodProvider:
    """NASA APOD — ``{"title": …, "explanation": …,
    "media_type": "image"|"video", "url": …, "hdurl": …,
    "thumbnail_url": …}``.  The key is DEMO_KEY unless the
    owner configured one (settings/env ride the mixin); a 429
    is a NAMED failure — the rate limit is the fact, and the
    user hears it named, not a generic outage.  A video day
    delivers its thumbnail still (the API's own still for the
    video) and says so in the caption; a video day with no
    thumbnail answers text-only — never a broken attachment."""

    name = "nasa-apod"

    def __init__(self, *, api_key: str = "",
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self.api_key = str(api_key or "").strip() or "DEMO_KEY"
        self._fetch = fetcher or self._fetch_json

    def _fetch_json(self, url: str) -> Any:
        try:
            return _http_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise FunError(
                    "NASA APOD is rate-limited right now "
                    f"({self.api_key} quota) — try again later"
                ) from exc
            raise

    def image(self, arg: str = "") -> dict[str, Any]:
        url = ("https://api.nasa.gov/planetary/apod?"
               + urllib.parse.urlencode({"api_key": self.api_key}))
        data = self._fetch(url)
        if not isinstance(data, dict):
            raise FunError("nasa apod answered a non-object")
        title = _need_text(data.get("title"), "title")
        date = str(data.get("date") or "").strip()
        explanation = str(data.get("explanation") or "").strip()
        head = f"🌌 {title}" + (f" ({date})" if date else "")
        if data.get("media_type") == "video":
            video = str(data.get("url") or "").strip()
            thumb = str(data.get("thumbnail_url") or "").strip()
            if not thumb:
                return {"text_only": (
                    f"{head}\n{explanation}\n"
                    f"▶ today's APOD is a video: {video}").strip()}
            caption = _trim_caption(
                f"{head}\n{explanation}\n"
                f"▶ video day — this is its still; watch: {video}")
            return {"image_url": thumb, "caption": caption}
        image_url = str(data.get("url") or "").strip()
        if not image_url:
            raise FunError("nasa apod carried no image url")
        return {"image_url": image_url,
                "caption": _trim_caption(f"{head}\n{explanation}")}


class PokeApiProvider:
    """pokeapi.co — the record (``/pokemon/<name|id>``) plus
    its species flavor text.  The caption carries the record's
    REAL fields: dex number, types, height, weight, base stats
    and an English flavor line.  A miss is a named Pokédex
    failure (text-only), never an outage claim.  Random draws
    an id in 1..1025 (the full National Dex as of Gen IX)."""

    name = "pokeapi"
    MAX_DEX = 1025

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None,
                 rng: random.Random | None = None) -> None:
        self._fetch = fetcher or self._fetch_json
        self.rng = rng or random.Random()

    @staticmethod
    def _fetch_json(url: str) -> Any:
        try:
            return _http_json(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise FunError("pokeapi 404") from exc
            raise

    def image(self, arg: str = "") -> dict[str, Any]:
        key = str(arg or "").strip().lower().replace(" ", "-")
        if not key:
            key = str(self.rng.randint(1, self.MAX_DEX))
        try:
            data = self._fetch(
                f"https://pokeapi.co/api/v2/pokemon/{key}")
        except FunError as exc:
            if "404" in str(exc):
                return {"text_only": (
                    f"no Pokémon called “{key}” in the Pokédex — "
                    "check the name, or give a number 1–1025.")}
            raise
        if not isinstance(data, dict) or not data.get("name"):
            raise FunError("pokeapi answered a shapeless record")
        dex = int(data.get("id") or 0)
        name = str(data["name"]).replace("-", " ").title()
        types = " / ".join(
            str(t.get("type", {}).get("name") or "")
            for t in (data.get("types") or [])).strip(" /")
        stats = {}
        for row in data.get("stats") or []:
            label = str((row.get("stat") or {}).get("name") or "")
            short = {"hp": "HP", "attack": "ATK", "defense": "DEF",
                     "special-attack": "SPA", "special-defense": "SPD",
                     "speed": "SPE"}.get(label)
            if short:
                stats[short] = int(row.get("base_stat") or 0)
        sprites = data.get("sprites") or {}
        art = ((sprites.get("other") or {}).get("official-artwork")
               or {}).get("front_default") \
            or sprites.get("front_default") or ""
        if not art:
            raise FunError(f"pokeapi has no artwork for {name}")
        flavor = ""
        try:
            species = self._fetch(
                f"https://pokeapi.co/api/v2/pokemon-species/{dex}")
            for entry in (species or {}).get(
                    "flavor_text_entries") or []:
                if (entry.get("language") or {}).get("name") == "en":
                    flavor = re.sub(
                        r"\s+", " ", str(
                            entry.get("flavor_text") or "")).strip()
                    break
        except Exception:  # noqa: BLE001 - flavor is garnish; the record stands
            flavor = ""
        height_m = (int(data.get("height") or 0)) / 10.0
        weight_kg = (int(data.get("weight") or 0)) / 10.0
        stat_line = " · ".join(f"{k} {v}" for k, v in stats.items())
        lines = [f"#{dex} {name}" + (f" — {types}" if types else ""),
                 f"HT {height_m:.1f} m · WT {weight_kg:.1f} kg"]
        if stat_line:
            lines.append(stat_line)
        if flavor:
            lines.append(f"“{flavor}”")
        return {"image_url": str(art),
                "caption": _trim_caption("\n".join(lines))}


# ── riddles ──────────────────────────────────────────────────────

class RiddlesApiProvider:
    """riddles-api (vercel) — ``{"riddle": …, "answer": …}``."""

    name = "riddles-api"

    def __init__(self, *,
                 fetcher: Callable[[str], Any] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def riddle(self) -> dict[str, Any]:
        data = self._fetch(
            "https://riddles-api-eight.vercel.app/random")
        if not isinstance(data, dict):
            raise FunError("riddles-api answered a non-object")
        return {"question": _need_text(data.get("riddle"), "riddle"),
                "answer": _need_text(data.get("answer"), "answer")}


#: The code-rung floor: classic riddles, shipped with the
#: organ.  When every riddle source is down the game still
#: plays — real content, never fabricated on the spot.
LOCAL_RIDDLES: tuple[tuple[str, str], ...] = (
    ("I speak without a mouth and hear without ears. I have no "
     "body, but I come alive with wind. What am I?", "An echo"),
    ("The more of me you take, the more you leave behind. "
     "What am I?", "Footsteps"),
    ("I have cities but no houses, forests but no trees, and "
     "rivers without water. What am I?", "A map"),
    ("What has keys but can't open a single lock?", "A piano"),
    ("The person who makes me has no need of me; the person "
     "who buys me has no use for me; the person who uses me "
     "can neither see nor feel me. What am I?", "A coffin"),
    ("What runs all around a backyard, yet never moves?",
     "A fence"),
    ("What has a head and a tail but no body?", "A coin"),
    ("I'm tall when I'm young, and I'm short when I'm old. "
     "What am I?", "A candle"),
    ("What has many needles, but doesn't sew?", "A Christmas tree"),
    ("What gets wet while drying?", "A towel"),
    ("What can you catch, but never throw?", "A cold"),
    ("What has one eye, but can't see?", "A needle"),
    ("What belongs to you, but everyone else uses it?",
     "Your name"),
    ("What has hands, but can't clap?", "A clock"),
    ("What kind of room has no doors?", "A mushroom"),
    ("The more there is of me, the less you see. What am I?",
     "Darkness"),
)


# ── local logic (the code rung) ──────────────────────────────────

#: The canonical Magic 8-Ball answer set: ten affirmative,
#: five non-committal, five negative.
EIGHT_BALL_ANSWERS: tuple[str, ...] = (
    "It is certain.", "It is decidedly so.", "Without a doubt.",
    "Yes definitely.", "You may rely on it.", "As I see it, yes.",
    "Most likely.", "Outlook good.", "Yes.",
    "Signs point to yes.",
    "Reply hazy, try again.", "Ask again later.",
    "Better not tell you now.", "Cannot predict now.",
    "Concentrate and ask again.",
    "Don't count on it.", "My reply is no.",
    "My sources say no.", "Outlook not so good.", "Very doubtful.",
)


def eight_ball_answer(rng: random.Random) -> str:
    return rng.choice(EIGHT_BALL_ANSWERS)


def split_choices(text: str) -> list[str]:
    """The user's own options, split the way people write them:
    "a or b", "a, b or c", "a, b, c"."""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r"\s+or\s+", raw, flags=re.IGNORECASE)
    out: list[str] = []
    for part in parts:
        out.extend(p.strip() for p in part.split(","))
    return [p for p in out if p]


def choose_pick(text: str, rng: random.Random) -> str | None:
    options = split_choices(text)
    if len(options) < 2:
        return None
    return rng.choice(options)


_MONTHS = {name: i + 1 for i, name in enumerate((
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november",
    "december"))}
_MONTHS.update({name[:3]: num for name, num in list(_MONTHS.items())})


def parse_history_date(text: str, today: datetime.date
                       ) -> datetime.date | None:
    """Parse a /history date argument against ``today``:
    empty → today; "nov 5" / "5 nov" / "november 5" / "11/5" /
    "2026-11-05".  A bare day name gives that month/day in
    today's year.  None when the text names no readable date."""
    raw = str(text or "").strip().lower()
    if not raw:
        return today
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", raw)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)),
                                 int(m.group(3)))
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})/(\d{1,2})", raw)
    if m:
        try:
            return datetime.date(today.year, int(m.group(1)),
                                 int(m.group(2)))
        except ValueError:
            return None
    m = re.fullmatch(r"([a-z]+)\s+(\d{1,2})", raw) \
        or re.fullmatch(r"(\d{1,2})\s+([a-z]+)", raw)
    if m:
        first, second = m.group(1), m.group(2)
        month = _MONTHS.get(first) or _MONTHS.get(second)
        day_txt = second if first.isalpha() else first
        if month is None or not day_txt.isdigit():
            return None
        try:
            return datetime.date(today.year, month, int(day_txt))
        except ValueError:
            return None
    return None


# ── the service ──────────────────────────────────────────────────

def default_fun_chains(*, nasa_api_key: str = "",
                       rng: random.Random | None = None
                       ) -> dict[str, list[Any]]:
    """The seed chains, kind → ordered providers.  Seeds, not
    a closed list: the service walks whatever chain it is
    handed, and the chains can grow from settings later."""
    return {
        "joke": [JokeApiProvider(), OfficialJokeProvider()],
        "dadjoke": [DadJokeProvider()],
        "progjoke": [JokeApiProvider()],
        "chuck": [ChuckProvider()],
        "fact": [UselessFactsProvider(mode="random")],
        "fact2": [UselessFactsProvider(mode="today")],
        "catfact": [CatFactProvider()],
        "numfact": [NumbersApiProvider()],
        "advice": [AdviceSlipProvider()],
        "quote": [DummyQuotesProvider()],
        "activity": [
            BoredProvider("bored-api",
                          "https://bored-api.appbrewery.com/random"),
            BoredProvider("boredapi-legacy",
                          "https://www.boredapi.com/api/activity"),
        ],
        "history": [WikimediaOnThisDayProvider(),
                    MuffinLabsHistoryProvider()],
        "dog": [DogCeoProvider()],
        "cat": [TheCatApiProvider(), CataasProvider()],
        "fox": [RandomFoxProvider()],
        "meme": [MemeApiProvider()],
        "nasa": [NasaApodProvider(api_key=nasa_api_key)],
        "pokemon": [PokeApiProvider(rng=rng)],
        "riddle": [RiddlesApiProvider()],
    }


#: kind → (provider method, fixed argument or None for the
#: caller's own).  progjoke's fixed argument is its JokeAPI
#: category — the programming shelf of the same seed.
_TEXT_CALLS: dict[str, tuple[str, str | None]] = {
    "joke": ("joke", ""),
    "dadjoke": ("joke", ""),
    "progjoke": ("joke", "Programming"),
    "chuck": ("fact", None),
    "fact": ("fact", None),
    "fact2": ("fact", None),
    "catfact": ("fact", None),
    "numfact": ("fact", None),
    "advice": ("advice", None),
    "quote": ("quote", None),
    "activity": ("activity", None),
}


class FunService:
    """Walks the seed chains for the fun one-shots.  Every
    outcome is a plain dict — ``ok``/``sent`` says what
    ACTUALLY happened and ``reason``/``text`` says the rest.
    Never raises, like every expression-path service."""

    def __init__(self, *, media_dir: Path | str = "",
                 chains: dict[str, list[Any]] | None = None,
                 nasa_api_key: str = "",
                 health: ProviderHealth | None = None,
                 rng: random.Random | None = None,
                 bytes_fetcher: Callable[[str], bytes] | None = None,
                 today_fn: Callable[[], datetime.date] | None = None
                 ) -> None:
        self.media_dir = Path(media_dir) if media_dir else None
        self.rng = rng or random.Random()
        self.health = health or ProviderHealth()
        self.bytes_fetcher = bytes_fetcher or _http_bytes
        self.today_fn = today_fn or datetime.date.today
        merged = default_fun_chains(nasa_api_key=nasa_api_key,
                                    rng=self.rng)
        for kind, chain in (chains or {}).items():
            merged[kind] = list(chain)
        self.chains: dict[str, list[Any]] = merged
        self._riddles: dict[str, tuple[str, str]] = {}

    # -- the chain walk ----------------------------------------
    def _walk(self, kind: str, method: str, *args: Any
              ) -> tuple[dict[str, Any] | None, str]:
        """(result, source-name) from the first healthy source
        that answers in shape; (None, last-error) when the
        whole chain is down or shapeless."""
        last = ""
        for provider in self.chains.get(kind) or []:
            name = str(getattr(provider, "name", "?"))
            if not self.health.available(name):
                continue
            try:
                result = getattr(provider, method)(*args)
            except Exception as exc:  # noqa: BLE001 - a dead source sheds
                self.health.note(name, False)
                last = str(exc)[:160]
                continue
            if not isinstance(result, dict) or not result:
                self.health.note(name, False)
                last = f"{name} answered a shapeless result"
                continue
            self.health.note(name, True)
            return result, name
        return None, last

    # -- text one-shots -----------------------------------------
    def text(self, kind: str, arg: str = "") -> dict[str, Any]:
        kind = str(kind or "").strip().lower()
        try:
            if kind == "history":
                return self._history(arg)
            call = _TEXT_CALLS.get(kind)
            if call is None:
                return {"ok": False,
                        "reason": f"no fun text called /{kind}"}
            method, fixed = call
            result, source = self._walk(
                kind, method, fixed if fixed is not None else arg)
            if result is None:
                return {"ok": False, "reason": (
                    f"every /{kind} source is down right now — "
                    "try again in a bit")}
            text = str(result.get("text") or "").strip()
            if not text:
                return {"ok": False, "reason": (
                    f"every /{kind} source answered empty — "
                    "try again in a bit")}
            return {"ok": True, "text": text, "source": source}
        except Exception as exc:  # noqa: BLE001 - never breaks its host
            return {"ok": False, "reason": f"fun error: {exc}"[:160]}

    def _history(self, arg: str) -> dict[str, Any]:
        today = self.today_fn()
        day = parse_history_date(arg, today)
        if day is None:
            return {"ok": False, "reason": (
                "I couldn't read that date — try /history nov 5 "
                "or /history 2026-11-05")}
        result, source = self._walk("history", "events", day)
        if result is None:
            return {"ok": False, "reason": (
                "every /history source is down right now — "
                "try again in a bit")}
        events = list(result.get("events") or [])
        if len(events) > 5:
            events = self.rng.sample(events, 5)
        events.sort(key=lambda e: int(e.get("year") or 0))
        lines = [f"on this day — {day.strftime('%B')} {day.day}:"]
        for event in events:
            year = int(event.get("year") or 0)
            text = str(event.get("text") or "").strip()
            lines.append(f"• {year} — {text}" if year
                         else f"• {text}")
        return {"ok": True, "text": "\n".join(lines),
                "source": source}

    # -- image one-shots ----------------------------------------
    def image(self, kind: str, arg: str = "", *,
              send: Callable[[Any, str], Any]) -> dict[str, Any]:
        """Fetch → sniff → attach.  ``send`` takes (MediaRef,
        caption) and returns a SendResult-shaped object.
        Memes retry past flagged hits (groups play here);
        a text_only result (a Pokédex miss, a video-day APOD
        with no still) comes back as ``text`` for the handler
        to post instead.  Never raises."""
        kind = str(kind or "").strip().lower()
        try:
            last = ""
            for provider in self.chains.get(kind) or []:
                source = str(getattr(provider, "name", "?"))
                if not self.health.available(source):
                    continue
                # memes re-draw past flagged hits (groups play
                # here); every other kind draws once per source
                draws = 3 if kind == "meme" else 1
                for _ in range(draws):
                    try:
                        result = provider.image(arg)
                    except Exception as exc:  # noqa: BLE001 - a dead source sheds
                        self.health.note(source, False)
                        last = str(exc)[:160]
                        break
                    if not isinstance(result, dict) or not result:
                        self.health.note(source, False)
                        last = f"{source} answered a shapeless result"
                        break
                    self.health.note(source, True)
                    if result.get("text_only"):
                        return {"sent": False,
                                "text": str(result["text_only"])}
                    if result.get("nsfw") or result.get("spoiler"):
                        last = ("only flagged memes came up "
                                "just now")
                        continue
                    url = str(result.get("image_url") or "")
                    caption = _trim_caption(
                        str(result.get("caption") or ""))
                    try:
                        data = self.bytes_fetcher(url)
                    except Exception as exc:  # noqa: BLE001
                        self.health.note(source, False)
                        last = f"download failed: {exc}"[:160]
                        break
                    sniffed = _sniff_image(data)
                    if sniffed is None:
                        # The source answered with something that
                        # is not a picture — a wrong asset; noted
                        # against the source and NEVER sent.
                        self.health.note(source, False)
                        last = f"{source} served a non-image"
                        break
                    if self.media_dir is None:
                        return {"sent": False,
                                "reason": "no media dir configured"}
                    ext, mime = sniffed
                    digest = hashlib.sha256(data).hexdigest()[:16]
                    dest = self.media_dir / f"fun_{digest}{ext}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        dest.write_bytes(data)
                    from .base import MediaRef

                    ref = MediaRef(path=str(dest), kind="image",
                                   mime=mime,
                                   name=f"{kind} via {source}")
                    outcome = send(ref, caption)
                    if not getattr(outcome, "ok", False):
                        return {"sent": False, "reason": (
                            f"send failed: "
                            f"{getattr(outcome, 'error', '')}")[:160]}
                    return {"sent": True, "source": source}
            return {"sent": False, "reason": (
                f"no /{kind} picture reached me — "
                + (last or "every source is down"))}
        except Exception as exc:  # noqa: BLE001 - never breaks its host
            return {"sent": False, "reason": f"fun error: {exc}"[:160]}

    # -- riddles (held state, one per room) ----------------------
    def new_riddle(self, room: str) -> dict[str, Any]:
        """Post a fresh riddle in ``room`` and HOLD its answer
        (a new one replaces the held one).  Source chain first;
        her shipped bank of classics is the floor."""
        room = str(room or "")
        result, source = self._walk("riddle", "riddle")
        if result is not None:
            question = str(result.get("question") or "").strip()
            answer = str(result.get("answer") or "").strip()
        else:
            question, answer = self.rng.choice(LOCAL_RIDDLES)
            source = "her own bank"
        if not question or not answer:
            return {"ok": False, "reason": (
                "every riddle source answered empty — "
                "try again in a bit")}
        self._riddles[room] = (question, answer)
        return {"ok": True, "question": question, "source": source}

    def riddle_answer(self, room: str) -> dict[str, Any]:
        """Reveal (and clear) the held answer for ``room``."""
        held = self._riddles.pop(str(room or ""), None)
        if held is None:
            return {"ok": False}
        return {"ok": True, "question": held[0], "answer": held[1]}

    def has_riddle(self, room: str) -> bool:
        return str(room or "") in self._riddles
