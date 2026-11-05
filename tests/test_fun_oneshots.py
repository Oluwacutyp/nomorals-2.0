"""FEI gap wave D — the fun one-shot commands (/joke, /fact,
/dog, /pokemon, /nasa, /riddle, /8ball, /choose, /calc, …)
at every seam:

- the registry: every shipped kind is in all five catalog
  tables with exactly one LIST_GROUPS membership, and the
  runtime's literal dispatch set can never drift from
  fun.FUN_COMMANDS; /animequote is NOT wave D's (the anime
  wave owns it — a name Devon has is never hijacked);
- the providers' real response shapes (fixtures shaped from
  the published APIs), shape validation as named failures,
  the health-checked failover order, and all-down honesty;
- image delivery: bytes sniffed as a real image, attached
  through the send path with the caption riding the media —
  a delivered picture returns "" (MEDIA_DELIVERED), never the
  never-empty law's "produced no reply" fallback;
- the local rung: canonical 8-ball answers, choose splitting,
  /calc through the games organ's own evaluator (never eval);
- /riddle's held answer per room; /history's date handling
  on a fixed clock.

Network-free throughout: every fetcher is a fake.
"""

from __future__ import annotations

import datetime
import os
import random
import re
import urllib.error
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_expression import (
    RuntimeExpressionMixin,
)
from nomorals.agents.partner.runtime_fun import RuntimeFunMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat import fun as fun_mod
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.expression import (
    MEDIA_DELIVERED_KINDS,
    ProviderHealth,
)
from nomorals.social.chat.fun import (
    EIGHT_BALL_ANSWERS,
    FUN_COMMANDS,
    LOCAL_RIDDLES,
    AdviceSlipProvider,
    BoredProvider,
    CataasProvider,
    CatFactProvider,
    ChuckProvider,
    DadJokeProvider,
    DogCeoProvider,
    DummyQuotesProvider,
    FunError,
    FunService,
    JokeApiProvider,
    MemeApiProvider,
    MuffinLabsHistoryProvider,
    NasaApodProvider,
    NumbersApiProvider,
    OfficialJokeProvider,
    PokeApiProvider,
    RandomFoxProvider,
    RiddlesApiProvider,
    TheCatApiProvider,
    UselessFactsProvider,
    WikimediaOnThisDayProvider,
    choose_pick,
    eight_ball_answer,
    parse_history_date,
    split_choices,
)

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100999"

JPEG_BYTES = b"\xff\xd8\xff\xe0" + bytes(range(32))
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
HTML_BYTES = b"<html><body>not a picture</body></html>"


# ── the registry ─────────────────────────────────────────────


def test_fun_commands_are_the_22():
    # the brief's set enumerates 23 names; animequote is the
    # anime wave's, so wave D ships the other 22
    assert len(FUN_COMMANDS) == 22
    assert "animequote" not in FUN_COMMANDS  # wave B owns it
    assert "coinflip" not in FUN_COMMANDS  # the casino owns it


def test_registry_presence_for_every_command():
    from nomorals.social.chat import control

    for name in FUN_COMMANDS:
        assert name in control.CONTROL_COMMANDS, name
        assert name in control.PUBLIC_COMMANDS, name
        assert name in control.COMMAND_DETAILS, name
        assert name in control.LIST_ONELINERS, name
        memberships = sum(
            name in members for _g, members in control.LIST_GROUPS)
        assert memberships == 1, (name, memberships)
        assert control.detailed_help(name), name
    menu = control.help_menu("whatsapp")
    for name in FUN_COMMANDS:
        assert f"/{name}" in menu, name
    assert len(menu) < 65536  # the one-message WhatsApp cap


def test_dispatch_literal_set_matches_fun_commands():
    src = Path("nomorals/agents/partner/runtime.py").read_text()
    start = src.index('"joke", "dadjoke"')
    block = src[start:src.index("}:", start)]
    kinds = set(re.findall(r'"([a-z0-9]+)"', block))
    assert kinds == set(FUN_COMMANDS)


def test_image_kinds_ride_media_delivered():
    assert frozenset({
        "steal", "sticker", "convert", "gif", "react",
        "dog", "cat", "fox", "meme", "nasa", "pokemon"}) == MEDIA_DELIVERED_KINDS
    from nomorals.social.chat.fun import FUN_IMAGE_KINDS

    assert FUN_IMAGE_KINDS <= MEDIA_DELIVERED_KINDS


# ── providers: jokes & short text ────────────────────────────


def test_jokeapi_twopart_and_single_shapes():
    seen = []

    def fetcher(url):
        seen.append(url)
        return {"error": False, "category": "Misc",
                "type": "twopart",
                "setup": "Why did the chicken cross the road?",
                "delivery": "To get to the other side."}

    hit = JokeApiProvider(fetcher=fetcher).joke()
    assert seen == ["https://v2.jokeapi.dev/joke/Any?safe-mode"]
    # one message, punchline on its own line
    assert hit["text"] == ("Why did the chicken cross the road?\n"
                           "To get to the other side.")
    single = JokeApiProvider(fetcher=lambda url: {
        "error": False, "type": "single",
        "joke": "I only know 25 letters of the alphabet. "
                "I don't know y."}).joke("Programming")
    assert single["text"].startswith("I only know 25 letters")


def test_jokeapi_programming_category_and_error_shape():
    seen = []
    JokeApiProvider(fetcher=lambda url: seen.append(url) or {
        "error": False, "type": "single", "joke": "UDP joke."}
    ).joke("Programming")
    assert seen == [
        "https://v2.jokeapi.dev/joke/Programming?safe-mode"]
    try:
        JokeApiProvider(fetcher=lambda url: {
            "error": True, "message": "no jokes found"}).joke()
    except FunError:
        pass
    else:
        raise AssertionError("error shape must be a named failure")


def test_official_joke_and_dadjoke_shapes():
    hit = OfficialJokeProvider(fetcher=lambda url: {
        "id": 1, "type": "general",
        "setup": "What do you call a fish without eyes?",
        "punchline": "A fsh."}).joke()
    assert hit["text"] == ("What do you call a fish without "
                           "eyes?\nA fsh.")
    seen = []
    dad = DadJokeProvider(fetcher=lambda url: seen.append(url) or {
        "id": "x", "joke": "I'm afraid for the calendar. "
        "Its days are numbered.", "status": 200}).joke()
    assert seen == ["https://icanhazdadjoke.com/"]
    assert dad["text"].startswith("I'm afraid for the calendar")


def test_chuck_fact_quote_advice_shapes():
    chuck = ChuckProvider(fetcher=lambda url: {
        "value": "Chuck Norris counted to infinity. Twice."}).fact()
    assert chuck["text"].endswith("Twice.")
    fact = UselessFactsProvider(fetcher=lambda url: {
        "id": "1", "text": "Honey never spoils.",
        "source": "djtech.net"}).fact()
    assert fact["text"] == "Honey never spoils."
    seen = []
    UselessFactsProvider(
        mode="today",
        fetcher=lambda url: seen.append(url) or {"text": "T."}).fact()
    assert seen == ["https://uselessfacts.jsph.pl/api/v2/facts/"
                    "today?language=en"]
    cat = CatFactProvider(fetcher=lambda url: {
        "fact": "Cats sleep 70% of their lives.", "length": 32}).fact()
    assert cat["text"].startswith("Cats sleep")
    advice = AdviceSlipProvider(fetcher=lambda url: {
        "slip": {"id": 117, "advice": "Always bet on black."}}).advice()
    assert advice["text"] == "Always bet on black."
    quote = DummyQuotesProvider(fetcher=lambda url: {
        "id": 7, "quote": "Stay hungry, stay foolish.",
        "author": "Steve Jobs"}).quote()
    assert quote["text"] == "“Stay hungry, stay foolish.”\n— Steve Jobs"
    try:
        AdviceSlipProvider(fetcher=lambda url: {}).advice()
    except FunError:
        pass
    else:
        raise AssertionError("shapeless slip must fail by name")


def test_numbersapi_text_forms():
    seen = []

    def fetcher(url):
        seen.append(url)
        return "42 is the number of laws of cricket."

    hit = NumbersApiProvider(fetcher=fetcher).fact("42")
    assert seen == ["http://numbersapi.com/42/trivia"]
    assert hit["text"].startswith("42 is the number")
    NumbersApiProvider(fetcher=fetcher).fact("")
    assert seen[-1] == "http://numbersapi.com/random/trivia"
    try:
        NumbersApiProvider(
            fetcher=lambda url: "<html>error</html>").fact("1")
    except FunError:
        pass
    else:
        raise AssertionError("an HTML page is not a number fact")


def test_bored_activity_shapes():
    hit = BoredProvider("bored-api", "https://x/random",
                        fetcher=lambda url: {
                            "activity": "Learn a new recipe",
                            "type": "cooking", "participants": 1,
                            "price": 0, "key": "1"}).activity()
    assert hit["text"] == "Learn a new recipe\n(cooking · 1 person)"
    solo = BoredProvider("boredapi-legacy", "https://y",
                         fetcher=lambda url: {
                             "activity": "Meditate",
                             "type": "", "participants": 0}).activity()
    assert solo["text"] == "Meditate"


# ── providers: history ───────────────────────────────────────


def test_history_provider_shapes():
    seen = []

    def fetch_wiki(url):
        seen.append(url)
        return {"events": [
            {"text": "Apollo 11 lands.", "year": 1969,
             "pages": [{"title": "Apollo 11"}]},
            {"text": "Something older.", "year": 1066,
             "pages": []}]}

    out = WikimediaOnThisDayProvider(fetcher=fetch_wiki).events(
        datetime.date(2026, 7, 20))
    assert seen == ["https://api.wikimedia.org/feed/v1/wikipedia/"
                    "en/onthisday/events/07/20"]
    assert out["events"][0] == {"year": 1969,
                                "text": "Apollo 11 lands."}
    muff = MuffinLabsHistoryProvider(fetcher=lambda url: {
        "date": "July 20", "data": {"Events": [
            {"year": "1969", "text": "Apollo 11 lands.",
             "links": []}]}}).events(datetime.date(2026, 7, 20))
    assert muff["events"] == [{"year": 1969,
                               "text": "Apollo 11 lands."}]


# ── providers: images ────────────────────────────────────────


def test_dog_cat_fox_providers():
    dog = DogCeoProvider(fetcher=lambda url: {
        "message": "https://images.dog.ceo/breeds/hound-afghan/"
                   "n02088094_1003.jpg", "status": "success"}).image()
    assert dog["caption"] == "🐶 hound afghan · via dog.ceo"
    try:
        DogCeoProvider(fetcher=lambda url: {
            "message": "x", "status": "error"}).image()
    except FunError:
        pass
    else:
        raise AssertionError("non-success dog.ceo must fail by name")
    cat = TheCatApiProvider(fetcher=lambda url: [
        {"id": "a", "url": "https://cdn2.thecatapi.com/a.jpg",
         "width": 800, "height": 600}]).image()
    assert cat["image_url"].endswith("a.jpg")
    cataas = CataasProvider(fetcher=lambda url: {
        "id": "abc", "url": "/cat/abc"}).image()
    assert cataas["image_url"] == "https://cataas.com/cat/abc"
    fox = RandomFoxProvider(fetcher=lambda url: {
        "image": "https://randomfox.ca/images/12.jpg",
        "link": "https://randomfox.ca/?i=12"}).image()
    assert fox["image_url"].endswith("12.jpg")


def test_meme_provider_carries_flags():
    hit = MemeApiProvider(fetcher=lambda url: {
        "postLink": "https://redd.it/x", "subreddit": "memes",
        "title": "when the tests pass", "nsfw": False,
        "spoiler": False, "author": "ada", "ups": 42,
        "url": "https://i.redd.it/x.jpg", "preview": []}).image()
    assert hit["caption"] == "when the tests pass\n— r/memes · via meme-api"
    assert hit["nsfw"] is False
    flagged = MemeApiProvider(fetcher=lambda url: {
        "title": "spicy", "subreddit": "x", "nsfw": True,
        "spoiler": False, "url": "https://i.redd.it/y.jpg"}).image()
    assert flagged["nsfw"] is True


def test_nasa_apod_image_video_and_trimming():
    img = NasaApodProvider(fetcher=lambda url: {
        "date": "2026-11-05", "title": "Orion Rising",
        "explanation": "A nebula does its thing.",
        "media_type": "image",
        "url": "https://apod.nasa.gov/orion.jpg"}).image()
    assert img["image_url"].endswith("orion.jpg")
    assert img["caption"].startswith("🌌 Orion Rising (2026-11-05)")
    long = NasaApodProvider(fetcher=lambda url: {
        "date": "2026-11-05", "title": "Long",
        "explanation": "word " * 400, "media_type": "image",
        "url": "https://apod.nasa.gov/long.jpg"}).image()
    assert len(long["caption"]) <= 1000
    assert long["caption"].endswith("…")
    vid = NasaApodProvider(fetcher=lambda url: {
        "date": "2026-11-05", "title": "Earth in Motion",
        "explanation": "A timelapse.", "media_type": "video",
        "url": "https://youtube.com/watch?v=x",
        "thumbnail_url": "https://img.youtube.com/x.jpg"}).image()
    assert vid["image_url"] == "https://img.youtube.com/x.jpg"
    assert "video day" in vid["caption"]
    still_less = NasaApodProvider(fetcher=lambda url: {
        "date": "2026-11-05", "title": "Earth in Motion",
        "explanation": "A timelapse.", "media_type": "video",
        "url": "https://youtube.com/watch?v=x"}).image()
    assert "text_only" in still_less
    assert "youtube.com/watch?v=x" in still_less["text_only"]


def test_nasa_rate_limit_is_named(monkeypatch):
    def boom(url, **kw):
        raise urllib.error.HTTPError(
            url, 429, "Too Many Requests", {}, None)

    monkeypatch.setattr(fun_mod, "_http_json", boom)
    try:
        NasaApodProvider().image()
    except FunError as exc:
        assert "rate-limited" in str(exc)
    else:
        raise AssertionError("a 429 must surface as a named limit")


_POKE = {
    "id": 25, "name": "pikachu", "height": 4, "weight": 60,
    "types": [{"slot": 1, "type": {"name": "electric",
                                   "url": "https://x/type/13"}}],
    "stats": [
        {"base_stat": 35, "stat": {"name": "hp"}},
        {"base_stat": 55, "stat": {"name": "attack"}},
        {"base_stat": 40, "stat": {"name": "defense"}},
        {"base_stat": 50, "stat": {"name": "special-attack"}},
        {"base_stat": 50, "stat": {"name": "special-defense"}},
        {"base_stat": 90, "stat": {"name": "speed"}}],
    "sprites": {
        "front_default": "https://img/25.png",
        "other": {"official-artwork": {
            "front_default": "https://img/official/25.png"}}},
}
_SPECIES = {"flavor_text_entries": [
    {"flavor_text": "When several of\nthese Pokémon\fgather, their "
                    "electricity could build and cause lightning "
                    "storms.",
     "language": {"name": "en"}, "version": {"name": "red"}}]}


def _poke_fetcher(seen):
    def fetcher(url):
        seen.append(url)
        if "pokemon-species" in url:
            return _SPECIES
        return _POKE
    return fetcher


def test_pokeapi_record_caption():
    seen = []
    hit = PokeApiProvider(fetcher=_poke_fetcher(seen)).image("Pikachu")
    assert seen[0] == "https://pokeapi.co/api/v2/pokemon/pikachu"
    assert hit["image_url"] == "https://img/official/25.png"
    caption = hit["caption"]
    assert caption.startswith("#25 Pikachu — electric")
    assert "HT 0.4 m · WT 6.0 kg" in caption
    assert "HP 35 · ATK 55 · DEF 40 · SPA 50 · SPD 50 · SPE 90" \
        in caption
    assert "When several of these Pokémon gather" in caption
    flavor = caption.split("“", 1)[1]
    assert "\n" not in flavor and "\f" not in flavor  # one clean line


def test_pokeapi_by_id_random_and_not_found():
    seen = []
    PokeApiProvider(fetcher=_poke_fetcher(seen)).image("25")
    assert seen[0].endswith("/pokemon/25")
    seen2 = []
    PokeApiProvider(fetcher=_poke_fetcher(seen2),
                    rng=random.Random(3)).image("")
    dex = int(seen2[0].rsplit("/", 1)[1])
    assert 1 <= dex <= 1025

    def missing(url):
        raise FunError("pokeapi 404")

    out = PokeApiProvider(fetcher=missing).image("missingno")
    assert "text_only" in out
    assert "no Pokémon called “missingno”" in out["text_only"]


def test_riddles_api_shape():
    hit = RiddlesApiProvider(fetcher=lambda url: {
        "riddle": "What has a neck but no head?",
        "answer": "A bottle"}).riddle()
    assert hit == {"question": "What has a neck but no head?",
                   "answer": "A bottle"}


# ── local logic ──────────────────────────────────────────────


def test_eight_ball_canonical_set():
    assert len(EIGHT_BALL_ANSWERS) == 20
    for member in ("It is certain.", "Without a doubt.",
                   "Reply hazy, try again.", "Cannot predict now.",
                   "Don't count on it.", "Very doubtful."):
        assert member in EIGHT_BALL_ANSWERS
    rng = random.Random(0)
    draws = {eight_ball_answer(rng) for _ in range(200)}
    assert draws <= set(EIGHT_BALL_ANSWERS)
    assert len(draws) > 10  # the draw actually varies


def test_choose_splitting_forms():
    assert split_choices("pizza or sushi") == ["pizza", "sushi"]
    assert split_choices("a, b or c") == ["a", "b", "c"]
    assert split_choices("a, b, c") == ["a", "b", "c"]
    assert split_choices("tea OR coffee") == ["tea", "coffee"]
    assert split_choices("") == []
    assert split_choices("solo") == ["solo"]
    rng = random.Random(1)
    assert choose_pick("a or b", rng) in {"a", "b"}
    assert choose_pick("solo", rng) is None
    assert choose_pick("", rng) is None


def test_parse_history_date_forms():
    today = datetime.date(2026, 11, 5)
    assert parse_history_date("", today) == today
    assert parse_history_date("nov 5", today) == today
    assert parse_history_date("5 nov", today) == today
    assert parse_history_date("november 5", today) == today
    assert parse_history_date("7/20", today) == datetime.date(2026, 7, 20)
    assert parse_history_date("2026-11-05", today) == today
    assert parse_history_date("feb 30", today) is None
    assert parse_history_date("garbage", today) is None


# ── the service: chains, honesty, delivery ───────────────────


class _FakeTextProvider:
    def __init__(self, name, method, *, result=None, error=None):
        self.name = name
        self.method = method
        self.result = result
        self.error = error
        self.calls = []

    def __getattr__(self, item):
        if item == self.method:
            def call(*args):
                self.calls.append(args)
                if self.error is not None:
                    raise self.error
                return self.result
            return call
        raise AttributeError(item)


def _service(tmp_path, chains, **kw):
    kw.setdefault("health", ProviderHealth())
    kw.setdefault("rng", random.Random(0))
    return FunService(media_dir=tmp_path / "fun_media",
                      chains=chains, **kw)


def test_service_joke_failover_and_health(tmp_path):
    down = _FakeTextProvider("jokeapi", "joke",
                             error=RuntimeError("down"))
    up = _FakeTextProvider("official-joke-api", "joke", result={
        "text": "Setup line\nPunchline"})
    svc = _service(tmp_path, {"joke": [down, up]})
    out = svc.text("joke")
    assert out["ok"] is True
    assert out["text"] == "Setup line\nPunchline"  # one message
    assert out["source"] == "official-joke-api"
    assert down.calls and up.calls
    assert svc.health.status("jokeapi")["fails"] == 1


def test_service_all_down_is_one_honest_line(tmp_path):
    down = _FakeTextProvider("jokeapi", "joke",
                             error=RuntimeError("down"))
    down2 = _FakeTextProvider("official-joke-api", "joke",
                              error=RuntimeError("down"))
    svc = _service(tmp_path, {"joke": [down, down2]})
    out = svc.text("joke")
    assert out["ok"] is False
    assert out["reason"] == ("every /joke source is down right "
                             "now — try again in a bit")


def test_service_progjoke_asks_the_programming_shelf(tmp_path):
    probe = _FakeTextProvider("jokeapi", "joke",
                              result={"text": "UDP joke."})
    svc = _service(tmp_path, {"progjoke": [probe]})
    out = svc.text("progjoke")
    assert out["ok"] is True
    assert probe.calls == [("Programming",)]


def test_service_numfact_passes_the_number(tmp_path):
    probe = _FakeTextProvider("numbersapi", "fact",
                              result={"text": "42 is …"})
    svc = _service(tmp_path, {"numfact": [probe]})
    svc.text("numfact", "42")
    assert probe.calls == [("42",)]


def _image_service(tmp_path, kind, providers, bytes_by_url):
    return _service(
        tmp_path, {kind: providers},
        bytes_fetcher=lambda url: bytes_by_url[url])


def _recording_send(box):
    def send(ref, caption):
        box.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")
    return send


def test_service_dog_attaches_with_caption(tmp_path):
    fake = _FakeTextProvider("dog.ceo", "image", result={
        "image_url": "https://img/dog.jpg",
        "caption": "🐶 hound · via dog.ceo"})
    svc = _image_service(tmp_path, "dog", [fake],
                         {"https://img/dog.jpg": JPEG_BYTES})
    sent = []
    out = svc.image("dog", send=_recording_send(sent))
    assert out["sent"] is True
    assert len(sent) == 1
    ref, caption = sent[0]
    assert ref.kind == "image"
    assert ref.path.endswith(".jpg")
    assert Path(ref.path).exists()
    assert caption == "🐶 hound · via dog.ceo"


def test_service_never_sends_a_non_image(tmp_path):
    fake = _FakeTextProvider("dog.ceo", "image", result={
        "image_url": "https://img/error-page",
        "caption": "🐶 via dog.ceo"})
    svc = _image_service(tmp_path, "dog", [fake],
                         {"https://img/error-page": HTML_BYTES})
    sent = []
    out = svc.image("dog", send=_recording_send(sent))
    assert out["sent"] is False
    assert sent == []
    assert "non-image" in out["reason"]
    assert svc.health.status("dog.ceo")["fails"] == 1


def test_service_bad_asset_falls_through_the_chain(tmp_path):
    bad = _FakeTextProvider("thecatapi", "image", result={
        "image_url": "https://img/error-page",
        "caption": "🐱 via thecatapi"})
    good = _FakeTextProvider("cataas", "image", result={
        "image_url": "https://img/cat.jpg",
        "caption": "🐱 via cataas"})
    svc = _service(
        tmp_path, {"cat": [bad, good]},
        bytes_fetcher=lambda url: {
            "https://img/error-page": HTML_BYTES,
            "https://img/cat.jpg": JPEG_BYTES}[url])
    sent = []
    out = svc.image("cat", send=_recording_send(sent))
    assert out["sent"] is True
    assert out["source"] == "cataas"  # the bad asset shed its source
    assert sent[0][1] == "🐱 via cataas"
    assert svc.health.status("thecatapi")["fails"] == 1


def test_service_meme_skips_flagged_draws(tmp_path):
    draws = iter([
        {"image_url": "https://m/1.jpg", "caption": "spicy",
         "nsfw": True, "spoiler": False},
        {"image_url": "https://m/2.jpg", "caption": "clean",
         "nsfw": False, "spoiler": False}])
    fake = _FakeTextProvider("meme-api", "image", result=None)
    fake.method = "image"

    def _image(arg=""):
        fake.calls.append((arg,))
        return next(draws)

    fake.image = _image
    svc = _image_service(tmp_path, "meme", [fake], {
        "https://m/1.jpg": JPEG_BYTES, "https://m/2.jpg": JPEG_BYTES})
    sent = []
    out = svc.image("meme", send=_recording_send(sent))
    assert out["sent"] is True
    assert sent[0][1] == "clean"  # the flagged draw never rode


def test_service_meme_all_flagged_is_honest(tmp_path):
    fake = _FakeTextProvider("meme-api", "image", result={
        "image_url": "https://m/1.jpg", "caption": "spicy",
        "nsfw": True, "spoiler": False})
    svc = _image_service(tmp_path, "meme", [fake],
                         {"https://m/1.jpg": JPEG_BYTES})
    sent = []
    out = svc.image("meme", send=_recording_send(sent))
    assert out["sent"] is False
    assert sent == []
    assert "flagged" in out["reason"]


def test_service_text_only_results_pass_through(tmp_path):
    fake = _FakeTextProvider("pokeapi", "image", result={
        "text_only": "no Pokémon called “missingno” in the Pokédex"})
    svc = _image_service(tmp_path, "pokemon", [fake], {})
    sent = []
    out = svc.image("pokemon", "missingno",
                    send=_recording_send(sent))
    assert out["sent"] is False
    assert out["text"].startswith("no Pokémon called")
    assert sent == []


def test_service_riddle_hold_reveal_replace(tmp_path):
    fake = _FakeTextProvider("riddles-api", "riddle", result={
        "question": "What has a neck but no head?",
        "answer": "A bottle"})
    svc = _service(tmp_path, {"riddle": [fake]})
    assert svc.riddle_answer("room1")["ok"] is False  # none held
    out = svc.new_riddle("room1")
    assert out["ok"] is True
    assert svc.has_riddle("room1")
    assert svc.has_riddle("room2") is False  # held per room
    revealed = svc.riddle_answer("room1")
    assert revealed["answer"] == "A bottle"
    assert svc.riddle_answer("room1")["ok"] is False  # cleared
    # a fresh riddle replaces the held one
    fake.result = {"question": "New?", "answer": "New answer"}
    svc.new_riddle("room1")
    fake.result = {"question": "Newer?", "answer": "Newer answer"}
    svc.new_riddle("room1")
    assert svc.riddle_answer("room1")["answer"] == "Newer answer"


def test_service_riddle_local_bank_floor(tmp_path):
    down = _FakeTextProvider("riddles-api", "riddle",
                             error=RuntimeError("down"))
    svc = _service(tmp_path, {"riddle": [down]})
    out = svc.new_riddle("room1")
    assert out["ok"] is True
    assert out["source"] == "her own bank"
    bank = dict(LOCAL_RIDDLES)
    assert out["question"] in bank
    assert svc.riddle_answer("room1")["answer"] == bank[out["question"]]


def test_service_history_fixed_clock_and_arg(tmp_path):
    events = [{"year": year, "text": f"Event {year}"}
              for year in (1066, 1969, 2001, 1990, 1850, 1776, 1945)]
    probe = _FakeTextProvider("wikimedia-onthisday", "events",
                              result={"events": events})
    svc = _service(tmp_path, {"history": [probe]},
                   today_fn=lambda: datetime.date(2026, 11, 5))
    out = svc.text("history")
    assert out["ok"] is True
    assert probe.calls == [(datetime.date(2026, 11, 5),)]
    lines = out["text"].splitlines()
    assert lines[0] == "on this day — November 5:"
    assert len(lines) == 6  # header + five entries
    years = [int(line.split(" — ")[0][2:]) for line in lines[1:]]
    assert years == sorted(years)
    svc.text("history", "jul 20")
    assert probe.calls[-1] == (datetime.date(2026, 7, 20),)
    bad = svc.text("history", "garbage!!")
    assert bad["ok"] is False
    assert "couldn't read that date" in bad["reason"]


# ── /calc on the games evaluator (the mixin) ─────────────────


def test_calc_through_the_real_evaluator():
    rt = RuntimeFunMixin()
    assert rt._fun_calc("2+3*4") == "2+3*4 = 14"
    assert rt._fun_calc("(2 + 3) * 4") == "(2 + 3) * 4 = 20"
    assert rt._fun_calc("10 / 4") == "10 / 4 = 2.5"
    assert rt._fun_calc("2 ** 10") == "2 ** 10 = 1024"
    assert rt._fun_calc("17 % 5") == "17 % 5 = 2"
    assert rt._fun_calc("") .startswith("usage: /calc")
    bad = rt._fun_calc("2 +")
    assert "can't compute" in bad
    evil = rt._fun_calc("__import__('os').system('echo hi')")
    assert "can't compute" in evil  # never a raw eval
    div0 = rt._fun_calc("1 / 0")
    assert "can't compute" in div0


# ── the runtime handler (mixin rig, wave-C pattern) ──────────


class _Chat:
    def __init__(self, platform, key, kind=ChatKind.GROUP):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = "Builders"
        self.kind = kind


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.media = kw.get("media") or []


class _Adapter:
    def __init__(self, admin_ids=(), bot_status="administrator"):
        self.media_sent = []
        self.admin_ids = set(admin_ids)
        self._bot_id = 4242
        self.bot_status = bot_status

    def admin_member(self, chat, user_id):
        if str(user_id) == str(self._bot_id):
            status = self.bot_status
        elif str(user_id) in self.admin_ids:
            status = "administrator"
        else:
            status = "member"
        return {"ok": True, "result": {"status": status}}

    def send_media(self, chat, ref, caption=""):
        self.media_sent.append((ref, caption))
        return SimpleNamespace(ok=True, message_id="9",
                               media_file_id="", error="")


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeFunMixin, RuntimeExpressionMixin,
               RuntimeGroupMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators


def _rig(tmp_path, chains=None, platform="telegram-bot",
         key=BOT_KEY):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    chat = _Chat(platform, key)
    rt._fun_service_obj = FunService(
        media_dir=tmp_path / "fun_media", chains=chains or {},
        health=ProviderHealth(), rng=random.Random(0),
        bytes_fetcher=lambda url: JPEG_BYTES,
        today_fn=lambda: datetime.date(2026, 11, 5))
    return rt, adapter, chat


def test_handler_text_kind_returns_the_text(tmp_path):
    joke = _FakeTextProvider("jokeapi", "joke",
                             result={"text": "Setup\nPunchline"})
    rt, _adapter, chat = _rig(tmp_path, {"joke": [joke]})
    out = rt._control_fun("joke", "", _Msg(chat))
    assert out == "Setup\nPunchline"


def test_handler_image_kind_returns_empty_and_attaches(tmp_path):
    dog = _FakeTextProvider("dog.ceo", "image", result={
        "image_url": "https://img/dog.jpg",
        "caption": "🐶 via dog.ceo"})
    rt, adapter, chat = _rig(tmp_path, {"dog": [dog]})
    out = rt._control_fun("dog", "", _Msg(chat))
    assert out == ""  # the media WAS the reply
    assert len(adapter.media_sent) == 1
    ref, caption = adapter.media_sent[0]
    assert ref.kind == "image"
    assert caption == "🐶 via dog.ceo"


def test_handler_unclaimed_room_is_gated(tmp_path):
    joke = _FakeTextProvider("jokeapi", "joke",
                             result={"text": "Setup\nPunchline"})
    rt, adapter, chat = _rig(tmp_path, {"joke": [joke]},
                             platform="telegram", key=USER_KEY)
    out = rt._control_fun("joke", "", _Msg(chat))
    assert "isn't claimed" in str(out)
    assert adapter.media_sent == []


def test_handler_dm_is_never_gated(tmp_path):
    joke = _FakeTextProvider("jokeapi", "joke",
                             result={"text": "Setup\nPunchline"})
    rt, _adapter, _chat = _rig(tmp_path, {"joke": [joke]})
    dm = _Chat("telegram", "telegram:777", kind=ChatKind.DM)
    out = rt._control_fun("joke", "", _Msg(dm))
    assert out == "Setup\nPunchline"


def test_handler_riddle_flow(tmp_path):
    fake = _FakeTextProvider("riddles-api", "riddle", result={
        "question": "What has a neck but no head?",
        "answer": "A bottle"})
    rt, _adapter, chat = _rig(tmp_path, {"riddle": [fake]})
    none = rt._control_fun("riddle", "answer", _Msg(chat))
    assert "no riddle is open" in none
    posted = rt._control_fun("riddle", "", _Msg(chat))
    assert posted.startswith("🧩 What has a neck")
    assert "riddle answer reveals" in posted
    revealed = rt._control_fun("riddle", "answer", _Msg(chat))
    assert revealed == "the answer was: A bottle"
    again = rt._control_fun("riddle", "answer", _Msg(chat))
    assert "no riddle is open" in again


def test_handler_choose_and_8ball_and_numfact_guard(tmp_path):
    rt, _adapter, chat = _rig(tmp_path)
    pick = rt._control_fun("choose", "tea or coffee", _Msg(chat))
    assert pick.startswith("🎯 ")
    assert pick[2:] in {"tea", "coffee"}
    one = rt._control_fun("choose", "tea", _Msg(chat))
    assert one == "only one option there — tea it is."
    bare = rt._control_fun("choose", "", _Msg(chat))
    assert bare.startswith("usage: /choose")
    ball = rt._control_fun("8ball", "will it rain", _Msg(chat))
    assert ball.startswith("🎱 ")
    assert ball[2:] in EIGHT_BALL_ANSWERS
    guard = rt._control_fun("numfact", "abc", _Msg(chat))
    assert guard.startswith("usage: /numfact")


def test_handler_all_down_text_is_the_honest_line(tmp_path):
    down = _FakeTextProvider("jokeapi", "joke",
                             error=RuntimeError("down"))
    rt, _adapter, chat = _rig(tmp_path, {"joke": [down]})
    out = rt._control_fun("joke", "", _Msg(chat))
    assert out == ("every /joke source is down right now — "
                   "try again in a bit")


# ── the composed control path ────────────────────────────────


def _composed_runtime():
    import tempfile

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.TemporaryDirectory(
        prefix="nm-waved-", dir=os.environ.get("NM_TEST_TMPDIR")
        or None)
    settings = load_settings(overrides={
        "home": tmp.name, "partner.platforms": "local",
        "chat.local_enabled": "true", "partner.enabled": "false"})
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    return PartnerRuntime(context), tmp


def test_composed_dog_delivery_is_not_a_no_reply(tmp_path):
    from nomorals.social.chat.base import ChatMessage, ChatRef

    rt, tmp = _composed_runtime()
    try:
        dog = _FakeTextProvider("dog.ceo", "image", result={
            "image_url": "https://img/dog.jpg",
            "caption": "🐶 via dog.ceo"})
        rt._fun_service_obj = FunService(
            media_dir=Path(tmp.name) / "fun_media",
            chains={"dog": [dog]}, health=ProviderHealth(),
            rng=random.Random(0),
            bytes_fetcher=lambda url: JPEG_BYTES)
        chat = ChatRef(platform="local", chat_id="room1",
                       kind=ChatKind.DM)
        msg = ChatMessage(chat=chat, incoming=True, text="/dog",
                          sender="Ada", sender_id="7",
                          message_id="55")
        reply = rt.handle_control("/dog", "local:room1",
                                  message=msg)
        # the picture was the reply: "" — and crucially NOT the
        # never-empty fallback text after a successful send
        assert reply == ""
        landed = list((Path(tmp.name) / "fun_media").glob("*.jpg"))
        assert landed  # the fetch really landed on disk
    finally:
        tmp.cleanup()


def test_composed_joke_and_calc_through_handle_control(tmp_path):
    rt, tmp = _composed_runtime()
    try:
        joke = _FakeTextProvider("jokeapi", "joke", result={
            "text": "Setup line\nPunchline"})
        rt._fun_service_obj = FunService(
            media_dir=Path(tmp.name) / "fun_media",
            chains={"joke": [joke]}, health=ProviderHealth(),
            rng=random.Random(0))
        assert rt.handle_control("/joke", "local:room1") \
            == "Setup line\nPunchline"
        assert rt.handle_control("/calc 2+3*4", "local:room1") \
            == "2+3*4 = 14"
        bad = rt.handle_control("/calc open('x')", "local:room1")
        assert "can't compute" in bad
    finally:
        tmp.cleanup()
