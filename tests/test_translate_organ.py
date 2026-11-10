"""The translation organ (llm/translate.py + llm/languages.py).

Pins the owner ask (2026-11-03): foreign-language news arrives in the
owner's language, and /translate works on demand — with the ladder's
honesty intact: rungs fall in order (local → brain → providers →
marked identity), untranslated text is never dressed up as a
translation, and English content passes through untouched.
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

from nomorals.llm.languages import detect_language, normalize_code
from nomorals.llm.translate import Translator, translator_for

FR_HEADLINE = ("Les premières images de la simulation Lumina lèvent "
               "le voile sur l'enfance de l'Univers")
FR_GIST = ("Cette simulation numérique géante a permis à une équipe "
           "d'astronomes américains de reconstituer l'Univers après "
           "le Big Bang afin d'étudier les premières étoiles.")
EN_HEADLINE = ("The first images from the Lumina simulation lift "
               "the veil on the infancy of the Universe")
EN_GIST = ("The giant numerical simulation let a team of American "
           "astronomers rebuild the Universe after the Big Bang to "
           "study the first stars.")

#: (src, dst, source text) -> translation, for the fake providers.
_PAIRS: dict[tuple[str, str, str], str] = {
    ("fr", "en", FR_HEADLINE): EN_HEADLINE,
    ("fr", "en", FR_GIST): EN_GIST,
    ("fr", "en", "Bonjour le monde, ceci est un test"):
        "Hello world, this is a test",
    ("en", "es", "The morning is good and the sun is over the hills"):
        "La mañana es buena y el sol está sobre las colinas",
    ("fr", "en", "Bonjour le monde"): "Hello world",
}


class FakeResp:
    def __init__(self, payload=None, text: str = "", status: int = 200):
        self._payload = payload
        self.text = text
        self.status = status

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self):
        return self._payload


def _fake_translate(src: str, dst: str, text: str):
    """Segment-wise fake translation: exact pair hits, plus the two
    Lumina segments wherever they appear inside a larger chunk."""
    out = _PAIRS.get((src, dst, text))
    if out is not None:
        return out
    if (src, dst) == ("fr", "en"):
        swapped = text.replace(FR_HEADLINE, EN_HEADLINE).replace(
            FR_GIST, EN_GIST)
        if swapped != text:
            return swapped
    return None


class FakeHttp:
    """Provider-shaped fake transport: MyMemory + Lingva + pages.

    ``fail=True`` makes every call raise (the dead-network case);
    ``pages`` maps URL -> HTML for the URL-gist path.
    """

    def __init__(self, *, fail: bool = False, pages=None):
        self.fail = fail
        self.pages = pages or {}
        self.calls: list[str] = []

    def get(self, url: str, **kw):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network down")
        if url in self.pages:
            return FakeResp(text=self.pages[url])
        if "api.mymemory.translated.net" in url:
            qs = parse_qs(urlparse(url).query)
            text = qs["q"][0]
            src, _, dst = qs["langpair"][0].partition("|")
            out = _fake_translate(src, dst, text)
            if out is None:
                return FakeResp(status=429, payload={})
            return FakeResp(payload={
                "responseData": {"translatedText": out},
                "responseStatus": 200})
        if "/api/v1/" in url:  # Lingva shape
            parts = urlparse(url).path.split("/")
            src, dst = parts[3], parts[4]
            text = unquote(parts[5])
            out = _fake_translate(
                "fr" if src == "auto" else src, dst, text)
            if out is None:
                return FakeResp(status=404, payload={})
            return FakeResp(payload={"translation": out})
        return FakeResp(status=404, payload={})

    def post_json(self, url: str, payload, **kw):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network down")
        return FakeResp(status=403, payload={})


class FakeBrain:
    """Brain-rung fake: translates the chunk after the blank line."""

    def __init__(self):
        self.calls: list[str] = []

    def chat(self, messages, task_kind: str = ""):
        self.calls.append(task_kind)
        prompt = messages[0].content
        chunk = prompt.split("\n\n", 1)[1]
        out = _PAIRS.get(("fr", "en", chunk)) or _PAIRS.get(
            ("en", "es", chunk))
        if out is None:
            raise RuntimeError("brain has no answer for this chunk")
        return SimpleNamespace(text=out)


class FakeLocalBackend:
    def __init__(self):
        self.seen: list[tuple[str, str, str]] = []

    def translate(self, text: str, src_nllb: str, dst_nllb: str) -> str:
        self.seen.append((text, src_nllb, dst_nllb))
        out = _PAIRS.get(("fr", "en", text))
        if out is None:
            raise RuntimeError("fake backend: unknown text")
        return out


def _translator(**kw) -> Translator:
    kw.setdefault("client", FakeHttp())
    return Translator(None, **kw)


class DetectionTests(unittest.TestCase):
    def test_french(self):
        code, conf = detect_language(FR_HEADLINE + " " + FR_GIST)
        self.assertEqual(code, "fr")
        self.assertGreater(conf, 0.5)

    def test_spanish(self):
        code, _ = detect_language(
            "El presidente anunció que el gobierno presentará un "
            "nuevo plan para la economía del país durante los "
            "próximos meses y los mercados reaccionaron con calma")
        self.assertEqual(code, "es")

    def test_english(self):
        code, _ = detect_language(
            "The committee said the new plan will take effect over "
            "the coming months and officials expect more changes "
            "after the summit ends")
        self.assertEqual(code, "en")

    def test_german(self):
        code, _ = detect_language(
            "Die Regierung hat am Dienstag ein neues Gesetz für die "
            "Wirtschaft vorgestellt und die Minister sprachen über "
            "die nächsten Schritte nach dem Gipfel")
        self.assertEqual(code, "de")

    def test_russian_script(self):
        code, _ = detect_language(
            "Правительство объявило о новом плане развития "
            "экономики страны на ближайшие годы")
        self.assertEqual(code, "ru")

    def test_undetermined_is_honest(self):
        self.assertEqual(detect_language("xkcd zqwv kjpl")[0], "und")
        self.assertEqual(detect_language("")[0], "und")
        self.assertEqual(detect_language("42")[0], "und")

    def test_normalize(self):
        self.assertEqual(normalize_code("fr-CA"), "fr")
        self.assertEqual(normalize_code("French"), "fr")
        self.assertEqual(normalize_code("pt-BR"), "pt")
        self.assertEqual(normalize_code("klingon"), "")


class LadderTests(unittest.TestCase):
    def test_local_rung_wins_when_backend_present(self):
        backend = FakeLocalBackend()
        t = _translator(brain=FakeBrain(), local_backend=backend)
        out = t.translate(FR_HEADLINE)
        self.assertTrue(out.translated)
        self.assertEqual(out.rung, "local")
        self.assertEqual(out.text, EN_HEADLINE)
        self.assertEqual(backend.seen[0][1:], ("fra_Latn", "eng_Latn"))

    def test_local_rung_absent_without_weights_or_deps(self):
        t = _translator()
        ok, reason = t.local.available()
        self.assertFalse(ok)
        self.assertTrue(reason)  # an honest reason, never silence

    def test_brain_rung_when_no_local(self):
        brain = FakeBrain()
        t = _translator(brain=brain)
        out = t.translate(FR_GIST)
        self.assertTrue(out.translated)
        self.assertEqual(out.rung, "brain")
        self.assertEqual(out.text, EN_GIST)
        self.assertEqual(brain.calls, ["translate"])

    def test_provider_rung_when_no_brain(self):
        t = _translator()  # context None -> no router -> brain off
        out = t.translate(FR_HEADLINE)
        self.assertTrue(out.translated)
        self.assertTrue(out.rung.startswith("provider:"))
        self.assertEqual(out.text, EN_HEADLINE)

    def test_identity_fallback_is_marked(self):
        t = _translator(client=FakeHttp(fail=True))
        out = t.translate(FR_HEADLINE)
        self.assertFalse(out.translated)
        self.assertEqual(out.rung, "identity")
        self.assertEqual(out.text, FR_HEADLINE)
        self.assertIn("ORIGINAL", out.note)

    def test_english_passes_through_untouched(self):
        t = _translator(brain=FakeBrain())
        out = t.translate("The committee said the plan will take "
                          "effect over the coming months")
        self.assertFalse(out.translated)
        self.assertEqual(out.rung, "same-language")

    def test_explicit_target_overrides_once(self):
        t = _translator(brain=FakeBrain())
        out = t.translate(
            "The morning is good and the sun is over the hills",
            target="es")
        self.assertTrue(out.translated)
        self.assertEqual(out.target_lang, "es")
        self.assertIn("mañana", out.text)

    def test_provider_health_stands_down_a_dead_seed(self):
        http = FakeHttp(fail=True)
        t = _translator(client=http)
        texts = [
            "Le gouvernement a annoncé un nouveau plan pour le pays",
            "Les marchés ont réagi avec calme après la réunion",
            "Une équipe de chercheurs a publié les résultats hier",
        ]
        for text in texts:
            out = t.translate(text)
            self.assertEqual(out.rung, "identity")
        ok, reason = t.providers[0].available()
        self.assertFalse(ok)
        self.assertIn("stood down", reason)


class OwnerLanguageTests(unittest.TestCase):
    def test_default_english_and_memory_set(self):
        t = _translator()
        self.assertEqual(t.owner_language(), "en")
        self.assertEqual(t.set_owner_language("fr"), "fr")
        self.assertEqual(t.owner_language(), "fr")
        with self.assertRaises(ValueError):
            t.set_owner_language("klingon")

    def test_db_persistence(self):
        import sqlite3

        class _Db:
            def __init__(self):
                self.conn = sqlite3.connect(":memory:")
                self.conn.row_factory = sqlite3.Row

            def execute(self, sql, params=()):
                self.conn.execute(sql, params)
                self.conn.commit()

            def query_one(self, sql, params=()):
                row = self.conn.execute(sql, params).fetchone()
                return dict(row) if row else None

        db = _Db()
        t1 = Translator(None, db=db, client=FakeHttp())
        t1.set_owner_language("es")
        t2 = Translator(None, db=db, client=FakeHttp())
        self.assertEqual(t2.owner_language(), "es")


class UrlGistTests(unittest.TestCase):
    def test_short_page_full_translation(self):
        page = ("<html><head><title>" + FR_HEADLINE + "</title></head>"
                "<body><p>" + FR_GIST + "</p></body></html>")
        http = FakeHttp(pages={"https://lemonde.example/a": page})
        t = _translator(client=http)
        res = t.translate_url("https://lemonde.example/a")
        self.assertTrue(res["ok"])
        self.assertTrue(res["full"])
        self.assertTrue(res["translated"])
        self.assertEqual(res["title"], EN_HEADLINE)
        self.assertIn("giant numerical simulation", res["text"])

    def test_long_page_is_a_labelled_gist(self):
        body = " ".join([FR_GIST] * 12)
        page = ("<html><head><title>" + FR_HEADLINE + "</title></head>"
                "<body><p>" + body + "</p></body></html>")
        http = FakeHttp(pages={"https://lemonde.example/long": page})
        t = _translator(client=http)
        res = t.translate_url("https://lemonde.example/long")
        self.assertTrue(res["ok"])
        self.assertFalse(res["full"])
        self.assertTrue(res["translated"])
        self.assertGreater(res["words"], 100)

    def test_dead_page_reports_error(self):
        t = _translator(client=FakeHttp(fail=True))
        res = t.translate_url("https://lemonde.example/x")
        self.assertFalse(res["ok"])
        self.assertIn("fetch failed", res["error"])


def _french_row(**kw):
    row = {
        "id": "n1", "title": FR_HEADLINE, "summary": FR_GIST,
        "url": "https://lemonde.example/a", "source_name": "Le Monde",
        "published": 1000.0, "delivery_state": "undelivered",
        "interest": None,
    }
    row.update(kw)
    return row


class DeliveryPathTests(unittest.TestCase):
    """The news digest + the pulse story block, with a fake provider."""

    def _agent(self):
        from nomorals.agents.news import NewsAgent

        ctx = SimpleNamespace(settings=None, db=None, router=None)
        ctx._translator_for_ctx = _translator()
        agent = NewsAgent(ctx)
        agent._pending = [
            _french_row(),
            _french_row(id="n2", title="The markets rose on Tuesday",
                        summary="Stocks climbed as investors welcomed "
                                "the news from the summit.",
                        url="https://example.com/markets",
                        source_name="Example"),
        ]
        return agent

    def test_digest_translates_and_keeps_original_and_rows(self):
        agent = self._agent()
        res = agent.brief()
        self.assertTrue(res["ok"])
        digest = res["digest"]
        self.assertIn(EN_HEADLINE, digest)
        self.assertIn(f"original: {FR_HEADLINE}", digest)
        # the English story is untouched — no original line for it
        self.assertIn("The markets rose on Tuesday", digest)
        self.assertNotIn("original: The markets rose", digest)
        # ledger rows are never mutated by delivery composition
        self.assertEqual(agent._pending[0]["title"], FR_HEADLINE)

    def test_story_block_keeps_original_line_and_link(self):
        from nomorals.agents.morning_pulse import _story_block
        from nomorals.llm.translate import localize_news_rows

        rows = localize_news_rows(
            SimpleNamespace(settings=None, db=None, router=None,
                            _translator_for_ctx=_translator()),
            [_french_row()])
        block = _story_block(rows[0])
        self.assertIn(EN_HEADLINE, block)
        self.assertIn(f"original: {FR_HEADLINE}", block)
        self.assertIn("https://lemonde.example/a", block)


class CommandTests(unittest.TestCase):
    """The /translate command through the real PartnerRuntime."""

    @classmethod
    def setUpClass(cls):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.llm.translate import BrainRung
        from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class _FakeAdapter(ChatAdapter):
            def __init__(self, name: str) -> None:
                super().__init__(media_dir="/tmp/nm-test-media")
                self.name = name

            def run(self, handler) -> None:
                pass

            def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
                     buttons=None, **kwargs) -> SendResult:
                return SendResult(ok=True, platform=self.name,
                                  message_id="m1")

        cls._tmp = tempfile.TemporaryDirectory(prefix="nm-translate-")
        settings = load_settings(overrides={
            "home": cls._tmp.name,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        cls.context = build_context(settings, with_executor=False,
                                    with_tools=False)
        gateway = ChatGateway({"telegram": _FakeAdapter("telegram")},
                              db=cls.context.db)
        cls.runtime = PartnerRuntime(cls.context, gateway=gateway)
        # the ladder the runtime sees: fake provider transport, the
        # brain rung neutralized (no live router in a test context),
        # no local model.  Bound to the real context so the owner
        # language persists in the real db.
        seeded = Translator(cls.context, client=FakeHttp())
        seeded.brain = BrainRung(None)
        cls.context._translator_for_ctx = seeded

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_translate_text(self):
        reply = self.runtime.handle_control(
            "/translate Bonjour le monde, ceci est un test",
            "telegram:1", message=None)
        self.assertIn("Hello world, this is a test", reply)
        self.assertIn("translated from French", reply)

    def test_translate_alias_and_target(self):
        reply = self.runtime.handle_control(
            "/tl es The morning is good and the sun is over the hills",
            "telegram:1", message=None)
        self.assertIn("La mañana es buena", reply)
        self.assertIn("to Spanish", reply)

    def test_translate_reply(self):
        from nomorals.social.chat.base import (
            ChatKind, ChatMessage, ChatRef)

        msg = ChatMessage(
            chat=ChatRef(platform="telegram", chat_id="1",
                         kind=ChatKind.DM),
            incoming=True, text="/translate",
            meta={"reply_to_text": "Bonjour le monde"})
        reply = self.runtime.handle_control(
            "/translate", "telegram:1", message=msg)
        self.assertIn("Hello world", reply)

    def test_translate_lang_roundtrip(self):
        reply = self.runtime.handle_control(
            "/translate lang fr", "telegram:1", message=None)
        self.assertIn("French", reply)
        fresh = Translator(self.context, client=FakeHttp())
        self.assertEqual(fresh.owner_language(), "fr")
        # restore the shared context's default for other tests
        fresh.set_owner_language("en")

    def test_translate_status_is_honest(self):
        reply = self.runtime.handle_control(
            "/translate status", "telegram:1", message=None)
        self.assertIn("translation ladder", reply)
        self.assertIn("identity", reply)

    def test_translate_nothing(self):
        reply = self.runtime.handle_control(
            "/translate", "telegram:1", message=None)
        self.assertIn("nothing to translate", reply)


# ── Yoruba fixtures (docs 19+21, N1) ─────────────────────────────────
# Real Yoruba — the owner's flagship pair — with hand-checked English.
# The rungs are faked like everywhere else in this file; the language
# data is not.  Tone marks are written the way everyday Yoruba text
# writes them (underdots kept, tones light), which is also the shape
# the detector's yo profile is built from.

YO_GREETING = "Kaasan o, ṣe alaafia ni?"
EN_GREETING = "Good afternoon, are you well?"
YO_MARKET = ("Ọjà Ìláwé ti kún fun ẹ̀wà ati ìrẹsì lónìí, "
             "awọn ènìyàn si n ra wọn ni owó rere.")
EN_MARKET = ("Ilawe market is full of beans and rice today, "
             "and people are buying at good prices.")
EN_FIELD = ("The farmers brought yams and peppers to the market "
            "this morning")
YO_FIELD = ("Awọn agbẹ ni o mu iṣu ati ata wa si ọja "
            "ni owurọ yii.")

_PAIRS.update({
    ("yo", "en", YO_MARKET): EN_MARKET,
    ("yo", "en", YO_GREETING): EN_GREETING,
    ("en", "yo", EN_FIELD): YO_FIELD,
})


class _YoLocalBackend(FakeLocalBackend):
    """Local-rung fake keyed by (text, destination NLLB code)."""

    def translate(self, text: str, src_nllb: str, dst_nllb: str) -> str:
        self.seen.append((text, src_nllb, dst_nllb))
        out = {(YO_GREETING, "eng_Latn"): EN_GREETING}.get(
            (text, dst_nllb))
        if out is None:
            raise RuntimeError("fake backend: unknown text")
        return out


class YorubaTests(unittest.TestCase):
    def test_detection(self):
        code, conf = detect_language(YO_MARKET)
        self.assertEqual(code, "yo")
        self.assertGreater(conf, 0.5)
        code, _ = detect_language(YO_FIELD)
        self.assertEqual(code, "yo")

    def test_normalize_and_nllb_mapping(self):
        from nomorals.llm.languages import NLLB_CODES

        self.assertEqual(normalize_code("Yoruba"), "yo")
        self.assertEqual(NLLB_CODES["yo"], "yor_Latn")
        self.assertEqual(NLLB_CODES["en"], "eng_Latn")

    def test_local_rung_carries_yor_latn(self):
        backend = _YoLocalBackend()
        t = _translator(local_backend=backend)
        out = t.translate(YO_GREETING, source="yo", target="en")
        self.assertTrue(out.translated)
        self.assertEqual(out.rung, "local")
        self.assertEqual(out.text, EN_GREETING)
        self.assertEqual(backend.seen[0][1:], ("yor_Latn", "eng_Latn"))

    def test_provider_rung_yo_to_en(self):
        t = _translator()  # context None -> providers answer
        out = t.translate(YO_MARKET, target="en")
        self.assertTrue(out.translated)
        self.assertTrue(out.rung.startswith("provider:"))
        self.assertEqual(out.source_lang, "yo")
        self.assertEqual(out.text, EN_MARKET)

    def test_provider_rung_en_to_yo(self):
        t = _translator()
        out = t.translate(EN_FIELD, target="yo")
        self.assertTrue(out.translated)
        self.assertEqual(out.target_lang, "yo")
        self.assertEqual(out.text, YO_FIELD)

    def test_ticket_path_yoruba_for_news(self):
        from nomorals.llm.ticket import TranslationTicket, run_ticket

        t = _translator()
        outcome = run_ticket(t, TranslationTicket(
            text=YO_MARKET, target_lang="en", exactness="gist",
            requester="news"))
        self.assertEqual(outcome.outcome.text, EN_MARKET)
        self.assertTrue(outcome.outcome.translated)
        self.assertEqual(outcome.requester, "news")
        self.assertTrue(outcome.receipt_id.startswith("tr-"))


if __name__ == "__main__":
    unittest.main()
