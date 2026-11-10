"""Docs 19+21 pass T1 — TranslationTicket facade, the wired quality
gate, and durable translation receipts.

Pins: every translation through the organ mints a receipt with the
rung's real name; the bidirectional gate (llm.translation_quality)
runs for fidelity-class requests and its verdict is a MARK — a
failed round trip never swaps or blocks the translation, it marks
it unverified and refuses the exact label (doc 19's rule, the
i18n bridge's shipped consumer shape); identity receipts read
identity + gate not_run, never a fake pass; cache hits are recorded
as hits with no rung re-execution; a broken receipt store never
breaks a translation.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

from nomorals.llm.receipts import TranslationReceiptStore
from nomorals.llm.ticket import (
    TicketOutcome,
    TranslationTicket,
    run_ticket,
)
from nomorals.llm.translate import Translator
from nomorals.storage.db import Database

FR_GOOD = "le chat mange le poisson sur la table"
EN_GOOD = "the cat eats the fish on the table"
FR_BAD = "la banque centrale releve ses taux"
EN_BAD = "the central bank raises its rates"
FR_GARBAGE_BACK = "quelque chose sans aucun rapport vraiment"
FR_HEADLINE = "les marches ouvrent en hausse ce matin"
EN_HEADLINE = "markets open higher this morning"

#: (chunk, target language name) -> translation for the fake brain.
_BRAIN_PAIRS: dict[tuple[str, str], str] = {
    (FR_GOOD, "english"): EN_GOOD,
    (EN_GOOD, "french"): FR_GOOD,
    (FR_BAD, "english"): EN_BAD,
    (EN_BAD, "french"): FR_GARBAGE_BACK,
    (FR_HEADLINE, "english"): EN_HEADLINE,
    ("qqq zzz www xxx", "english"): "some english words here",
}


class RoundTripBrain:
    """Brain fake answering both directions from a pair table."""

    def __init__(self, pairs=None):
        self.calls: list[str] = []
        self.pairs = dict(_BRAIN_PAIRS if pairs is None else pairs)

    def chat(self, messages, task_kind: str = ""):
        prompt = messages[0].content
        self.calls.append(task_kind)
        match = re.search(r"to (\w+)\.", prompt)
        target = match.group(1).lower() if match else ""
        chunk = prompt.split("\n\n", 1)[1]
        out = self.pairs.get((chunk, target))
        if out is None:
            raise RuntimeError("brain has no answer for this chunk")
        return SimpleNamespace(text=out)


class FakeResp:
    def __init__(self, payload=None, status: int = 200):
        self._payload = payload
        self.status = status

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self):
        return self._payload


class FakeHttp:
    """Minimal provider transport: MyMemory shape, or dead network."""

    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.calls: list[str] = []

    def get(self, url: str, **kw):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network down")
        if "api.mymemory.translated.net" in url:
            qs = parse_qs(urlparse(url).query)
            text = qs["q"][0]
            src, _, dst = qs["langpair"][0].partition("|")
            if (src, dst, text) == ("fr", "en", "Bonjour le monde"):
                return FakeResp(payload={
                    "responseData": {"translatedText": "Hello world"},
                    "responseStatus": 200})
            return FakeResp(status=429, payload={})
        if "/api/v1/" in url:  # Lingva shape
            parts = urlparse(url).path.split("/")
            text = unquote(parts[5])
            if text == "Bonjour le monde":
                return FakeResp(payload={"translation": "Hello world"})
            return FakeResp(status=404, payload={})
        return FakeResp(status=404, payload={})

    def post_json(self, url: str, payload, **kw):
        self.calls.append(url)
        if self.fail:
            raise ConnectionError("network down")
        return FakeResp(status=403, payload={})


class FakeLocalBackend:
    """Local-rung fake keyed by (text, destination NLLB code)."""

    def __init__(self, pairs):
        self.pairs = dict(pairs)
        self.seen: list[tuple[str, str]] = []

    def translate(self, text: str, src_nllb: str, dst_nllb: str) -> str:
        self.seen.append((text, dst_nllb))
        out = self.pairs.get((text, dst_nllb))
        if out is None:
            raise RuntimeError("fake backend: unknown text")
        return out


class _TempHome(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            dir="/home/hatch/workspace/d1921t1-verify/tmp")
        self.addCleanup(self._tmp.cleanup)
        self.home = self._tmp.name
        self.store = TranslationReceiptStore(db=None, home=self.home)

    def translator(self, **kw) -> Translator:
        kw.setdefault("client", FakeHttp())
        kw.setdefault("receipt_store", self.store)
        return Translator(None, **kw)


class ReceiptMintingTests(_TempHome):
    def test_brain_rung_receipt_with_real_name(self):
        t = self.translator(brain=RoundTripBrain())
        out = t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.rung, "brain")
        self.assertEqual(out.text, EN_GOOD)
        receipt = t.last_receipt
        self.assertIsNotNone(receipt)
        self.assertTrue(receipt["receipt_id"].startswith("tr-"))
        self.assertEqual(receipt["rung"], "brain")
        self.assertEqual(receipt["source_lang"], "fr")
        self.assertEqual(receipt["target_lang"], "en")
        self.assertEqual(receipt["chars"], len(FR_GOOD))
        self.assertEqual(receipt["cache"], "miss")
        self.assertEqual(receipt["requester"], "")  # typed absent
        self.assertEqual(receipt["exactness"], "gist")
        # gist: the gate honestly did not run
        self.assertEqual(receipt["gate"]["status"], "not_run")
        self.assertIn("gist", receipt["gate"]["reason"])
        self.assertEqual(receipt["quality_mark"], "not_checked")
        self.assertEqual(len(self.store.recent()), 1)

    def test_provider_rung_receipt(self):
        t = self.translator()  # no brain, no local: providers answer
        out = t.translate("Bonjour le monde", target="en")
        self.assertTrue(out.translated)
        self.assertEqual(out.text, "Hello world")
        self.assertTrue(out.rung.startswith("provider:"))
        receipt = self.store.recent()[0]
        self.assertEqual(receipt["rung"], out.rung)
        self.assertEqual(receipt["source_lang"], "fr")

    def test_identity_receipt_is_honest(self):
        t = self.translator(client=FakeHttp(fail=True))
        out = t.translate(FR_GOOD, target="en", exactness="fidelity")
        self.assertFalse(out.translated)
        self.assertEqual(out.rung, "identity")
        receipt = t.last_receipt
        self.assertEqual(receipt["quality_mark"], "identity")
        self.assertEqual(receipt["gate"]["status"], "not_run")
        self.assertIn("identity", receipt["gate"]["reason"])

    def test_same_language_receipt(self):
        t = self.translator(brain=RoundTripBrain())
        out = t.translate("The morning is good and the sun is over "
                          "the hills", target="en")
        self.assertEqual(out.rung, "same-language")
        receipt = t.last_receipt
        self.assertEqual(receipt["quality_mark"], "same-language")
        self.assertEqual(receipt["gate"]["status"], "not_run")

    def test_news_requester_attribution(self):
        t = self.translator(brain=RoundTripBrain())
        item = t.translate_item(FR_HEADLINE, "", target="en")
        self.assertTrue(item.translated)
        self.assertEqual(item.title, EN_HEADLINE)
        receipts = self.store.recent()
        self.assertTrue(receipts)
        self.assertTrue(
            all(r["requester"] == "news" for r in receipts))


class GateWiringTests(_TempHome):
    def test_fidelity_gate_pass_labels_exact(self):
        t = self.translator(brain=RoundTripBrain())
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity", requester="chat"))
        self.assertIsInstance(res, TicketOutcome)
        self.assertEqual(res.text, EN_GOOD)
        self.assertEqual(res.gate["status"], "passed")
        self.assertEqual(res.gate["f1"], 1.0)
        self.assertEqual(res.gate["quality"], 1.0)
        self.assertEqual(res.gate["threshold"], 0.5)
        self.assertTrue(res.exact_label)
        self.assertTrue(res.receipt_id.startswith("tr-"))
        receipt = self.store.recent()[0]
        self.assertEqual(receipt["receipt_id"], res.receipt_id)
        self.assertEqual(receipt["quality_mark"], "verified")
        self.assertEqual(receipt["requester"], "chat")

    def test_gate_failure_marks_but_never_swaps(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain)
        res = run_ticket(t, TranslationTicket(
            text=FR_BAD, source_lang="fr", target_lang="en",
            exactness="fidelity"))
        # The translation is the ladder's candidate, unchanged —
        # the gate is a mark, not a fallback trigger.
        self.assertEqual(res.text, EN_BAD)
        self.assertTrue(res.translated)
        self.assertEqual(res.gate["status"], "failed")
        self.assertEqual(res.gate["quality"], 0.0)
        self.assertFalse(res.exact_label)
        self.assertTrue(any("not labelled exact" in n
                            for n in res.notes))
        receipt = self.store.recent()[0]
        self.assertEqual(receipt["quality_mark"], "unverified")

    def test_gate_not_run_when_source_undetected(self):
        t = self.translator(brain=RoundTripBrain())
        res = run_ticket(t, TranslationTicket(
            text="qqq zzz www xxx", target_lang="en",
            exactness="legal"))
        self.assertTrue(res.translated)
        self.assertEqual(res.gate["status"], "not_run")
        self.assertIn("undetected", res.gate["reason"])
        self.assertFalse(res.exact_label)

    def test_unknown_exactness_class_raises(self):
        t = self.translator(brain=RoundTripBrain())
        with self.assertRaises(ValueError):
            run_ticket(t, TranslationTicket(
                text=FR_GOOD, exactness="perfekt"))
        # …while the organ itself degrades an unknown class to gist
        out = t.translate(FR_GOOD, source="fr", target="en",
                          exactness="perfekt")
        self.assertTrue(out.translated)
        self.assertEqual(t.last_receipt["exactness"], "gist")


class CacheReceiptTests(_TempHome):
    def test_cache_hit_recorded_without_rerun(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain)
        t.translate(FR_GOOD, source="fr", target="en")
        calls_after_first = len(brain.calls)
        out = t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.text, EN_GOOD)
        self.assertEqual(len(brain.calls), calls_after_first)
        receipts = self.store.recent()
        self.assertEqual([r["cache"] for r in receipts],
                         ["hit", "miss"])  # newest first

    def test_fidelity_verdict_survives_cache_hit(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain)
        first = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity"))
        self.assertEqual(first.gate["status"], "passed")
        calls_after_first = len(brain.calls)  # forward + round trip
        second = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity"))
        self.assertEqual(second.gate["status"], "passed")
        self.assertTrue(second.exact_label)
        self.assertEqual(len(brain.calls), calls_after_first)
        self.assertEqual(t.last_receipt["cache"], "hit")

    def test_fidelity_on_gist_warmed_cache_earns_verdict(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain)
        t.translate(FR_GOOD, source="fr", target="en")  # gist: no gate
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity"))
        self.assertEqual(res.gate["status"], "passed")
        self.assertTrue(res.exact_label)
        self.assertEqual(t.last_receipt["cache"], "hit")


class LocalOnlyTicketTests(_TempHome):
    def _backend(self):
        return FakeLocalBackend({
            (FR_GOOD, "eng_Latn"): EN_GOOD,
            (EN_GOOD, "fra_Latn"): FR_GOOD,
        })

    def test_local_only_ticket_uses_only_the_local_rung(self):
        brain = RoundTripBrain()
        backend = self._backend()
        t = self.translator(brain=brain, local_backend=backend)
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity", local_only=True))
        self.assertEqual(res.text, EN_GOOD)
        self.assertEqual(res.outcome.rung, "local")
        self.assertEqual(res.gate["status"], "passed")
        self.assertTrue(res.exact_label)
        self.assertEqual(brain.calls, [])  # the brain never consulted

    def test_local_only_declines_to_marked_identity(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain,
                            local_backend=FakeLocalBackend({}))
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            local_only=True))
        self.assertFalse(res.translated)
        self.assertEqual(res.outcome.rung, "identity")
        self.assertIn("local-only ticket", res.outcome.note)
        self.assertEqual(brain.calls, [])
        receipt = self.store.recent()[0]
        self.assertEqual(receipt["quality_mark"], "identity")


class ReceiptStoreTests(_TempHome):
    def test_read_api_filters_jsonl(self):
        brain = RoundTripBrain()
        t = self.translator(brain=brain)
        t.translate(FR_GOOD, source="fr", target="en")
        t.translate(FR_HEADLINE, source="fr", target="en")
        t.translate(FR_GOOD, source="fr", target="en")  # cache hit
        self.assertEqual(len(self.store.recent()), 3)
        self.assertEqual(len(self.store.recent(source_lang="fr")), 3)
        self.assertEqual(len(self.store.recent(source_lang="es")), 0)
        self.assertEqual(
            len(self.store.recent(source_lang="fr",
                                  target_lang="en")), 3)
        self.assertEqual(len(self.store.recent(rung="brain")), 3)
        self.assertEqual(len(self.store.recent(rung="local")), 0)

    def test_db_store_is_durable_across_instances(self):
        db_path = Path(self.home) / "receipts.db"
        db = Database(db_path)
        store = TranslationReceiptStore(db=db)
        t = self.translator(brain=RoundTripBrain(),
                            receipt_store=store)
        t.translate(FR_GOOD, source="fr", target="en")
        t.translate(FR_HEADLINE, source="fr", target="en")
        fresh = TranslationReceiptStore(db=Database(db_path))
        rows = fresh.recent()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["chars"], len(FR_HEADLINE))
        self.assertEqual(
            len(fresh.recent(source_lang="fr", rung="brain")), 2)

    def test_broken_store_never_breaks_translation(self):
        class BrokenStore:
            def append(self, receipt):
                raise RuntimeError("store is down")

            def recent(self, **kw):
                raise RuntimeError("store is down")

        t = self.translator(brain=RoundTripBrain(),
                            receipt_store=BrokenStore())
        out = t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.text, EN_GOOD)
        self.assertTrue(out.translated)
        # the in-memory receipt still exists for facades
        self.assertTrue(t.last_receipt["receipt_id"].startswith("tr-"))
        self.assertEqual(t.recent_receipts(), [])


if __name__ == "__main__":
    unittest.main()
