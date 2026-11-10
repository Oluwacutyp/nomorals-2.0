"""Docs 19+21 pass T2 — section-wise translation, the glossary
store, the durable cache, and the translate tool.

Pins: a document's skeleton (fence contents, heading levels, list
shape, URLs, table shape) is byte-identical after translation with
fake rungs, and the section map makes every section's rung/gate/
receipt legible; glossary terms are protect-restored and verified —
a mangled placeholder is disclosed by name, never left in the
output, never silently dropped; the glossary fingerprint rides the
cache key (same text, different glossary → miss); the durable cache
survives a restart (new Translator, same db → durable-hit, zero
rung execution) with TTL and local-model invalidation; the tool is
registered and returns the ticket shape end to end.
"""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.llm.cache import TranslationCacheStore
from nomorals.llm.glossary import GlossaryStore
from nomorals.llm.receipts import TranslationReceiptStore
from nomorals.llm.sections import parse_document
from nomorals.llm.ticket import TranslationTicket, run_ticket
from nomorals.llm.translate import Translator
from nomorals.storage.db import Database

# -- fake rungs (the T1 fakes' shapes) --------------------------------

_FR_EN = {
    "bonjour": "hello", "monde": "world", "le": "the", "la": "the",
    "chat": "cat",
    "mange": "eats", "poisson": "fish", "sur": "on", "table": "table",
    "guide": "guide", "rapide": "quick", "bienvenue": "welcome",
    "chez": "at", "nous": "us", "voir": "see", "ici": "here",
    "premier": "first", "point": "point", "ville": "city",
    "nom": "name", "chien": "dog", "une": "a", "citation": "quote",
    "importante": "important", "dernier": "last",
    "paragraphe": "paragraph", "avec": "with", "dedans": "inside",
    "deuxième": "second", "note": "note", "banque": "bank",
    "taux": "rates", "montent": "rise",
}
_EN_FR = {v: k for k, v in _FR_EN.items()}


def _word_map(chunk: str, table: dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        word = match.group(0)
        out = table.get(word.lower())
        if out is None:
            return word
        if word.isupper():
            return out.upper()
        if word[:1].isupper():
            return out.capitalize()
        return out

    return re.sub(r"\w+", repl, chunk, flags=re.UNICODE)


class WordMapBrain:
    """Brain fake translating word by word from a fixed table;
    unknown words and ⟦tokens⟧ pass through unchanged."""

    def __init__(self, *, mangle_tokens: bool = False):
        self.calls: list[str] = []
        self.mangle_tokens = mangle_tokens

    def chat(self, messages, task_kind: str = ""):
        prompt = messages[0].content
        self.calls.append(task_kind)
        match = re.search(r"to (\w+)\.", prompt)
        target = match.group(1).lower() if match else ""
        chunk = prompt.split("\n\n", 1)[1]
        table = _FR_EN if target == "english" else _EN_FR
        out = _word_map(chunk, table)
        if self.mangle_tokens:
            out = re.sub(r"⟦[^⟧]*⟧", "", out)
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
    """Dead provider transport — the brain rung must do the work."""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url: str, **kw):
        self.calls.append(url)
        return FakeResp(status=404, payload={})

    def post_json(self, url: str, payload, **kw):
        self.calls.append(url)
        return FakeResp(status=403, payload={})


FR_GOOD = "le chat mange le poisson sur la table"
EN_GOOD = "the cat eats the fish on the table"

DOC = """# Guide Rapide {#start}

Bienvenue chez nous. Voir [le guide](https://example.com/guide) ici.

- Premier point
- Deuxième point avec https://example.com/brut dedans

```python
# commentaire
x = "ne pas traduire"
```

| Nom | Ville |
| --- | --- |
| Chat | Paris |
| Chien | Lyon |

> Une citation importante.

Dernier paragraphe.
"""

FENCE = '```python\n# commentaire\nx = "ne pas traduire"\n```'


class _TempHome(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            dir="/home/hatch/workspace/d1921t2-verify/tmp")
        self.addCleanup(self._tmp.cleanup)
        self.home = self._tmp.name
        self.store = TranslationReceiptStore(db=None, home=self.home)

    def translator(self, **kw) -> Translator:
        kw.setdefault("client", FakeHttp())
        kw.setdefault("receipt_store", self.store)
        kw.setdefault("brain", WordMapBrain())
        return Translator(None, **kw)

    def db(self, name: str = "t2.db") -> Database:
        return Database(Path(self.home) / name)


# -- structure ---------------------------------------------------------

class SectionStructureTests(_TempHome):
    def test_parse_reassemble_is_byte_identical(self):
        doc = parse_document(DOC)
        self.assertIsNotNone(doc)
        self.assertEqual(doc.reassemble({}), DOC)
        html = parse_document(
            "<h1>Titre</h1><p>Bonjour <b>monde</b></p>"
            "<pre>x = 1</pre>")
        self.assertIsNotNone(html)
        self.assertEqual(
            html.reassemble({}),
            "<h1>Titre</h1><p>Bonjour <b>monde</b></p><pre>x = 1</pre>")

    def test_plain_prose_is_not_a_document(self):
        self.assertIsNone(parse_document("Bonjour le monde ici."))
        self.assertIsNone(parse_document(""))

    def test_document_round_trip_skeleton(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text=DOC, source_lang="fr", target_lang="en",
            requester="test"))
        self.assertEqual(res.outcome.rung, "document")
        self.assertTrue(res.translated)
        out = res.text
        # code fence byte-identical, never translated
        self.assertIn(FENCE, out)
        # heading stays a heading, anchor preserved, text translated
        self.assertIn("# Guide Quick {#start}", out)
        # link destination untouched, link text translated
        self.assertIn("[the guide](https://example.com/guide)", out)
        # bare URL untouched inside its list item
        self.assertIn("- Second point with https://example.com/brut "
                      "inside", out)
        self.assertIn("- First point", out)
        # table shape + separator row intact, cells translated
        self.assertIn("| --- | --- |", out)
        self.assertIn("| Name | City |", out)
        self.assertIn("| Cat | Paris |", out)
        self.assertIn("| Dog | Lyon |", out)
        # quote marker preserved
        self.assertIn("> A quote important.", out)
        # the section map is legible: kinds + per-section proof
        kinds = [s["kind"] for s in res.sections]
        self.assertEqual(kinds[0], "heading")
        self.assertIn("code", kinds)
        self.assertIn("table", kinds)
        self.assertIn("quote", kinds)
        self.assertEqual(kinds.count("list-item"), 2)
        code = next(s for s in res.sections if s["kind"] == "code")
        self.assertEqual(code["rung"], "preserved")
        self.assertFalse(code["translated"])
        table = next(s for s in res.sections if s["kind"] == "table")
        self.assertTrue(table["translated"])
        self.assertEqual(len(table["cells"]), 6)
        prose_sections = [s for s in res.sections
                          if s["kind"] != "code"]
        self.assertTrue(all(s["receipt_id"].startswith("tr-")
                            for s in prose_sections))
        self.assertEqual(res.receipt_id, "")  # composite by design
        self.assertEqual(len(res.receipt_ids),
                         sum(len(s["receipt_ids"])
                             for s in res.sections))
        self.assertTrue(any("one receipt per section" in n
                            for n in res.notes))

    def test_mangled_structure_placeholder_keeps_original(self):
        t = self.translator(brain=WordMapBrain(mangle_tokens=True))
        text = ("# Guide Rapide\n\nBienvenue chez nous. "
                "Voir [le guide](https://example.com/guide) ici.\n")
        res = run_ticket(t, TranslationTicket(
            text=text, source_lang="fr", target_lang="en"))
        # the heading (no placeholders) still translated…
        self.assertIn("# Guide Quick", res.text)
        # …the linked paragraph kept its original text, undisguised
        self.assertIn("Bienvenue chez nous.", res.text)
        self.assertIn("https://example.com/guide", res.text)
        self.assertNotIn("⟦", res.text)
        para = next(s for s in res.sections
                    if s["kind"] == "paragraph")
        self.assertFalse(para["translated"])
        self.assertIn("placeholders did not survive", para["note"])

    def test_html_document_cells_and_pre(self):
        t = self.translator()
        html = ("<p>Bonjour monde</p>"
                "<table><tr><td>Nom</td><td>Ville</td></tr></table>"
                "<pre>le chat</pre>")
        res = run_ticket(t, TranslationTicket(
            text=html, source_lang="fr", target_lang="en"))
        self.assertEqual(res.outcome.rung, "document")
        self.assertIn("<p>Hello world</p>", res.text)
        self.assertIn("<td>Name</td><td>City</td>", res.text)
        self.assertIn("<pre>le chat</pre>", res.text)  # verbatim

    def test_plain_text_stays_single_shot(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en"))
        self.assertEqual(res.text, EN_GOOD)
        self.assertEqual(res.outcome.rung, "brain")
        self.assertEqual(res.sections, [])
        self.assertTrue(res.receipt_id.startswith("tr-"))
        self.assertEqual(res.receipt_ids, [res.receipt_id])

    def test_preserve_tables_passes_table_through(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text=DOC, source_lang="fr", target_lang="en",
            preserve=("tables",)))
        self.assertIn("| Chat | Paris |", res.text)  # untranslated
        self.assertIn("# Guide Quick {#start}", res.text)
        table = next(s for s in res.sections if s["kind"] == "table")
        self.assertEqual(table["rung"], "preserved")
        self.assertFalse(table["translated"])

    def test_preserve_citations_on_plain_text(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text="la banque monte [12] ici", source_lang="fr",
            target_lang="en", preserve=("citations",)))
        self.assertIn("[12]", res.text)
        self.assertNotIn("⟦", res.text)
        self.assertTrue(res.translated)

    def test_unknown_preserve_rule_is_carried_honestly(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            preserve=("footnotes-x",)))
        self.assertEqual(res.text, EN_GOOD)
        self.assertTrue(any("no mechanism applies it" in n
                            for n in res.notes))

    def test_fidelity_document_labels_exact_when_sections_pass(self):
        t = self.translator()
        text = "# Guide Rapide\n\n" + FR_GOOD + "\n"
        res = run_ticket(t, TranslationTicket(
            text=text, source_lang="fr", target_lang="en",
            exactness="fidelity"))
        self.assertTrue(res.translated)
        self.assertEqual(res.gate["status"], "passed")
        self.assertTrue(res.exact_label)
        self.assertIn("# Guide Quick", res.text)
        self.assertIn(EN_GOOD, res.text)


# -- glossary ----------------------------------------------------------

class GlossaryStoreTests(_TempHome):
    def test_db_store_pair_filtering_and_scope_precedence(self):
        store = GlossaryStore(self.db())
        self.assertTrue(store.put_term(
            "g1", "chat", "cat", source_lang="fr", target_lang="en"))
        self.assertTrue(store.put_term("g1", "banque", "bank"))
        self.assertTrue(store.put_term(
            "g1", "banque", "BANQUE", scope="owner"))
        terms = store.terms_for("g1", "fr", "en")
        by_source = {t.source_term: t.target_term for t in terms}
        self.assertEqual(by_source["chat"], "cat")
        # owner scope beats global for the same source term
        self.assertEqual(by_source["banque"], "BANQUE")
        # pair filtering: the fr→en-only term stays out of es→en
        es_terms = store.terms_for("g1", "es", "en")
        self.assertNotIn("chat",
                         {t.source_term for t in es_terms})
        self.assertEqual(store.terms_for("nope", "fr", "en"), [])

    def test_file_fallback_store_is_durable(self):
        store = GlossaryStore(None, self.home)
        self.assertTrue(store.put_term(
            "g1", "chat", "KITTY", source_lang="fr",
            target_lang="en"))
        fresh = GlossaryStore(None, self.home)
        terms = fresh.terms_for("g1", "fr", "en")
        self.assertEqual([(t.source_term, t.target_term)
                          for t in terms], [("chat", "KITTY")])
        self.assertTrue(fresh.remove_term(
            "g1", "chat", source_lang="fr", target_lang="en"))
        self.assertEqual(fresh.terms_for("g1", "fr", "en"), [])

    def test_fingerprint_changes_with_terms(self):
        store = GlossaryStore(self.db())
        before = store.fingerprint("g1", "fr", "en")
        store.put_term("g1", "chat", "cat", source_lang="fr",
                       target_lang="en")
        after = store.fingerprint("g1", "fr", "en")
        self.assertNotEqual(before, after)
        store.put_term("g1", "chat", "KITTY", source_lang="fr",
                       target_lang="en")
        self.assertNotEqual(after,
                            store.fingerprint("g1", "fr", "en"))


class GlossaryApplicationTests(_TempHome):
    def _translator_with_glossary(self, **brain_kw):
        db = self.db()
        store = GlossaryStore(db)
        store.put_term("pets", "chat", "KITTY", source_lang="fr",
                       target_lang="en")
        t = Translator(None, db=db, client=FakeHttp(),
                       brain=WordMapBrain(**brain_kw),
                       receipt_store=TranslationReceiptStore(db=db))
        return t

    def test_glossary_term_is_enforced(self):
        t = self._translator_with_glossary()
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            glossary_id="pets"))
        self.assertEqual(
            res.text, "the KITTY eats the fish on the table")
        self.assertTrue(any("enforced 1 term(s)" in n
                            and "chat" in n for n in res.notes))
        self.assertFalse(any("carried on the ticket" in n
                             for n in res.notes))

    def test_mangled_placeholder_disclosed_and_clean(self):
        t = self._translator_with_glossary(mangle_tokens=True)
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            exactness="fidelity", glossary_id="pets"))
        # fell back to the unprotected translation — the rung's own
        # word for the term, no placeholder residue anywhere
        self.assertEqual(res.text, EN_GOOD)
        self.assertNotIn("⟦", res.text)
        self.assertTrue(any("NOT enforced" in n and "'chat'" in n
                            for n in res.notes))
        # …and the exact label is refused even though the text is
        # the gate-passing fallback
        self.assertFalse(res.exact_label)

    def test_glossary_named_but_empty_is_honest(self):
        t = self.translator()
        res = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            glossary_id="ghost"))
        self.assertEqual(res.text, EN_GOOD)
        self.assertTrue(any("no terms stored" in n
                            and "nothing to apply" in n
                            for n in res.notes))
        self.assertFalse(any("carried on the ticket" in n
                             for n in res.notes))

    def test_fingerprint_separates_cache_entries(self):
        db = self.db()
        store = GlossaryStore(db)
        store.put_term("a", "chat", "KITTY", source_lang="fr",
                       target_lang="en")
        store.put_term("b", "chat", "FELINE", source_lang="fr",
                       target_lang="en")
        brain = WordMapBrain()
        t = Translator(None, db=db, client=FakeHttp(), brain=brain,
                       receipt_store=TranslationReceiptStore(db=db))
        first = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            glossary_id="a"))
        self.assertIn("KITTY", first.text)
        second = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            glossary_id="b"))
        # a different glossary must MISS the first entry's cache
        self.assertIn("FELINE", second.text)
        self.assertNotIn("KITTY", second.text)
        calls = len(brain.calls)
        third = run_ticket(t, TranslationTicket(
            text=FR_GOOD, source_lang="fr", target_lang="en",
            glossary_id="a"))
        self.assertIn("KITTY", third.text)
        self.assertEqual(len(brain.calls), calls)  # served from cache


# -- durable cache -----------------------------------------------------

class DurableCacheTests(_TempHome):
    def _db_translator(self, db, brain):
        return Translator(None, db=db, client=FakeHttp(), brain=brain,
                          receipt_store=TranslationReceiptStore(db=db))

    def test_durable_hit_across_instances(self):
        db = self.db()
        brain_a = WordMapBrain()
        first_t = self._db_translator(db, brain_a)
        out = first_t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.text, EN_GOOD)
        self.assertEqual(first_t.last_receipt["cache"], "miss")
        self.assertGreater(len(brain_a.calls), 0)

        brain_b = WordMapBrain()
        second_t = self._db_translator(db, brain_b)
        out2 = second_t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out2.text, EN_GOOD)
        self.assertEqual(brain_b.calls, [])  # zero rung execution
        self.assertEqual(second_t.last_receipt["cache"],
                         "durable-hit")
        # …and the promoted entry is a plain hot hit afterwards
        second_t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(second_t.last_receipt["cache"], "hit")
        self.assertEqual(brain_b.calls, [])

    def test_expired_entry_reads_as_miss(self):
        db = self.db()
        store = TranslationCacheStore(db)
        import time as _time

        from nomorals.llm.cache import cache_key_for
        store.put(cache_key_for(FR_GOOD, "fr", "en"), source_lang="fr",
                  target_lang="en", rung="brain",
                  payload={"outcome": {
                      "text": EN_GOOD, "source_lang": "fr",
                      "target_lang": "en", "rung": "brain",
                      "translated": True, "note": ""},
                      "gate": {"status": "not_run"},
                      "model_ref": "",
                      "stored_at": _time.time() - 40 * 24 * 3600})
        brain = WordMapBrain()
        t = self._db_translator(db, brain)
        out = t.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.text, EN_GOOD)
        self.assertGreater(len(brain.calls), 0)  # the ladder ran
        self.assertEqual(t.last_receipt["cache"], "miss")

    def test_local_model_change_invalidates_entry(self):
        db = self.db()
        store = TranslationCacheStore(db)
        from nomorals.llm.cache import cache_key_for

        def plant(model_ref):
            store.put(cache_key_for(FR_GOOD, "fr", "en"),
                      source_lang="fr", target_lang="en", rung="local",
                      payload={"outcome": {
                          "text": EN_GOOD, "source_lang": "fr",
                          "target_lang": "en", "rung": "local",
                          "translated": True, "note": ""},
                          "gate": {"status": "not_run"},
                          "model_ref": model_ref})

        plant("some-other-model")
        brain = WordMapBrain()
        t = self._db_translator(db, brain)  # configured ref is ""
        t.translate(FR_GOOD, source="fr", target="en")
        self.assertGreater(len(brain.calls), 0)  # invalidated: miss
        self.assertEqual(t.last_receipt["cache"], "miss")

        plant("")  # matching model reference
        brain2 = WordMapBrain()
        t2 = self._db_translator(db, brain2)
        out = t2.translate(FR_GOOD, source="fr", target="en")
        self.assertEqual(out.rung, "local")
        self.assertEqual(brain2.calls, [])
        self.assertEqual(t2.last_receipt["cache"], "durable-hit")


# -- the tool ----------------------------------------------------------

class TranslateToolTests(_TempHome):
    def _registry(self):
        db = self.db()
        translator = Translator(
            None, db=db, client=FakeHttp(), brain=WordMapBrain(),
            receipt_store=TranslationReceiptStore(db=db))
        context = SimpleNamespace(
            settings=SimpleNamespace(home=self.home),
            db=db, _translator_for_ctx=translator)
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(context)
        from nomorals.tools.translate import register
        register(registry)
        return registry

    def test_tool_registered_in_builtins(self):
        from nomorals.tools.registry import ToolRegistry
        registry = ToolRegistry(SimpleNamespace(
            settings=SimpleNamespace(home=self.home)))
        registry.register_builtins()
        self.assertIn("translate", registry.names())
        spec = registry.get("translate")
        self.assertEqual(spec.capability, "model.call")

    def test_tool_returns_the_ticket_shape(self):
        registry = self._registry()
        outcome = registry.call(
            "translate", text=FR_GOOD, source="fr", target="en",
            exactness="fidelity", requester="test")
        self.assertTrue(outcome.ok, getattr(outcome, "error", None))
        value = outcome.value
        self.assertEqual(value["text"], EN_GOOD)
        self.assertEqual(value["rung"], "brain")
        self.assertTrue(value["receipt_id"].startswith("tr-"))
        self.assertEqual(value["gate"]["status"], "passed")
        self.assertTrue(value["exact_label"])
        self.assertEqual(value["requester"], "test")
        self.assertEqual(value["sections"], [])

    def test_tool_document_ticket_carries_section_map(self):
        registry = self._registry()
        outcome = registry.call(
            "translate", text=DOC, source="fr", target="en")
        self.assertTrue(outcome.ok, getattr(outcome, "error", None))
        value = outcome.value
        self.assertEqual(value["rung"], "document")
        self.assertIn(FENCE, value["text"])
        self.assertTrue(value["sections"])
        self.assertTrue(value["receipt_ids"])

    def test_tool_rejects_empty_text_and_bad_exactness(self):
        registry = self._registry()
        empty = registry.call("translate", text="  ", target="en")
        self.assertFalse(empty.ok)
        bad = registry.call("translate", text=FR_GOOD,
                            exactness="perfekt")
        self.assertFalse(bad.ok)


if __name__ == "__main__":
    unittest.main()
