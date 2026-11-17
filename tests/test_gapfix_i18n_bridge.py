"""Gapfix R3-65 — the i18n bridge (§65): score through the bridge, keep the pack seam.

Verdict (see ~/workspace/gapfix65-verify/GAPFIX65_SURVEY.md):

* WIRE ``score_contribution`` into the translation quality gate
  (``llm/translate.py::_gate_verdict``). The gate's own docstring names
  the bridge as its shipped consumer; the gate scored inline through
  ``translate_with_quality`` instead. The bridge is generalized with
  ``source_lang`` / ``target_lang`` so the gate's ladder receives
  byte-identical call triples.
* KEEP ``pack_locales`` / ``pack_primary_locale`` as the documented
  family-pack seam, pinned: no production seam exists where pack
  resolution changes an outcome (English is first in every pack, the
  ``en`` catalog is complete, the family surface renders no catalog
  keys, and the family check verb's ``"ng"`` is a pack code used
  correctly as one — resolving it to a locale would silently drop the
  Nigerian kid-safe terms).
"""

from __future__ import annotations

from nomorals.family.locales import blocked_terms_for
from nomorals.i18n import bridge
from nomorals.llm.translate import TranslationOutcome, Translator


class _FakeHttp:
    """Fake transport so Translator construction never touches the net."""
    def get(self, url, **kw):
        raise RuntimeError("no network in tests")

    def post_json(self, url, payload, **kw):
        raise RuntimeError("no network in tests")


def _translator() -> Translator:
    return Translator(None, client=_FakeHttp())


def _lexicon_translate(lexicon):
    def _fn(text: str, src: str, dst: str) -> str:
        return lexicon.get((src, dst, text), text)
    return _fn


# ── WIRE: the gate scores through the bridge ──────────────────────────

def test_gate_scores_through_bridge(monkeypatch):
    """The quality gate's scoring call goes through score_contribution.

    Failing-first: on the unmodified tree the gate calls
    translate_with_quality inline and never touches the bridge.
    """
    calls: list[dict] = []
    real = bridge.score_contribution

    def _spy(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return real(*args, **kwargs)

    monkeypatch.setattr("nomorals.llm.translate.score_contribution",
                        _spy, raising=False)
    tr = _translator()
    outcome = TranslationOutcome(
        text="Hello world", source_lang="fr", target_lang="en",
        rung="local", translated=True)
    gate = tr._gate_verdict(
        "Bonjour le monde", outcome, exactness="fidelity",
        reverse_fn=lambda text, s, d: "Bonjour le monde")
    assert gate["status"] == "passed"
    assert gate["f1"] == 1.0
    assert len(calls) == 1
    assert calls[0]["kwargs"]["source_lang"] == "fr"
    assert calls[0]["kwargs"]["target_lang"] == "en"


def test_gate_ladder_triples_unchanged():
    """The back leg still walks the ladder in reverse, same triple.

    The gate's reverse leg is the only new-work call the scoring makes;
    it must receive (draft, dst, src) exactly as before the wire.
    """
    seen: list[tuple[str, str, str]] = []

    def _reverse(text: str, s: str, d: str) -> str:
        seen.append((text, s, d))
        return "Bonjour le monde"

    tr = _translator()
    outcome = TranslationOutcome(
        text="Hello world", source_lang="fr", target_lang="en",
        rung="local", translated=True)
    gate = tr._gate_verdict(
        "Bonjour le monde", outcome, exactness="fidelity",
        reverse_fn=_reverse)
    assert gate["status"] == "passed"
    assert gate["f1"] == 1.0
    assert seen == [("Hello world", "en", "fr")]


def test_gate_non_english_source_still_scores():
    """The wire must not narrow the gate to English-source only.

    The bridge's forward-leg adapter used to hardcode "en"; the gate
    handles any source language, and a German→French round trip must
    still score (and pass on a perfect round trip).
    """
    tr = _translator()
    outcome = TranslationOutcome(
        text="Bonjour le monde", source_lang="de", target_lang="fr",
        rung="local", translated=True)
    gate = tr._gate_verdict(
        "Guten Morgen Welt", outcome, exactness="fidelity",
        reverse_fn=lambda text, s, d: "Guten Morgen Welt")
    assert gate["status"] == "passed"
    assert gate["f1"] == 1.0


def test_score_contribution_source_lang_param():
    """score_contribution accepts a non-English source language.

    Failing-first: on the unmodified tree the parameter does not exist
    (TypeError) and the forward leg only returns the draft verbatim
    for an English source.
    """
    lexicon = {
        ("de", "fr", "Guten Morgen Welt"): "Bonjour le monde",
        ("fr", "de", "Bonjour le monde"): "Guten Morgen Welt",
    }
    result = bridge.score_contribution(
        "Guten Morgen Welt", "Bonjour le monde",
        target_locale="fr", source_lang="de",
        translate_fn=_lexicon_translate(lexicon))
    assert result.passed
    assert result.back_translation_f1 == 1.0


def test_score_contribution_default_signature_unchanged():
    """The catalog-row shape (English source, locale target) still works."""
    lexicon = {
        ("en", "fr", "Welcome"): "Bienvenue",
        ("fr", "en", "Bienvenue"): "Welcome",
    }
    good = bridge.score_contribution(
        "Welcome", "Bienvenue", target_locale="fr",
        translate_fn=_lexicon_translate(lexicon))
    assert good.passed
    assert good.back_translation_f1 == 1.0
    bad = bridge.score_contribution(
        "Welcome", "Zqxjkv nonsense", target_locale="fr",
        translate_fn=_lexicon_translate(lexicon))
    assert not bad.passed


def test_score_contribution_target_lang_verbatim():
    """A caller that already speaks MT codes gets them used verbatim.

    A regional target ("pt-BR") must reach the ladder as "pt-BR", not
    collapsed to "pt" by the bridge's locale resolution.
    """
    seen: list[tuple[str, str, str]] = []

    def _fn(text: str, src: str, dst: str) -> str:
        seen.append((text, src, dst))
        if (src, dst) == ("en", "pt-BR"):
            return "Olá mundo"
        if (src, dst) == ("pt-BR", "en"):
            return "Hello world"
        return text

    result = bridge.score_contribution(
        "Hello world", "Olá mundo",
        target_locale="pt-BR", target_lang="pt-BR", translate_fn=_fn)
    assert result.passed
    # The forward leg returns the draft verbatim (it never reaches the
    # ladder), so the proof is two-sided: the back leg ran with the
    # verbatim code, and no collapsed "pt" call was ever made.
    assert ("Olá mundo", "pt-BR", "en") in seen
    assert ("Hello world", "en", "pt") not in seen
    assert ("Olá mundo", "pt", "en") not in seen


# ── KEEP: the family-pack seam stays, pinned ─────────────────────────

def test_pack_seam_stays_importable():
    """pack_locales / pack_primary_locale remain the documented seam."""
    assert bridge.pack_locales("ng") == ["en", "pcm", "yo", "ha", "ig"]
    assert bridge.pack_primary_locale("ng") == "en"
    assert bridge.pack_locales("gb") == ["en"]
    assert bridge.pack_primary_locale("zz") == "en"


def test_pack_code_is_not_a_locale_for_kid_safe():
    """The family check verb must keep passing the PACK code, not a locale.

    Resolving "ng" through pack_primary_locale to "en" would silently
    drop the Nigerian extra terms from the kid-safe check — a
    regression, not a wire. Pin the organ fact the verdict rests on.
    """
    assert "ashawo" in blocked_terms_for("ng")
    assert "yahoo boy" in blocked_terms_for("ng")
    assert "ashawo" not in blocked_terms_for("en")
    assert "yahoo boy" not in blocked_terms_for("en")
