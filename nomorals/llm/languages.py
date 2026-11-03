"""Language table + the CODE detection rung for the translation organ.

Detection is laddered like everything else in the organ: a real
library (``langdetect`` / ``lingua``) wins when it is installed, and
the built-in heuristic rung below carries every environment without
one.  The heuristic rung is a genuine detector, not a gesture:

* **Script first** — Cyrillic / Arabic / Han / Hangul / Hebrew /
  Greek / Devanagari / Thai / Armenian / Georgian ranges decide the
  language family before any Latin statistics run (with the usual
  disambiguators: Ukrainian-specific Cyrillic letters, kana → Japanese
  over Chinese, Persian-specific Arabic letters).
* **Stopword profiles** for the Latin-script languages Devon actually
  meets in feeds and chats — the highest-frequency function words of
  each language, scored against the text, with diacritic bonuses.
  Confidence comes from the winner's margin over the runner-up, so a
  two-word headline can still resolve (French *les/de/la* is decisive)
  while genuinely ambiguous text honestly reports ``und``.

``und`` (undetermined) is a real answer: the translation ladder treats
it as "ask a smarter rung" (the brain / a provider's auto-detect),
never as a guess dressed up as a detection.
"""

from __future__ import annotations

import re
from typing import Any

__all__ = [
    "LANGUAGES",
    "NLLB_CODES",
    "normalize_code",
    "language_name",
    "detect_language",
    "detect_with_library",
    "detect_heuristics",
]

#: ISO 639-1 code -> (English name, NLLB-200 code).  The NLLB code is
#: "" for languages NLLB-200 does not carry — the local rung declines
#: those pairs honestly and the ladder moves on.
LANGUAGES: dict[str, tuple[str, str]] = {
    "en": ("English", "eng_Latn"),
    "fr": ("French", "fra_Latn"),
    "es": ("Spanish", "spa_Latn"),
    "de": ("German", "deu_Latn"),
    "it": ("Italian", "ita_Latn"),
    "pt": ("Portuguese", "por_Latn"),
    "nl": ("Dutch", "nld_Latn"),
    "sv": ("Swedish", "swe_Latn"),
    "da": ("Danish", "dan_Latn"),
    "nb": ("Norwegian Bokmål", "nob_Latn"),
    "fi": ("Finnish", "fin_Latn"),
    "pl": ("Polish", "pol_Latn"),
    "ro": ("Romanian", "ron_Latn"),
    "ca": ("Catalan", "cat_Latn"),
    "tr": ("Turkish", "tur_Latn"),
    "id": ("Indonesian", "ind_Latn"),
    "ms": ("Malay", "zsm_Latn"),
    "sw": ("Swahili", "swh_Latn"),
    "yo": ("Yoruba", "yor_Latn"),
    "ha": ("Hausa", "hau_Latn"),
    "ig": ("Igbo", "ibo_Latn"),
    "ru": ("Russian", "rus_Cyrl"),
    "uk": ("Ukrainian", "ukr_Cyrl"),
    "ar": ("Arabic", "arb_Arab"),
    "fa": ("Persian", "pes_Arab"),
    "he": ("Hebrew", "heb_Hebr"),
    "el": ("Greek", "ell_Grek"),
    "hi": ("Hindi", "hin_Deva"),
    "zh": ("Chinese", "zho_Hans"),
    "ja": ("Japanese", "jpn_Jpan"),
    "ko": ("Korean", "kor_Hang"),
    "th": ("Thai", "tha_Thai"),
    "vi": ("Vietnamese", "vie_Latn"),
    "ur": ("Urdu", "urd_Arab"),
    "bn": ("Bengali", "ben_Beng"),
    "ta": ("Tamil", "tam_Taml"),
    "te": ("Telugu", "tel_Telu"),
    "mr": ("Marathi", "mar_Deva"),
    "gu": ("Gujarati", "guj_Gujr"),
    "pa": ("Punjabi", "pan_Guru"),
    "am": ("Amharic", "amh_Ethi"),
    "ka": ("Georgian", "kat_Geor"),
    "hy": ("Armenian", "hye_Armn"),
    "cs": ("Czech", "ces_Latn"),
    "sk": ("Slovak", "slk_Latn"),
    "hu": ("Hungarian", "hun_Latn"),
    "bg": ("Bulgarian", "bul_Cyrl"),
    "sr": ("Serbian", "srp_Cyrl"),
    "hr": ("Croatian", "hrv_Latn"),
    "sl": ("Slovenian", "slv_Latn"),
    "et": ("Estonian", "est_Latn"),
    "lv": ("Latvian", "lvs_Latn"),
    "lt": ("Lithuanian", "lit_Latn"),
}

#: code -> NLLB-200 code ("" when the local model cannot carry it).
NLLB_CODES: dict[str, str] = {c: n for c, (_n, n) in LANGUAGES.items()}

_NAME_TO_CODE: dict[str, str] = {}
for _code, (_name, _nllb) in LANGUAGES.items():
    _NAME_TO_CODE[_name.lower()] = _code
    _NAME_TO_CODE[_code] = _code
# common aliases owners actually type
_NAME_TO_CODE.update({
    "bokmål": "nb", "bokmal": "nb", "norwegian": "nb",
    "mandarin": "zh", "chinese simplified": "zh",
    "farsi": "fa", "castilian": "es", "brasil": "pt",
    "portuguese (brazil)": "pt", "bahasa": "id",
    "bahasa indonesia": "id", "bahasa melayu": "ms",
})


def normalize_code(raw: str) -> str:
    """Free-form language reference -> ISO code ("" when unknown).

    Accepts codes (``fr``, ``pt-BR``), full names (``French``) and the
    common aliases above.  Regional variants collapse to their base
    language (``fr-CA`` -> ``fr``) — the rungs translate languages,
    not locales.
    """
    text = (raw or "").strip().lower().replace("_", "-")
    if not text:
        return ""
    if text in _NAME_TO_CODE:
        return _NAME_TO_CODE[text]
    base = text.split("-", 1)[0]
    if base in LANGUAGES:
        return base
    return _NAME_TO_CODE.get(base, "")


def language_name(code: str) -> str:
    """Display name for a code (the code itself when unknown)."""
    entry = LANGUAGES.get((code or "").strip().lower())
    return entry[0] if entry else (code or "")


# ── script detection ─────────────────────────────────────────────────

_SCRIPT_RANGES: list[tuple[str, int, int]] = [
    ("cyrillic", 0x0400, 0x04FF),
    ("arabic", 0x0600, 0x06FF),
    ("hebrew", 0x0590, 0x05FF),
    ("greek", 0x0370, 0x03FF),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0xAC00, 0xD7AF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("han", 0x4E00, 0x9FFF),
    ("armenian", 0x0530, 0x058F),
    ("georgian", 0x10A0, 0x10FF),
    ("ethiopic", 0x1200, 0x137F),
]

_UKRAINIAN_MARKS = set("іїєґІЇЄҐ")
_PERSIAN_MARKS = set("پچژگکی")


def _script_of(ch: str) -> str:
    cp = ord(ch)
    for name, lo, hi in _SCRIPT_RANGES:
        if lo <= cp <= hi:
            return name
    if ch.isalpha() and cp < 0x0250:
        return "latin"
    return ""


def _script_language(text: str) -> tuple[str, float] | None:
    """(code, confidence) when a non-Latin script dominates, else None."""
    counts: dict[str, int] = {}
    letters = 0
    for ch in text:
        if not ch.isalpha():
            continue
        letters += 1
        script = _script_of(ch)
        if script:
            counts[script] = counts.get(script, 0) + 1
    if not letters or not counts:
        return None
    top, n = max(counts.items(), key=lambda kv: kv[1])
    share = n / letters
    if top == "latin" or share < 0.5:
        return None
    if top == "cyrillic":
        code = "uk" if any(c in _UKRAINIAN_MARKS for c in text) else "ru"
        return code, min(0.99, 0.7 + share / 2)
    if top == "arabic":
        code = "fa" if any(c in _PERSIAN_MARKS for c in text) else "ar"
        return code, min(0.99, 0.7 + share / 2)
    if top in ("hiragana", "katakana"):
        return "ja", 0.99
    if top == "han":
        # Han alone: Chinese (Japanese without kana is rare in feeds).
        return "zh", min(0.95, 0.6 + share / 2)
    fixed = {
        "hangul": "ko", "hebrew": "he", "greek": "el",
        "devanagari": "hi", "thai": "th", "bengali": "bn",
        "tamil": "ta", "telugu": "te", "gujarati": "gu",
        "gurmukhi": "pa", "armenian": "hy", "georgian": "ka",
        "ethiopic": "am",
    }
    code = fixed.get(top)
    if code:
        return code, min(0.99, 0.7 + share / 2)
    return None


# ── Latin stopword profiles ──────────────────────────────────────────

#: Highest-frequency function words per Latin-script language.  Kept
#: deliberately small and distinctive: these are the words headlines
#: and short messages cannot avoid.
_STOPWORDS: dict[str, frozenset[str]] = {
    "en": frozenset("the of and a an in to is are was were for on with as by at from or not but be this that it its has have had will would can could said says new after over under more than into".split()),
    "fr": frozenset("le la les de des du un une et est sont dans pour qui au aux ce cette ces sur plus pas par il elle ils nous vous ont être avec après aussi comme chez entre vers lors même très".split()),
    "es": frozenset("el la los las de del un una unos unas y o en es son con por para que los del las como más pero sus le ya o este esta esto entre cuando muy sin sobre también me hasta donde".split()),
    "de": frozenset("der die das den dem des ein eine einen einem und oder in im am zu von mit für ist sind war waren nicht sich es er sie wir auf bei nach über unter mehr als auch noch wie".split()),
    "it": frozenset("il lo la i gli le di un una uno e è sono in con per che del della dei delle non si più anche come da a al alla nel nella sul sulla tra fra questo questa".split()),
    "pt": frozenset("o a os as de do da dos das um uma uns umas e é são em com por para que não se como mais mas ao aos à às pelo pela este esta isso entre sobre após".split()),
    "nl": frozenset("de het een van en in op te dat die zijn is was aan voor met als bij ook nog wel naar uit over onder meer tussen dit deze aan het er zich bij".split()),
    "sv": frozenset("och det att i en ett av för som är var på med till har inte den de sig om ett efter över under mellan när men även från vid utan".split()),
    "da": frozenset("og det at i en et af til som er var på med har ikke den de sig om efter over under mellem når men også fra ved uden sig".split()),
    "nb": frozenset("og det at i en et av til som er var på med har ikke den de seg om etter over under mellom når men også fra ved uten".split()),
    "fi": frozenset("ja se että on ovat oli ei kun niin kuin mutta myös tämä nämä sen hänen heidän jälkeen yli alle kesken kunnes jos jo vielä sekä tai".split()),
    "pl": frozenset("i w na z ze do od po za się nie to jest są był była jak ale też już tylko jeszcze ich jego jej tym tej przez między podczas według".split()),
    "ro": frozenset("și în la de un o este sunt că nu cu pe pentru mai din care fost după între despre acest această aceste prin decât atunci când".split()),
    "ca": frozenset("el la els les de del un una i o en és són amb per pera que com més però seus li ja aquest aquesta entre sobre després molt sense on quan".split()),
    "tr": frozenset("bir ve bu da de ne için gibi çok daha en olarak sonra önce ama ancak ya da ile ilgili üzere değil ki şey her iki üç dört beş".split()),
    "id": frozenset("yang dan di ke dari untuk dengan ini itu adalah tidak akan pada juga atau dalam lebih setelah antara oleh karena telah saat ketika atas bawah mereka kami".split()),
    "ms": frozenset("yang dan di ke dari untuk dengan ini itu adalah tidak akan pada juga atau dalam lebih selepas antara oleh kerana telah saat ketika atas bawah mereka kami".split()),
    "sw": frozenset("ya wa na katika kwa ni kutoka juu chini baada kabla lakini pia sana zaidi huyu hii hizi wale wakati ambao kuwa kuna".split()),
    "yo": frozenset("ni ti si ki o jẹ lati fun pẹlu ṣugbọn tabi ju lẹhin ṣaaju nigba ti awọn eyi yii wọn a mi rẹ lori labẹ laarin".split()),
    "ha": frozenset("na da a cikin daga zuwa ga ya yi ne ce su wa wannan haka kuma amma ko bayan kafin lokacin da suke kan karkashin tsakanin".split()),
    "ig": frozenset("na nke bụ si maka ọ ndị ha mgbe tupu ọzọ karịa n'okpuru n'etiti mana ma ka dị ga n'ime n'elu ugbua ebe ahụ".split()),
    "vi": frozenset("và của là một các không có trong cho được với này đó khi đã về như hơn sau trước giữa theo tại bởi vì nên cũng rất".split()),
    "cs": frozenset("a v na z ze do od po za se ne to je jsou byl byla jak ale také již jen ještě jejich jeho její tím té přes mezi během podle".split()),
    "hu": frozenset("a az és hogy nem van vannak volt mint is el meg de csak még már igen őket őt után között alatt fölött szerint amikor".split()),
    "hr": frozenset("i u na z ze do od po za se ne to je su bio bila kako ali također već samo još njihov njegov njezin tim tom kroz između tijekom prema".split()),
}

#: Diacritic signatures: a strong secondary signal per language.
_DIACRITICS: dict[str, str] = {
    "fr": "éèêàâçîôû",
    "es": "ñáéíóúü¡¿",
    "de": "äöüß",
    "pt": "ãõáéíóúâêôç",
    "it": "àèéìòù",
    "sv": "åäö",
    "da": "æøå",
    "nb": "æøå",
    "fi": "äöå",
    "pl": "ąćęłńóśźż",
    "ro": "ăâîșț",
    "ca": "àèéíòóúç·",
    "tr": "çğışöü",
    "vi": "ăâđêôơưàảãáạ",
    "cs": "áčďéěíňóřšťúůýž",
    "hu": "áéíóöőúüű",
    "hr": "čćžšđ",
    "yo": "ẹọṣ",
    "ig": "ịọụ",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_heuristics(text: str) -> tuple[str, float]:
    """The built-in detector: (ISO code | "und", confidence 0..1)."""
    sample = (text or "").strip()
    if len(sample) < 2:
        return "und", 0.0
    scripted = _script_language(sample)
    if scripted is not None:
        return scripted
    words = [w.lower() for w in _WORD_RE.findall(sample)]
    if not words:
        return "und", 0.0
    lowered = sample.lower()
    scores: dict[str, float] = {}
    word_set = set(words)
    for code, stops in _STOPWORDS.items():
        hits = sum(1 for w in words if w in stops)
        if hits == 0:
            continue
        score = hits / max(6, len(words))
        dia = _DIACRITICS.get(code, "")
        if dia:
            d_hits = sum(1 for ch in lowered if ch in dia)
            if d_hits:
                score += min(0.25, 0.05 * d_hits)
        # exact-phrase bonus: two consecutive stopwords of one language
        # is far more distinctive than scattered single hits.
        for a, b in zip(words, words[1:]):
            if a in stops and b in stops:
                score += 0.03
        scores[code] = score
    if not scores:
        return "und", 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_code, best = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    hits = sum(1 for w in words if w in _STOPWORDS[best_code])
    if hits < 2 and len(words) > 4:
        return "und", 0.0
    if best < 0.06:
        return "und", 0.0
    confidence = max(0.35, min(0.98, 0.45 + (best - runner) * 2.2 + best / 2))
    # id/ms and da/nb share most stopwords: without diacritics or
    # distinctive words the honest answer is the stronger prior.
    if best_code == "ms" and scores.get("id", 0) >= best - 0.02:
        best_code = "id"
    if best_code == "nb" and scores.get("da", 0) >= best - 0.02:
        best_code = "da" if any(c in lowered for c in "æø") and "æ" in lowered else best_code
    return best_code, round(confidence, 3)


def detect_with_library(text: str) -> tuple[str, float] | None:
    """Library rung (langdetect / lingua) when installed, else None."""
    try:
        from langdetect import detect_langs  # type: ignore[import-not-found]

        langs = detect_langs(text)
        if langs:
            top = langs[0]
            code = normalize_code(str(top.lang))
            if code:
                return code, float(top.prob)
        return None
    except Exception:  # noqa: BLE001 — absent or broken library: next rung
        pass
    try:
        from lingua import LanguageDetectorBuilder  # type: ignore[import-not-found]

        detector = (LanguageDetectorBuilder.from_all_languages()
                    .with_preloaded_language_models().build())
        lang = detector.detect_language_of(text)
        if lang is not None:
            code = normalize_code(str(lang.iso_code_639_1.name).lower())
            if code:
                conf = detector.compute_language_confidence(text, lang)
                return code, float(conf)
        return None
    except Exception:  # noqa: BLE001
        return None


def detect_language(text: str) -> tuple[str, float]:
    """Detect the language of ``text``: (ISO code | "und", confidence).

    Laddered: an installed detection library first, the built-in
    heuristic rung behind it.  ``und`` means genuinely undetermined —
    callers must treat it as unknown, not as English.
    """
    sample = (text or "").strip()
    if len(sample) < 2:
        return "und", 0.0
    lib = detect_with_library(sample)
    if lib is not None:
        return lib
    return detect_heuristics(sample)


def any_language_hint(obj: Any) -> str:
    """Best-effort stored language hint on a feed item ("" when none)."""
    if isinstance(obj, dict):
        for key in ("language", "lang", "locale"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                code = normalize_code(val)
                if code:
                    return code
    return ""
