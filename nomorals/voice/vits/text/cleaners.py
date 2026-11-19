"""Text cleaners for the VITS front end.

Faithful port of upstream ``text/cleaners.py`` with the import-time
failures removed:

* Upstream instantiates ``ARPAPhonemizer()`` at import time (which
  downloads the dictionary from Google Drive when absent) and defines
  ``english_cleaners2`` / ``spanish_cleaners`` / ``f_english_cleaners2``
  against phonemizer backends that are never constructed
  (``HAS_ESPEAK`` / ``HAS_FESTIVAL`` are ``False``), so calling them
  raises ``NameError``. Here every phonemizer is built lazily on first
  use and a missing backend raises a named error saying what to install.
* ``convert_to_ascii`` uses ``unidecode`` when installed and falls back to
  a stdlib NFKD strip otherwise, so the ARPA and basic pipelines run
  with no optional dependency at all.
* ``expand_numbers`` upstream calls ``normalize_numbers``, which is never
  imported (``text/numbers.py`` does not exist in the clone) — calling it
  raised ``NameError``. It now raises a named error saying number
  expansion is unavailable.
"""

import re

from .arpa import ARPAPhonemizer

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile(f'\\b{x[0]}\\.', re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]

_arpa_phonemizer = None


def _get_arpa_phonemizer():
    global _arpa_phonemizer
    if _arpa_phonemizer is None:
        _arpa_phonemizer = ARPAPhonemizer()
    return _arpa_phonemizer


def _espeak_backend(language):
    try:
        import phonemizer
    except ImportError as exc:
        raise ImportError(
            f"phonemizer backend {language!r} needs the phonemizer package and an "
            "espeak/espeak-ng binary; neither is installed"
        ) from exc
    return phonemizer.backend.EspeakBackend(
        language=language, preserve_punctuation=True, with_stress=True)


def _festival_backend():
    try:
        import phonemizer
    except ImportError as exc:
        raise ImportError(
            "the festival backend needs the phonemizer package and festival; "
            "neither is installed"
        ) from exc
    return phonemizer.backend.FestivalBackend(
        language='en-us', preserve_punctuation=True)


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    raise NotImplementedError(
        "number expansion is unavailable: upstream cleaners.py calls "
        "normalize_numbers, which is never imported (text/numbers.py does "
        "not exist in the clone)"
    )


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    try:
        from unidecode import unidecode
    except ImportError:
        import unicodedata
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English text that transliterates to ASCII.'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English text, including abbreviation expansion.'''
    from phonemizer import phonemize
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def english_cleaners2(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = _espeak_backend('en-us').phonemize([text], strip=True, njobs=1)
    phonemes = phonemes[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def f_english_cleaners2(text):
    '''Pipeline for English text with Festival phonemizer, including abbreviation expansion. + punctuation'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = _festival_backend().phonemize([text], strip=True, njobs=1)
    phonemes = phonemes[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def arpa_precleaners(text):
    '''Pipeline for English text before ARPA'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    return text


def arpa_cleaners(text):
    '''Pipeline for ARPA English text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = _get_arpa_phonemizer().doARPA(text)
    return phonemes


def spanish_cleaners(text):
    '''Pipeline for Spanish text, punctuation + stress'''
    text = lowercase(text)
    phonemes = _espeak_backend('es-419').phonemize([text], strip=True, njobs=1)
    phonemes = phonemes[0].replace("(es-419)", "").replace("(en)", "")
    phonemes = collapse_whitespace(phonemes)
    return phonemes
