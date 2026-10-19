"""WisdomKeeper readability helpers — pure functions, no I/O.

Everything the owner needs to actually *read* a text lives behind these
helpers so the CLI, the Telegram runtime, and the corpus all parse and
paginate identically:

- :func:`normalize_tradition` — case/hyphen/space-tolerant tradition names.
- :func:`parse_timeline_args` — ``timeline <tradition> [n]`` without
  swallowing the count into the tradition name.
- :func:`parse_read_args` — ``read <work> [chapter] [page]``.
- :func:`dehyphenate` — undo OCR line-break hyphenation (``liter-\\n ature``
  → ``literature``) without touching real hyphenated words.
- :func:`paginate_text` — deterministic sentence-boundary pages: page 2
  starts exactly where page 1 ended, no gap, no overlap.
- :data:`TRADITION_INFO` / :data:`SEEDS` — what a tradition is, and every
  seed source knowledge can come from.
"""
from __future__ import annotations

import re
from typing import Any

# Default page size for ``/wisdom read`` — fits one Telegram message
# (hard cap 4096 chars) with room for the header + continuation hint.
PAGE_CHARS = 3500

# ── traditions ────────────────────────────────────────────────────────
# Canonical slug -> (readable label, what this tradition is). A
# "tradition" is a family of texts + teachings that grew together around
# shared scriptures, teachers, and practices — e.g. everything filed
# under "christian-gnostic" is early-Christian esoteric writing, not
# the mainstream canon. Works are grouped by tradition so an answer can
# say *whose* scripture a passage comes from.
TRADITION_INFO: dict[str, tuple[str, str]] = {
    "buddhism": ("Buddhism",
                 "The teaching of the Buddha: suffering, impermanence, "
                 "and the path out — sutras, monastic and Mahayana texts."),
    "christian-gnostic": ("Christian Gnosticism",
                          "Early Christian esoteric writings (Nag Hammadi "
                          "and related): hidden knowledge (gnosis), the "
                          "divine spark, and sayings outside the canon."),
    "christianity": ("Christianity",
                     "Mainstream Christian scripture and mysticism: the "
                     "Bible, the Apocrypha, and contemplative writers."),
    "confucianism": ("Confucianism",
                     "Confucius and his successors: ethics, ritual, "
                     "government, and the cultivated person."),
    "egyptian": ("Egyptian",
                "Ancient Egyptian funerary and wisdom literature — "
                "the Book of the Dead and related texts."),
    "hermeticism": ("Hermeticism",
                    "Texts attributed to Hermes Trismegistus and their "
                    "heirs: correspondence, polarity, mentalism — the "
                    "Kybalion, Corpus Hermeticum, Emerald Tablet."),
    "hinduism": ("Hinduism",
                 "The Vedic and post-Vedic scriptures: Upanishads, "
                 "Gita, Yoga Sutras, and devotional literature."),
    "interfaith": ("Interfaith",
                   "Comparative / cross-tradition works that study "
                   "religious experience itself."),
    "islam": ("Islam",
              "The Quran and Islamic scripture in translation."),
    "jainism": ("Jainism",
                "Jain scripture and ethics: non-violence, non-attachment, "
                "liberation of the soul."),
    "judaism": ("Judaism",
                "Jewish scripture and mysticism: pseudepigrapha, the "
                "Zohar, Sefer Yetzirah, Dead Sea Scrolls."),
    "mesoamerican": ("Mesoamerican",
                    "Pre-Columbian American sacred narrative — the "
                    "Popol Vuh and related creation literature."),
    "stoicism": ("Stoicism",
                 "Greco-Roman practical philosophy: Marcus Aurelius, "
                 "Epictetus — control, assent, and the examined life."),
    "sufism": ("Sufism",
               "Islamic mysticism: Rumi, Ibn Arabi — love, annihilation "
               "in the divine, and the inner path."),
    "taoism": ("Taoism",
               "The Tao Te Ching, Chuang Tzu, I Ching — the Way that "
               "cannot be named, wu-wei, and naturalness."),
    "theosophy": ("Theosophy",
                  "Blavatsky and the Theosophical synthesis: Secret "
                  "Doctrine, Isis Unveiled — esoteric evolution and "
                  "hidden masters."),
    "vedanta": ("Vedanta",
                "Advaita Vedanta: non-dual philosophy of the Upanishads — "
                "Atman is Brahman."),
}

_TRADITION_ALIASES = {
    "gnostic": "christian-gnostic",
    "gnosticism": "christian-gnostic",
    "gnostic-christian": "christian-gnostic",
    "christian-gnosticism": "christian-gnostic",
    "christian": "christianity",
    "hermetic": "hermeticism",
    "sufi": "sufism",
    "taoist": "taoism",
    "tao": "taoism",
    "confucius": "confucianism",
    "muslim": "islam",
    "hindu": "hinduism",
    "buddhist": "buddhism",
    "jewish": "judaism",
    "stoic": "stoicism",
    "egypt": "egyptian",
    "maya": "mesoamerican",
    "aztec": "mesoamerican",
    "theosophical": "theosophy",
    "advaita": "vedanta",
    "jain": "jainism",
}


def normalize_tradition(raw: str, known: Any = None) -> str:
    """Canonical tradition slug for free-form input.

    Case-insensitive and tolerant of hyphens, underscores, and spaces:
    ``"Christian-gnostic"``, ``"CHRISTIAN GNOSTIC"`` and
    ``"christian_gnostic"`` all become ``"christian-gnostic"``.
    Common aliases (``"gnostic"`` → ``"christian-gnostic"``) resolve too.
    Unknown names pass through normalized (callers decide whether to
    fail fast with the known list).
    """
    cleaned = re.sub(r"[\s_]+", "-", (raw or "").strip().lower())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    if not cleaned:
        return ""
    if known is not None:
        known_norm = {normalize_tradition(k) for k in known}
        if cleaned in known_norm:
            return cleaned
        aliased = _TRADITION_ALIASES.get(cleaned)
        if aliased and aliased in known_norm:
            return aliased
        return cleaned
    return _TRADITION_ALIASES.get(cleaned, cleaned)


def tradition_label(slug: str) -> str:
    info = TRADITION_INFO.get(normalize_tradition(slug))
    return info[0] if info else (slug or "unknown")


def tradition_description(slug: str) -> str:
    info = TRADITION_INFO.get(normalize_tradition(slug))
    if info:
        return info[1]
    return ("A family of texts and teachings grouped by shared "
            "scriptures and practices.")


# ── seeds ─────────────────────────────────────────────────────────────
# Every source WisdomKeeper knowledge can come from. A "seed" is a
# starting source list / discovery channel — NOT the ingested texts
# themselves. Works record which seed they came from (``ManifestEntry.seed``)
# so the owner can always see the receipt: seed → work → source URL →
# ingested characters.
SEEDS: list[dict[str, str]] = [
    {
        "id": "packaged-seed",
        "name": "Packaged seed manifest (curated starter library)",
        "what": "nomorals/wisdom/data/manifest.json — 47 curated texts "
                "across 14 traditions, each with title, tradition, "
                "translator, and source URL. Registered by "
                "`/wisdom seed`, fetched by `/wisdom ingest --all`.",
        "how_to_add": "Edit nomorals/wisdom/data/manifest.json (new entry "
                      "with slug/title/tradition/source_url) and run "
                      "`/wisdom seed`, or add one work with "
                      "`/wisdom ingest <slug>` after registering it.",
    },
    {
        "id": "timeline-seed",
        "name": "Seed timeline dataset (history only)",
        "what": "nomorals/wisdom/data/timeline.json — 52 sourced history "
                "events across 14 traditions. Feeds `/wisdom timeline` "
                "and compare's timeline context. History, not readable "
                "texts — timeline events have no full text to open.",
        "how_to_add": "Append a sourced event to "
                      "nomorals/wisdom/data/timeline.json (start/end, "
                      "tradition, region, title, summary, sources).",
    },
    {
        "id": "hunt:gutendex",
        "name": "Gutendex / Project Gutenberg (live hunt)",
        "what": "Structured Gutenberg catalog search via the Gutendex "
                "API — public-domain books with authors/subjects and a "
                "direct plain-text URL. A discovery channel: candidates "
                "are NOT ingested until the owner picks one.",
        "how_to_add": "Run `/wisdom search <query>`; candidates from "
                      "this seed carry source 'gutendex'. Ingest a "
                      "picked candidate and it is recorded with "
                      "seed 'hunt:gutendex'.",
    },
    {
        "id": "hunt:wikisource",
        "name": "Wikisource (live hunt)",
        "what": "English Wikisource via the MediaWiki API — proofread, "
                "work-level public-domain texts (poems, treatises). "
                "Discovery channel; see hunt:gutendex for the flow.",
        "how_to_add": "Run `/wisdom search <query>`; candidates from "
                      "this seed carry source 'wikisource'.",
    },
    {
        "id": "hunt:archive",
        "name": "archive.org (live hunt)",
        "what": "archive.org advanced search over mediatype:texts — the "
                "largest open-text archive. Discovery channel.",
        "how_to_add": "Run `/wisdom search <query>`; candidates from "
                      "this seed carry source 'archive'.",
    },
    {
        "id": "hunt:web",
        "name": "Web fallback (live hunt)",
        "what": "Plain web search fallback (DuckDuckGo lite) when the "
                "structured archives have nothing. Discovery channel; "
                "robots.txt respected.",
        "how_to_add": "Run `/wisdom search <query>`; candidates from "
                      "this seed carry source 'web'.",
    },
    {
        "id": "manual",
        "name": "Manual / owner-added",
        "what": "Texts the owner (or the brain) registers directly — "
                "paste-in text, a local file, or a URL handed to the "
                "ingestor. Recorded with seed 'manual' so they are "
                "never confused with the packaged list.",
        "how_to_add": "Register a ManifestEntry (seed='manual') and "
                      "ingest its text, or ask Devon to ingest a URL.",
    },
]

# ── arg parsing ───────────────────────────────────────────────────────


def parse_timeline_args(rest: list[str]) -> tuple[str, int | None]:
    """Split ``timeline <tradition> [n]`` into (tradition, count).

    A trailing bare integer is the event count, NOT part of the
    tradition name — ``["christian-gnostic", "50"]`` → ``("christian-gnostic", 50)``.
    A lone integer means "first N events, all traditions".
    """
    tokens = [t for t in (rest or []) if t != ""]
    if not tokens:
        return "", None
    if tokens[-1].isdigit() and len(tokens) >= 1:
        count = int(tokens[-1])
        head = tokens[:-1]
        # A lone number is a count; a tradition + number is tradition + count.
        return " ".join(head), count
    return " ".join(tokens), None


def parse_read_args(tokens: list[str]) -> tuple[Any, int]:
    """Parse the tail of ``read <work> [chapter] [page]``.

    Returns ``(chapter, page)`` where chapter is an int (chapter number),
    a str (chapter-title search), or None, and page is 1-based.
    ``["3"]`` → page 3 (no chapter); ``["2", "3"]`` → chapter 2, page 3;
    ``["chapter", "2"]`` / ``["page", "3"]`` / ``["Parables"]`` all work.
    """
    toks = list(tokens or [])
    chapter: Any = None
    page = 1
    bare: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i].lower()
        if t in ("chapter", "ch", "section", "sec"):
            if i + 1 < len(toks):
                nxt = toks[i + 1]
                chapter = int(nxt) if nxt.isdigit() else nxt
                i += 2
                continue
        elif t in ("page", "p", "pg"):
            if i + 1 < len(toks) and toks[i + 1].isdigit():
                page = int(toks[i + 1])
                i += 2
                continue
        bare.append(toks[i])
        i += 1
    if bare:
        if all(b.isdigit() for b in bare):
            if len(bare) == 1:
                page = int(bare[0])
            else:
                chapter, page = int(bare[0]), int(bare[1])
        elif len(bare) == 1 and bare[0].isdigit():
            page = int(bare[0])
        elif bare and bare[-1].isdigit() and len(bare) >= 2:
            page = int(bare[-1])
            chapter = " ".join(bare[:-1])
        else:
            chapter = " ".join(bare)
    return chapter, max(1, page)


# ── text cleaning / pagination ────────────────────────────────────────

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_SOFT_HYPHEN = "\u00ad"


def dehyphenate(text: str) -> str:
    """Undo OCR line-break hyphenation.

    ``"liter-\\n ature"`` → ``"literature"``. Real hyphenated words are
    untouched: only a hyphen immediately followed by a line break (the
    OCR wrap) is removed — ``"well-known"`` on one line keeps its hyphen.
    """
    out = (text or "").replace(_SOFT_HYPHEN, "")
    return _HYPHEN_BREAK.sub(r"\1\2", out)


_SENTENCE_END = re.compile(r"[.!?…]+[\"'\u201d)\]]?")


def _sentence_breaks(text: str) -> list[int]:
    """End offsets (exclusive) of every sentence in ``text``."""
    return [m.end() for m in _SENTENCE_END.finditer(text)]


def trim_to_sentences(text: str) -> str:
    """Drop a leading/trailing partial sentence from a truncated snippet.

    The library marks a cut window with a leading/trailing ellipsis
    (``…``): only a side carrying that mark is trimmed — a snippet that
    starts at a chapter start or ends at a chapter end is already whole
    and passes through untouched. A leading ``…`` drops the fragment up
    to the first sentence end; a trailing ``…`` cuts back to the last
    sentence end.
    """
    text = dehyphenate((text or "")).strip()
    if not text:
        return text
    if text.startswith("…"):
        body = text[1:].lstrip()
        first = _SENTENCE_END.search(body)
        if first and first.end() < len(body) - 20:
            text = body[first.end():].strip()
        else:
            text = body
    if text.endswith("…"):
        body = text[:-1].rstrip()
        breaks = _sentence_breaks(body)
        if breaks:
            text = body[:breaks[-1]].strip()
        else:
            text = body
    return text.strip()


def sentence_passage(full_text: str, query: str, *,
                     before_chars: int = 500, after_chars: int = 450,
                     max_chars: int = 1100) -> str:
    """Extract a sentence-complete passage around the query's match.

    Finds the first query term in the full chapter text, then expands
    to whole sentences on both sides. Never returns a mid-sentence cut
    (unless the chapter itself has no sentence punctuation at all, in
    which case the raw window is returned dehyphenated).
    """
    text = dehyphenate(full_text or "")
    if not text.strip():
        return ""
    terms = [w.lower() for w in re.findall(r"[A-Za-z0-9']+", query or "")
             if len(w) >= 3]
    low = text.lower()
    pos = -1
    for term in terms:
        pos = low.find(term)
        if pos != -1:
            break
    if pos == -1:
        return trim_to_sentences(text[:max_chars])
    start = max(0, pos - before_chars)
    end = min(len(text), pos + after_chars)
    # Back up to the sentence start before `start`: the last sentence
    # end at or before start marks the boundary.
    if start > 0:
        prev = [b for b in _sentence_breaks(text[:start])]
        if prev:
            start = prev[-1]
            # Skip whitespace/quotes after the terminator.
            while start < len(text) and text[start] in " \t\n\"'“‘([»":
                start += 1
    # Forward to the sentence end at or after `end`.
    tail = _SENTENCE_END.search(text, end)
    if tail:
        end = tail.end()
    else:
        # No sentence end ahead — keep to the last end before pos+max.
        prev = [b for b in _sentence_breaks(text[:min(len(text), pos + max_chars)])]
        if prev and prev[-1] > pos:
            end = prev[-1]
    out = text[start:end].strip()
    if len(out) > max_chars:
        # Too long: keep whole sentences from the match outward by
        # re-trimming to the last sentence end within the cap.
        capped = out[:max_chars]
        breaks = _sentence_breaks(capped)
        if breaks:
            out = capped[:breaks[-1]].strip()
    return out


def paginate_text(text: str, *, page_chars: int = 3500) -> list[str]:
    """Split ``text`` into pages that end at sentence boundaries.

    Deterministic: re-paginating the same text yields the same pages,
    page N+1 starts exactly where page N ended (no gap, no overlap).
    A page break is the last sentence end / paragraph break at or
    before the target, falling back to the next sentence end shortly
    after, then to a word boundary. The final page is whatever remains.
    """
    text = dehyphenate(text or "").strip()
    if not text:
        return [""]
    pages: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        if n - start <= page_chars:
            pages.append(text[start:].strip())
            break
        target = start + page_chars
        # Prefer a paragraph break, else a sentence end, at/before target.
        window_start = max(start + 200, target - 700)
        para = text.rfind("\n\n", window_start, target)
        if para != -1 and para > start + 200:
            cut = para + 2
        else:
            breaks = [b for b in _sentence_breaks(text[window_start:target])]
            if breaks:
                cut = window_start + breaks[-1]
            else:
                # Look ahead for the next sentence end within 700 chars.
                nxt = _SENTENCE_END.search(text, target,
                                           min(n, target + 700))
                if nxt:
                    cut = nxt.end()
                else:
                    # Last resort: word boundary.
                    sp = text.rfind(" ", window_start, target)
                    cut = sp + 1 if sp > start else target
        page = text[start:cut].strip()
        if not page:  # defensive: never emit an empty page / loop forever
            cut = target
            page = text[start:cut].strip()
        pages.append(page)
        start = cut
        # Skip blank separators between pages (they belong to neither).
        while start < n and text[start] in " \n\t":
            start += 1
    return pages
