"""Formatting conformance (GROUPS Tier 0) — one platform's markup must
never leak into another's message.

The renderers already produce platform-native output, but nothing
*checked* it: a Telegram ``<b>`` could sail into a WhatsApp send (where
it renders literally as ``<b>``), and WhatsApp ``*bold*`` into Telegram
HTML mode (where it renders literally as ``*bold*``). This module is
the check + safe splitter at the send boundary:

* :func:`check_message` — returns a violations list (empty = clean).
  HTML in WhatsApp/Discord/SMS, markdown in Telegram, unclosed tags,
  unbalanced markers, unknown tags, over-length.
* :func:`chunk` — splits long text for a platform on structure
  (paragraph / sentence / word), never mid-markup, never mid-``<b>``,
  never mid-emoji (ZWJ / variation-selector / skin-tone sequences
  stay whole). A markup span that fits in one chunk stays in one
  chunk; a span larger than the budget is split at inner safe
  boundaries and re-wrapped so every chunk is balanced.
* :func:`convert` / :func:`sanitize` — transcode markup between
  platforms (HTML <-> WhatsApp/Discord markdown <-> plain) and verify
  the result.

Unlike :mod:`nomorals.social.chat.chunking` (lossless, content chunks
concatenate exactly), :func:`chunk` may re-wrap an oversized markup
span, so the join guarantee is: chunks concatenate to the original
whenever every markup span fits in one chunk (the normal case); for an
oversized span each chunk is balanced and the *unwrapped* content
concatenates to the original.
"""

from __future__ import annotations

import html
import re
from typing import Any

__all__ = [
    "check_message",
    "chunk",
    "convert",
    "sanitize",
    "max_text_for",
    "KNOWN_PLATFORMS",
]

#: Canonical platform names this module reasons about.
KNOWN_PLATFORMS = ("telegram", "telegram-bot", "whatsapp", "discord",
                   "sms", "web", "local", "webhook")

#: Fallback per-message caps when platform_limits is unavailable.
_FALLBACK_CAPS = {
    "telegram": 4096, "telegram-bot": 4096, "whatsapp": 4000,
    "discord": 2000, "sms": 160, "web": 8000, "local": 8000,
    "webhook": 8000,
}

#: Tags Telegram HTML mode accepts.
TELEGRAM_TAGS = {
    "b", "strong", "i", "em", "u", "ins", "s", "del", "strike",
    "code", "pre", "a", "blockquote", "tg-spoiler",
}

_TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9\-]*)\b([^>]*)>")
_ENTITY_RE = re.compile(r"&(?:lt|gt|amp;lt|amp;gt|quot|amp;quot);")
_MD_LINK_RE = re.compile(r"\[[^\]\n]+\]\([^)\n]+\)")
_MD_BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
_MD_UNDER_RE = re.compile(r"__[^_\n]+__")
_MD_STRIKE_RE = re.compile(r"~~[^~\n]+~~")
_MD_SINGLE_STAR_RE = re.compile(r"(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)")
_MD_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def _plat(platform: str) -> str:
    p = (platform or "").strip().lower()
    if p in ("wa", "whatsapp"):
        return "whatsapp"
    if p in ("tg", "telegram", "telegram-bot", "tgbot"):
        return "telegram" if p != "tgbot" else "telegram-bot"
    if p.startswith("discord"):
        return "discord"
    if p in ("sms", "twilio"):
        return "sms"
    if p in ("web", "local", "webhook", "console"):
        return p
    return "telegram" if p.startswith("telegram") else p


def max_text_for(platform: str) -> int:
    """Per-message cap for ``platform`` (descriptor table, else fallback)."""
    p = _plat(platform)
    try:
        from .chat.platform_limits import for_platform

        return int(for_platform(p).max_text)
    except Exception:  # noqa: BLE001 - descriptor table is best-effort
        return _FALLBACK_CAPS.get(p, 4000)


def _telegram_tag_violations(text: str) -> list[str]:
    violations: list[str] = []
    stack: list[str] = []
    for m in _TAG_RE.finditer(text):
        closing, name, attrs = m.group(1), m.group(2).lower(), m.group(3)
        if name not in TELEGRAM_TAGS:
            violations.append(
                f"unsupported-html-tag <{name}>: not allowed in Telegram "
                "HTML mode")
            continue
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
            elif name in stack:
                violations.append(
                    f"misnested-html-tag </{name}>: closes out of order "
                    f"(open: {stack})")
                while stack and stack[-1] != name:
                    stack.pop()
                stack.pop()
            else:
                violations.append(
                    f"stray-closing-tag </{name}>: no matching opener")
        else:
            if name == "a" and not re.search(
                    r"href\s*=\s*\"[^\"]+\"|href\s*=\s*'[^']+'", attrs):
                violations.append(
                    "<a> without href: Telegram requires href on links")
            stack.append(name)
    for name in reversed(stack):
        violations.append(f"unclosed-html-tag <{name}>: missing </{name}>")
    return violations


def _leftover_markers(text: str, patterns: list[re.Pattern[str]],
                      label: str) -> list[str]:
    """Markers left over after removing every paired span = unbalanced."""
    cleaned = text
    for pat in patterns:
        cleaned = pat.sub("", cleaned)
    violations: list[str] = []
    for marker in ("```", "**", "__", "~~"):
        if marker in cleaned:
            violations.append(
                f"unbalanced-markdown {marker}: {label}")
    return violations


def check_message(platform: str, text: str) -> list[str]:
    """Violations of ``platform``'s formatting rules in ``text``.

    Empty list = clean. Each violation is a ``code: detail`` string
    so callers can filter (``v.startswith("html-in-")``) or display.
    Never raises — garbage input yields a violation, not an exception.
    """
    try:
        p = _plat(platform)
        text = text or ""
        violations: list[str] = []
        tags = list(_TAG_RE.finditer(text))

        if p in ("whatsapp", "discord", "sms", "web", "local", "webhook"):
            for m in tags:
                violations.append(
                    f"html-in-{p}: {m.group(0)[:40]} — {p} does not "
                    "render HTML tags; they show literally")
            if _ENTITY_RE.search(text):
                violations.append(
                    f"html-entity-in-{p}: &lt; / &gt; escaped HTML "
                    "leaking into plain-text output")

        if p in ("telegram", "telegram-bot"):
            violations.extend(_telegram_tag_violations(text))
            for label, pat in (
                    ("markdown-bold **..**", _MD_BOLD_RE),
                    ("markdown-underline __..__", _MD_UNDER_RE),
                    ("markdown-strike ~~..~~", _MD_STRIKE_RE),
                    ("markdown-link [..](..)", _MD_LINK_RE),
                    ("markdown-code `..`", _MD_CODE_RE),
                    ("markdown-bold *..*", _MD_SINGLE_STAR_RE)):
                for m in pat.finditer(text):
                    violations.append(
                        f"markdown-in-telegram: {label} "
                        f"{m.group(0)[:40]!r} renders literally in "
                        "Telegram HTML mode — use <b>/<i>/<code> instead")
            for m in _MD_FENCE_RE.finditer(text):
                violations.append(
                    "markdown-in-telegram: ```fence``` "
                    f"{m.group(0)[:40]!r} renders literally — use "
                    "<pre> instead")

        if p == "whatsapp":
            violations.extend(_leftover_markers(
                text, [_MD_FENCE_RE, _MD_CODE_RE,
                       re.compile(r"\*[^*\n]+\*"),
                       re.compile(r"_[^_\n]+_"),
                       re.compile(r"~[^~\n]+~")],
                "WhatsApp markers must pair up"))

        if p == "discord":
            violations.extend(_leftover_markers(
                text, [_MD_FENCE_RE, _MD_CODE_RE, _MD_BOLD_RE,
                       _MD_UNDER_RE, _MD_STRIKE_RE],
                "Discord markers must pair up"))

        if p == "sms":
            for label, pat in (
                    ("markdown-bold **..**", _MD_BOLD_RE),
                    ("markdown-underline __..__", _MD_UNDER_RE),
                    ("markdown-strike ~~..~~", _MD_STRIKE_RE),
                    ("markdown-link [..](..)", _MD_LINK_RE),
                    ("markdown-code `..`", _MD_CODE_RE),
                    ("markdown-bold *..*", _MD_SINGLE_STAR_RE)):
                if pat.search(text):
                    violations.append(
                        f"markup-in-sms: {label} — SMS is plain text")

        cap = max_text_for(p)
        if len(text) > cap:
            violations.append(
                f"over-length: {len(text)} chars > {cap} cap for {p} "
                "— chunk before sending")
        return violations
    except Exception as exc:  # noqa: BLE001 - checker never raises
        return [f"check-failed: {exc}"]


# ── safe splitting ───────────────────────────────────────────────────

#: Characters a chunk boundary must never follow / precede.
_NO_CUT_AFTER = set("\u200d\ufe0f\u20e3") | {
    chr(c) for c in range(0x1F3FB, 0x1F400)} | {
    chr(c) for c in range(0x0300, 0x0370)}
_NO_CUT_BEFORE = set("\u200d\ufe0f\u20e3") | {
    chr(c) for c in range(0x1F3FB, 0x1F400)} | {
    chr(c) for c in range(0x0300, 0x0370)}


def _cut_ok(s: str, pos: int) -> bool:
    """True when cutting ``s`` at ``pos`` splits nothing atomic."""
    if pos <= 0 or pos >= len(s):
        return True
    if s[pos - 1] in _NO_CUT_AFTER or s[pos] in _NO_CUT_BEFORE:
        return False
    # Never cut inside an <...> tag: more '<' than '>' before the cut.
    head = s[:pos]
    if head.count("<") > head.count(">"):
        return False
    # Never split a regional-indicator pair (flag emoji = 2 RI chars).
    if "\U0001F1E6" <= s[pos - 1] <= "\U0001F1FF":
        ri_before = 0
        i = pos - 1
        while i >= 0 and "\U0001F1E6" <= s[i] <= "\U0001F1FF":
            ri_before += 1
            i -= 1
        if ri_before % 2 == 1:
            return False
    return True


def _safe_cut(s: str, pos: int) -> int:
    """Move ``pos`` left to the nearest safe cut (0 if none)."""
    pos = max(0, min(len(s), pos))
    while pos > 0 and not _cut_ok(s, pos):
        pos -= 1
    return pos


def _span_regex(platform: str) -> re.Pattern[str] | None:
    if platform in ("telegram", "telegram-bot"):
        return re.compile(
            r"<(b|strong|i|em|u|ins|s|del|strike|code|pre|a|blockquote|"
            r"tg-spoiler)(\s[^>]*)?>.*?</\1>",
            re.DOTALL)
    if platform == "whatsapp":
        return re.compile(
            r"```.*?```|`[^`\n]+`|\*[^*\n]+\*|_[^_\n]+_|~[^~\n]+~",
            re.DOTALL)
    if platform == "discord":
        return re.compile(
            r"```.*?```|`[^`\n]+`|\*\*[^*\n]+\*\*|__[^_\n]+__|"
            r"~~[^~\n]+~~|\*[^*\n]+\*|_[^_\n]+_",
            re.DOTALL)
    return None


def _word_atoms(text: str) -> list[str]:
    """Word + trailing-whitespace atoms; never splits a word or emoji."""
    return [a for a in re.findall(r"\S+\s*|\s+", text) if a]


def _hard_split_word(word: str, budget: int) -> list[str]:
    """Split one over-budget word at safe character boundaries."""
    out: list[str] = []
    rest = word
    while len(rest) > budget:
        cut = _safe_cut(rest, budget)
        if cut <= 0:  # no safe cut (e.g. a ZWJ run) — force one char
            cut = 1
        out.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        out.append(rest)
    return out


def _split_span(span: str, budget: int, platform: str) -> list[str]:
    """Split one over-budget markup span, re-wrapping every piece."""
    if platform in ("telegram", "telegram-bot"):
        m = re.match(r"^(<[^>]+>)(.*)(</[^>]+>)$", span, re.DOTALL)
        if m:
            opener, inner, closer = m.groups()
            inner_budget = budget - len(opener) - len(closer)
            if inner_budget >= 16:
                pieces: list[str] = []
                for atom in _word_atoms(inner):
                    if len(atom) > inner_budget:
                        pieces.extend(_hard_split_word(atom, inner_budget))
                    else:
                        pieces.append(atom)
                chunks: list[str] = []
                cur = ""
                for piece in pieces:
                    if cur and len(cur) + len(piece) > inner_budget:
                        chunks.append(f"{opener}{cur}{closer}")
                        cur = piece
                    else:
                        cur += piece
                if cur:
                    chunks.append(f"{opener}{cur}{closer}")
                return chunks or [span]
        return _hard_split_word(span, budget)
    # Markdown spans: opener / inner / closer (fence or 1–2 char marker).
    fence = re.match(r"^(```)(.*)(```)$", span, re.DOTALL)
    md = re.match(r"^(\*\*|__|~~|\*|_|`|~)(.*)(\1)$", span, re.DOTALL)
    m = fence or md
    if m:
        opener, inner, closer = m.groups()
        inner_budget = budget - len(opener) - len(closer)
        if inner_budget >= 16:
            pieces: list[str] = []
            for atom in _word_atoms(inner):
                if len(atom) > inner_budget:
                    pieces.extend(_hard_split_word(atom, inner_budget))
                else:
                    pieces.append(atom)
            chunks = []
            cur = ""
            for piece in pieces:
                if cur and len(cur) + len(piece) > inner_budget:
                    chunks.append(f"{opener}{cur}{closer}")
                    cur = piece
                else:
                    cur += piece
            if cur:
                chunks.append(f"{opener}{cur}{closer}")
            return chunks or [span]
    return _hard_split_word(span, budget)


def _atoms(text: str, platform: str) -> list[str]:
    """Markup spans as single atoms; everything else as word atoms."""
    span_re = _span_regex(platform)
    atoms: list[str] = []
    if span_re is None:
        return _word_atoms(text)
    pos = 0
    for m in span_re.finditer(text):
        if m.start() > pos:
            atoms.extend(_word_atoms(text[pos:m.start()]))
        atoms.append(m.group(0))
        pos = m.end()
    if pos < len(text):
        atoms.extend(_word_atoms(text[pos:]))
    return [a for a in atoms if a]


def chunk(text: str, platform: str = "telegram",
          max_len: int | None = None) -> list[str]:
    """Split ``text`` for ``platform`` into balanced, safe chunks.

    Splits happen on structure (whitespace/word boundaries) and never
    mid-markup, mid-tag, or mid-emoji. A markup span that fits stays
    whole; an oversized span is split at inner safe boundaries and
    re-wrapped so every chunk is balanced on its own.
    """
    text = text or ""
    if not text:
        return []
    p = _plat(platform)
    budget = int(max_len or max_text_for(p))
    budget = max(budget, 64)
    if len(text) <= budget:
        return [text]
    atoms = _atoms(text, p)
    chunks: list[str] = []
    cur = ""
    for atom in atoms:
        if len(atom) > budget:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(_split_span(atom, budget, p))
            continue
        if cur and len(cur) + len(atom) > budget:
            chunks.append(cur)
            cur = atom
        else:
            cur += atom
    if cur:
        chunks.append(cur)
    return chunks or [text]


# ── transcoding ──────────────────────────────────────────────────────

def _html_to_target(text: str, target: str) -> str:
    """HTML markup -> ``target``'s native markup (or plain)."""
    def _link(m: re.Match[str]) -> str:
        href = m.group(1)
        inner = m.group(2)
        if target == "discord":
            return f"[{inner}]({href})"
        if target in ("whatsapp", "sms"):
            return f"{inner} ({href})"
        return inner

    out = re.sub(r"<a\s+[^>]*href\s*=\s*\"([^\"]+)\"[^>]*>(.*?)</a>",
                 _link, text, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<a\s+[^>]*href\s*=\s*'([^']+)'[^>]*>(.*?)</a>",
                 _link, out, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<br\s*/?>", "\n", out, flags=re.IGNORECASE)
    subs: dict[str, list[tuple[str, str]]] = {
        "whatsapp": [
            (r"<(?:b|strong)>(.*?)</(?:b|strong)>", r"*\1*"),
            (r"<(?:i|em)>(.*?)</(?:i|em)>", r"_\1_"),
            (r"<(?:u|ins)>(.*?)</(?:u|ins)>", r"_\1_"),
            (r"<(?:s|del|strike)>(.*?)</(?:s|del|strike)>", r"~\1~"),
            (r"<code>(.*?)</code>", r"`\1`"),
            (r"<pre>(.*?)</pre>", r"```\1```"),
            (r"<blockquote>(.*?)</blockquote>", r"> \1"),
            (r"<tg-spoiler>(.*?)</tg-spoiler>", r"\1"),
        ],
        "discord": [
            (r"<(?:b|strong)>(.*?)</(?:b|strong)>", r"**\1**"),
            (r"<(?:i|em)>(.*?)</(?:i|em)>", r"*\1*"),
            (r"<(?:u|ins)>(.*?)</(?:u|ins)>", r"__\1__"),
            (r"<(?:s|del|strike)>(.*?)</(?:s|del|strike)>", r"~~\1~~"),
            (r"<code>(.*?)</code>", r"`\1`"),
            (r"<pre>(.*?)</pre>", r"```\1```"),
            (r"<blockquote>(.*?)</blockquote>", r"> \1"),
            (r"<tg-spoiler>(.*?)</tg-spoiler>", r"||\1||"),
        ],
        "sms": [
            (r"<(?:b|strong|i|em|u|ins|s|del|strike|code|pre|blockquote|"
             r"tg-spoiler)>(.*?)</(?:b|strong|i|em|u|ins|s|del|strike|"
             r"code|pre|blockquote|tg-spoiler)>", r"\1"),
        ],
    }
    for pat, rep in subs.get(target, subs["sms"]):
        out = re.sub(pat, rep, out, flags=re.DOTALL | re.IGNORECASE)
    out = re.sub(r"<[^>]+>", "", out)  # any tag left: drop it, keep text
    return html.unescape(out)


def _to_telegram(text: str) -> str:
    """WhatsApp/Discord markdown -> Telegram HTML (escaping the rest)."""
    placeholders: dict[str, str] = {}

    def _stash(markup: str) -> str:
        key = f"\x00TG{len(placeholders)}\x00"
        placeholders[key] = markup
        return key

    out = _MD_FENCE_RE.sub(
        lambda m: _stash(f"<pre>{html.escape(m.group(0)[3:-3])}</pre>"),
        text)
    out = _MD_CODE_RE.sub(
        lambda m: _stash(f"<code>{html.escape(m.group(0)[1:-1])}</code>"),
        out)
    out = _MD_BOLD_RE.sub(
        lambda m: _stash(f"<b>{html.escape(m.group(0)[2:-2])}</b>"), out)
    out = _MD_UNDER_RE.sub(
        lambda m: _stash(f"<u>{html.escape(m.group(0)[2:-2])}</u>"), out)
    out = _MD_STRIKE_RE.sub(
        lambda m: _stash(f"<s>{html.escape(m.group(0)[2:-2])}</s>"), out)
    out = _MD_SINGLE_STAR_RE.sub(
        lambda m: _stash(f"<b>{html.escape(m.group(1))}</b>"), out)
    out = _MD_LINK_RE.sub(
        lambda m: _stash(
            f"<a href=\"{html.escape(m.group(0).split('](')[1][:-1], quote=True)}\">"
            f"{html.escape(m.group(0).split('](')[0][1:])}</a>"), out)
    out = html.escape(out)
    for key, markup in placeholders.items():
        out = out.replace(html.escape(key), markup)
    return out


def convert(text: str, platform: str) -> str:
    """Transcode ``text``'s markup into ``platform``'s native markup."""
    p = _plat(platform)
    text = text or ""
    if p in ("telegram", "telegram-bot"):
        if "<" in text and _TAG_RE.search(text):
            return text  # already Telegram HTML — leave it alone
        return _to_telegram(text)
    if p in ("whatsapp", "discord", "sms"):
        if "<" in text and _TAG_RE.search(text):
            return _html_to_target(text, p)
        if p == "sms":
            # Markdown -> plain for SMS.
            out = _MD_FENCE_RE.sub(lambda m: m.group(0)[3:-3], text)
            out = _MD_CODE_RE.sub(lambda m: m.group(0)[1:-1], out)
            out = _MD_BOLD_RE.sub(lambda m: m.group(0)[2:-2], out)
            out = _MD_UNDER_RE.sub(lambda m: m.group(0)[2:-2], out)
            out = _MD_STRIKE_RE.sub(lambda m: m.group(0)[2:-2], out)
            out = _MD_SINGLE_STAR_RE.sub(lambda m: m.group(1), out)
            out = _MD_LINK_RE.sub(
                lambda m: f"{m.group(0).split('](')[0][1:]} "
                          f"({m.group(0).split('](')[1][:-1]})", out)
            return out
        if p == "whatsapp" and ("**" in text or "__" in text
                                or "~~" in text):
            # Discord-style -> WhatsApp-style.
            out = _MD_BOLD_RE.sub(lambda m: f"*{m.group(0)[2:-2]}*", text)
            out = _MD_UNDER_RE.sub(lambda m: f"_{m.group(0)[2:-2]}_", out)
            out = _MD_STRIKE_RE.sub(lambda m: f"~{m.group(0)[2:-2]}~", out)
            return out
        if p == "discord" and _MD_SINGLE_STAR_RE.search(text):
            # WhatsApp *bold* -> Discord **bold** (single * = italic
            # in Discord, so a bare *span* would change meaning).
            return _MD_SINGLE_STAR_RE.sub(
                lambda m: f"**{m.group(1)}**", text)
        return text
    return re.sub(r"<[^>]+>", "", text)  # plain surfaces: strip tags


def sanitize(text: str, platform: str) -> tuple[str, list[str]]:
    """Transcode ``text`` for ``platform`` + verify: (clean, violations)."""
    clean = convert(text, platform)
    return clean, check_message(platform, clean)


def conformance_report(text: str) -> dict[str, list[str]]:
    """Violations per platform for one ``text`` — the cross-check table."""
    return {p: check_message(p, text) for p in KNOWN_PLATFORMS}


def _unused() -> Any:  # pragma: no cover - keeps typing import used
    return None
