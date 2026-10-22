"""Platform-aware output formatting.

One canonical text in, correct rendering per platform out::

    from nomorals.social.chat.platforms import format_for_platform, detect_platform

    text = format_for_platform(raw, detect_platform(chat_key))

Platform rules (verified against real client behavior):

* **telegram** — HTML (``<b>``, ``<code>``, ``<i>``) with ``parse_mode="HTML"``.
  Canonical ``**bold**`` / `` `code` `` spans are upgraded to HTML and
  everything else is escaped, so literal text can never break the
  entity parse or inject tags. Paired HTML elements a producer
  authored directly pass through verbatim — never double-processed.
* **whatsapp** — ``*bold*``, ``_italic_``, `` ```mono``` ``, ``~strike~``.
  HTML tags are converted (``<b>`` → ``*`` …), unknown tags stripped,
  ``[text](url)`` links flattened to ``text: url`` (WhatsApp auto-links
  bare URLs). No inline buttons, no HTML — channels render plain text.
* **sms** — all markup stripped to plain text.
* **web** — HTML converted to light markdown (``**bold**``, `` `code` ``).
* **discord** — markdown passthrough (``**bold**``, `` `code` ``), HTML
  converted to markdown.

Every function never raises: on any failure the input is returned
unchanged (or best-effort stripped).
"""

from __future__ import annotations

import html as _html
import re
from typing import Any

from .platform_limits import for_platform as _for_platform

__all__ = [
    "TELEGRAM", "WHATSAPP", "SMS", "WEB", "DISCORD", "LOCAL",
    "PLATFORM_LIMITS",
    "detect_platform",
    "format_for_platform",
    "to_whatsapp", "to_telegram", "to_plain", "to_markdown",
    "chunk_text",
    "strip_html",
    "escape_markdown_v2", "link", "code_block", "mention",
]

TELEGRAM = "telegram"
WHATSAPP = "whatsapp"
SMS = "sms"
WEB = "web"
DISCORD = "discord"
LOCAL = "local"

#: Per-message character budgets per platform — derived from the
#: :class:`PlatformLimits` descriptor table (``platform_limits.py``), the
#: single source of truth for platform caps. Callers that want a
#: conservative rendering budget below the hard cap pass their own limit
#: to :func:`chunk_text`.
PLATFORM_LIMITS: dict[str, int] = {
    name: _for_platform(name).max_text
    for name in (WHATSAPP, TELEGRAM, DISCORD, SMS, WEB, LOCAL)
}

#: chat_key prefixes (``platform:chat_id``) → canonical platform.
_PREFIX_MAP: dict[str, str] = {
    "telegram": TELEGRAM, "tgbot": TELEGRAM, "tg": TELEGRAM,
    "whatsapp": WHATSAPP, "wa": WHATSAPP,
    "sms": SMS, "twilio": SMS,
    "web": WEB, "webapp": WEB,
    "discord": DISCORD,
    "local": LOCAL, "console": LOCAL, "cli": LOCAL,
}


def detect_platform(chat_key: Any) -> str:
    """Canonical platform for a ``platform:chat_id`` key. Never raises."""
    try:
        prefix = str(chat_key or "").split(":", 1)[0].strip().lower()
        return _PREFIX_MAP.get(prefix, TELEGRAM)
    except Exception:  # noqa: BLE001
        return TELEGRAM


# ── tag conversion ───────────────────────────────────────────────────────────

_BOLD_HTML = re.compile(r"<b>(.*?)</b>", re.S | re.I)
_ITALIC_HTML = re.compile(r"<i>(.*?)</i>", re.S | re.I)
_CODE_HTML = re.compile(r"<code>(.*?)</code>", re.S | re.I)
_LINK_HTML = re.compile(r'<a\s+href="([^"]*)"(?:\s[^>]*)?>(.*?)</a>', re.S | re.I)
_ANY_TAG = re.compile(r"<[^>]+>")
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
#: lone `code` — not adjacent to another backtick (leaves ``` blocks alone)
_MD_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def strip_html(text: str) -> str:
    """Remove all HTML tags, unescape entities. Never raises."""
    try:
        return _html.unescape(_ANY_TAG.sub("", str(text or "")))
    except Exception:  # noqa: BLE001
        return str(text or "")


def to_plain(text: str) -> str:
    """Everything → plain readable text. Never raises."""
    try:
        out = strip_html(text)
        out = _MD_LINK.sub(r"\1: \2", out)
        out = _MD_BOLD.sub(r"\1", out)
        out = _MD_CODE.sub(r"\1", out)
        # whatsapp-style leftovers
        out = re.sub(r"\*([^*\n]+)\*", r"\1", out)
        out = re.sub(r"_([^_\n]+)_", r"\1", out)
        out = re.sub(r"```([^`]+)```", r"\1", out)
        out = re.sub(r"~([^~\n]+)~", r"\1", out)
        return out
    except Exception:  # noqa: BLE001
        return str(text or "")


def to_whatsapp(text: str) -> str:
    """Canonical/Telegram-HTML text → WhatsApp markdown. Never raises."""
    try:
        out = str(text or "")
        # markdown links → "text: url" (WhatsApp auto-links bare URLs)
        out = _LINK_HTML.sub(r"\2: \1", out)
        out = _MD_LINK.sub(r"\1: \2", out)
        # canonical **bold** → *bold* (WhatsApp has no **)
        out = _MD_BOLD.sub(r"*\1*", out)
        # lone `code` → ```code``` — do this BEFORE the HTML <code> pass so
        # the triple backticks we emit are not re-matched (```x``` bug)
        out = _MD_CODE.sub(r"```\1```", out)
        # telegram HTML → whatsapp markdown
        out = _BOLD_HTML.sub(r"*\1*", out)
        out = _ITALIC_HTML.sub(r"_\1_", out)
        out = _CODE_HTML.sub(r"```\1```", out)
        out = _ANY_TAG.sub("", out)  # any other tag: drop, keep content
        return _html.unescape(out)
    except Exception:  # noqa: BLE001
        return str(text or "")


#: Paired Telegram-HTML elements a producer may have authored directly
#: (``social/chat/style.py`` helpers emit these). They pass through
#: :func:`to_telegram` verbatim — their inner text is already markup.
_HTML_SPAN = re.compile(
    r"<(b|strong|i|em|u|ins|s|strike|del|code|pre|blockquote|tg-spoiler)"
    r"(\s[^>]*)?>.*?</\1>",
    re.S | re.I)
_HTML_LINK = re.compile(
    r"<a\s[^>]*href\s*=\s*\"[^\"]*\"[^>]*>.*?</a>", re.S | re.I)
_MD_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*\n)?(.*?)```", re.S)


def to_telegram(text: str) -> str:
    """Canonical house dialect → Telegram HTML, escaping-correct.

    The house dialect is markdown-ish: ``**bold**``, `` `code` ``,
    ```` ``` ```` fences, ``[text](url)`` links. Everything that is not
    one of those spans — and not a paired Telegram-HTML element a
    producer authored directly — is HTML-escaped, so user-derived
    content interpolated into a title or body can neither break the
    entity parse (a stray ``<`` used to fail the whole Bot API send)
    nor inject tags. Paired HTML elements pass through verbatim.
    Unpaired markers (a stray ``**``) stay literal — content fidelity
    over cleverness. Never raises.
    """
    try:
        out = str(text or "")
        stash: dict[str, str] = {}

        def _keep(html_text: str) -> str:
            key = f"\x00TG{len(stash)}\x00"
            stash[key] = html_text
            return key

        # 1. Producer-authored HTML elements pass through untouched.
        out = _HTML_LINK.sub(lambda m: _keep(m.group(0)), out)
        out = _HTML_SPAN.sub(lambda m: _keep(m.group(0)), out)
        # 2. Code first: its content is literal, never markdown.
        out = _MD_FENCE.sub(
            lambda m: _keep(
                f"<pre>{_html.escape(m.group(1))}</pre>"), out)
        out = _MD_CODE.sub(
            lambda m: _keep(
                f"<code>{_html.escape(m.group(1))}</code>"), out)
        # 3. Links: label may itself carry **bold**; the URL is escaped
        #    for the attribute.
        def _link(m: re.Match[str]) -> str:
            label = _MD_BOLD.sub(r"<b>\1</b>", _html.escape(m.group(1)))
            url = _html.escape(m.group(2), quote=True)
            return _keep(f'<a href="{url}">{label}</a>')

        out = _MD_LINK.sub(_link, out)
        # 4. Bold spans (content escaped — user text inside a banner
        #    title is still user text).
        out = _MD_BOLD.sub(
            lambda m: _keep(f"<b>{_html.escape(m.group(1))}</b>"), out)
        # 5. Everything left is literal text: escape it.
        out = _html.escape(out)
        # 6. Restore the stashed markup, outermost first: a span
        #    stashed later may contain an earlier span's key inside it
        #    (code inside bold), and that inner key is only reachable
        #    once the outer span is back in the text.
        for key in reversed(list(stash)):
            out = out.replace(key, stash[key])
        return out
    except Exception:  # noqa: BLE001
        return str(text or "")


def to_markdown(text: str) -> str:
    """HTML → light markdown (web/discord). Never raises."""
    try:
        out = str(text or "")
        out = _LINK_HTML.sub(r"[\2](\1)", out)
        out = _BOLD_HTML.sub(r"**\1**", out)
        out = _ITALIC_HTML.sub(r"_\1_", out)
        out = _CODE_HTML.sub(r"`\1`", out)
        out = _ANY_TAG.sub("", out)
        return _html.unescape(out)
    except Exception:  # noqa: BLE001
        return str(text or "")


def format_for_platform(text: str, platform: str) -> str:
    """Adapt text for a platform. Never raises; unknown platform → telegram."""
    try:
        p = (platform or TELEGRAM).strip().lower()
        if p == WHATSAPP:
            return to_whatsapp(text)
        if p == SMS:
            return to_plain(text)
        if p in (WEB, DISCORD):
            return to_markdown(text)
        return to_telegram(text)
    except Exception:  # noqa: BLE001
        return str(text or "")


# MarkdownV2's reserved set — every one must be escaped with a backslash.
# HTML parse_mode is preferred (escape surface is just <>&), but when a
# caller needs MarkdownV2, this is the correct escape. Never raises.
_MD_V2_SPECIAL = re.compile(r"([_*\[\]()~`>#+\-=|{}.!])")


def escape_markdown_v2(text: str) -> str:
    """Escape text for Telegram's MarkdownV2 parse mode.

    Forgetting this is the #1 silent-failure trap in Telegram bots — an
    unescaped ``_`` or ``[`` makes the whole message fail to parse.
    Prefer HTML (``to_telegram``) for new code; use this for legacy
    MarkdownV2 surfaces.
    """
    try:
        return _MD_V2_SPECIAL.sub(r"\\\1", str(text or ""))
    except Exception:  # noqa: BLE001
        return str(text or "")


def link(label: str, url: str, platform: str = TELEGRAM) -> str:
    """A clickable link in the platform's native syntax. Never raises."""
    try:
        label, url = str(label or ""), str(url or "")
        p = (platform or TELEGRAM).strip().lower()
        if p == WHATSAPP:
            return f"{label}: {url}"  # WhatsApp auto-links bare URLs
        if p == SMS:
            return f"{label}: {url}"
        if p in (WEB, DISCORD):
            return f"[{label}]({url})"
        return f'<a href="{_html.escape(url, quote=True)}">{_html.escape(label)}</a>'
    except Exception:  # noqa: BLE001
        return str(label or "")


def code_block(text: str, language: str = "", platform: str = TELEGRAM) -> str:
    """A fenced code block in the platform's native syntax. Never raises."""
    try:
        body = str(text or "")
        p = (platform or TELEGRAM).strip().lower()
        if p == TELEGRAM:
            lang = f' class="language-{_html.escape(language)}"' if language else ""
            return f"<pre><code{lang}>{_html.escape(body)}</code></pre>"
        if p == WHATSAPP:
            return f"```{body}```"
        if p == SMS:
            return body
        fence = f"```{language}" if language else "```"
        return f"{fence}\n{body}\n```"
    except Exception:  # noqa: BLE001
        return str(text or "")


def mention(name: str, user_id: str = "", platform: str = TELEGRAM) -> str:
    """An @mention that actually pings on the platform. Never raises."""
    try:
        name = str(name or "")
        uid = str(user_id or "")
        p = (platform or TELEGRAM).strip().lower()
        if p == TELEGRAM and uid:
            return f'<a href="tg://user?id={_html.escape(uid, quote=True)}">{_html.escape(name)}</a>'
        if p == DISCORD and uid:
            return f"<@{uid}>"
        return f"@{name.lstrip('@')}" if name else ""
    except Exception:  # noqa: BLE001
        return str(name or "")


def chunk_text(text: str, limit: int = 3000) -> list[str]:
    """Split text so no chunk exceeds *limit*.

    Delegates to the canonical lossless chunker
    (:mod:`nomorals.social.chat.chunking`): paragraph → sentence → hard
    boundary, never truncates, ``"".join(chunks) == text``. The ``limit``
    here is a caller-chosen rendering budget, not a platform cap — the
    platform caps live in :mod:`platform_limits`.
    """
    from .chunking import chunk_message

    try:
        return chunk_message(str(text or ""), max_text=limit)
    except Exception:  # noqa: BLE001
        return [str(text or "") or text] if text else [""]
