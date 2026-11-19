"""PDF primitives (Craft-Libs port of the ``pdf`` crate).

The shared PDF surface the organs use: which link targets may
become clickable links, the input bounds, the document metadata
and the writer/reader bridge. Writing goes through Devon's real
PDF writer (``nomorals/core/pdf.py``); reading goes through the
real COS engine (``nomorals/documents/pdfcraft/``). No new writer
was built: the organs already had one.
"""

from __future__ import annotations

__all__ = [
    "MAX_INPUT_BYTES",
    "Metadata",
    "check_input_size",
    "read_pdf",
    "safe_url",
    "uri_target",
    "write_pdf",
]

#: The largest input the reader accepts (upstream ``MAX_INPUT_BYTES``).
MAX_INPUT_BYTES = 512 * 1024 * 1024


def check_input_size(size: int) -> None:
    """Refuse an input over :data:`MAX_INPUT_BYTES`."""
    if size > MAX_INPUT_BYTES:
        raise ValueError(
            f"PDF input is {size} bytes "
            f"(the limit is {MAX_INPUT_BYTES})")


def safe_url(url: str) -> bool:
    """Whether a link target from a (possibly untrusted) document
    may be opened by whoever opens the file: ``http``/``https``
    URLs with a host, and ``mailto`` with an address.

    Everything else is refused: ``file:``, ``javascript:``,
    ``data:``, relative paths, network shares (``\\\\host\\share``,
    ``//host/share``, ``smb:``) and other schemes, which an
    operating system may hand to a protocol handler. Leading and
    trailing whitespace is ignored; schemes compare
    case-insensitively.
    """
    text = url.strip()
    if ":" not in text:
        return False
    scheme, rest = text.split(":", 1)
    scheme = scheme.lower()
    if scheme in ("http", "https"):
        if not rest.startswith("//"):
            return False
        return _host(rest[2:]) != ""
    if scheme == "mailto":
        return _has_recipient(rest)
    return False


def _host(rest: str) -> str:
    """The host of an authority-first URL remainder: before the
    path, query or fragment, after any userinfo, without the
    port."""
    authority = rest
    for sep in ("/", "?", "#", "\\"):
        authority = authority.split(sep, 1)[0]
    host_port = (authority.rsplit("@", 1)[1] if "@" in authority
                 else authority)
    if host_port.startswith("["):
        after = host_port[1:]
        if "]" in after:
            return after.split("]", 1)[0]
        return ""
    return host_port.split(":", 1)[0]


def _has_recipient(rest: str) -> bool:
    """Whether a ``mailto:`` remainder names an address, before
    its headers or in a ``to`` header."""
    if "?" in rest:
        to, headers = rest.split("?", 1)
    else:
        to, headers = rest, ""
    if to.strip():
        return True
    for header in headers.split("&"):
        if "=" in header:
            key, value = header.split("=", 1)
            if key.lower() == "to" and value.strip():
                return True
    return False


class Metadata:
    """A document's metadata: who made it, what it is, when, in
    which language. Only the fields the writer can carry reach
    the file; the rest stays on the record."""

    def __init__(self, creator: str | None = None,
                 producer: str | None = None,
                 title: str | None = None,
                 authors: list | None = None,
                 description: str | None = None,
                 keywords: list | None = None,
                 created=None, modified=None,
                 language: str | None = None) -> None:
        self.creator = creator
        self.producer = producer
        self.title = title
        self.authors = list(authors or [])
        self.description = description
        self.keywords = list(keywords or [])
        self.created = created
        self.modified = modified
        self.language = language

    def to_dict(self) -> dict:
        data: dict = {}
        if self.creator is not None:
            data["creator"] = self.creator
        if self.producer is not None:
            data["producer"] = self.producer
        if self.title is not None:
            data["title"] = self.title
        if self.authors:
            data["authors"] = list(self.authors)
        if self.description is not None:
            data["description"] = self.description
        if self.keywords:
            data["keywords"] = list(self.keywords)
        if self.created is not None:
            data["created"] = self.created
        if self.modified is not None:
            data["modified"] = self.modified
        if self.language is not None:
            data["language"] = self.language
        return data


def write_pdf(text: str, metadata: Metadata | None = None) -> bytes:
    """Write ``text`` as a PDF. ``metadata`` carries the title,
    creator and friends into the file's info dictionary."""
    from nomorals.core.pdf import render_pdf

    info: dict = {}
    if metadata is not None:
        if metadata.title is not None:
            info["title"] = metadata.title
        if metadata.creator is not None:
            info["creator"] = metadata.creator
        if metadata.authors:
            info["author"] = ", ".join(metadata.authors)
        if metadata.description is not None:
            info["subject"] = metadata.description
        if metadata.keywords:
            info["keywords"] = ", ".join(metadata.keywords)
    return render_pdf(text, metadata=info or None)


def read_pdf(data: bytes) -> dict:
    """Read a PDF back: how many pages, its text, and its title.
    Encrypted documents are refused by the COS engine."""
    check_input_size(len(data))
    from nomorals.core.pdf import read_pdf_text
    from nomorals.documents.pdfcraft.model import PdfDocument

    document = PdfDocument.from_bytes(data)
    return {"pages": len(document.pages),
            "text": read_pdf_text(data),
            "title": document.info.get("Title", ""),
            "info": dict(document.info)}


def uri_target(url: str) -> str:
    """The link target a written PDF may carry: the URL itself
    when :func:`safe_url` allows it, else a refusal."""
    if not safe_url(url):
        raise ValueError(f"not a safe link target: {url!r}")
    return url.strip()
