"""Errors of the INDD reader (``src/lib.rs``: ``Error``).

A malformed input gives an error or a warning, never a crash: the
kinds below mirror upstream's enum exactly, including the payloads
that name the offset or the count that failed.
"""

from __future__ import annotations


class InddError(Exception):
    """One of upstream's ``Error`` variants, as a Python exception.

    ``kind`` is the variant name in snake_case (``not_indd``,
    ``truncated``, ``unknown_byte_order``, ``bad_contig_object``,
    ``corrupt``, ``unsupported``, ``no_database``); ``detail`` carries
    the numbers upstream's message carries (needed/got bytes, the
    offset, the message).
    """

    def __init__(self, kind: str, detail: str = "") -> None:
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}" if detail else kind)

    @classmethod
    def not_indd(cls) -> InddError:
        return cls("not_indd", "not an INDD file")

    @classmethod
    def truncated(cls, needed: int, got: int) -> InddError:
        return cls("truncated",
                   f"file truncated: need {needed} bytes, got {got}")

    @classmethod
    def unknown_byte_order(cls, flag: int) -> InddError:
        return cls("unknown_byte_order",
                   f"unknown byte-order flag {flag:#04x}")

    @classmethod
    def bad_contig_object(cls, offset: int) -> InddError:
        return cls("bad_contig_object",
                   f"contiguous object at {offset:#x} has no matching "
                   "trailer")

    @classmethod
    def corrupt(cls, message: str) -> InddError:
        return cls("corrupt", f"corrupt database: {message}")

    @classmethod
    def unsupported(cls, what: str) -> InddError:
        return cls("unsupported", f"not supported yet: {what}")

    @classmethod
    def no_database(cls, db_pages: int, directory: int) -> InddError:
        return cls("no_database",
                   f"no object database: the page directory is page "
                   f"{directory}, but the file has only {db_pages} "
                   "database pages")
