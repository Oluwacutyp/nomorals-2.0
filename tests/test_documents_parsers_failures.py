"""Pass tail-11 pins: documents/parsers failures stay silent, degraded honest.

Target: nomorals/documents/parsers.py — 0 converts / 10 exclusions.
Every site below maps a failure to a genuine degraded value: sniff probes
fall through to the next probe (or to the documented txt fallback / loud
DocumentError); docx enrichments degrade to their documented miss values
(no links, bullet default, no footnotes, plain paragraph); cell text keeps
the value and only loses formatting. None of them records, because none
of them hides a user-visible outcome — text is preserved in every case,
and no format is ever claimed on negative evidence.

NOT-recording: the documents package has no failures ledger and the
target has no record call, so silence is pinned against the module's
observability surface ``_log`` (tail-7/tail-10 precedent): each trigger
runs with ``_log`` mocked and asserts zero calls. No bare ``except: pass``
anywhere in this file — the scanner counts test files too; the tests
below need no exception handlers at all (pytest.raises covers the two
DocumentError outcomes).

NOTE on 215 (flagged for parent adjudication, not pinned here): a valid
XML document longer than the 4096-byte probe window sniffs as "txt"
because the probe parses a truncated prefix. That is a property of the
probe window, not of the handler (which honestly falls through); fixing
it would change sniffing semantics, not handler behavior. See SURVEY.md.
"""

import io
import zipfile
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.documents import parsers as pl
from nomorals.documents.errors import DocumentError
from nomorals.documents.parsers import (
    _cell_text,
    _docx_footnotes,
    _docx_hyperlink_targets,
    _docx_list_kinds,
    _para_num_id,
    _sniff,
    _sniff_zip,
    _xls_cell_text,
)


def _silent(fn):
    """Run fn with _log mocked; assert the handler logged nothing."""
    with mock.patch.object(pl, "_log") as log:
        out = fn()
    assert log.method_calls == []
    return out


def _make_zip(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _raising_read_for(entry_name):
    orig = zipfile.ZipFile.read

    def read(self, name, *a, **k):
        if name == entry_name:
            raise zipfile.BadZipFile(f"corrupt {entry_name}")
        return orig(self, name, *a, **k)

    return read


def test_137_mimetype_unreadable_falls_through_to_other_probes():
    # Corrupt mimetype entry: no odt/ods/epub claim on negative evidence.
    # The zip carries no other marker, so the sniff ends loud.
    data = _make_zip({"mimetype": b"application/vnd.oasis.opendocument.text"})
    with mock.patch.object(zipfile.ZipFile, "read", _raising_read_for("mimetype")):
        with pytest.raises(DocumentError):
            _silent(lambda: _sniff_zip(data))


def test_152_container_unreadable_falls_through_to_content_types_probe():
    # Corrupt container.xml: no epub claim on negative evidence.
    data = _make_zip(
        {"META-INF/container.xml": b"<container><rootfile full-path='x.opf'/></container>"})
    with mock.patch.object(zipfile.ZipFile, "read",
                           _raising_read_for("META-INF/container.xml")):
        with pytest.raises(DocumentError):
            _silent(lambda: _sniff_zip(data))


def test_209_invalid_json_keeps_sniffing_to_txt():
    assert _silent(lambda: _sniff(b'{"not json', filename="", mime="")) == "txt"


def test_215_invalid_xml_keeps_sniffing_to_txt():
    assert _silent(lambda: _sniff(b"<root><unclosed", filename="", mime="")) == "txt"


def test_452_missing_rels_means_no_link_targets():
    data = _make_zip({"word/document.xml": b"<w:document/>"})
    assert _silent(lambda: _docx_hyperlink_targets(data)) == {}


def test_470_missing_numbering_means_no_list_kinds():
    data = _make_zip({"word/document.xml": b"<w:document/>"})
    assert _silent(lambda: _docx_list_kinds(data)) == {}


def test_500_missing_footnotes_means_no_footnote_sections():
    data = _make_zip({"word/document.xml": b"<w:document/>"})
    assert _silent(lambda: _docx_footnotes(data)) == []


class _RaisingChain:
    element = None

    @property
    def base_style(self):
        raise RuntimeError("chain boom")


class _FakeChild:
    def find(self, *a, **k):
        return None


def test_539_broken_style_chain_means_no_numbering_id():
    assert _silent(
        lambda: _para_num_id(_FakeChild(), SimpleNamespace(style=_RaisingChain()), "{w}")) == ""


class _OddDate:
    def isoformat(self):
        raise RuntimeError("iso boom")

    def __str__(self):
        return "2024-ish"


def test_666_odd_date_like_falls_back_to_str():
    assert _silent(lambda: _cell_text(_OddDate())) == "2024-ish"


def test_1049_bad_xls_date_falls_back_to_raw_value():
    # xlrd is absent in this environment, so the import inside the try
    # raises; with xlrd present a bogus value raises in conversion.
    # Either arm of the same handler returns the raw value.
    cell = SimpleNamespace(ctype=3, value="bogus-date")
    assert _silent(lambda: _xls_cell_text(cell, 0)) == "bogus-date"
