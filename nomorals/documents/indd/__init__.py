"""Indd-Utils port — InDesign (.indd) → IDML conversion with measured
fidelity, in Devon's documents organ.

A faithful port of storytold/indd-utils (dual MIT/Apache-2.0, commit
136f973): the binary container (4096-byte pages, two master pages),
the little-endian database (logical directory + three B+ trees), the
object cursor, the document model, and the IDML writer (designmap,
fonts, styles, preferences, backing story, tags, spreads, stories,
graphic — a real .idml ZIP package), plus the measured-fidelity differ
(``tools/compare.py``: paired-file diffing, value coverage % as the
headline). Devon already had a measured-fidelity corpus
(:mod:`nomorals.documents.corpus`); this organ adds what it lacked:
the INDD reader and the IDML writer itself. Attribution in
``nomorals/documents/inddutils-attribution.toml``; machine-checked
parity in ``nomorals/documents/inddutils-parity.toml``.

Entry points: :func:`convert` (INDD bytes → IDML package),
:func:`convert_file` (from disk), :func:`convert_into` (to a file),
:func:`read_header` (header only), :func:`read_document` (model only),
:func:`measure_fidelity` (paired-file diffing, coverage % headline),
:func:`run_parity`, :func:`verify_attribution`.
"""

from __future__ import annotations

from .convert import (
                      Conversion,
                      Preview,
                      convert,
                      convert_file,
                      convert_into,
                      read_document,
                      read_header,
)
from .errors import InddError
from .fidelity import FidelityResult, measure_fidelity

__all__ = ["Conversion", "FidelityResult", "InddError", "Preview",
           "convert", "convert_file", "convert_into", "measure_fidelity",
           "read_document", "read_header"]
