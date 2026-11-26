"""PhotoCraft testkit — the shared test support for the photo organ.

Ports storytold/photocraft's ``crates/testkit`` habit into Devon's
idiom: one module every photo test imports, instead of each test
file re-defining the same oracle helpers. What it carries:

* fixture paths — the vendored fixtures
  (``tests/fixtures/photocraft/``) and the vendored corpus
  (``tests/fixtures/photocraft-corpus/``, lane 118);
* the oracle helpers — ``read_bytes``, ``max_diff``,
  ``mean_diff``, ``flatten_vs_composite``,
  ``mean_diff_vs_composite`` — the same measurements the oracle
  itself uses, so a test's "within a few levels of Photoshop"
  means the same thing everywhere;
* ``require_corpus`` — corpus absence is a test FAILURE naming
  the missing file, never a silent skip (census §A.25);
* ``Sweep`` — the long-sweep splitter from
  ``crates/testkit/src/sweep.rs``: a run covers one of ``parts``
  slices, chosen by the UTC date, so every case runs at least
  once every ``parts`` days; ``NM_PHOTOCRAFT_FULL_SWEEP=1`` runs
  every case, ``NM_PHOTOCRAFT_SWEEP_PART=<n>`` replays slice
  ``n``; the parts cover every case exactly once;
* the full-corpus paths — ``default_full_corpus_dir`` /
  ``full_corpus_files`` for the 258-file mining-checkout corpus
  the extended oracle run covers (the vendored 50 always run;
  the full 258 runs sliced by ``Sweep``).

Upstream's ``testkit`` also carries a GPU lock and perf
benchmarks; Devon has no GPU rung here, and the perf habit is
carried by the oracle tests reporting their elapsed time, not
by a benchmark harness.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Any

__all__ = [
    "CORPUS",
    "FIXTURES",
    "Sweep",
    "default_full_corpus_dir",
    "flatten_vs_composite",
    "full_corpus_files",
    "max_diff",
    "mean_diff",
    "mean_diff_vs_composite",
    "read_bytes",
    "require_corpus",
]

#: Vendored fixtures (lane 76) — always run.
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "photocraft"
#: Vendored corpus (lane 118) — always run; 50 PSDs.
CORPUS = Path(__file__).resolve().parent / "fixtures" / "photocraft-corpus"

#: The mining checkout's full corpus, when present on this box.
_DEFAULT_FULL_CORPUS = Path.home() / "workspace" / "echelon-mining" \
    / "repos" / "storytold" / "photocraft-corpus" / "photoshop"

#: Default sweep width: one slice per day of the week, so the
#: full corpus is covered every seven days.
DEFAULT_SWEEP_PARTS = 7


def read_bytes(path: Path) -> bytes:
    """A file's bytes — the one way tests read fixtures."""
    return Path(path).read_bytes()


def max_diff(a: Any, b: Any) -> int:
    """Max per-channel absolute difference between two RGBA images.

    A shape mismatch is a failure, not a number: it raises
    instead of returning a sentinel, so a wrong-sized render can
    never pass as "differs by 255".
    """
    import numpy as np

    x = np.asarray(a.convert("RGBA")).astype(int)
    y = np.asarray(b.convert("RGBA")).astype(int)
    if x.shape != y.shape:
        raise AssertionError(
            f"shape mismatch {x.shape} vs {y.shape}")
    return int(abs(x - y).max()) if x.size else 0


def mean_diff(a: Any, b: Any) -> float:
    """Mean per-channel absolute difference, in levels (0..255).

    The honest metric for renders whose mean is rounding-level
    but whose max has a few outlier pixels at sharp edges and
    step boundaries. A shape mismatch raises, as in
    :func:`max_diff`.
    """
    import numpy as np

    x = np.asarray(a.convert("RGBA")).astype(int)
    y = np.asarray(b.convert("RGBA")).astype(int)
    if x.shape != y.shape:
        raise AssertionError(
            f"shape mismatch {x.shape} vs {y.shape}")
    return float(abs(x - y).mean()) if x.size else 0.0


def flatten_vs_composite(path: Path) -> int:
    """The oracle in miniature for one PSD file: Devon's flatten
    against Photoshop's own merged composite, max per-channel
    difference in levels."""
    from nomorals.media.photo_editor.psd import PsdFile

    psd = PsdFile.from_bytes(read_bytes(path))
    return max_diff(psd.flatten_layers(), psd.composite_rgba())


def mean_diff_vs_composite(path: Path) -> float:
    """The oracle in miniature, mean metric: Devon's flatten
    against Photoshop's own merged composite, mean per-channel
    difference in levels."""
    from nomorals.media.photo_editor.psd import PsdFile

    psd = PsdFile.from_bytes(read_bytes(path))
    return mean_diff(psd.flatten_layers(), psd.composite_rgba())


def require_corpus(paths: list[Path]) -> None:
    """Fail naming the first missing corpus file — never skip
    when a corpus file is gone."""
    for path in paths:
        path = Path(path)
        if not path.is_file():
            raise AssertionError(f"missing corpus fixture {path}")


def default_full_corpus_dir() -> str:
    """The full 258-file corpus: ``NM_PHOTOCRAFT_CORPUS`` wins,
    else the mining checkout on this box."""
    return os.environ.get("NM_PHOTOCRAFT_CORPUS",
                          str(_DEFAULT_FULL_CORPUS))


def full_corpus_files(
        corpus_dir: str | os.PathLike[str] | None = None) -> list[Path]:
    """Every ``.psd`` under the full corpus dir, sorted. Empty
    when the dir is absent — the caller decides whether that is
    a skip or a failure (requested runs fail, unrequested skip)."""
    root = Path(corpus_dir or default_full_corpus_dir())
    if not root.is_dir():
        return []
    return sorted(root.rglob("*.psd"))


def _mix(x: int) -> int:
    """splitmix64's finalizer — the exact hash upstream's
    ``Sweep`` slices by, so a slice here is the same slice
    there."""
    mask = (1 << 64) - 1
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9 & mask
    x = (x ^ (x >> 27)) * 0x94D049BB133111EB & mask
    return (x ^ (x >> 31)) & mask


class Sweep:
    """The cases of a long sweep that this run covers.

    By default a run takes one of ``parts`` slices, chosen by
    the UTC date, so every case runs at least once every
    ``parts`` days. ``NM_PHOTOCRAFT_FULL_SWEEP=1`` runs every
    case, and ``NM_PHOTOCRAFT_SWEEP_PART=<n>`` replays slice
    ``n``. Cases go to slices by a hash of their index, so each
    slice mixes every position of the sweep instead of taking
    every ``parts``-th one — and the parts cover every case
    exactly once.
    """

    def __init__(self, parts: int = DEFAULT_SWEEP_PARTS) -> None:
        if os.environ.get("NM_PHOTOCRAFT_FULL_SWEEP") == "1":
            self.part = 0
            self.parts = 1
            return
        parts = max(1, int(parts))
        raw = os.environ.get("NM_PHOTOCRAFT_SWEEP_PART")
        if raw is not None:
            try:
                part = int(raw)
            except ValueError:
                part = -1
            if part < 0 or part >= parts:
                raise ValueError(
                    f"NM_PHOTOCRAFT_SWEEP_PART={raw!r} is not a "
                    f"part number below {parts}")
            self.part = part
            self.parts = parts
            return
        today = datetime.datetime.now(datetime.UTC).date().toordinal()
        self.part = today % parts
        self.parts = parts

    def includes(self, i: int) -> bool:
        """Whether case ``i`` runs in this slice."""
        return self.parts == 1 or _mix(int(i)) % self.parts == self.part

    def describe(self) -> str:
        """The slice this run covers, for the test output."""
        if self.parts == 1:
            return "full sweep (NM_PHOTOCRAFT_FULL_SWEEP=1)"
        return (f"sweep part {self.part} of {self.parts}: replay "
                f"with NM_PHOTOCRAFT_SWEEP_PART={self.part}, run "
                f"every part with NM_PHOTOCRAFT_FULL_SWEEP=1")
