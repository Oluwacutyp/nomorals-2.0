"""Dataset preparation: cleaned filelists, context sidecars, test data.

Numpy port of upstream ``preprocess.py``. Upstream needs TorchMoji
weights + TinyBERT + torch just to clean a filelist; here the cleaned
filelist (the part training actually reads) is produced with no optional
dependency at all. Sidecars (``.torchmoji`` / ``.bert``) are built only
when the frontends are available; otherwise they are listed as skipped
in the report and the loader will refuse honestly at training time.
"""

import os

import numpy as np

from . import data as _data
from . import mathutils
from .text import _clean_text, cleaned_text_to_sequence

TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "In a galaxy far, far away, a young hero embarks on an epic adventure",
    "Ladies and gentlemen, welcome to the annual science fair!",
    "Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
    "The crowd erupted in cheers as the team scored the winning goal in the final seconds of the match",
]


def clean_filelist(filelist, text_index=1, text_cleaners=("arpa_cleaners",),
                   out_extension="cleaned"):
    """Write ``<filelist>.<out_extension>`` with cleaned text.

    In dual ARPA mode (upstream behaviour) every row is written twice:
    once with ``arpa_cleaners`` and once with ``arpa_precleaners``.
    Returns (out_path, rows_written).
    """
    rows = _data.load_filepaths_and_text(filelist)
    dual = "arpa_cleaners" in text_cleaners
    out_path = filelist + "." + out_extension
    new_rows = []
    for row in rows:
        original_text = row[text_index]
        cleaned = _clean_text(original_text, list(text_cleaners))
        new_row = list(row)
        new_row[text_index] = cleaned
        new_rows.append(new_row)
        if dual:
            # Upstream's comment: a plain assignment would alias the list,
            # so the second row is an explicit copy.
            second = list(row)
            second[text_index] = _clean_text(original_text, ["arpa_precleaners"])
            new_rows.append(second)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in new_rows:
            f.write("|".join(row) + "\n")
    return out_path, len(new_rows)


def build_sidecars(filelist, text_index=1, skip_existing=True):
    """Build ``.torchmoji`` / ``.bert`` sidecars for every row.

    Returns {"built": [...], "skipped": [...], "missing_frontend": bool}.
    When neither frontend is available nothing is built and the reason is
    recorded instead of failing — the trainer refuses later, naming the
    exact missing piece.
    """
    from .bertfe import BERTFrontEnd
    from .moji import TorchMoji

    rows = _data.load_filepaths_and_text(filelist)
    report = {"built": [], "skipped": [], "missing_frontend": False}
    try:
        moji = TorchMoji()
    except (ImportError, FileNotFoundError) as exc:
        report["missing_frontend"] = True
        report["missing_reason"] = f"TorchMoji unavailable: {exc}"
        moji = None
    try:
        bert = BERTFrontEnd()
    except (ImportError, OSError) as exc:
        report["missing_frontend"] = True
        report["missing_reason"] = f"TinyBERT unavailable: {exc}"
        bert = None
    if moji is None and bert is None:
        report["skipped"] = [row[0] for row in rows]
        return report
    for row in rows:
        text = row[text_index]
        wav = row[0]
        if moji is not None:
            out = wav.replace(".wav", ".torchmoji")
            if not skip_existing or not os.path.exists(out):
                arr = moji(text)
                np.savez_compressed(out + ".npz", embedding=np.asarray(arr))
                report["built"].append(out)
        if bert is not None:
            out = wav.replace(".wav", ".bert")
            if not skip_existing or not os.path.exists(out):
                emb, _ = bert.infer(text)
                np.savez_compressed(out + ".npz", embedding=np.asarray(emb))
                report["built"].append(out)
    return report


def build_test_preproc(sentences, text_cleaners, out_dir,
                       add_blank=True):
    """Write one test entry per sentence (upstream ``test_preproc``).

    Each entry holds the interspersed text ids plus the moji/bert sidecars
    when the frontends are available; without them the entry holds the
    text ids alone and records the gap.
    """
    from .bertfe import BERTFrontEnd
    from .moji import TorchMoji

    os.makedirs(out_dir, exist_ok=True)
    try:
        moji = TorchMoji()
    except (ImportError, FileNotFoundError):
        moji = None
    try:
        bert = BERTFrontEnd()
    except (ImportError, OSError):
        bert = None
    written = []
    for i, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence:
            continue
        cleaned = _clean_text(sentence, list(text_cleaners))
        ids = cleaned_text_to_sequence(cleaned)
        if add_blank:
            ids = mathutils.intersperse(ids, 0)
        entry = {"text": np.array(ids, dtype=np.int64)}
        if moji is not None:
            entry["moji"] = np.asarray(moji(sentence))
        if bert is not None:
            emb, _ = bert.infer(sentence)
            entry["bert"] = np.asarray(emb)
        path = os.path.join(out_dir, f"test{i}.npz")
        np.savez_compressed(path, **entry)
        written.append(path)
    return written


def preprocess_filelists(filelists, text_cleaners, out_extension="cleaned",
                         build_sidecars_flag=False, test_sentences=None,
                         test_dir=None, text_index=1):
    """Full upstream-equivalent preprocess run; returns a report dict."""
    report = {"cleaned": {}, "sidecars": {}, "test": []}
    for filelist in filelists:
        out_path, n_rows = clean_filelist(
            filelist, text_index=text_index,
            text_cleaners=text_cleaners, out_extension=out_extension)
        report["cleaned"][filelist] = {"out": out_path, "rows": n_rows}
        if build_sidecars_flag:
            report["sidecars"][filelist] = build_sidecars(
                filelist, text_index=text_index)
    if test_dir is not None:
        sentences = test_sentences if test_sentences is not None else TEST_SENTENCES
        report["test"] = build_test_preproc(
            sentences, text_cleaners, test_dir)
    return report
