"""Learned English grapheme-to-phoneme — the Arpabet prediction model.

A Python port of echelon/grapheme_to_phoneme.rs (BSD-4-Clause,
Copyright (c) 2020 Brandon Thomas), itself based on Kyubyong Park
and Jongseok Kim's g2p.py. The Rust library is a prediction tool
that turns graphemes (words) into Arpabet phonemes; it focuses only
on the prediction model (OOV prediction) — CMUDict lookup and
heteronym handling belong elsewhere, which in Devon is
:mod:`nomorals.media.g2p_en` (lexicon first, this model second,
letter-to-sound rules last).

The model is a seq2seq GRU encoder–decoder (hidden 256, embedding
256), trained on CMUdict. The weights ship as ``model/model.npz``
(the same bundle the Rust binary embeds with ``include_bytes!``)
and are carried as data — never invented, never re-fitted here.

Redistribution note (BSD-4-Clause, clause 3): this product
includes software developed by Brandon Thomas (bt@brand.io,
echelon@gmail.com). The full licence text ships alongside as
``LICENSE`` in this package.

The core keeps the Rust semantics exactly:
- no lowercasing (an uppercase letter maps to ``<unk>`` — callers
  normalise first; :func:`predict_word` does);
- unknown characters map to ``<unk>``;
- the encoder reads the word plus ``</s>`` and the decoder runs at
  most 20 steps, stopping at ``</s>``.

Honest degradation: numpy missing → ``numpy_unavailable``; the
bundle missing or unreadable → ``model_unavailable``; wrong array
shapes → ``model_shape_mismatch``. Nothing here raises on import;
failures surface as :class:`G2PLearnedError` with a ``kind`` name,
or as data from :func:`model_status`.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import numpy as _np
except ImportError:  # noqa: BLE001 - numpy is optional on thin hosts
    _np = None  # type: ignore[assignment]

__all__ = [
    "GRAPHEMES",
    "PHONEME_TABLE",
    "G2PLearnedError",
    "Model",
    "get_model",
    "model_status",
    "predict_word",
]

#: Grapheme vocabulary, in training order: <pad>, <unk>, </s>, a–z.
GRAPHEMES: tuple[str, ...] = ("<pad>", "<unk>", "</s>") + tuple(
    chr(c) for c in range(ord("a"), ord("z") + 1))

#: The 70 Arpabet phonemes, in the order the network was trained on
#: (src/tables.rs ``build_phoneme_string_table`` — the order MUST
#: match training, otherwise the predictions are corrupted).
_ARPABET: tuple[str, ...] = (
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2",
    "AO0", "AO1", "AO2", "AW0", "AW1", "AW2", "AY0", "AY1", "AY2",
    "B", "CH", "D", "DH", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2", "F", "G", "HH", "IH0", "IH1", "IH2",
    "IY0", "IY1", "IY2", "JH", "K", "L", "M", "N", "NG", "OW0",
    "OW1", "OW2", "OY0", "OY1", "OY2", "P", "R", "S", "SH", "T",
    "TH", "UH0", "UH1", "UH2", "UW", "UW0", "UW1", "UW2", "V", "W",
    "Y", "Z", "ZH",
)

#: Full decoder vocabulary: <pad>, <unk>, <s>, </s>, then Arpabet.
PHONEME_TABLE: tuple[str, ...] = ("<pad>", "<unk>", "<s>", "</s>") + _ARPABET

_HIDDEN = 256
_MAX_DECODE_STEPS = 20
_END_INDEX = 3  # </s> in the decoder vocabulary
_START_INDEX = 2  # <s> in the decoder vocabulary
_UNKNOWN_GRAPHEME = 1  # <unk> in the grapheme vocabulary

_EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "enc_emb": (29, 256),
    "enc_w_ih": (768, 256),
    "enc_w_hh": (768, 256),
    "enc_b_ih": (768,),
    "enc_b_hh": (768,),
    "dec_emb": (74, 256),
    "dec_w_ih": (768, 256),
    "dec_w_hh": (768, 256),
    "dec_b_ih": (768,),
    "dec_b_hh": (768,),
    "fc_w": (74, 256),
    "fc_b": (74,),
}

_BUNDLED_PATH = os.path.join(os.path.dirname(__file__), "model",
                             "model.npz")


class G2PLearnedError(Exception):
    """A named failure of the learned G2P rung (never a bare stub)."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.message = message

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}: {self.message}"


def _require_numpy():
    if _np is None:
        raise G2PLearnedError(
            "numpy_unavailable",
            "numpy is not installed, so the learned G2P rung cannot "
            "run — the letter-to-sound rules answer instead")
    return _np


class Model:
    """The GRU encoder–decoder, loaded from an .npz weight bundle."""

    def __init__(self, arrays: dict[str, Any]):
        self.enc_emb = arrays["enc_emb"]
        self.enc_w_ih = arrays["enc_w_ih"]
        self.enc_w_hh = arrays["enc_w_hh"]
        self.enc_b_ih = arrays["enc_b_ih"]
        self.enc_b_hh = arrays["enc_b_hh"]
        self.dec_emb = arrays["dec_emb"]
        self.dec_w_ih = arrays["dec_w_ih"]
        self.dec_w_hh = arrays["dec_w_hh"]
        self.dec_b_ih = arrays["dec_b_ih"]
        self.dec_b_hh = arrays["dec_b_hh"]
        self.fc_w = arrays["fc_w"]
        self.fc_b = arrays["fc_b"]
        self._g2idx = {g: i for i, g in enumerate(GRAPHEMES)}

    @classmethod
    def load(cls, path: str | None = None) -> Model:
        """Load the weights from ``path`` (default: the bundled npz)."""
        np = _require_numpy()
        target = path or _BUNDLED_PATH
        try:
            bundle = np.load(target, allow_pickle=False)
        except (OSError, ValueError, EOFError) as exc:
            raise G2PLearnedError(
                "model_unavailable",
                f"cannot read the learned G2P weights at {target!r}: "
                f"{exc}") from exc
        arrays: dict[str, Any] = {}
        try:
            for name, shape in _EXPECTED_SHAPES.items():
                if name not in bundle.files:
                    raise G2PLearnedError(
                        "model_unavailable",
                        f"weights bundle {target!r} has no array {name!r}")
                arr = bundle[name]
                if tuple(arr.shape) != shape:
                    raise G2PLearnedError(
                        "model_shape_mismatch",
                        f"array {name!r} is {tuple(arr.shape)}, "
                        f"expected {shape}")
                arrays[name] = arr
        finally:
            bundle.close()
        return cls(arrays)

    # ── GRU (PyTorch gate layout: [r, z, n]) ───────────────────────

    @staticmethod
    def _sigmoid(np, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _grucell(self, np, x, h, w_ih, w_hh, b_ih, b_hh):
        rzn_ih = x @ w_ih.T + b_ih
        rzn_hh = h @ w_hh.T + b_hh
        split = rzn_ih.shape[1] * 2 // 3
        rz = self._sigmoid(np, rzn_ih[:, :split] + rzn_hh[:, :split])
        r, z = rz[:, :_HIDDEN], rz[:, _HIDDEN:]
        n = np.tanh(rzn_ih[:, split:] + r * rzn_hh[:, split:])
        return (1.0 - z) * n + z * h

    def _gru_encode(self, np, embedded, steps: int):
        h = np.zeros((1, _HIDDEN), dtype=np.float32)
        outputs = np.zeros((1, steps, _HIDDEN), dtype=np.float32)
        for i in range(steps):
            h = self._grucell(np, embedded[:, i, :], h,
                              self.enc_w_ih, self.enc_w_hh,
                              self.enc_b_ih, self.enc_b_hh)
            outputs[:, i, :] = h
        return outputs

    def predict(self, grapheme: str) -> list[str]:
        """Word → Arpabet phonemes, Rust semantics (no lowercasing).

        Unknown characters map to ``<unk>``; the decoder stops at
        ``</s>`` or after 20 steps. An empty word still runs the
        model (the Rust does the same) — callers that want an empty
        answer for empty input guard it themselves.
        """
        if not isinstance(grapheme, str):
            raise TypeError("grapheme must be a str")
        np = _require_numpy()
        chars = list(grapheme) + ["</s>"]
        encoded = [self._g2idx.get(c, _UNKNOWN_GRAPHEME) for c in chars]
        embedded = np.zeros((1, len(encoded), _HIDDEN),
                            dtype=np.float32)
        for i, idx in enumerate(encoded):
            embedded[0, i, :] = self.enc_emb[idx]
        enc = self._gru_encode(np, embedded, len(grapheme) + 1)
        h = enc[0, len(grapheme), :].reshape(1, _HIDDEN)
        dec = self.dec_emb[_START_INDEX].reshape(1, _HIDDEN)
        preds: list[int] = []
        for _ in range(_MAX_DECODE_STEPS):
            h = self._grucell(np, dec, h, self.dec_w_ih, self.dec_w_hh,
                              self.dec_b_ih, self.dec_b_hh)
            logits = h @ self.fc_w.T + self.fc_b
            pred = int(np.argmax(logits))
            if pred == _END_INDEX:
                break
            preds.append(pred)
            dec = self.dec_emb[pred].reshape(1, _HIDDEN)
        return [PHONEME_TABLE[i] for i in preds]


_MODEL: Model | None = None
_MODEL_ERROR: G2PLearnedError | None = None


def get_model() -> Model:
    """The bundled model, loaded once and cached.

    Raises :class:`G2PLearnedError` (named) when the rung cannot
    run — callers that must not raise use :func:`model_status`.
    """
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    if _MODEL_ERROR is not None:
        raise _MODEL_ERROR
    try:
        _MODEL = Model.load()
    except G2PLearnedError as exc:
        _MODEL_ERROR = exc
        raise
    return _MODEL


def model_status() -> dict[str, Any]:
    """Availability of the learned rung, as data. Never raises."""
    try:
        model = get_model()
    except G2PLearnedError as exc:
        return {"available": False, "reason": exc.kind,
                "detail": exc.message, "path": _BUNDLED_PATH,
                "arrays": 0, "shapes": None}
    except Exception as exc:  # noqa: BLE001 - status never raises
        return {"available": False, "reason": "load_failed",
                "detail": f"{type(exc).__name__}: {exc}",
                "path": _BUNDLED_PATH, "arrays": 0, "shapes": None}
    shapes = {name: list(arr.shape) for name, arr in (
        ("enc_emb", model.enc_emb), ("enc_w_ih", model.enc_w_ih),
        ("enc_w_hh", model.enc_w_hh), ("enc_b_ih", model.enc_b_ih),
        ("enc_b_hh", model.enc_b_hh), ("dec_emb", model.dec_emb),
        ("dec_w_ih", model.dec_w_ih), ("dec_w_hh", model.dec_w_hh),
        ("dec_b_ih", model.dec_b_ih), ("dec_b_hh", model.dec_b_hh),
        ("fc_w", model.fc_w), ("fc_b", model.fc_b))}
    return {"available": True, "reason": "ok", "detail": "",
            "path": _BUNDLED_PATH, "arrays": len(shapes),
            "shapes": shapes}


def predict_word(word: str) -> list[str]:
    """Normalised word → Arpabet phonemes from the learned model.

    Lowercases and strips token edges first (the core keeps Rust
    semantics and does not lowercase). Returns [] for tokens with
    nothing pronounceable. Raises :class:`G2PLearnedError` when the
    rung cannot run — the caller falls back to the rules rung.
    """
    raw = (word or "").strip("\"'.,;:!?()[]{}“”‘’—–-").lower()
    if not raw or not any(c.isalpha() for c in raw):
        return []
    return get_model().predict(raw)
