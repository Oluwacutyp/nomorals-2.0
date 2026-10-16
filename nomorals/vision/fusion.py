"""Vision-language fusion: joint image + text reasoning.

``vision/native.py`` describes what an image IS (OCR, EXIF, colors,
quality, hashes) but never reasons about it.  This module fuses that
native description with the user's text and a vision-capable model
into one joint answer — and grounds the model's claims back against
the OCR text, so a hallucinated figure that never appears in the image
is flagged rather than passed on.

Document kinds (invoice vs contract vs other) are classified from the
OCR text by deterministic keyword scoring — no model call needed for
the classification itself, which keeps it testable and honest.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "FusionResult",
    "classify_document_kind",
    "build_fusion_prompt",
    "grounded_claims",
    "fuse_image_text",
]

#: Keyword signals per document kind (lowercase substring match).
_KIND_SIGNALS: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice", "amount due", "total due", "bill to", "subtotal",
                "tax", "payment due", "invoice no", "invoice #", "balance due"),
    "contract": ("agreement", "hereby agree", "party of the first part",
                 "whereas", "terms and conditions", "hereinafter",
                 "signatory", "effective date", "breach", "governing law"),
    "receipt": ("receipt", "change due", "cashier", "tendered", "thank you for"),
}


def classify_document_kind(ocr_text: str) -> dict[str, Any]:
    """Classify OCR text as invoice / contract / receipt / other.

    Deterministic keyword scoring: each signal hit is +1 for its kind.
    Ties break toward the kind with the earliest hit in the text.
    Returns ``{"kind", "scores", "hits"}`` — ``kind`` is ``"other"``
    when nothing hits.
    """
    text = (ocr_text or "").lower()
    scores: dict[str, int] = {}
    first_hit: dict[str, int] = {}
    for kind, signals in _KIND_SIGNALS.items():
        hits = 0
        first = -1
        for signal in signals:
            pos = text.find(signal)
            if pos >= 0:
                hits += 1
                if first < 0 or pos < first:
                    first = pos
        scores[kind] = hits
        if hits:
            first_hit[kind] = first
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] <= 0:
        return {"kind": "other", "scores": scores, "hits": []}
    # Tie-break: earliest first hit wins.
    top = max(scores.values())
    tied = [k for k, v in scores.items() if v == top]
    kind = min(tied, key=lambda k: first_hit.get(k, 0)) if len(tied) > 1 else best[0]
    hits = [s for s in _KIND_SIGNALS[kind] if s in text]
    return {"kind": kind, "scores": scores, "hits": hits}


def build_fusion_prompt(native_report: dict[str, Any], text: str) -> str:
    """Structured prompt: native image facts + user text, one reasoning job."""
    ocr = ""
    try:
        ocr = str((native_report.get("ocr") or {}).get("text") or "")
    except Exception:  # noqa: BLE001
        ocr = ""
    meta = native_report.get("metadata") or {}
    colors = native_report.get("colors") or {}
    quality = native_report.get("quality") or {}
    kind = classify_document_kind(ocr)
    lines = [
        "You are reasoning jointly about an image and a user message.",
        "Use ONLY the image facts below plus the user message. Do not invent",
        "figures, dates, or names that do not appear in the image facts.",
        "",
        f"Document kind (from image text): {kind['kind']}",
        f"Image: {meta.get('format', '')} {meta.get('width', '')}x"
        f"{meta.get('height', '')}",
        f"Colors: {colors.get('dominant', '')} "
        f"brightness={colors.get('brightness', '')}",
        f"Quality: sharpness={quality.get('sharpness', '')}",
        "",
        "Text found in the image (OCR):",
        ocr[:4000] or "(no text found in image)",
        "",
        "User message:",
        text,
        "",
        "Answer the user's message using the image facts. Quote the exact",
        "image text for every figure or date you rely on.",
    ]
    return "\n".join(lines)


_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")
_NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


def grounded_claims(model_answer: str, ocr_text: str) -> list[dict[str, Any]]:
    """Check each sentence of the model answer against the OCR text.

    A sentence is ``grounded`` when every number it contains also
    appears in the OCR text (numbers are the classic hallucination
    vector in invoice/contract reasoning) — or when it contains no
    numbers at all (``ungrounded`` never applies without a checkable
    claim).  Sentences whose numbers are absent from OCR are flagged
    ``unverified`` with the missing numbers listed.
    """
    ocr = ocr_text or ""
    out: list[dict[str, Any]] = []
    for raw in _SENTENCE_RE.findall(model_answer or ""):
        sentence = raw.strip()
        if not sentence:
            continue
        numbers = _NUMBER_RE.findall(sentence)
        missing = [n for n in numbers if n not in ocr]
        out.append({
            "sentence": sentence,
            "numbers": numbers,
            "grounded": not missing,
            "missing_numbers": missing,
        })
    return out


@dataclass
class FusionResult:
    """Joint image + text reasoning result."""

    document_kind: str
    ocr_text: str
    native_report: dict[str, Any] = field(default_factory=dict)
    model_answer: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    model_used: bool = False
    reason: str = ""

    @property
    def ungrounded(self) -> list[dict[str, Any]]:
        return [c for c in self.claims if not c.get("grounded")]

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_kind": self.document_kind,
            "ocr_text": self.ocr_text,
            "model_answer": self.model_answer,
            "claims": list(self.claims),
            "ungrounded": self.ungrounded,
            "model_used": self.model_used,
            "reason": self.reason,
        }


def fuse_image_text(
    image_bytes: bytes,
    text: str,
    *,
    brain: Any = None,
    describe_fn: Any = None,
    native_report: dict[str, Any] | None = None,
) -> FusionResult:
    """Jointly reason about ``image_bytes`` + ``text``.

    1. Native description via ``vision/native.py`` (or the supplied
       ``native_report`` — injectable for tests/offline use).
    2. Document-kind classification from OCR (no model needed).
    3. When a brain / ``describe_fn`` is supplied, joint reasoning with
       the fused prompt; otherwise the native facts + kind are returned
       with ``model_used=False`` — never a faked model answer.
    4. The model's sentences are grounded against the OCR text.
    """
    from .native import analyze as _native_analyze

    if native_report is None:
        try:
            native_report = _native_analyze(image_bytes)
        except Exception as exc:  # noqa: BLE001 — native failure is a result
            return FusionResult(document_kind="other", ocr_text="",
                                reason=f"native vision failed: {exc}")
    ocr_text = ""
    try:
        ocr_text = str((native_report.get("ocr") or {}).get("text") or "")
    except Exception:  # noqa: BLE001
        ocr_text = ""
    kind = classify_document_kind(ocr_text)
    prompt = build_fusion_prompt(native_report, text)
    answer = ""
    model_used = False
    reason = ""
    if describe_fn is not None:
        try:
            answer = str(describe_fn(prompt, image_bytes) or "")
            model_used = True
        except Exception as exc:  # noqa: BLE001
            reason = f"describe_fn failed: {exc}"
    elif brain is not None:
        try:
            describe = getattr(brain, "describe_image", None)
            if callable(describe):
                resp = describe(image_bytes, prompt)
                answer = str(getattr(resp, "text", "") or "")
                model_used = bool(answer)
            else:
                reason = "brain has no describe_image"
        except Exception as exc:  # noqa: BLE001
            reason = f"brain describe_image failed: {exc}"
    else:
        reason = "no model supplied — native facts only"
    claims = grounded_claims(answer, ocr_text) if answer else []
    return FusionResult(
        document_kind=kind["kind"], ocr_text=ocr_text,
        native_report=dict(native_report), model_answer=answer,
        claims=claims, model_used=model_used, reason=reason)
