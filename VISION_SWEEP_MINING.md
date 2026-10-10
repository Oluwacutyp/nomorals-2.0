# Vision Sweep — External Mining

Mined 2026-10-10 for `nomorals/vision/` (seer.py, screenshot.py, native.py,
`__init__.py`). Only real, verifiable sources; no invented techniques.
Each item below names what was taken and where it landed.

## 1. Microsoft OmniParser (github.com/microsoft/OmniParser, 25k stars, CC-BY-4.0)

What it does: turns a screenshot into a structured list of interactable
elements. Pipeline = fine-tuned YOLOv8 interactable-icon/region detector +
fine-tuned icon-description captioner (BLIP-2 / Florence-2) + OCR module.
Outputs labeled elements with boxes; the LLM then grounds to those
elements instead of raw pixels. V2: ~0.6s/frame on A100, 39.6 avg on
ScreenSpot Pro. (HF: microsoft/OmniParser-v2.0.)

**Gold taken:** the *architecture idea* — detect-then-label beats asking a
model for raw coordinates. Native, offline approximation implemented:
`nomorals/vision/native.py::screen_marks` builds a Set-of-Mark overlay
from OCR word boxes (tesseract TSV) + contour regions (OpenCV
findContours) + deduplication, draws numbered marks, returns PNG bytes +
`[{id, bbox_1000, text}]`. The model picks a number; the caller maps the
number back to coordinates. No YOLO weights shipped — the detector layer
is a documented seam (`detect_regions` accepts injected elements) so a
future YOLO/Florence pass can slot in behind the same contract.

Also from the OmniParser Responsible-AI notes: human judgement stays in
the loop — every locate result keeps its disclaimer.

## 2. Set-of-Mark prompting (Yang et al., 2023 — "Set-of-Mark Prompting
Unleashes Extraordinary Visual Grounding in GPT-4V", arxiv 2310.11441)

Core finding: drawing numbered segmentation marks on the image and letting
the VLM answer with a mark ID is dramatically more reliable for visual
grounding than raw coordinate prediction, especially on dense UIs.
Confirmed by practitioners: dev.to/nughes "Building Screen Automation
Agents" (Oct 2026): "Choosing a number is far easier for a model than
estimating exact coordinates"; ganainy/ai-mobile-ui-crawler spec
FR-001…FR-006 (local OCR → labeled screenshot → VLM picks label ID →
map ID back to box center; raw coordinates only as fallback for
icon-only elements); mizchi/vlmkit "check grounding" (Oct 2026):
marks must be drawn at the *downscaled resolution the model actually
sees*, or positions drift.

**Gold taken:** `screen_marks` draws marks on the *same downscaled bytes*
the model will receive (`resize_for_model` shares the budget), label
style = white-on-black chip with black outline for legibility at small
sizes (the SoM paper's convention), plus a `mark_prompt_snippet()` helper
that injects the "answer with the element number" instruction. FR-006's
fallback advice (icons OCR misses → raw coords) is mirrored in
`screen_marks` by including contour candidates even without text.

## 3. UI-TARS / OS-Atlas grounding conventions (ByteDance-Seed UI-TARS,
arxiv 2501.12326; OS-Atlas, arxiv 2410.23218)

Both normalize GUI coordinates to a fixed 0–1000 grid
(`denormalize: px = norm/1000 * dimension`) — the same convention this
repo already used for `template_locate`/`detect_faces` bboxes. UI-TARS
associates each element's box center as its click point.

**Gold taken:** validation that the repo's 0–1000 convention is the
industry-standard one; `screen_marks` elements therefore also expose
`center_1000` (click point = box center), matching UI-TARS's action
derivation.

## 4. Anthropic computer-use reference (anthropics/anthropic-quickstarts,
computer-use-demo) + calebnewtonusc/chewbacca 04-VISION.md

Pattern: never ask for raw coordinates when an accessibility tree/DOM
exists — check the tree first (Playwright `page.accessibility.snapshot()`),
fall back to vision only where the tree is empty (canvas, games, remote
desktops).

**Gold taken:** documented as the strategy chain in `screenshot.py`'s new
`capture_screen` docstring and in `screen_marks`'s docstring (OCR boxes
are the "tree"; contours are the fallback). Not implemented as code
because the browser module owns the a11y snapshot — out of this module's
scope; left as a documented integration point.

## 5. Cross-platform host screenshots (cross-capture by jonschlinkert;
pyscreenshot 3.1; Robot Framework Screenshot lib; pq.hosting Ubuntu guide)

Backend matrix verified across sources:
- macOS: `screencapture` binary (always present).
- Windows: Pillow `ImageGrab` (works on Windows without extra deps).
- Linux X11: `scrot`, `maim`, ImageMagick `import -window root`
  (scrot/`import` capture *black* on Wayland — must be gated by session type).
- Linux Wayland: `grim` (native; `slurp` only for region selection).
- Android/Termux: `termux-screenshot` from termux-api.
- `mss` python lib as the cross-platform in-process option when installed.

**Gold taken:** `screenshot.py::capture_screen()` implements exactly this
as a strategy chain — termux-screenshot → mss → screencapture (darwin)
→ ImageGrab (win32) → Wayland grim / X11 scrot|maim|import (gated on
`XDG_SESSION_TYPE`, with DISPLAY check). Order = cheapest/most-likely
first; every miss reports *which* backend failed and the concrete
install hint. Headless/Termux-without-api → `ScreenshotUnavailable`
with the exact package to install. Previously the module could only
screenshot a *browser tab* or reuse a file — the owner could never say
"look at my screen".

## 6. Tesseract PSM strategy (0xdarkmatter/axiom `.claude/skills/ocr-tesseract`;
NanoNets/ocr-with-tesseract tutorial; tesseract man page)

- PSM 3 = fully automatic (default, full pages); PSM 6 = single uniform
  block; PSM 11 = sparse text (screenshots); PSM 7 = single line.
- OEM 1 (LSTM) is usually best; OEM 3 = default auto.
- Preprocessing that matters: upscale small text (tesseract expects
  ~300 DPI equivalent), grayscale, and a thin white border helps tight
  crops.

**Gold taken:** new `nomorals/vision/native.py::ocr_text()` (the module
*advertised* OCR in `capabilities()` but had no OCR function — a real
gap). Strategy: auto-upscale short side < 800px → grayscale → 8px white
border → `--oem 1 --psm {psm}`; default psm=11 for screenshots,
configurable; returns text + mean confidence (from `tsv` word confs)
+ `method` chain record. Verbatim transcription only — never "cleaned
up" by heuristics.

## 7. Prompt-injection posture for vision (already in-repo, validated)

The repo's `UNTRUSTED_VISION_PREFIX` (mark every vision/OCR return as
untrusted data) matches the standing guidance for tool-using agents:
extracted image text must be framed as *data* before it reaches the
model or an audit trail. Kept and extended: `screen_marks` OCR text
and `ocr_text` payloads are flagged the same way at the tool boundary,
and `read_qr`'s `untrusted_note` stays.

## 8. Bugs found during mining (fixed in this sweep)

- `nomorals/vision/seer.py:150` — `brain_for(self.context)` but
  `Seer.__init__` never sets `self.context` → **every** router-path
  `see()` raised `AttributeError`, silently fell through to the local
  path, and failed there too. Fix: `context` constructor param stored
  as `self._context` (None ⇒ shared env brain, matching `brain_for`'s
  own fallback contract).
- Same copy-paste bug exists in `nomorals/tools/vision.py:467`
  (`brain_for(self.context)` inside module-level `_vision_call`,
  no `self` in scope) — **sibling worker's file, NOT touched**; flagged
  in the sweep report for the tools worker.

## 9. Deliberately NOT taken

- OmniParser's YOLOv8/Florence-2 weights: 100MB+ downloads, GPU-hungry,
  wrong for a phone-first agent; the seam is there if a provisioned
  model lands later.
- UI-TARS/OS-Atlas end-to-end grounding models: same reason; documented
  in `capabilities()["needs_model"]["word_locate"]` as the upgrade path.
- EasyOCR/PaddleOCR: heavier than tesseract, worse on Termux; tesseract
  binary stays the native OCR backend.
- Anthropic-style raw coordinate prediction prompts: superseded by
  Set-of-Mark for this codebase's reliability bar.

## What this class SHOULD have (gap analysis → implemented)

| Class / area | Gap | Landed |
|---|---|---|
| `Seer` | primary path crashed (`self.context`) | fixed; `context=` param |
| `Seer` | no image normalization (BMP/TIFF sent to VLMs, 4K screenshots burn tokens) | `_prepare_bytes`: EXIF-transpose + RGB + downscale to 1568px budget, PNG out |
| `Seer` | transient 429/timeout = instant failover, no retry | `_is_transient` + 3 attempts, exponential backoff |
| `Seer` | bytes input unsupported (only file paths) | `see()` accepts `bytes`; new `see_bytes()` |
| `Seer` | local model never released on GC | `__enter__/__exit__` + `__del__` guard |
| `screenshot` | no host screen capture at all | `capture_screen()` strategy chain |
| `native` | `capabilities()` advertised OCR, no OCR function | `ocr_text()` |
| `native` | no visual-grounding aid | `screen_marks()` (SoM), `resize_for_model()`, `mark_prompt_snippet()` |
| `native` | `analyze()` lacks OCR text | `analyze()` now includes `ocr_text` best-effort (fast path, psm 11) |
