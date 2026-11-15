"""One-call poster pipeline: generated art + exact typography.

Septorch-grade posters need two things no generative model does alone:
the art, and HEAVY EXACT TEXT. The model renders the scene without
text; real typography is composited on top with PIL — the upgrade
tier's "text composited never generated" invariant
(:mod:`nomorals.media.upgrade.typography`).

:func:`make_poster` is the single call the two halves never had: it
generates the art through the imggen :class:`Studio`, composites the
text layers onto it with :func:`composite_text`, and saves the
finished poster PNG. ``generate`` is an injectable seam with exactly
:func:`Studio.generate`'s shape (prompt + kwargs in, [saved art
path] out) — the same shape :func:`upgrade.pipeline.run_pipeline`
takes — so tests and callers with their own art source drive the
composition without a model. The default is the real Studio: when
generation cannot run (no torch, no checkpoint) the real error
propagates and no poster is produced — this module never fakes art.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Portrait poster ratio — the classic poster shape.
DEFAULT_ASPECT = "9:16"
#: A bare-string text layer is a footer line at this size.
BODY_SIZE = 48
#: Where a poster lands when no ``save_to`` is given (same out dir
#: the Studio writes its generations to).
DEFAULT_OUT_DIR = "~/.nomorals/imggen/output"


def _normalize_layers(texts: Any) -> list:
    """Text specs → a list of :class:`TextComposite` layers.

    Each entry may be a :class:`TextComposite`, a dict of its fields,
    or a bare string (a footer line: bottom, body size). A single
    layer need not be wrapped in a list.
    """
    from .upgrade.typography import TextComposite

    if isinstance(texts, (str, dict, TextComposite)):
        texts = [texts]
    layers = []
    for entry in texts:
        if isinstance(entry, TextComposite):
            layers.append(entry)
        elif isinstance(entry, dict):
            layers.append(TextComposite(**entry))
        elif isinstance(entry, str):
            layers.append(TextComposite(text=entry, position="bottom",
                                        size=BODY_SIZE))
        else:
            raise ValueError(
                "poster text layers are strings, dicts or "
                f"TextComposite — not {type(entry).__name__}")
    return layers


def _poster_target(save_to: str | None) -> str:
    """Where the finished poster lands.

    A directory (existing, or trailing ``/``) takes a unique poster
    name inside it; a file path is used exactly; None lands in the
    Studio's output dir.
    """
    if save_to and (os.path.isdir(save_to)
                    or save_to.endswith(("/", os.sep))):
        os.makedirs(save_to, exist_ok=True)
        return os.path.join(save_to, f"poster_{time.time_ns()}.png")
    if save_to:
        parent = os.path.dirname(save_to)
        if parent:
            os.makedirs(parent, exist_ok=True)
        return save_to
    base = os.path.expanduser(DEFAULT_OUT_DIR)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, f"poster_{time.time_ns()}.png")


def make_poster(prompt: str, texts: Any, *, aspect: str = DEFAULT_ASPECT,
                seed: int = 0, steps: int = 0,
                save_to: str | None = None,
                generate: Any = None) -> dict[str, Any]:
    """Generate art and composite exact typography onto it.

    ``prompt`` is the art prompt; ``texts`` is the text spec (see
    :func:`_normalize_layers`). ``aspect`` reaches the generator as
    ``aspect_ratio`` (``9:16`` by default — portrait poster);
    ``seed``/``steps`` pass through to the generator. ``save_to``
    is the poster destination (directory or file path). ``generate``
    defaults to the real :class:`Studio` — pass a callable with
    ``Studio.generate``'s shape only to drive the composition from
    another art source.

    Returns ``{"path", "art_path", "prompt", "texts", "aspect",
    "size"}``. The art file is never mutated; a sidecar ``.txt``
    beside the poster records the prompt and the rendered layers.
    """
    from PIL import Image

    from .upgrade.typography import composite_text

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("make_poster needs a non-empty art prompt")
    if not texts:
        raise ValueError("make_poster needs at least one text layer")
    layers = _normalize_layers(texts)
    if generate is None:
        from .imggen.studio import Studio

        generate = Studio().generate
    gen_kwargs: dict[str, Any] = {"aspect_ratio": aspect}
    if seed:
        gen_kwargs["seed"] = seed
    if steps:
        gen_kwargs["steps"] = steps
    art_paths = generate(prompt, **gen_kwargs)
    if not art_paths:
        raise ValueError("image generation returned no art")
    art_path = str(art_paths[0])
    with Image.open(art_path) as art:
        poster = composite_text(art, layers)
        size = poster.size
        target = _poster_target(save_to)
        poster.save(target)
    with open(target + ".txt", "w") as f:
        f.write(prompt)
        for layer in layers:
            f.write(f"\ntext: {layer.text} "
                    f"[{layer.position} {layer.size}]")
    _log.info("poster saved: %s (art %s, %d layer(s))",
              target, art_path, len(layers))
    return {"path": target, "art_path": art_path, "prompt": prompt,
            "texts": [layer.to_dict() for layer in layers],
            "aspect": aspect, "size": size}
