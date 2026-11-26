"""Imggen's rider on the shared provision seam.

The stock checkpoint Devon fetches for herself on first use with no
loadable image model: SD1.5 (runwayml/stable-diffusion-v1-5,
v1-5-pruned-emaonly.safetensors — free, open, no key, no login),
into the checkpoint dir. A checkpoint the owner drops in himself
always wins: provision runs only when the registry has NO loadable
entry, so an owner file is never overwritten or shadowed.

The mechanism lives in :mod:`nomorals.core.provision`; this module
holds only imggen's knowledge — which file, where it lands, and how
to tell a good landing (the registry's own header classification
must come back ``sd15`` + LDM + loadable).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...core.provision import ProvisionError, ResourceSpec, ensure_resource
from . import ImgGenError, checkpoint_dir

__all__ = [
    "STOCK_REPO",
    "STOCK_FILE",
    "STOCK_URL",
    "stock_spec",
    "ensure_stock_checkpoint",
]

STOCK_REPO = "runwayml/stable-diffusion-v1-5"
STOCK_FILE = "v1-5-pruned-emaonly.safetensors"
STOCK_URL = ("https://huggingface.co/runwayml/stable-diffusion-v1-5/"
             "resolve/main/v1-5-pruned-emaonly.safetensors")
#: The file the CDN serves (verified live 2026-11-25: 200 OK,
#: Accept-Ranges: bytes, Content-Length 4265146304).
STOCK_BYTES = 4265146304


def _validate_stock(path: Path) -> bool | str:
    """The landed file must classify as a loadable SD1.5 (LDM)."""
    from .registry import resolve

    entry = resolve(str(path))
    if entry is None or entry["model_class"] != "sd15" \
            or not entry["loadable"]:
        reason = (entry or {}).get("loadable_reason", "unresolvable")
        return (f"landed file classifies as "
                f"{(entry or {}).get('model_class', 'unknown')} "
                f"({reason}) — expected a loadable SD1.5 checkpoint")
    return True


def stock_spec(dest_dir: str | Path | None = None) -> ResourceSpec:
    """The stock-checkpoint spec: what, where, how to tell it landed."""
    dest = Path(dest_dir) if dest_dir is not None \
        else Path(checkpoint_dir())
    return ResourceSpec(
        name="stock SD1.5 checkpoint",
        urls=(STOCK_URL,),
        dest=dest / STOCK_FILE,
        expected_bytes=STOCK_BYTES,
        validate=_validate_stock,
    )


def ensure_stock_checkpoint(
        *, progress: Callable[[int, int | None], None] | None = None,
        downloader: Callable[..., dict[str, Any]] | None = None,
        dest_dir: str | Path | None = None) -> dict[str, Any]:
    """Fetch the stock checkpoint when the registry has none loadable.

    Returns the registry entry for the (new or already-present)
    stock file. When a loadable entry already exists — the owner's
    own checkpoint, a previous provision, anything — returns that
    entry at once with no download. Raises :class:`ImgGenError`
    (wrapping the seam's :class:`ProvisionError` receipts) when
    nothing loadable exists and the fetch fails.
    """
    from .registry import list_models

    loadable = [e for e in list_models() if e["loadable"]]
    if loadable:
        return dict(loadable[0], already=True)
    try:
        facts = ensure_resource(
            stock_spec(dest_dir), progress=progress,
            downloader=downloader)
    except ProvisionError as exc:
        raise ImgGenError(
            "no image checkpoint and the stock checkpoint could "
            f"not be fetched: {exc}") from exc
    from .registry import resolve

    entry = resolve(facts["path"])
    if entry is None or not entry["loadable"]:
        raise ImgGenError(
            "stock checkpoint fetched but the registry still does "
            "not list it as loadable — refusing to pretend it "
            "landed")
    return dict(entry, **{k: v for k, v in facts.items()
                          if k not in entry})
