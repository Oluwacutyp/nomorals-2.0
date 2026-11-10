"""Pack operations: the M1 proof surface over the contracts.

Two package-level operations tie the M1 pieces together
end-to-end — verify a pack directory (manifest contract → verdict
→ receipt) and render a preview from a pack (manifest → renderer
boundary → image + receipt).  This is deliberately the smallest
proof surface: no CLI, no chat verb — those are M3/M7 territory,
and they will call exactly these functions.
"""

from __future__ import annotations

from pathlib import Path

from .coords import BBox
from .manifest import PackVerification, load_manifest, verify_pack
from .receipts import OP_PACK_VERIFY, OUTCOME_FAILED, OUTCOME_OK, MapReceipt, MapReceiptStore
from .render import FixtureRenderer, PngBackend, RenderResult

__all__ = ["render_pack_preview", "verify_pack_dir"]


def verify_pack_dir(pack_dir: Path | str, *,
                    store: MapReceiptStore | None = None,
                    ) -> tuple[PackVerification, MapReceipt]:
    """Verify a pack directory and mint its ``pack_verify`` receipt.

    The receipt's detail is the verification summary — attribution
    and signature state included, so the evidence trail carries
    what a surface would render.
    """
    verification = verify_pack(pack_dir)
    receipt = MapReceipt.mint(
        OP_PACK_VERIFY,
        OUTCOME_OK if verification.ok else OUTCOME_FAILED,
        pack_refs=(verification.pack_id,),
        pack_digests=(verification.digest,),
        detail=verification.summary()
        + " · attribution: " + verification.attribution)
    if store is not None:
        store.append(receipt)
    return verification, receipt


def render_pack_preview(pack_dir: Path | str, bbox: BBox, zoom: int,
                        size: tuple[int, int] = (512, 512), *,
                        store: MapReceiptStore | None = None,
                        backend: PngBackend | None = None,
                        ) -> RenderResult:
    """Render one static preview from a pack, with its receipt."""
    manifest = load_manifest(pack_dir)
    renderer = FixtureRenderer(pack_dir, manifest, store=store,
                               backend=backend)
    return renderer.render(bbox, zoom, size)
