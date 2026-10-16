"""Storefront: product → cover-image brief + prompt, via imggen.

A product cover is built in two honest layers:

1. **Brief** — always available, pure text: subject, setting, lighting,
   composition, brand voice. ``brief_only`` is the truthful default.
2. **Generated image** — only when the caller supplies a real imggen
   callable (``imggen(prompt, ...) -> path | bytes | None``), e.g.
   ``nomorals.media.imggen.pipeline.NativePipeline.generate``. When no
   callable is supplied, ``generate_cover()`` returns the brief with
   ``generated=False`` and ``image=None`` — it NEVER fabricates an image
   path, a fake URL, or placeholder bytes.

Product page data (``product_page_data``) is plain structured data for
the storefront renderer: title, price, description, cover brief, tags,
stock state — no invented images, no invented reviews.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Product",
    "cover_brief",
    "cover_prompt",
    "generate_cover",
    "product_page_data",
]


@dataclass
class Product:
    """A sellable product (storefront view, not the Medusa record)."""
    id: str
    name: str
    price_minor: int
    currency: str = "NGN"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""
    image_spec: str = ""  # operator's own visual direction, if any
    stock: int = 0
    brand_voice: str = ""  # e.g. "warm Lagos market stall"

    def in_stock(self) -> bool:
        return self.stock > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "name": self.name,
            "price_minor": self.price_minor, "currency": self.currency,
            "description": self.description, "tags": list(self.tags),
            "category": self.category, "image_spec": self.image_spec,
            "stock": self.stock, "brand_voice": self.brand_voice,
        }


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "product"


def cover_brief(product: Product) -> dict[str, Any]:
    """Build the honest text brief for a product cover image.

    Always available, always text-only. No image is claimed to exist.
    """
    name = (product.name or "").strip() or "Untitled product"
    subject = product.image_spec.strip() or f"a clean product photo of {name}"
    voice = product.brand_voice.strip() or "warm, honest Nigerian storefront"
    setting = ("market-stall warmth, natural daylight" if not product.tags
               else f"styled around: {', '.join(product.tags[:4])}")
    return {
        "product_id": product.id,
        "slug": _slug(name),
        "subject": subject,
        "setting": setting,
        "lighting": "soft natural daylight, no harsh flash",
        "composition": "product centred, generous negative space, "
                       "readable at thumbnail size",
        "brand_voice": voice,
        "avoid": "watermarks, fake badges, misleading 'before/after', "
                 "text baked into the image",
        "aspect": "square (1:1) for feeds, 4:5 for product pages",
        "generated": False,
        "image": None,
    }


def cover_prompt(product: Product) -> str:
    """One-line diffusion prompt distilled from the brief."""
    brief = cover_brief(product)
    return (f"{brief['subject']}, {brief['setting']}, {brief['lighting']}, "
            f"{brief['composition']}, {brief['brand_voice']} "
            f"— product photography, no text, no watermark")


def generate_cover(product: Product, *,
                   imggen: Callable[..., Any] | None = None,
                   **imggen_kwargs: Any) -> dict[str, Any]:
    """Generate (or honestly brief-only) a product cover.

    - With ``imggen``: calls ``imggen(prompt, **kwargs)`` and returns its
      real result in ``image`` with ``generated=True``. An imggen failure
      is reported (``generated=False``, ``error`` set) — never papered
      over with a fake path.
    - Without ``imggen``: returns the brief with ``generated=False`` and
      ``image=None`` — the caller renders the brief, not an image.
    """
    brief = cover_brief(product)
    prompt = cover_prompt(product)
    brief["prompt"] = prompt
    if imggen is None:
        brief["note"] = ("No image generator supplied — brief only. "
                         "Pass imggen=... (e.g. Devon's imggen pipeline) "
                         "to generate a real cover.")
        return brief
    try:
        result = imggen(prompt, **imggen_kwargs)
    except Exception as exc:  # noqa: BLE001 — generation failure is data
        brief["generated"] = False
        brief["image"] = None
        brief["error"] = f"imggen failed: {exc}"
        return brief
    if result is None:
        brief["generated"] = False
        brief["image"] = None
        brief["error"] = "imggen returned no image"
        return brief
    brief["generated"] = True
    brief["image"] = result
    return brief


def product_page_data(product: Product, *,
                      imggen: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Structured product-page data: copy + cover (brief or generated)."""
    cover = generate_cover(product, imggen=imggen)
    return {
        "id": product.id,
        "slug": _slug(product.name),
        "title": product.name,
        "price_minor": product.price_minor,
        "currency": product.currency,
        "description": product.description or product.name,
        "tags": list(product.tags),
        "category": product.category,
        "in_stock": product.in_stock(),
        "stock": product.stock,
        "cover": cover,
        "reviews": [],  # never invented — populated from real orders only
    }
