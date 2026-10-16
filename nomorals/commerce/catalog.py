"""Catalog / feed sync: product catalog, feed export, sync diff report.

The catalog is the store's own list of products (plain dicts — the
same shape as :mod:`storefront.Product.to_dict`). ``feed_export`` dumps
it as CSV or JSON for marketplaces / ad platforms. ``sync`` compares
the local catalog against a remote listing (fetched by the caller —
this module never invents a remote) and returns a diff report:
``added`` (remote-only), ``removed`` (local-only), ``changed``
(same id, different fields), ``unchanged``.

No record is ever silently overwritten: sync reports the diff; the
caller decides what to apply via :meth:`Catalog.apply_diff`.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Catalog",
    "diff_catalogs",
]

#: Fields compared for change detection (id excluded — it is the key).
_COMPARED_FIELDS: tuple[str, ...] = (
    "name", "price_minor", "currency", "description",
    "category", "stock", "tags",
)


def _norm(product: dict[str, Any]) -> dict[str, Any]:
    p = dict(product or {})
    p.setdefault("tags", [])
    p["tags"] = sorted(str(t) for t in (p.get("tags") or []))
    return p


def diff_catalogs(local: list[dict[str, Any]],
                  remote: list[dict[str, Any]]) -> dict[str, Any]:
    """Diff two product lists by ``id``. Pure function, no I/O."""
    local_map = {str(p.get("id")): _norm(p) for p in local if p.get("id")}
    remote_map = {str(p.get("id")): _norm(p) for p in remote if p.get("id")}
    added, removed, changed, unchanged = [], [], [], []
    for pid, rp in remote_map.items():
        if pid not in local_map:
            added.append(rp)
        else:
            lp = local_map[pid]
            diffs = {f: {"local": lp.get(f), "remote": rp.get(f)}
                     for f in _COMPARED_FIELDS
                     if lp.get(f) != rp.get(f)}
            if diffs:
                changed.append({"id": pid, "fields": diffs})
            else:
                unchanged.append(lp)
    for pid, lp in local_map.items():
        if pid not in remote_map:
            removed.append(lp)
    return {
        "added": added, "removed": removed,
        "changed": changed, "unchanged": unchanged,
        "summary": {
            "local": len(local_map), "remote": len(remote_map),
            "added": len(added), "removed": len(removed),
            "changed": len(changed), "unchanged": len(unchanged),
        },
    }


class Catalog:
    """In-memory product catalog with feed export + sync diff."""

    def __init__(self, products: list[dict[str, Any]] | None = None) -> None:
        self._products: dict[str, dict[str, Any]] = {}
        for p in products or []:
            self.upsert(p)

    # ── CRUD ──

    def upsert(self, product: dict[str, Any]) -> dict[str, Any]:
        if not str(product.get("id") or "").strip():
            raise ValueError("product needs an id")
        if int(product.get("price_minor", 0) or 0) < 0:
            raise ValueError("price_minor must be non-negative")
        record = _norm(product)
        self._products[str(record["id"])] = record
        return record

    def remove(self, product_id: str) -> bool:
        return self._products.pop(str(product_id), None) is not None

    def get(self, product_id: str) -> dict[str, Any] | None:
        return self._products.get(str(product_id))

    def list(self) -> list[dict[str, Any]]:
        return [dict(p) for p in self._products.values()]

    def __len__(self) -> int:
        return len(self._products)

    # ── feed export ──

    def feed_export(self, fmt: str = "json") -> str:
        """Export the catalog as a feed. ``fmt``: ``json`` | ``csv``."""
        products = self.list()
        if fmt == "json":
            return json.dumps(products, indent=2, sort_keys=True)
        if fmt == "csv":
            buf = io.StringIO()
            fields = ["id", "name", "price_minor", "currency", "description",
                      "category", "stock", "tags"]
            writer = csv.DictWriter(buf, fieldnames=fields)
            writer.writeheader()
            for p in products:
                row = {f: p.get(f, "") for f in fields}
                row["tags"] = ";".join(str(t) for t in (p.get("tags") or []))
                writer.writerow(row)
            return buf.getvalue()
        raise ValueError(f"unknown feed format {fmt!r} — json|csv")

    # ── sync ──

    def sync(self, remote: list[dict[str, Any]]) -> dict[str, Any]:
        """Diff local catalog vs a remote listing. Reports only."""
        return diff_catalogs(self.list(), remote)

    def apply_diff(self, diff: dict[str, Any], *,
                   apply_added: bool = True, apply_changed: bool = True,
                   apply_removed: bool = False) -> dict[str, int]:
        """Apply a diff from :meth:`sync`. Removals are opt-in (default off).

        Applying ``changed`` overwrites local fields with the remote
        values for the differing fields only. Returns counts applied.
        """
        counts = {"added": 0, "changed": 0, "removed": 0}
        if apply_added:
            for p in diff.get("added", []):
                self.upsert(p)
                counts["added"] += 1
        if apply_changed:
            for change in diff.get("changed", []):
                pid = change.get("id")
                current = self._products.get(str(pid))
                if current is None:
                    continue
                for f, vals in (change.get("fields") or {}).items():
                    current[f] = vals.get("remote")
                self._products[str(pid)] = _norm(current)
                counts["changed"] += 1
        if apply_removed:
            for p in diff.get("removed", []):
                if self.remove(str(p.get("id"))):
                    counts["removed"] += 1
        return counts
