"""Share tool: ``share_file`` — expiring file shares (File.io build-new).

Devon mints a share over a file she already holds and hands back
a link that dies on its own: after the download budget is spent
or the expiry passes, the link 404s and the bytes are gone from
storage. One tool, action-dispatched, over
:mod:`nomorals.storage.share`; the organ owns the record, the
budget, the expiry and the reaper; this module is only the
surface.

``action=mint`` shares a file (``file=<path or blob:<sha>>``,
``downloads=<budget, default 1>``, ``expire=<14d default | 90s |
30m | 6h | 2d | 1w | friday | 2026-12-25 | never>``) and returns
the link. ``action=list`` lists live shares as metadata with
links, never contents. ``action=revoke`` kills a share early
(``share_id=<id>``). ``action=status`` reports the live count,
the journal length and the sweep job.

Owner-minted only: the tool never accepts an upload, and the
recipient only downloads. Anyone holding the link can download
until the budget dies or the expiry passes — that is the design,
and the mint reply says so. The link is host-network-only unless
the owner fronts the API with a tunnel; the reply says that too.
The tool never logs a share's token.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.policy import Capability
from ..storage.share import (
    SHARE_JOB,
    SHARE_SPEC,
    ShareStore,
    list_shares,
    mint_share,
    parse_share_expiry,
    revoke_share,
)

__all__ = ["register"]


def _home_of(context: Any) -> str | None:
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    home = (getattr(settings, "home", None)
            if settings is not None else None)
    return str(home) if home else None


def _base_url_of(context: Any) -> str:
    """The API server's public prefix for share links.

    No hostnames/ports are hardcoded: the configured API bind
    address wins, ``0.0.0.0`` falls back to loopback (host-network
    only — the mint reply says so), and an empty setting means the
    link is just the path. A tunnel in front of the API is the
    owner's call; the tool never invents a public host.
    """
    settings = (getattr(context, "settings", None)
                if context is not None else None)
    api = getattr(settings, "api", None) if settings is not None else None
    host = str(getattr(api, "host", "") or "").strip()
    port = getattr(api, "port", None)
    if not host or not port:
        return ""
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return f"http://{host}:{int(port)}"


def register(registry: Any) -> None:
    """Attach the ``share_file`` tool."""
    context = registry.context

    @registry.register(
        "share_file",
        description=(
            "Share a file with a link that dies on its own: mint a "
            "share over a file Devon already holds and get back a "
            "link — after the download budget is spent or the expiry "
            "passes, the link 404s and the bytes are gone from "
            "storage. action=mint shares a file (file=<path or "
            "blob:<sha256>>, downloads=<budget, default 1>, "
            "expire=<14d default | 90s | 30m | 6h | 2d | 1w | a weekday "
            "like friday | an ISO date like 2026-12-25 | never>) and "
            "returns the link. action=list lists live shares as "
            "metadata with links, never contents. action=revoke kills "
            "a share early (share_id=<id>). action=status reports the "
            "live count, the journal length and the sweep job. "
            "Owner-minted only — the tool never accepts an upload, "
            "recipients only download. Anyone holding the link can "
            "download until the budget dies or the expiry passes, and "
            "the link is host-network-only unless the API is fronted "
            "with a tunnel."
        ),
        capability=Capability.FS_WRITE,
    )
    def share_file_tool(action: str = "mint", file: str = "",
                        downloads: int = 1, expire: str = "",
                        share_id: str = "") -> dict[str, Any]:
        verb = (action or "mint").strip().lower()
        home = _home_of(context)
        if home is None:
            raise ValueError("share_file needs a home (settings.home)")
        try:
            store = ShareStore(home)
        except Exception as exc:  # noqa: BLE001 - share never crashes the caller
            return {"ok": False, "action": verb,
                    "error": f"share store unavailable: {exc}"}
        base_url = _base_url_of(context)
        if verb == "mint":
            if not (file or "").strip():
                raise ValueError("share_file mint needs a file (path or blob:<sha256>)")
            try:
                parsed = parse_share_expiry(expire if expire else None)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            try:
                if parsed is None:
                    out = mint_share(store, file.strip(), downloads=downloads,
                                     never=True)
                else:
                    out = mint_share(store, file.strip(), downloads=downloads,
                                     expires_in_s=max(0.0, parsed - time.time()))
            except Exception as exc:  # noqa: BLE001 - a failed mint is a refusal, not a crash
                return {"ok": False, "action": "mint",
                        "error": f"could not mint share: {exc}"}
            if not out.get("ok"):
                return out
            if base_url and out.get("link"):
                out["link"] = base_url.rstrip("/") + str(out["link"])
            out["note"] = ("anyone holding this link can download until "
                           "the budget dies or the expiry passes; the link "
                           "is host-network-only unless the API is fronted "
                           "with a tunnel")
            return out
        if verb == "list":
            return {"ok": True, "action": "list",
                    "shares": list_shares(store, base_url=base_url)}
        if verb == "revoke":
            if not (share_id or "").strip():
                raise ValueError("share_file revoke needs a share_id")
            revoked = revoke_share(store, share_id.strip())
            return {"ok": revoked, "action": "revoke",
                    "share_id": share_id.strip(),
                    "error": "" if revoked else "no such live share"}
        if verb == "status":
            return {"ok": True, "action": "status",
                    "live": len(list_shares(store)),
                    "journal": len(store.journal_entries()),
                    "job": SHARE_JOB, "spec": SHARE_SPEC}
        raise ValueError(f"unknown share_file action {verb!r} "
                         "(mint | list | revoke | status)")
