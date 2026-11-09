"""Owner system organ (DESIGN §19 + §20 expansion): the
owner's presence across Devon as ONE system.

§19's center is a single owner record.  Before this build the
pieces were scattered: the runtime's operator check (surface
rule + gateway verdict + a config sender-id allowlist), the
core identity seal (handles + a passphrase proof), and the
§15 pairing registry (his accounts as sessions).  This organ
is the record they all read from:

- **Identities**: the owner's platform ids, per platform —
  seeded from the existing sources (config sender ids, the
  pairing registry's accounts) and editable ONLY by the owner,
  ONLY outside a group (§20e: the record is never editable
  from a group, by anyone).
- **The authority ladder** (§20a): owner → sudo → group-scoped
  deputies.  Sudo is recognition, never ownership; deputies
  hold their rung in ONE group.  Every edit lands in the
  record's held audit ring.
- **Recovery** (§20e): if the owner's platform id changes, a
  sealed DM rebind — identity claim + passphrase proof
  against the core seal — rewrites the binding.  The
  passphrase passes through and is never stored or logged.
- **The media watermark** (§19/§20c): the owner's mark as a
  REAL image stamp, drawn by :func:`apply_media_watermark`;
  whether it applies is decided by the typography organ from
  the presentation settings (one settings home) — this module
  only draws.
- **Impersonation sensing** (§20d): a member wearing the
  owner's handle or Devon's name is flagged for the owner —
  a sensor with stated features, not a routing shortcut.

The store follows the house discipline: atomic writes, and an
unreadable file is quarantined and RAISED — never silently
reset, because a reset would unbind the owner from his own
system.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .failures import record_failure

__all__ = [
    "OwnerRegistry", "OwnerStoreError", "OwnerAuthorityError",
    "canon_platform", "same_person", "apply_media_watermark",
    "impersonation_check", "verification_card", "status_card",
    "default_registry", "ROLE_OWNER", "ROLE_SUDO",
    "ROLE_DEPUTY", "LADDER_TEXT",
]

ROLE_OWNER = "owner"
ROLE_SUDO = "sudo"
ROLE_DEPUTY = "deputy"

_AUDIT_RING = 200

#: The ladder, stated once — the owner card prints it and the
#: registry enforces exactly this (powers are enumerated, not
#: implied).
LADDER_TEXT = (
    "owner — everything: edits this record, the sudo list and "
    "deputies; sets the mark; rebinds by seal; never moderated.\n"
    "sudo — recognized everywhere, exempt from automatic "
    "actions; edits nothing here; is never the owner.\n"
    "deputy — sudo's exemption inside ONE named group only; "
    "nothing beyond it."
)


class OwnerStoreError(RuntimeError):
    """The owner record exists but cannot be read."""


class OwnerAuthorityError(PermissionError):
    """An edit was attempted without the owner's authority."""


def canon_platform(platform: str) -> str:
    """Group keys and adapters spell platforms several ways
    (``telegram-bot``, ``telegram-userbot``…); the record
    keeps one name per platform."""
    p = str(platform or "").strip().lower()
    if p.startswith("telegram"):
        return "telegram"
    if p.startswith("whatsapp"):
        return "whatsapp"
    if p.startswith("discord"):
        return "discord"
    return p


def _digits(value: str) -> str:
    return "".join(c for c in str(value or "") if c.isdigit())


def same_person(platform: str, a: str, b: str) -> bool:
    """Identity equality on one platform.  WhatsApp ids arrive
    as bare digits and as JIDs (``<digits>@s.whatsapp.net``) —
    the digit form is the person; everywhere else the id is
    compared exactly (after trim/case-fold)."""
    a, b = str(a or "").strip(), str(b or "").strip()
    if not a or not b:
        return False
    if canon_platform(platform) == "whatsapp":
        da, db = _digits(a), _digits(b)
        return bool(da) and da == db
    return a.lower() == b.lower()


def _bound(ids: list[str], platform: str, user_id: str) -> bool:
    return any(same_person(platform, bound, user_id)
               for bound in ids)


@dataclass
class OwnerRegistry:
    """The one owner record, persisted beside the group store
    as ``owner.json``."""

    path: Path
    data: dict[str, Any] = field(default_factory=dict)

    # ── persistence ──────────────────────────────────────────
    def load(self) -> "OwnerRegistry":
        if not self.path.exists():
            self.data = self._empty()
            return self
        try:
            raw = json.loads(self.path.read_text(
                encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("owner record root is not an "
                                 "object")
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError as _exc:
                record_failure("owner_system.load", _exc)
            raise OwnerStoreError(
                f"owner record at {self.path} is unreadable "
                f"({type(exc).__name__}: {exc}); quarantined "
                f"to {quarantine} — refusing to start from an "
                f"empty record") from exc
        merged = self._empty()
        merged.update({k: raw.get(k, merged[k]) for k in merged})
        self.data = merged
        return self

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {"identities": {}, "handles": [], "sudo": {},
                "deputies": {}, "mark_text": "", "audit": []}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # ── reads (anyone may read a role; reads are not power) ──
    def identities(self, platform: str) -> list[str]:
        plat = canon_platform(platform)
        return list((self.data.get("identities") or {})
                    .get(plat) or [])

    def handles(self) -> list[str]:
        return list(self.data.get("handles") or [])

    def mark_text(self) -> str:
        return str(self.data.get("mark_text") or "")

    def is_owner(self, platform: str, user_id: str) -> bool:
        plat = canon_platform(platform)
        return _bound(self.identities(plat), plat, user_id)

    def is_sudo(self, platform: str, user_id: str) -> bool:
        plat = canon_platform(platform)
        ids = (self.data.get("sudo") or {}).get(plat) or []
        return _bound(ids, plat, user_id)

    def is_deputy(self, group_key: str, user_id: str) -> bool:
        plat = canon_platform(str(group_key).split(":", 1)[0])
        ids = (self.data.get("deputies") or {}) \
            .get(str(group_key)) or []
        return _bound(ids, plat, user_id)

    def role_of(self, platform: str, user_id: str, *,
                group_key: str = "") -> str:
        """owner | sudo | deputy | "" — the ladder rung this
        person stands on here.  Deputy is group-scoped: the
        same id elsewhere is a plain member."""
        if self.is_owner(platform, user_id):
            return ROLE_OWNER
        if self.is_sudo(platform, user_id):
            return ROLE_SUDO
        if group_key and self.is_deputy(group_key, user_id):
            return ROLE_DEPUTY
        return ""

    def owner_elsewhere(self, platform: str, user_id: str,
                          username: str, name: str) -> str:
        """Impersonation sensing (§20d) for ONE member: "" when
        clean, else the flag reason.  The id check comes
        first — the owner wearing his own handle is not an
        impersonator."""
        if self.is_owner(platform, user_id):
            return ""
        handles = list(self.handles())
        try:
            from ...core.owner import OWNER_IDENTITIES

            handles.extend(h for h in OWNER_IDENTITIES
                           if h not in handles)
        except Exception:  # noqa: BLE001 - seal module is
            pass               # additive, never required
        return impersonation_check(name, username,
                                   owner_handles=handles)

    # ── seeding (first formation, from existing sources) ─────
    def seed(self, *, telegram_ids: Any = (),
             accounts: Any = (), handles: Any = ()) -> bool:
        """Form the record from the sources that already bind
        the owner (config sender ids; the §15 registry's
        accounts — his paired numbers ARE his WhatsApp
        identities, a linked Telegram account's remote id is
        his Telegram id).  Seeding only fills EMPTY slots and
        never overwrites an edit; returns True when it wrote.
        Handles seed from the core seal's ingrained list."""
        changed = False
        ident = self.data.setdefault("identities", {})
        for raw in telegram_ids or ():
            tid = str(raw or "").strip()
            if tid and not _bound(ident.get("telegram") or [],
                                  "telegram", tid):
                ident.setdefault("telegram", []).append(tid)
                changed = True
        for rec in accounts or ():
            plat = canon_platform(
                str(getattr(rec, "platform", "") or ""))
            if plat == "whatsapp":
                digits = _digits(
                    str(getattr(rec, "identity", "") or ""))
                if digits and not _bound(
                        ident.get("whatsapp") or [],
                        "whatsapp", digits):
                    ident.setdefault("whatsapp", []) \
                        .append(digits)
                    changed = True
            elif plat == "telegram":
                rid = str(getattr(rec, "remote_id", "")
                          or "").strip()
                if rid and not _bound(
                        ident.get("telegram") or [],
                        "telegram", rid):
                    ident.setdefault("telegram", []).append(rid)
                    changed = True
        known = self.data.setdefault("handles", [])
        for handle in handles or ():
            h = str(handle or "").strip().lstrip("@")
            if h and h.lower() not in [k.lower() for k in known]:
                known.append(h)
                changed = True
        if changed:
            self._audit("seed", actor="system",
                        detail="record formed from existing "
                               "bindings")
            self.save()
        return changed

    # ── edits (owner-only, never from a group) ───────────────
    def _require_owner(self, actor_role: str, *,
                       via_group: bool) -> None:
        if via_group:
            raise OwnerAuthorityError(
                "the owner record is never editable from a "
                "group — DM only")
        if actor_role != ROLE_OWNER:
            raise OwnerAuthorityError(
                "only the owner edits the owner record")

    def _audit(self, action: str, *, actor: str,
               platform: str = "", target: str = "",
               detail: str = "") -> None:
        ring = self.data.setdefault("audit", [])
        ring.append({
            "ts": time.time(), "action": action,
            "actor": actor, "platform": platform,
            "target": target, "detail": detail[:200]})
        del ring[:-_AUDIT_RING]

    def audit(self, limit: int = 20) -> list[dict[str, Any]]:
        return list((self.data.get("audit") or [])[-limit:])

    def bind(self, actor_role: str, *, via_group: bool,
             platform: str, user_id: str,
             actor: str = "") -> bool:
        self._require_owner(actor_role, via_group=via_group)
        plat = canon_platform(platform)
        uid = str(user_id or "").strip()
        if not uid:
            raise OwnerAuthorityError("no identity to bind")
        ident = self.data.setdefault("identities", {})
        if _bound(ident.get(plat) or [], plat, uid):
            return False
        ident.setdefault(plat, []).append(uid)
        self._audit("bind", actor=actor or actor_role,
                    platform=plat, target=uid)
        self.save()
        return True

    def unbind(self, actor_role: str, *, via_group: bool,
               platform: str, user_id: str,
               actor: str = "") -> bool:
        self._require_owner(actor_role, via_group=via_group)
        plat = canon_platform(platform)
        ident = self.data.setdefault("identities", {})
        kept = [i for i in ident.get(plat) or []
                if not same_person(plat, i, user_id)]
        if len(kept) == len(ident.get(plat) or []):
            return False
        ident[plat] = kept
        self._audit("unbind", actor=actor or actor_role,
                    platform=plat, target=str(user_id))
        self.save()
        return True

    def set_sudo(self, actor_role: str, *, via_group: bool,
                 platform: str, user_id: str, on: bool,
                 actor: str = "") -> bool:
        self._require_owner(actor_role, via_group=via_group)
        plat = canon_platform(platform)
        sudo = self.data.setdefault("sudo", {})
        rows = list(sudo.get(plat) or [])
        has = _bound(rows, plat, user_id)
        if on and not has:
            rows.append(str(user_id).strip())
        elif not on and has:
            rows = [i for i in rows
                    if not same_person(plat, i, user_id)]
        else:
            return False
        sudo[plat] = rows
        self._audit("sudo_add" if on else "sudo_remove",
                    actor=actor or actor_role, platform=plat,
                    target=str(user_id))
        self.save()
        return True

    def set_deputy(self, actor_role: str, *, via_group: bool,
                   group_key: str, user_id: str, on: bool,
                   actor: str = "") -> bool:
        self._require_owner(actor_role, via_group=via_group)
        plat = canon_platform(str(group_key).split(":", 1)[0])
        deputies = self.data.setdefault("deputies", {})
        rows = list(deputies.get(str(group_key)) or [])
        has = _bound(rows, plat, user_id)
        if on and not has:
            rows.append(str(user_id).strip())
        elif not on and has:
            rows = [i for i in rows
                    if not same_person(plat, i, user_id)]
        else:
            return False
        deputies[str(group_key)] = rows
        self._audit("deputy_add" if on else "deputy_remove",
                    actor=actor or actor_role, platform=plat,
                    target=str(user_id),
                    detail=str(group_key))
        self.save()
        return True

    def set_mark_text(self, actor_role: str, *,
                      via_group: bool, text: str,
                      actor: str = "") -> None:
        self._require_owner(actor_role, via_group=via_group)
        self.data["mark_text"] = str(text or "").strip()[:120]
        self._audit("mark_set", actor=actor or actor_role,
                    detail=self.data["mark_text"])
        self.save()

    def rebind_with_seal(self, *, platform: str,
                         new_id: str, identity_claim: str,
                         passphrase: str) -> bool:
        """§20e recovery: the claimant proves the seal
        (identity claim + passphrase, the core owner's two
        halves) in DM, and the platform binding is rewritten
        to the new id.  The passphrase is verified and
        forgotten — never stored, never logged, never in the
        audit detail.  Refuses honestly when no seal is baked.
        """
        from ...core.owner import seal_configured, verify_owner

        if not seal_configured():
            raise OwnerAuthorityError(
                "no owner seal is baked into this build — "
                "recovery by seal is unavailable until the "
                "owner runs `nm owner seal`")
        if not verify_owner(identity_claim, passphrase):
            raise OwnerAuthorityError(
                "the seal did not verify — identity claim and "
                "passphrase must both be the owner's")
        plat = canon_platform(platform)
        uid = str(new_id or "").strip()
        if not uid:
            raise OwnerAuthorityError("no new identity given")
        self.data.setdefault("identities", {})[plat] = [uid]
        self._audit("rebind", actor="seal",
                    platform=plat, target=uid,
                    detail="sealed DM recovery")
        self.save()
        return True


def default_registry(workspace_dir: str | Path) -> OwnerRegistry:
    """The runtime's record: beside the group store."""
    root = Path(workspace_dir) / "data" / "social"
    return OwnerRegistry(root / "owner.json").load()


# ── the media watermark (§19/§20c: the mark, drawn) ──────────

def apply_media_watermark(data: bytes, text: str
                          ) -> tuple[bytes, bool]:
    """Draw the owner's mark onto an image: a small dark pill
    with the signature, bottom-right, sized to the image.
    Returns (bytes, applied) — non-image input and empty text
    come back untouched with applied=False; the caller
    journals which.  Output is PNG (the delivery formats all
    accept it)."""
    if not data or not str(text or "").strip():
        return data, False
    try:
        import io

        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return data, False
    try:
        with Image.open(io.BytesIO(data)) as img:
            base = img.convert("RGBA")
    except Exception:  # noqa: BLE001 - not an image: the
        return data, False          # text mark still applies
    width, height = base.size
    if width < 40 or height < 40:
        return data, False
    label = str(text).strip()
    font_size = max(11, min(28, width // 42))
    font = None
    try:
        from ...media.typography import get_service
        font = get_service().pil_font("caption", font_size, text=label)
    except Exception:  # noqa: BLE001 - legacy ladder below
        font = None
    if font is None:
        for candidate in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except (OSError, IOError):
                continue
    if font is None:
        font = ImageFont.load_default()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        box = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = box[2] - box[0], box[3] - box[1]
    except Exception:  # noqa: BLE001 - ancient PIL fallback
        text_w, text_h = (len(label) * font_size // 2,
                          font_size)
    pad = max(6, font_size // 2)
    margin = max(8, width // 100)
    x1 = width - margin
    y1 = height - margin
    x0 = x1 - text_w - pad * 2
    y0 = y1 - text_h - pad * 2
    if x0 < 0 or y0 < 0:
        return data, False
    draw.rounded_rectangle([x0, y0, x1, y1],
                           radius=(y1 - y0) // 2,
                           fill=(0, 0, 0, 110))
    draw.text((x0 + pad, y0 + pad), label, font=font,
              fill=(255, 255, 255, 235))
    out = Image.alpha_composite(base, overlay)
    buf = io.BytesIO()
    out.convert("RGB").save(buf, format="PNG")
    return buf.getvalue(), True


# ── impersonation sensing (§20d) ─────────────────────────────

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a",
                       "5": "s", "7": "t", "@": "a", "$": "s"})


def _normalize(value: str) -> str:
    text = str(value or "").lower().translate(_LEET)
    return re.sub(r"[^a-z0-9]", "", text)


def impersonation_check(name: str, username: str, *,
                        owner_handles: Any = (),
                        bot_names: Any = ("devon",)
                        ) -> str:
    """Flag a member wearing the owner's (or Devon's) face.
    Features, stated plainly: the username normalizes to an
    owner handle; the display name normalizes to an owner
    handle; the display name normalizes to the bot's own name.
    Anything weaker (a similar name, a shared first name) is
    NOT flagged — the sensor cries wolf only on a worn face.
    Returns the reason, or ""."""
    user_norm = _normalize(username)
    name_norm = _normalize(name)
    for handle in owner_handles or ():
        target = _normalize(handle)
        if not target:
            continue
        if user_norm and user_norm == target:
            return (f"username copies the owner's handle "
                    f"@{handle}")
        if name_norm and name_norm == target:
            return (f"display name copies the owner's handle "
                    f"@{handle}")
    for bot in bot_names or ():
        target = _normalize(bot)
        if target and name_norm == target:
            return "display name copies Devon's own name"
    return ""


# ── cards (composed from real state; the runtime sources it) ─

def _duration(seconds: float) -> str:
    total = int(max(0.0, seconds))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def verification_card(registry: OwnerRegistry, *,
                      seal_configured: bool,
                      account_lines: Any = (),
                      sources: Any = ()) -> str:
    """§19 verification, on one card: who the owner is on each
    platform (from the record), how each binding was proved,
    the ladder, and the seal's state.  Ids here are the
    OWNER's own, in his own DM — never member-facing text."""
    lines = ["👑 <b>Owner — verification</b>"]
    ident = registry.data.get("identities") or {}
    if ident:
        for plat in sorted(ident):
            ids = ident[plat]
            if ids:
                lines.append(
                    f"{plat}: " + ", ".join(str(i) for i in ids))
    else:
        lines.append("no platform identities bound yet")
    handles = registry.handles()
    if handles:
        lines.append("handles: " + ", ".join(
            f"@{h}" for h in handles))
    for line in sources or ():
        lines.append(f"· {line}")
    lines.append(
        "seal: " + ("baked — sealed DM recovery is armed"
                    if seal_configured else
                    "not baked — recovery by seal is "
                    "unavailable (nm owner seal)"))
    sudo = registry.data.get("sudo") or {}
    sudo_rows = [f"{p}: {', '.join(map(str, ids))}"
                 for p, ids in sorted(sudo.items()) if ids]
    lines.append("sudo: " + (" · ".join(sudo_rows)
                             if sudo_rows else "none"))
    deputies = registry.data.get("deputies") or {}
    dep_rows = sum(len(v or []) for v in deputies.values())
    lines.append(f"deputies: {dep_rows} grant(s) across "
                 f"{len([k for k, v in deputies.items() if v])} "
                 "group(s)")
    if account_lines:
        lines.append("<b>paired accounts (§15)</b>")
        lines.extend(f"· {line}" for line in account_lines)
    lines.append("<b>the ladder</b>")
    lines.extend(LADDER_TEXT.splitlines())
    return "\n".join(lines)


def status_card(*, uptime_s: float | None, version: str,
                commit: str = "", accounts: dict | None = None,
                queues: dict | None = None,
                last_briefing_ts: float | None = None,
                now: float | None = None) -> str:
    """The owner's status surface — uptime lives HERE (the
    /ping ruling), with the version, the §15 pool, the live
    queue depths and the last briefing.  Every field is
    sourced by the caller from real state; an unsourced field
    renders as absence, never as a plausible number."""
    lines = ["🛰️ <b>Owner — status</b>"]
    lines.append("uptime: " + (_duration(uptime_s)
                               if uptime_s is not None
                               else "unknown"))
    lines.append(f"version: {version or 'unknown'}"
                 + (f" · commit {commit}" if commit else
                    " · commit not stamped in this build"))
    acc = accounts or {}
    if acc:
        lines.append(
            f"accounts: {acc.get('registered', 0)} registered · "
            f"{acc.get('connected', 0)} connected · "
            f"{acc.get('dropped', 0)} dropped")
    else:
        lines.append("accounts: registry unread")
    if queues:
        lines.append("queues: " + " · ".join(
            f"{label} {int(n)}" for label, n in queues.items()))
    if last_briefing_ts:
        when = time.strftime("%Y-%m-%d %H:%M UTC",
                             time.gmtime(last_briefing_ts))
        ago = _duration((now or time.time()) - last_briefing_ts)
        lines.append(f"last briefing: {when} ({ago} ago)")
    else:
        lines.append("last briefing: none on record")
    return "\n".join(lines)
