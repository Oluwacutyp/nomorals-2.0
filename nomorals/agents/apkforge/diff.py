"""ApkForge diff — component-wise comparison of two dossiers.

The op MobSF doesn't have: permissions ±, components ± with exported
flips flagged, certificate same/different (the "same publisher?"
answer), dex deltas (classes added/removed/changed — "changed" by the
dossier's structural class hash: class name + member signatures, so a
pure bytecode-body change with identical signatures is *not* claimed),
endpoint deltas, native/SDK deltas, size delta.  Everything renders
as one readable change report and journals to both projects.
"""

from __future__ import annotations

from typing import Any

__all__ = ["diff_dossiers", "render_diff_md", "render_diff_card"]


def _set_delta(a: set, b: set) -> dict[str, list]:
    return {"added": sorted(b - a), "removed": sorted(a - b)}


def _components(a: dict, b: dict) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kind in ("activities", "services", "receivers", "providers"):
        am = {c["name"]: c for c in (a.get(kind) or [])}
        bm = {c["name"]: c for c in (b.get(kind) or [])}
        flips = []
        for name in sorted(set(am) & set(bm)):
            ea, eb = am[name].get("exported"), bm[name].get("exported")
            if ea != eb:
                flips.append({"name": name, "from": ea, "to": eb})
        out[kind] = {**_set_delta(set(am), set(bm)),
                     "exported_changed": flips}
    return out


def diff_dossiers(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """``a`` is the older/baseline dossier, ``b`` the newer one.
    "added" always means present in b, absent in a."""
    ia, ib = a.get("identity") or {}, b.get("identity") or {}
    identity = {}
    for field in ("package", "app_name", "version_name", "version_code",
                  "min_sdk", "target_sdk", "main_activity"):
        if ia.get(field) != ib.get(field):
            identity[field] = {"from": ia.get(field), "to": ib.get(field)}

    pa = {p["name"]: p for p in a.get("permissions") or []}
    pb = {p["name"]: p for p in b.get("permissions") or []}
    weight_changes = [
        {"name": n, "from": pa[n].get("weight"), "to": pb[n].get("weight")}
        for n in sorted(set(pa) & set(pb))
        if pa[n].get("weight") != pb[n].get("weight")]

    ca = (a.get("signing") or {}).get("certificates") or []
    cb = (b.get("signing") or {}).get("certificates") or []
    fa = ca[0].get("sha256") if ca else ""
    fb = cb[0].get("sha256") if cb else ""
    cert = {"same_signer": bool(fa) and fa == fb,
            "a_sha256": fa or "", "b_sha256": fb or "",
            "a_subject": ca[0].get("subject") if ca else "",
            "b_subject": cb[0].get("subject") if cb else ""}

    da, db = a.get("dex") or {}, b.get("dex") or {}
    ha, hb = da.get("class_hashes") or {}, db.get("class_hashes") or {}
    common = set(ha) & set(hb)
    changed = sorted(n for n in common if ha[n] != hb[n])
    classes = {**_set_delta(set(ha), set(hb)), "changed": changed}
    ta, tb = da.get("totals") or {}, db.get("totals") or {}

    na = {n["path"] for n in a.get("native") or []}
    nb = {n["path"] for n in b.get("native") or []}
    sa = {s["name"] for s in a.get("sdk_signatures") or []}
    sb = {s["name"] for s in b.get("sdk_signatures") or []}

    return {
        "a": {"source_name": a.get("source_name"),
              "source_sha256": a.get("source_sha256"),
              "version": ia.get("version_name")},
        "b": {"source_name": b.get("source_name"),
              "source_sha256": b.get("source_sha256"),
              "version": ib.get("version_name")},
        "identity": identity,
        "size_delta": int(b.get("size") or 0) - int(a.get("size") or 0),
        "permissions": {**_set_delta(set(pa), set(pb)),
                        "weight_changed": weight_changes},
        "components": _components(a.get("components") or {},
                                  b.get("components") or {}),
        "certificate": cert,
        "dex": {
            "classes": classes,
            "methods_delta": int(tb.get("methods") or 0)
                             - int(ta.get("methods") or 0),
            "strings_delta": int(tb.get("strings") or 0)
                             - int(ta.get("strings") or 0),
        },
        "endpoints": _set_delta(set(da.get("urls") or []),
                                set(db.get("urls") or [])),
        "native": _set_delta(na, nb),
        "sdk": _set_delta(sa, sb),
        "changed_note": "class 'changed' = structural hash changed "
                        "(name + member signatures); a bytecode-body "
                        "change with identical signatures is not "
                        "detected and is not claimed",
    }


def _delta_lines(title: str, delta: dict[str, list]) -> list[str]:
    out = []
    if delta.get("added"):
        out.append(f"- {title} added: {', '.join(delta['added'])}")
    if delta.get("removed"):
        out.append(f"- {title} removed: {', '.join(delta['removed'])}")
    return out


def render_diff_md(d: dict[str, Any]) -> str:
    a, b = d.get("a") or {}, d.get("b") or {}
    out = [f"# APK diff — {a.get('version') or a.get('source_name')} → "
           f"{b.get('version') or b.get('source_name')}", ""]
    if d.get("identity"):
        out.append("## Identity")
        for field, change in d["identity"].items():
            out.append(f"- {field}: {change['from']!r} → {change['to']!r}")
        out.append("")
    out.append(f"size delta: {d.get('size_delta', 0):+,} bytes")
    cert = d.get("certificate") or {}
    out.append(f"signer: {'SAME certificate' if cert.get('same_signer') else 'DIFFERENT certificate'}"
               f" ({cert.get('a_subject') or 'none'} → "
               f"{cert.get('b_subject') or 'none'})")
    out.append("")
    out.append("## Permissions")
    lines = _delta_lines("permission", d.get("permissions") or {})
    out.extend(lines or ["- no permission changes"])
    for w in (d.get("permissions") or {}).get("weight_changed") or []:
        out.append(f"- weight changed: {w['name']} {w['from']} → {w['to']}")
    out.append("")
    out.append("## Components")
    any_comp = False
    for kind, delta in (d.get("components") or {}).items():
        lines = _delta_lines(kind[:-1], delta)
        for flip in delta.get("exported_changed") or []:
            lines.append(f"- {kind[:-1]} exported flip: {flip['name']} "
                         f"{flip['from']} → {flip['to']}")
        if lines:
            any_comp = True
            out.extend(lines)
    if not any_comp:
        out.append("- no component changes")
    out.append("")
    dex = d.get("dex") or {}
    out.append("## Dex")
    cls = dex.get("classes") or {}
    out.extend(_delta_lines("class", cls) or ["- no class additions/removals"])
    if cls.get("changed"):
        out.append(f"- classes changed (structural): "
                   f"{', '.join(cls['changed'][:20])}")
    out.append(f"- methods delta: {dex.get('methods_delta', 0):+,} · "
               f"strings delta: {dex.get('strings_delta', 0):+,}")
    out.append(f"_{d.get('changed_note', '')}_")
    ep = _delta_lines("endpoint", d.get("endpoints") or {})
    if ep:
        out.append("")
        out.append("## Endpoints")
        out.extend(ep)
    nat = _delta_lines("native lib", d.get("native") or {})
    sdk = _delta_lines("SDK", d.get("sdk") or {})
    if nat or sdk:
        out.append("")
        out.append("## Native / SDK")
        out.extend(nat + sdk)
    out.append("")
    return "\n".join(out)


def render_diff_card(d: dict[str, Any]) -> str:
    a, b = d.get("a") or {}, d.get("b") or {}
    perms = d.get("permissions") or {}
    cls = (d.get("dex") or {}).get("classes") or {}
    cert = d.get("certificate") or {}
    bits = []
    if perms.get("added"):
        bits.append(f"+{len(perms['added'])} permission(s)")
    if perms.get("removed"):
        bits.append(f"-{len(perms['removed'])} permission(s)")
    if cls.get("added"):
        bits.append(f"+{len(cls['added'])} class(es)")
    if cls.get("removed"):
        bits.append(f"-{len(cls['removed'])} class(es)")
    if cls.get("changed"):
        bits.append(f"{len(cls['changed'])} class(es) changed")
    summary = ", ".join(bits) if bits else "no structural deltas"
    return (f"🔀 diff {a.get('version') or '?'} → {b.get('version') or '?'}: "
            f"{summary}; signer "
            f"{'same' if cert.get('same_signer') else 'DIFFERENT'}; "
            f"size {d.get('size_delta', 0):+,} bytes")
