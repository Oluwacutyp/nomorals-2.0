"""ApkForge dissect — androguard anatomy → the Dossier.

Every claim in a dossier traces to an actual parse of the artifact's
own bytes: manifest anatomy from the binary AXML (via androguard),
permission weight from androguard's AOSP permission database, dex
census from the Dalvik parser, certificate facts from the PKCS#7 /
signing-block parse, packer/compiler verdicts from APKiD's YARA rules.
Nothing is inferred from the filename or guessed.  Static analysis
only: a dossier says what an app *contains* and *can* do, never what
it *does* at runtime — and says so on its face.
"""

from __future__ import annotations

import hashlib
import re
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

__all__ = [
    "dissect_apk",
    "light_identity",
    "render_dossier_card",
    "render_dossier_md",
    "SDK_SIGNATURES",
    "androguard_available",
]

try:  # androguard logs verbosely via loguru; scope-silence its logger
    from loguru import logger as _loguru

    _loguru.disable("androguard")
except Exception:  # noqa: BLE001
    pass

_NS = "{http://schemas.android.com/apk/res/android}"
_CLASS_NAMES_CAP = 20000

# Seed SDK/tracker signature list (package prefixes, Exodus-style).
# Deliberately small in A1 and labeled as a seed in every dossier —
# A3 grows it under the open-sources law.  Absence of a hit is never
# proof of absence.
SDK_SIGNATURES: tuple[tuple[str, str, str], ...] = (
    ("com.google.android.gms.ads", "Google AdMob", "ads"),
    ("com.google.android.gms.analytics", "Google Analytics (GMS)", "analytics"),
    ("com.google.firebase.analytics", "Firebase Analytics", "analytics"),
    ("com.google.firebase.crashlytics", "Firebase Crashlytics", "crash-reporting"),
    ("com.google.firebase.messaging", "Firebase Cloud Messaging", "push"),
    ("com.facebook.ads", "Meta Audience Network", "ads"),
    ("com.facebook.android", "Meta Facebook SDK", "social"),
    ("com.appsflyer", "AppsFlyer", "attribution"),
    ("com.adjust.sdk", "Adjust", "attribution"),
    ("io.branch", "Branch", "attribution"),
    ("com.mixpanel", "Mixpanel", "analytics"),
    ("com.amplitude", "Amplitude", "analytics"),
    ("com.segment.analytics", "Segment", "analytics"),
    ("com.flurry", "Flurry", "analytics"),
    ("com.unity3d.ads", "Unity Ads", "ads"),
    ("com.applovin", "AppLovin", "ads"),
    ("com.bugsnag", "Bugsnag", "crash-reporting"),
    ("io.sentry", "Sentry", "crash-reporting"),
    ("io.intercom", "Intercom", "support"),
    ("com.count.android", "Countly", "analytics"),
    ("org.matomo", "Matomo", "analytics"),
    ("com.squareup.okhttp3", "OkHttp", "networking"),
    ("retrofit2", "Retrofit", "networking"),
    ("com.android.volley", "Volley", "networking"),
    ("io.reactivex", "RxJava", "reactive"),
    ("com.bumptech.glide", "Glide", "image-loading"),
    ("com.squareup.picasso", "Picasso", "image-loading"),
)

_URL_RE = re.compile(r"https?://[^\s\"'<>\\\]]+")
_DOMAIN_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:com|net|org|io|co|"
    r"app|dev|ai|cloud|info|biz|me|tv|xyz|online|site|tech)\b", re.I)


def androguard_available() -> bool:
    try:
        import androguard  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pkg_of(descriptor: str) -> str:
    """Lcom/example/app/Foo; → com.example.app"""
    d = descriptor.strip()
    if d.startswith("L"):
        d = d[1:]
    if d.endswith(";"):
        d = d[:-1]
    parts = d.split("/")
    return ".".join(parts[:-1]) if len(parts) > 1 else ""


def _dotted(descriptor: str) -> str:
    d = descriptor.strip()
    if d.startswith("L"):
        d = d[1:]
    if d.endswith(";"):
        d = d[:-1]
    return d.replace("/", ".")


# ── light identity (intake-time; tolerant) ───────────────────────────

def light_identity(path: str | Path) -> dict[str, Any]:
    """Package/version/cert in one cheap parse — the project card's
    facts from minute one.  Unparseable fields come back empty, never
    invented."""
    out: dict[str, Any] = {"package": "", "app_name": "",
                           "version_name": "", "version_code": None,
                           "cert_sha256": ""}
    try:
        from androguard.core.apk import APK

        a = APK(str(path))
        out["package"] = a.get_package() or ""
        out["app_name"] = a.get_app_name() or ""
        out["version_name"] = a.get_androidversion_name() or ""
        code = a.get_androidversion_code()
        out["version_code"] = int(code) if str(code).isdigit() else None
        certs = a.get_certificates()
        if certs:
            out["cert_sha256"] = _sha256_bytes(certs[0].dump())
    except Exception:  # noqa: BLE001 — light means tolerant
        pass
    return out


# ── manifest components from the decoded DOM ─────────────────────────

def _attr(el: Any, name: str) -> str | None:
    val = el.get(f"{_NS}{name}")
    if val is None:
        val = el.get(name)
    return val


def _bool_attr(el: Any, name: str) -> bool | None:
    val = _attr(el, name)
    if val is None:
        return None
    return val.strip().lower() == "true"


def _components_from_dom(a: Any, package: str) -> tuple[dict, dict]:
    """Components + application flags from the decoded manifest DOM —
    the only place `exported` and intent filters truly live."""
    components: dict[str, list[dict[str, Any]]] = {
        "activities": [], "services": [], "receivers": [], "providers": []}
    app_flags: dict[str, Any] = {}
    try:
        root = a.get_android_manifest_xml()
    except Exception:  # noqa: BLE001 — fall back to androguard lists
        return components, app_flags
    if root is None:
        return components, app_flags

    def resolve(name: str) -> str:
        if name.startswith("."):
            return package + name
        if "." not in name:
            return f"{package}.{name}"
        return name

    app_el = None
    for el in root.iter():
        if el.tag == "application":
            app_el = el
            break
    if app_el is None:
        return components, app_flags
    app_flags = {
        "debuggable": _bool_attr(app_el, "debuggable"),
        "allow_backup": _bool_attr(app_el, "allowBackup"),
        "uses_cleartext_traffic": _bool_attr(app_el, "usesCleartextTraffic"),
        "network_security_config": bool(
            _attr(app_el, "networkSecurityConfig")),
        "request_legacy_external_storage": _bool_attr(
            app_el, "requestLegacyExternalStorage"),
    }
    tag_map = {"activity": "activities", "activity-alias": "activities",
               "service": "services", "receiver": "receivers",
               "provider": "providers"}
    for el in app_el:
        kind = tag_map.get(el.tag)
        if kind is None:
            continue
        name = resolve(_attr(el, "name") or "")
        filters: list[dict[str, Any]] = []
        for child in el:
            if child.tag != "intent-filter":
                continue
            actions, categories = [], []
            for sub in child:
                if sub.tag == "action":
                    actions.append(_attr(sub, "name") or "")
                elif sub.tag == "category":
                    categories.append(_attr(sub, "name") or "")
            filters.append({"actions": actions, "categories": categories})
        row: dict[str, Any] = {
            "name": name,
            "exported": _bool_attr(el, "exported"),
            "intent_filters": filters,
        }
        if kind == "providers":
            row["authorities"] = _attr(el, "authorities") or ""
        components[kind].append(row)
    for rows in components.values():
        rows.sort(key=lambda r: r["name"])
    return components, app_flags


# ── APKiD verdict ────────────────────────────────────────────────────

_APKID_RULES: Any = None


def _apkid_rules() -> Any:
    global _APKID_RULES
    if _APKID_RULES is None:
        from apkid.apkid import RulesManager

        _APKID_RULES = RulesManager().load()
    return _APKID_RULES


def _verdict_bucket(rule: str, description: str) -> str:
    text = f"{rule} {description}".lower()
    if "packer" in text or "packed" in text:
        return "packer"
    if "protector" in text or "protection" in text:
        return "protector"
    if "obfuscat" in text:
        return "obfuscator"
    if "compiler" in text:
        return "compiler"
    if ("anti" in text or "emulator" in text or "virtual machine" in text
            or "root" in text or "debug" in text):
        return "anti_analysis"
    return "other"


def apkid_scan(path: str | Path, timeout: int = 120) -> dict[str, Any]:
    """APKiD fingerprint scan → per-entry matches + verdict buckets.

    Absence/failure is reported as a fact (``available: False`` with
    the reason), never papered over."""
    try:
        import apkid  # noqa: F401
        from apkid.apkid import Options, Scanner
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "reason": f"apkid not installed ({exc.__class__.__name__})"}
    try:
        scanner = Scanner(_apkid_rules(),
                          Options(timeout=timeout, verbose=False,
                                  recursive=True))
        raw = scanner.scan_file(str(path))
    except Exception as exc:  # noqa: BLE001
        return {"available": False,
                "reason": f"apkid scan failed: {exc.__class__.__name__}: {exc}"}
    matches: dict[str, list[dict[str, str]]] = {}
    verdicts: dict[str, list[str]] = {}
    for entry, found in raw.items():
        rows = []
        for m in found:
            meta = getattr(m, "meta", None) or {}
            desc = str(meta.get("description") or "")
            rule = str(getattr(m, "rule", ""))
            rows.append({"rule": rule,
                         "namespace": str(getattr(m, "namespace", "")),
                         "description": desc})
            bucket = _verdict_bucket(rule, desc)
            label = desc or rule
            if label and label not in verdicts.setdefault(bucket, []):
                verdicts[bucket].append(label)
        matches[str(entry)] = rows
    return {"available": True, "matches": matches, "verdicts": verdicts}


# ── dex census ───────────────────────────────────────────────────────

def _dex_census(a: Any) -> dict[str, Any]:
    from androguard.core.dex import DEX

    per_dex: list[dict[str, Any]] = []
    class_names: list[str] = []
    class_hashes: dict[str, str] = {}
    total_methods = 0
    total_fields = 0
    api_refs: Counter = Counter()
    api_methods: dict[str, set[str]] = {}
    ref_types: set[str] = set()
    all_strings: list[str] = []
    errors: list[str] = []
    names = []
    try:
        names = list(a.get_dex_names())
    except Exception:  # noqa: BLE001
        names = []
    blobs = []
    try:
        blobs = list(a.get_all_dex())
    except Exception:  # noqa: BLE001
        blobs = []
    for idx, blob in enumerate(blobs):
        name = names[idx] if idx < len(names) else f"classes{'' if idx == 0 else idx + 1}.dex"
        try:
            d = DEX(blob)
        except Exception as exc:  # noqa: BLE001 — one bad dex ≠ bad dossier
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
            continue
        d_classes = 0
        d_methods = 0
        for cdef in d.get_classes():
            cname = str(cdef.get_name())
            d_classes += 1
            class_names.append(cname)
            try:
                sup = cdef.get_superclassname()
                if sup:
                    ref_types.add(str(sup))
            except Exception:  # noqa: BLE001
                pass
            methods: list[str] = []
            fields: list[str] = []
            try:
                cdata = cdef.get_class_data()
            except Exception:  # noqa: BLE001
                cdata = None
            if cdata is not None:
                for m in (list(cdata.get_direct_methods())
                          + list(cdata.get_virtual_methods())):
                    methods.append(
                        f"{m.get_name()}{m.get_descriptor()}")
                for f in (list(cdata.get_static_fields())
                          + list(cdata.get_instance_fields())):
                    fields.append(f"{f.get_name()}:{f.get_descriptor()}")
            material = "\n".join([cname] + sorted(methods) + sorted(fields))
            class_hashes[cname] = _sha256_bytes(material.encode("utf-8"))
        try:
            mids = list(d.get_methods())
        except Exception:  # noqa: BLE001
            mids = []
        d_methods = len(mids)
        total_methods += d_methods
        try:
            fids = list(d.get_fields())
            total_fields += len(fids)
            for f in fids:
                try:
                    ref_types.add(str(f.get_class_name()))
                except Exception:  # noqa: BLE001
                    continue
        except Exception:  # noqa: BLE001
            pass
        for m in mids:
            try:
                cls = str(m.get_class_name())
            except Exception:  # noqa: BLE001
                continue
            ref_types.add(cls)
            if cls.startswith(("Landroid/", "Ljava/", "Ljavax/",
                               "Landroidx/")):
                api_refs[cls] += 1
                api_methods.setdefault(cls, set()).add(str(m.get_name()))
        try:
            all_strings.extend(str(s) for s in d.get_strings())
        except Exception:  # noqa: BLE001
            pass
        per_dex.append({"name": name, "classes": d_classes,
                        "methods": d_methods,
                        "sha256": _sha256_bytes(blob)})
    rollup = Counter(_pkg_of(c) for c in class_names)
    truncated = len(class_names) > _CLASS_NAMES_CAP
    urls = sorted({u for s in all_strings for u in _URL_RE.findall(s)})
    domains = sorted({dom.lower() for s in all_strings
                      for dom in _DOMAIN_RE.findall(s)} - {
                          u.split("/")[2].lower() if "//" in u else ""
                          for u in urls})
    notable = [{"class": _dotted(cls), "refs": n,
                "methods": sorted(api_methods.get(cls, set()))[:12]}
               for cls, n in api_refs.most_common(40)]
    # SDK census scans every type the dex REFERENCES (method/field
    # tables, superclasses) plus every defined class — an embedded
    # SDK shows up in the reference tables even where it defines no
    # classes of its own in this dex.
    dotted = [_dotted(c) for c in ref_types | set(class_names)]
    sdk_hits: list[dict[str, Any]] = []
    for prefix, label, kind in SDK_SIGNATURES:
        hits = [c for c in dotted
                if c == prefix or c.startswith(prefix + ".")]
        if hits:
            sdk_hits.append({"name": label, "kind": kind,
                             "package_prefix": prefix,
                             "class_count": len(hits)})
    return {
        "per_dex": per_dex,
        "errors": errors,
        "totals": {"classes": len(class_names), "methods": total_methods,
                   "fields": total_fields, "strings": len(all_strings)},
        "package_rollup": [
            {"package": pkg, "classes": n}
            for pkg, n in rollup.most_common(30) if pkg],
        "notable_apis": notable,
        "class_names": sorted(set(class_names))[:_CLASS_NAMES_CAP],
        "class_names_truncated": truncated,
        "class_hashes": class_hashes,
        "urls": urls,
        "domains": [d for d in domains if d][:200],
        "_all_strings": all_strings,
        "sdk_hits": sdk_hits,
    }


# ── main entry ───────────────────────────────────────────────────────

def dissect_apk(path: str | Path, *, include_apkid: bool = True,
                source_sha256: str = "") -> dict[str, Any]:
    """Full static dissection → the Dossier (a plain dict).

    Raises RuntimeError when the dissect engine is unavailable or the
    file is not a parseable APK — callers surface that verbatim."""
    if not androguard_available():
        raise RuntimeError(
            "androguard is not installed on this host — install the "
            "apk extra: pip install 'nomorals[apk]'")
    from androguard.core.apk import APK

    path = Path(path)
    data = path.read_bytes()
    sha = source_sha256 or _sha256_bytes(data)
    try:
        a = APK(str(path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"androguard could not parse {path.name} as an APK: "
            f"{exc.__class__.__name__}: {exc}") from exc

    # identity
    def _int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    identity = {
        "package": a.get_package() or "",
        "app_name": a.get_app_name() or "",
        "version_name": a.get_androidversion_name() or "",
        "version_code": _int(a.get_androidversion_code()),
        "min_sdk": _int(a.get_min_sdk_version()),
        "target_sdk": _int(a.get_target_sdk_version()),
        "max_sdk": _int(a.get_max_sdk_version()),
        "main_activity": a.get_main_activity() or "",
        "multidex": bool(a.is_multidex()) if hasattr(a, "is_multidex") else False,
    }

    # signing
    certs: list[dict[str, Any]] = []
    for cert in a.get_certificates():
        try:
            certs.append({
                "subject": cert.subject.human_friendly,
                "issuer": cert.issuer.human_friendly,
                "serial": str(cert.serial_number),
                "sha256": _sha256_bytes(cert.dump()),
                "sha1": hashlib.sha1(cert.dump()).hexdigest(),
                "not_before": str(cert.not_valid_before),
                "not_after": str(cert.not_valid_after),
                "self_signed": cert.subject == cert.issuer,
            })
        except Exception:  # noqa: BLE001 — a cert we cannot read is skipped
            continue
    signing = {
        "signed": bool(a.is_signed()),
        "v1": bool(a.is_signed_v1()),
        "v2": bool(a.is_signed_v2()),
        "v3": bool(a.is_signed_v3()),
        "signature_names": list(a.get_signature_names() or []),
        "certificates": certs,
    }

    # permissions with AOSP weight
    details: dict[str, Any] = {}
    try:
        details = a.get_details_permissions() or {}
    except Exception:  # noqa: BLE001
        details = {}
    permissions: list[dict[str, Any]] = []
    for perm in sorted(set(a.get_permissions() or [])):
        det = details.get(perm) or []
        protection = str(det[0]) if len(det) > 0 else ""
        label = str(det[1]) if len(det) > 1 else ""
        desc = " ".join(str(det[2]).split()) if len(det) > 2 else ""
        base = protection.split("|")[0].strip().lower()
        if base == "dangerous":
            weight = "high"
        elif base.startswith("signature"):
            weight = "signature"
        elif base == "normal":
            weight = "low"
        elif protection:
            weight = base
        else:
            weight = "unclassified"
        permissions.append({
            "name": perm, "protection": protection, "weight": weight,
            "label": label, "description": desc,
            "in_aosp_table": bool(det),
        })
    declared: list[str] = []
    try:
        declared = sorted(set(a.get_declared_permissions() or []))
    except Exception:  # noqa: BLE001
        declared = []

    components, app_flags = _components_from_dom(a, identity["package"])
    if not any(components.values()):
        # DOM parse yielded nothing (exotic AXML) — fall back to
        # androguard's own component lists, exported unknown.
        def _rows(names: list[str]) -> list[dict[str, Any]]:
            return [{"name": n, "exported": None, "intent_filters": []}
                    for n in sorted(names or [])]
        components = {
            "activities": _rows(a.get_activities()),
            "services": _rows(a.get_services()),
            "receivers": _rows(a.get_receivers()),
            "providers": _rows(a.get_providers()),
        }

    census = _dex_census(a)
    all_strings = census.pop("_all_strings")
    sdk_hits = census.pop("sdk_hits")

    # resources / native inventory from the zip itself
    res_counts: Counter = Counter()
    res_sizes: Counter = Counter()
    assets: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    file_count = 0
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            file_count += 1
            name = info.filename
            top = name.split("/", 1)[0] if "/" in name else "(root)"
            res_counts[top] += 1
            res_sizes[top] += info.file_size
            if name.startswith("assets/") and not name.endswith("/"):
                if len(assets) < 200:
                    assets.append({"path": name, "size": info.file_size})
            if name.startswith("lib/") and name.endswith(".so"):
                parts = name.split("/")
                native.append({"abi": parts[1] if len(parts) > 2 else "",
                               "name": parts[-1], "path": name,
                               "size": info.file_size})
    native.sort(key=lambda n: (n["abi"], n["name"]))
    arsc = {"present": "resources.arsc" in (a.get_files() or []),
            "parsed": False, "string_count": None}
    if arsc["present"]:
        try:
            res = a.get_android_resources()
            if res is not None:
                arsc["parsed"] = True
        except Exception:  # noqa: BLE001 — stub/custom arsc is a fact too
            arsc["parsed"] = False

    apkid = (apkid_scan(path) if include_apkid else
             {"available": False, "reason": "apkid scan skipped by caller"})
    # merge apkid descriptions onto native entries
    apkid_matches = apkid.get("matches") or {}
    for row in native:
        descs: list[str] = []
        for entry, found in apkid_matches.items():
            if entry.endswith(row["path"]):
                descs.extend(f.get("description") or f.get("rule", "")
                             for f in found)
        row["apkid"] = sorted(set(d for d in descs if d))

    findings: list[dict[str, str]] = []
    if app_flags.get("debuggable"):
        findings.append({"kind": "debuggable",
                         "detail": "android:debuggable=true — a debug "
                                   "build; runtime inspection is open."})
    dangerous = [p["name"] for p in permissions if p["weight"] == "high"]
    if dangerous:
        findings.append({"kind": "dangerous_permissions",
                         "detail": f"{len(dangerous)} dangerous "
                                   f"permission(s): {', '.join(dangerous)}"})
    exported = [c["name"] for rows in components.values() for c in rows
                if c.get("exported")]
    if exported:
        findings.append({"kind": "exported_components",
                         "detail": f"{len(exported)} exported component(s) "
                                   f"reachable by other apps: "
                                   f"{', '.join(exported[:8])}"})
    cleartext = [u for u in census["urls"] if u.startswith("http://")]
    if cleartext:
        findings.append({"kind": "cleartext_endpoints",
                         "detail": f"{len(cleartext)} cleartext http:// "
                                   f"endpoint(s) in dex strings, e.g. "
                                   f"{cleartext[0]}"})
    verdicts = apkid.get("verdicts") or {}
    if verdicts.get("packer") or verdicts.get("protector"):
        findings.append({"kind": "packed",
                         "detail": "APKiD reports a packer/protector — "
                                   "static view may show the loader shell, "
                                   "not the real dex."})
    if not certs:
        findings.append({"kind": "no_certificate",
                         "detail": "no signer certificate could be read "
                                   "(unsigned, or an unparseable "
                                   "signature)."})

    import androguard

    return {
        "tool": {"androguard": str(getattr(androguard, "__version__", "?")),
                 "apkid": (apkid.get("verdicts") is not None)},
        "source_name": path.name,
        "source_sha256": sha,
        "size": len(data),
        "file_count": file_count,
        "dissected_at": time.time(),
        "identity": identity,
        "signing": signing,
        "permissions": permissions,
        "declared_permissions": declared,
        "application": app_flags,
        "components": components,
        "dex": census,
        "strings_total": len(all_strings),
        "sdk_signatures": sdk_hits,
        "sdk_seed_note": "seed signature list (A1) — a hit is evidence; "
                         "no hit is not proof of absence",
        "resources": {
            "by_top_dir": {k: {"count": res_counts[k],
                               "bytes": res_sizes[k]}
                           for k in sorted(res_counts)},
            "assets": assets,
            "arsc": arsc,
        },
        "native": native,
        "apkid": apkid,
        "findings": findings,
        "static_note": "STATIC ANALYSIS ONLY — what this app contains "
                       "and can do, from its own bytes. Runtime "
                       "behavior is not proven by static evidence.",
    }


# ── renderers ────────────────────────────────────────────────────────

def render_dossier_card(d: dict[str, Any]) -> str:
    ident = d.get("identity") or {}
    totals = (d.get("dex") or {}).get("totals") or {}
    certs = (d.get("signing") or {}).get("certificates") or []
    verdicts = (d.get("apkid") or {}).get("verdicts") or {}
    lines = [
        f"🪓 **{ident.get('app_name') or ident.get('package') or 'APK'}** "
        f"— dissection",
        f"package: {ident.get('package') or '?'} · "
        f"v{ident.get('version_name') or '?'} "
        f"({ident.get('version_code') if ident.get('version_code') is not None else '?'}) · "
        f"SDK {ident.get('min_sdk') or '?'}→{ident.get('target_sdk') or '?'}",
        f"size: {d.get('size', 0):,} bytes · {d.get('file_count', 0)} files · "
        f"dex: {totals.get('classes', 0)} classes / "
        f"{totals.get('methods', 0)} methods",
        f"permissions: {len(d.get('permissions') or [])} "
        f"({sum(1 for p in d.get('permissions') or [] if p.get('weight') == 'high')} dangerous)",
    ]
    if certs:
        lines.append(f"signer: {certs[0].get('subject') or '?'} · "
                     f"sha256 {str(certs[0].get('sha256'))[:16]}…")
    else:
        lines.append("signer: none readable")
    if verdicts:
        flat = [v for rows in verdicts.values() for v in rows]
        lines.append(f"build verdict (APKiD): {', '.join(flat[:4])}")
    elif (d.get("apkid") or {}).get("available"):
        lines.append("build verdict (APKiD): no packer/compiler flags")
    for f in (d.get("findings") or [])[:4]:
        lines.append(f"⚠ {f.get('detail')}")
    lines.append("static analysis only — full dossier in dossier.md")
    return "\n".join(lines)


def render_dossier_md(d: dict[str, Any]) -> str:
    ident = d.get("identity") or {}
    signing = d.get("signing") or {}
    dex = d.get("dex") or {}
    totals = dex.get("totals") or {}
    out: list[str] = []
    out.append(f"# Dossier — {ident.get('app_name') or ident.get('package')}")
    out.append("")
    out.append(f"_{d.get('static_note', '')}_")
    out.append("")
    out.append("## Identity")
    out.append(f"- package: `{ident.get('package')}`")
    out.append(f"- version: {ident.get('version_name')} "
               f"(code {ident.get('version_code')})")
    out.append(f"- SDK: min {ident.get('min_sdk')} · target "
               f"{ident.get('target_sdk')} · max {ident.get('max_sdk')}")
    out.append(f"- main activity: {ident.get('main_activity') or '—'}")
    out.append(f"- source: {d.get('source_name')} · "
               f"{d.get('size', 0):,} bytes · sha256 "
               f"`{d.get('source_sha256')}`")
    out.append("")
    out.append("## Signing")
    schemes = [s for s, on in (("v1", signing.get("v1")),
                               ("v2", signing.get("v2")),
                               ("v3", signing.get("v3"))) if on]
    out.append(f"- schemes present: {', '.join(schemes) or 'none'}")
    for c in signing.get("certificates") or []:
        out.append(f"- signer: {c.get('subject')}")
        out.append(f"  - issuer: {c.get('issuer')} · serial {c.get('serial')}"
                   f" · self-signed: {c.get('self_signed')}")
        out.append(f"  - SHA-256: `{c.get('sha256')}`")
        out.append(f"  - valid: {c.get('not_before')} → {c.get('not_after')}")
    if not signing.get("certificates"):
        out.append("- no signer certificate readable")
    out.append("")
    out.append("## Permissions")
    for p in d.get("permissions") or []:
        out.append(f"- `{p['name']}` — {p.get('protection') or 'unknown'} "
                   f"(weight: {p.get('weight')})")
        if p.get("label"):
            out.append(f"  - {p['label']}: {p.get('description', '')}")
    if d.get("declared_permissions"):
        out.append(f"- declares its own: "
                   f"{', '.join(d['declared_permissions'])}")
    out.append("")
    out.append("## Components")
    for kind, rows in (d.get("components") or {}).items():
        out.append(f"### {kind} ({len(rows)})")
        for c in rows:
            exp = c.get("exported")
            exp_s = ("exported" if exp else
                     "not exported" if exp is False else "exported=?")
            out.append(f"- `{c['name']}` — {exp_s}")
            for f in c.get("intent_filters") or []:
                out.append(f"  - filter: actions={f.get('actions')} "
                           f"categories={f.get('categories')}")
    app = d.get("application") or {}
    flags = [f"{k}={v}" for k, v in app.items() if v is not None]
    if flags:
        out.append(f"application flags: {', '.join(flags)}")
    out.append("")
    out.append("## Dex census")
    out.append(f"- totals: {totals.get('classes')} classes · "
               f"{totals.get('methods')} methods · "
               f"{totals.get('fields')} fields · "
               f"{totals.get('strings')} strings")
    for row in dex.get("per_dex") or []:
        out.append(f"- {row['name']}: {row['classes']} classes, "
                   f"{row['methods']} methods")
    for row in dex.get("package_rollup") or []:
        out.append(f"- pkg {row['package']}: {row['classes']} classes")
    if dex.get("notable_apis"):
        out.append("### Notable platform APIs referenced")
        for row in dex["notable_apis"][:15]:
            out.append(f"- `{row['class']}` ×{row['refs']} "
                       f"({', '.join(row['methods'][:6])})")
    if dex.get("urls"):
        out.append("### Endpoints in strings")
        for u in dex["urls"][:25]:
            out.append(f"- {u}")
    if d.get("sdk_signatures"):
        out.append("## SDK / tracker census (seed signatures)")
        for s in d["sdk_signatures"]:
            out.append(f"- {s['name']} ({s['kind']}) — "
                       f"{s['class_count']} classes under "
                       f"`{s['package_prefix']}`")
        out.append(f"_{d.get('sdk_seed_note', '')}_")
    out.append("")
    out.append("## Native libraries")
    if d.get("native"):
        for n in d["native"]:
            extra = f" — APKiD: {', '.join(n['apkid'])}" if n.get("apkid") else ""
            out.append(f"- {n['path']} ({n['size']:,} bytes){extra}")
    else:
        out.append("- none")
    verdicts = (d.get("apkid") or {}).get("verdicts") or {}
    out.append("")
    out.append("## Build verdict (APKiD)")
    if (d.get("apkid") or {}).get("available"):
        if verdicts:
            for bucket, rows in verdicts.items():
                out.append(f"- {bucket}: {', '.join(rows)}")
        else:
            out.append("- no compiler/packer/protector flags matched")
    else:
        out.append(f"- unavailable: {(d.get('apkid') or {}).get('reason')}")
    if d.get("findings"):
        out.append("")
        out.append("## Findings (static)")
        for f in d["findings"]:
            out.append(f"- **{f['kind']}**: {f['detail']}")
    out.append("")
    return "\n".join(out)
