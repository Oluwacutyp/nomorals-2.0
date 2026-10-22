"""ApkForge teach — the learning layer over a Dossier.

Lessons are composed from *this APK's actual parsed anatomy*: what
this permission means here (cross-checked against the dex evidence),
what this exported component exposes, why this certificate matters,
why this packer verdict limits the static view.  With a brain, the
lesson is written in Devon's voice from the dossier facts only;
without one, a factual renderer produces the same walk from the same
fields — dissect and teach work fully offline.  Every lesson states
the static-analysis limit where it applies.
"""

from __future__ import annotations

from typing import Any

__all__ = ["teach_lesson", "factual_lesson", "TOPICS"]

TOPICS = ("overview", "manifest", "permissions", "components",
          "certificate", "code", "build")


def _permission_lesson(d: dict[str, Any]) -> list[str]:
    lines = ["**Permissions — what this app may ask Android for**", ""]
    dex = d.get("dex") or {}
    api_classes = {row["class"] for row in dex.get("notable_apis") or []}
    for p in d.get("permissions") or []:
        lines.append(f"- `{p['name']}` — protection "
                     f"{p.get('protection') or 'unknown'} "
                     f"(weight {p.get('weight')})")
        if p.get("label"):
            lines.append(f"  In plain terms: {p['label']} — "
                         f"{p.get('description', '')}")
        if not p.get("in_aosp_table"):
            lines.append("  Not in the AOSP table — a custom or "
                         "vendor permission; its meaning is defined "
                         "by whoever declares it.")
    loc = [c for c in api_classes if "location" in c.lower()]
    cam = [c for c in api_classes if "Camera" in c or "camera" in c]
    if any("LOCATION" in p["name"] for p in d.get("permissions") or []):
        lines.append("")
        lines.append(
            "This app holds a location permission"
            + (f" and its code references {', '.join(sorted(loc))} — "
               "the permission and the API use line up."
               if loc else
               ", though no location API references appear in the dex "
               "method census — the permission may be declared for an "
               "SDK, or used through a path static census can't see."))
    if any(p["name"].endswith("CAMERA")
           for p in d.get("permissions") or []):
        lines.append(
            "It holds CAMERA"
            + (f" and references {', '.join(sorted(cam))}."
               if cam else " (no camera API refs in the census)."))
    return lines


def _components_lesson(d: dict[str, Any]) -> list[str]:
    lines = ["**Components — the app's moving parts, and which doors "
             "are open**", ""]
    total = 0
    for kind, rows in (d.get("components") or {}).items():
        for c in rows:
            total += 1
            exp = c.get("exported")
            if exp:
                meaning = ("EXPORTED — any other app on the device "
                           "can start/reach it. That is a real attack "
                           "and integration surface.")
            elif exp is False:
                meaning = ("not exported — only this app (and the "
                           "system) can reach it.")
            else:
                meaning = ("no android:exported attribute in the "
                           "manifest — on modern Android, a component "
                           "with an intent filter defaults to exported "
                           "for target SDK ≥ 31, otherwise not.")
            lines.append(f"- {kind[:-1]} `{c['name']}` — {meaning}")
            for f in c.get("intent_filters") or []:
                lines.append(f"  - intent filter: "
                             f"{', '.join(f.get('actions') or [])} "
                             f"[{', '.join(f.get('categories') or [])}]")
    if not total:
        lines.append("- no components were readable from the manifest")
    lines.append("")
    lines.append("The main activity "
                 f"`{(d.get('identity') or {}).get('main_activity') or '—'}` "
                 "is the launcher entry point — where the app begins.")
    return lines


def _certificate_lesson(d: dict[str, Any]) -> list[str]:
    signing = d.get("signing") or {}
    lines = ["**Certificate — who signed this, and why it rules "
             "everything**", ""]
    for c in signing.get("certificates") or []:
        lines.append(f"- signer: {c.get('subject')}")
        lines.append(f"- issuer: {c.get('issuer')} "
                     f"({'self-signed — the signer vouches for itself, '
                     'typical of direct-distributed apps' if c.get('self_signed') else 'issued by a separate CA'})")
        lines.append(f"- SHA-256 fingerprint: `{c.get('sha256')}`")
        lines.append(f"- valid {c.get('not_before')} → {c.get('not_after')}")
    if not signing.get("certificates"):
        lines.append("- no certificate could be read from this file")
    lines += ["",
              "Android installs updates only when the new APK carries "
              "the SAME certificate as the installed one. That is why "
              "an edited APK (re-signed with a different key in A2) "
              "can never install over the original — the certificate "
              "IS the app's identity, the package name is just its "
              "label."]
    return lines


def _build_lesson(d: dict[str, Any]) -> list[str]:
    apkid = d.get("apkid") or {}
    verdicts = apkid.get("verdicts") or {}
    lines = ["**Build verdict — how this app was made**", ""]
    if not apkid.get("available"):
        lines.append(f"- APKiD unavailable on this host: "
                     f"{apkid.get('reason')}")
    elif verdicts:
        for bucket, rows in verdicts.items():
            lines.append(f"- {bucket}: {', '.join(rows)}")
        if verdicts.get("packer") or verdicts.get("protector"):
            lines += ["",
                      "A packer encrypts the real dex behind a loader "
                      "stub — most of what static dissect shows for a "
                      "packed app is the shell, not the app. Reading "
                      "the real code needs dynamic unpacking on a "
                      "device lab, which this class honestly does "
                      "not fake."]
        if verdicts.get("obfuscator"):
            lines += ["",
                      "Obfuscation renames classes and methods to "
                      "meaningless names (a, b, c). The smali still "
                      "works — it is the ground truth — but names no "
                      "longer teach you intent; structure has to."]
    else:
        lines.append("- APKiD matched no compiler/packer/protector "
                     "signatures — the dex looks like a plain build.")
    return lines


def _code_lesson(d: dict[str, Any]) -> list[str]:
    dex = d.get("dex") or {}
    totals = dex.get("totals") or {}
    lines = ["**Code census — the shape of the program**", "",
             f"- {totals.get('classes')} classes · "
             f"{totals.get('methods')} methods · "
             f"{totals.get('fields')} fields · "
             f"{totals.get('strings')} strings across "
             f"{len(dex.get('per_dex') or [])} dex file(s)"]
    for row in (dex.get("package_rollup") or [])[:10]:
        lines.append(f"- package `{row['package']}`: "
                     f"{row['classes']} classes")
    if dex.get("notable_apis"):
        lines.append("")
        lines.append("Platform APIs the code references (parsed from "
                     "the dex method table — references, not proof "
                     "of runtime calls):")
        for row in dex["notable_apis"][:10]:
            lines.append(f"- `{row['class']}` ×{row['refs']} — "
                         f"e.g. {', '.join(row['methods'][:4])}")
    if d.get("sdk_signatures"):
        lines.append("")
        lines.append("Embedded SDKs found by package census:")
        for s in d["sdk_signatures"]:
            lines.append(f"- {s['name']} ({s['kind']}) — "
                         f"{s['class_count']} classes")
    return lines


def _manifest_lesson(d: dict[str, Any]) -> list[str]:
    ident = d.get("identity") or {}
    app = d.get("application") or {}
    lines = ["**The manifest — the app's contract with Android**", "",
             f"- package `{ident.get('package')}` is the app's "
             "store-wide identity; version "
             f"{ident.get('version_name')} (code "
             f"{ident.get('version_code')}) is how updates order "
             "themselves.",
             f"- min SDK {ident.get('min_sdk')} = oldest Android it "
             f"runs on; target SDK {ident.get('target_sdk')} = the "
             "behavior rules it has opted into."]
    for key, val in app.items():
        if val is None:
            continue
        lines.append(f"- application flag {key} = {val}")
    if app.get("debuggable"):
        lines.append("  debuggable=true means anyone with the device "
                     "can inspect and drive this app at runtime — a "
                     "debug build trait, a finding on a shipped app.")
    lines.append("")
    lines.append("Everything Android enforces about this app — its "
                 "permissions, its components, its entry points — is "
                 "declared here, in binary XML, before a single line "
                 "of its code runs.")
    return lines


def factual_lesson(dossier: dict[str, Any], topic: str = "") -> str:
    """The no-brain lesson: the same walk, rendered from the same
    dossier fields.  Fully offline, fully factual."""
    topic = (topic or "overview").strip().lower()
    ident = dossier.get("identity") or {}
    head = [f"🪓 Learning from "
            f"**{ident.get('app_name') or ident.get('package')}** "
            f"v{ident.get('version_name') or '?'} — from its own bytes",
            ""]
    sections: list[str] = []
    if topic in ("overview", ""):
        sections = ["manifest", "permissions", "components",
                    "certificate", "build", "code"]
    elif topic in TOPICS:
        sections = [topic]
    else:
        sections = ["overview"]
        head.append(f"(topic {topic!r} isn't one of "
                    f"{', '.join(TOPICS)} — giving the full walk)")
        sections = ["manifest", "permissions", "components",
                    "certificate", "build", "code"]
    builders = {"manifest": _manifest_lesson,
                "permissions": _permission_lesson,
                "components": _components_lesson,
                "certificate": _certificate_lesson,
                "build": _build_lesson,
                "code": _code_lesson}
    body: list[str] = []
    for name in sections:
        body.extend(builders[name](dossier))
        body.append("")
    body.append("_Static analysis only: what this app contains and "
                "can do. What it does at runtime needs a device lab._")
    return "\n".join(head + body).strip()


def teach_lesson(context: Any, dossier: dict[str, Any],
                 topic: str = "") -> dict[str, Any]:
    """Brain-composed when a brain answers; the factual lesson
    otherwise.  Returns {text, source, topic} — source is honest
    about which path produced the words."""
    fallback = factual_lesson(dossier, topic)
    try:
        from ...llm.brain import brain_for

        brain = brain_for(context)
        if brain is None:
            return {"text": fallback, "source": "dossier",
                    "topic": topic or "overview"}
        import json as _json

        facts = _json.dumps({
            "identity": dossier.get("identity"),
            "permissions": dossier.get("permissions"),
            "application": dossier.get("application"),
            "components": dossier.get("components"),
            "signing": dossier.get("signing"),
            "dex_totals": (dossier.get("dex") or {}).get("totals"),
            "notable_apis": (dossier.get("dex") or {}).get(
                "notable_apis", [])[:15],
            "sdk_signatures": dossier.get("sdk_signatures"),
            "apkid_verdicts": (dossier.get("apkid") or {}).get("verdicts"),
            "findings": dossier.get("findings"),
        }, indent=1)[:9000]
        prompt = (
            "Teach your owner Android anatomy using THIS app's real "
            "dissection (below) — he is learning how APKs work from "
            "real specimens. Pick the "
            f"topic '{topic or 'overview'}' if it fits, else give the "
            "full walk. Rules: use ONLY facts present in the "
            "dissection; explain what each thing means and why it "
            "matters for this specific app; where static analysis "
            "cannot prove runtime behavior, say so plainly. Direct, "
            "concrete, no filler. 250–500 words.\n\n"
            f"DISSECTION:\n{facts}")
        resp = brain.chat([{"role": "user", "content": prompt}])
        text = resp.text if hasattr(resp, "text") else str(resp)
        text = (text or "").strip()
        words = len(text.split())
        if not (80 <= words <= 700):
            return {"text": fallback, "source": "dossier",
                    "topic": topic or "overview"}
        return {"text": text, "source": "brain",
                "topic": topic or "overview"}
    except Exception:  # noqa: BLE001 — no brain == factual lesson
        return {"text": fallback, "source": "dossier",
                "topic": topic or "overview"}
