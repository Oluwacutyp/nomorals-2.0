"""ApkForge teach — the learning layer over a Dossier.

Lessons are composed from *this APK's actual parsed anatomy*: what
this permission means here (cross-checked against the dex evidence),
what this exported component exposes, why this certificate matters,
why this packer verdict limits the static view.  With a brain, the
lesson is written in Devon's voice from the dossier facts only;
without one, a factual renderer produces the same walk from the same
fields — dissect and teach work fully offline.  Every lesson states
the static-analysis limit where it applies.

Teach v2 (A5) adds the code-level lesson: one class, one method,
walked in BOTH of its real representations — the smali from the
decoded tree (the ground truth that actually runs) and the
jadx-decompiled Java (the readable reconstruction, with R8 mapping
names restored when the project carries a mapping).  The lesson
is grounded only in those two texts: the brain path must cite the
method's real instructions to be accepted, and the no-brain path
is a structured side-by-side extract of the same pair.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = ["teach_lesson", "factual_lesson", "TOPICS",
           "teach_code_lesson", "find_smali_path",
           "normalize_class_name", "parse_mapping",
           "smali_method_names", "TeachError"]

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
        issuer_kind = ("self-signed — the signer vouches for itself, "
                       "typical of direct-distributed apps"
                       if c.get("self_signed")
                       else "issued by a separate CA")
        lines.append(f"- issuer: {c.get('issuer')} ({issuer_kind})")
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


# ── teach v2: one method in smali AND jadx Java (A5) ────────────────

class TeachError(RuntimeError):
    """A designed teach refusal — names the exact cause."""


def normalize_class_name(name: str) -> str:
    """Accept a dotted name (``com.x.Y``), a descriptor
    (``Lcom/x/Y;``) or a slash path, and return the dotted form."""
    d = (name or "").strip()
    if d.startswith("L") and d.endswith(";"):
        d = d[1:-1]
    return d.replace("/", ".")


def find_smali_path(decoded_dir: Path, dotted: str) -> Path | None:
    """The smali file for one class in a decoded tree — apktool
    writes ``smali/<path>.smali`` for the primary dex and
    ``smali_classes<N>/`` for the rest; inner classes are their
    own files (``MainActivity$1.smali``)."""
    rel = Path(*dotted.split(".")) if "." in dotted else Path(dotted)
    rel = rel.with_suffix(".smali")
    decoded_dir = Path(decoded_dir)
    for cand in sorted(decoded_dir.glob("smali*")):
        if cand.is_dir():
            hit = cand / rel
            if hit.exists():
                return hit
    for hit in sorted(decoded_dir.rglob(rel.name)):
        if hit.parent.name == rel.parent.name or len(rel.parts) < 2:
            return hit
    return None


_METHOD_LINE = re.compile(r"^\.method\s+(.+)$")


def _smali_blocks(smali_text: str) -> list[dict[str, str]]:
    """Split smali source into method blocks: name, full Dalvik
    signature (``onClick(Landroid/view/View;)V``) and the block
    text from ``.method`` through ``.end method``."""
    lines = smali_text.splitlines()
    blocks: list[dict[str, str]] = []
    i = 0
    while i < len(lines):
        m = _METHOD_LINE.match(lines[i])
        if not m:
            i += 1
            continue
        signature = m.group(1).strip().split()[-1]
        name = signature.split("(", 1)[0]
        j = i
        while j < len(lines) and lines[j].strip() != ".end method":
            j += 1
        block = "\n".join(lines[i:j + 1]) if j < len(lines) \
            else "\n".join(lines[i:])
        blocks.append({"name": name, "signature": signature,
                       "block": block})
        i = j + 1
    return blocks


def smali_method_names(smali_text: str) -> list[str]:
    return [b["name"] for b in _smali_blocks(smali_text)]


def _extract_java_method(java_text: str, method: str) -> str | None:
    """The Java declaration of one method, brace-matched from its
    opening brace to its close — annotation lines above the
    signature ride along."""
    m = re.search(rf"\b{re.escape(method)}\s*\(", java_text)
    if m is None:
        return None
    brace = java_text.find("{", m.end())
    if brace < 0:
        return None
    depth = 0
    end = -1
    for i in range(brace, len(java_text)):
        ch = java_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None
    start = java_text.rfind("\n", 0, m.start()) + 1
    # pull in annotation lines directly above the signature
    while start > 0:
        prev_end = start - 1
        prev_start = java_text.rfind("\n", 0, prev_end) + 1
        if java_text[prev_start:prev_end].strip().startswith("@"):
            start = prev_start
        else:
            break
    return java_text[start:end + 1].strip()


def parse_mapping(mapping_text: str) -> dict[str, str]:
    """ProGuard/R8 mapping.txt → {obfuscated dotted: original
    dotted} for class lines (``orig -> obf:``)."""
    out: dict[str, str] = {}
    for raw in (mapping_text or "").splitlines():
        if not raw.strip() or raw[0].isspace() \
                or raw.startswith("#"):
            continue
        m = re.match(r"^(\S+) -> (\S+):$", raw.strip())
        if m:
            out[m.group(2)] = m.group(1)
    return out


def _smali_facts(block: str) -> dict[str, Any]:
    """Mechanical facts of one smali method block — the reading
    the fallback lesson is built from, all read off the text."""
    facts: dict[str, Any] = {"registers": "", "invokes": [],
                             "consts": [], "fields": [],
                             "returns": ""}
    for line in block.splitlines():
        text = line.strip()
        if not text or text.startswith((".", "#", ":")):
            if text.startswith((".locals", ".registers")):
                facts["registers"] = text
            continue
        op = text.split(None, 1)[0]
        if op.startswith("invoke"):
            target = text.split("}, ", 1)[-1] if "}, " in text \
                else text
            facts["invokes"].append((op, target.strip()))
        elif op.startswith("const"):
            facts["consts"].append(text)
        elif op.split("/")[0] in ("iget", "iput", "sget", "sput"):
            target = text.split(", ", 1)[-1] if ", " in text else text
            facts["fields"].append((op, target.strip()))
        elif op.startswith("return"):
            facts["returns"] = text
    return facts


def _pair_notes(class_name: str, original_name: str,
                smali_block: str, java_block: str | None,
                facts: dict[str, Any]) -> list[str]:
    """Divergences and reading notes computed from the two texts —
    every note is checkable against them."""
    notes: list[str] = []
    if original_name and original_name != class_name:
        notes.append(
            f"names diverge: the dex calls this class "
            f"`{class_name}` — R8 renamed it; the mapping restores "
            f"the original `{original_name}`, which is the name "
            "the Java above uses.")
    if "$" in class_name.rsplit(".", 1)[-1]:
        notes.append(
            "the `$` in the class name marks a compiler-generated "
            "inner class (here, an anonymous listener) — it has no "
            "source-level name of its own; the smali is the only "
            "place it exists as itself.")
    if facts["registers"]:
        notes.append(
            f"registers: the smali declares `{facts['registers']}` "
            "— Dalvik code names registers, not variables; p0 is "
            "`this` in an instance method, parameters follow the "
            "locals.")
    for op, target in facts["invokes"]:
        notes.append(f"call: `{op}` → `{target}`")
        simple = target.rsplit("->", 1)[-1].split("(", 1)[0] \
            if "->" in target else ""
        if simple and java_block is not None \
                and f"{simple}(" not in java_block:
            notes.append(
                f"divergence: the smali calls `{simple}` but the "
                "Java shows no direct call to it — jadx folded, "
                "inlined or renamed it; the smali is the truth.")
    for const in facts["consts"]:
        notes.append(f"constant: `{const}` — a literal baked into "
                     "the instruction stream; the Java shows the "
                     "same value only if it survived decompilation "
                     "as an expression.")
    for op, target in facts["fields"]:
        notes.append(f"field access: `{op}` → `{target}`")
    if facts["returns"]:
        notes.append(f"exit: `{facts['returns']}`")
    if java_block is None:
        notes.append(
            "jadx produced no matching method for this class — "
            "the Java side of the pair is missing, so this lesson "
            "reads the smali alone (jadx output is a reconstruction "
            "and can drop methods it cannot decompile).")
    return notes


def _factual_code_lesson(class_name: str, original_name: str,
                         signature: str, smali_block: str,
                         java_block: str | None,
                         facts: dict[str, Any],
                         mapping_used: bool) -> str:
    head = [f"🪓 **Code lesson — `{class_name}`** (smali ↔ jadx "
            "Java)", ""]
    head.append(f"class in this APK: `{class_name}`"
                + (f" · original name (R8 mapping): "
                   f"`{original_name}`"
                   if original_name and original_name != class_name
                   else "")
                + f" · method: `{signature}`")
    head.append("")
    head.append("**The Java (jadx — what it reads like)**")
    if java_block:
        head.append("```java")
        head.append(java_block)
        head.append("```")
    else:
        head.append("_(jadx produced no method with this name for "
                    "this class)_")
    head.append("")
    head.append("**The smali (what actually runs)**")
    head.append("```smali")
    head.append(smali_block)
    head.append("```")
    head.append("")
    head.append("**Reading the pair**")
    for note in _pair_notes(class_name, original_name, smali_block,
                            java_block, facts):
        head.append(f"- {note}")
    if mapping_used:
        head.append("- the mapping file is what lets the two "
                    "sides be compared at all: without it, the "
                    "Java's restored names and the smali's "
                    "obfuscated ones never meet.")
    head.append("")
    head.append("_Static analysis only: what this code is, from "
                "its own bytes. What it does at runtime needs a "
                "device lab._")
    return "\n".join(head)


def teach_code_lesson(context: Any, *, class_name: str,
                      original_name: str = "", method: str = "",
                      smali_text: str, java_text: str = "",
                      mapping_used: bool = False) -> dict[str, Any]:
    """The v2 lesson: ONE method walked in its smali and its jadx
    Java.  Grounded only in the two real texts — with a brain, the
    narrative must cite the method's real instructions to be
    accepted; without one (or when the brain's answer fails that
    check), the structured side-by-side extract is the lesson and
    ``source`` says so."""
    blocks = _smali_blocks(smali_text)
    if not blocks:
        return {"error": f"no methods found in the smali for "
                         f"{class_name} — the class file is empty "
                         "or not smali"}
    chosen = None
    if method:
        chosen = next((b for b in blocks if b["name"] == method),
                      None)
        if chosen is None:
            return {"error": f"method {method!r} is not in "
                             f"{class_name} — its smali defines: "
                             + ", ".join(b["name"] for b in blocks)}
    else:
        for b in blocks:
            if b["name"] in ("<init>", "<clinit>"):
                continue
            if not java_text or f"{b['name']}(" in java_text:
                chosen = b
                break
        if chosen is None:
            chosen = next((b for b in blocks
                           if b["name"] != "<clinit>"), blocks[0])
    java_block = _extract_java_method(java_text, chosen["name"]) \
        if java_text else None
    facts = _smali_facts(chosen["block"])
    fallback = _factual_code_lesson(
        class_name, original_name, chosen["signature"],
        chosen["block"], java_block, facts, mapping_used)
    base = {"version": 2, "topic": "code", "class": class_name,
            "original_class": original_name or class_name,
            "method": chosen["name"], "signature": chosen["signature"],
            "mapping": mapping_used}
    try:
        from ...llm.brain import brain_for

        brain = brain_for(context)
        if brain is None:
            return {**base, "text": fallback,
                    "source": "side-by-side"}
        prompt = (
            "Teach your owner to read Android bytecode from ONE "
            "real method, shown in its two real forms below: the "
            "smali (ground truth — what the device executes) and "
            "the jadx-decompiled Java (a reconstruction). Walk the "
            "SAME method in both: what the Java says it does, what "
            "the smali actually does — name the registers, quote "
            "the actual invoke instructions and const values from "
            "the smali, and call out every divergence between the "
            "two (R8 renaming, inlined or synthetic code, anything "
            "jadx smoothed over). Rules: ground every claim in "
            "these two texts only; quote real instructions, never "
            "invent any; where static text cannot prove runtime "
            "behavior, say so. Direct, concrete, 200–450 words.\n\n"
            f"CLASS (in the APK): {class_name}\n"
            f"ORIGINAL NAME (R8 mapping): {original_name or '—'}\n"
            f"METHOD: {chosen['signature']}\n\n"
            f"SMALI:\n{chosen['block']}\n\n"
            f"JAVA (jadx):\n{java_block or '(jadx produced no matching method)'}")
        resp = brain.chat([{"role": "user", "content": prompt}])
        text = resp.text if hasattr(resp, "text") else str(resp)
        text = (text or "").strip()
        words = len(text.split())
        opcodes = {ln.strip().split(None, 1)[0]
                   for ln in chosen["block"].splitlines()
                   if ln.strip() and not ln.strip().startswith(
                       (".", "#", ":"))}
        cited = any(op in text for op in opcodes if op)
        if not (80 <= words <= 900) or chosen["name"] not in text \
                or not cited:
            return {**base, "text": fallback,
                    "source": "side-by-side"}
        return {**base, "text": text, "source": "brain"}
    except Exception:  # noqa: BLE001 — no brain == side-by-side
        return {**base, "text": fallback, "source": "side-by-side"}
