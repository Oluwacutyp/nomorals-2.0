"""ApkForge splits — bundle surgery over A1's inventory (design §3.4).

* ``list`` — the parts A1 inventoried, with sizes/hashes/packages.
* ``extract`` — pull one part out as a standalone file.
* ``merge`` — one installable APK from the set: APKEditor merge for
  .apks/.apkm/.xapk containers; bundletool ``build-apks
  --mode=universal`` for an .aab (signed with the vault key, since
  bundletool signs what it builds), then the merged output is
  zipaligned + signed through the one signing path and verified.
* ``installset`` — the exact split set for a device spec
  (abis / locales / density / sdk): bundletool ``extract-apks``
  when the set is bundletool-produced (toc.pb present); otherwise
  Devon-side selection over the inventory's config.* part names —
  the same answer, computed from the parts' own names, and labeled
  as such.  A bundletool failure on a foreign bundle is reported
  by name, never papered over.
* ``generate`` — split generation from an .aab via bundletool
  ``build-apks`` (default per-device mode).  From a plain APK there
  is nothing to split — the op says so.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any

from . import signing as signing_mod
from .project import Project, sha256_file
from .toolchain import ToolchainError, run_tool

__all__ = ["SplitsError", "list_splits", "extract_split",
           "merge_bundle", "installset", "generate_splits"]


class SplitsError(RuntimeError):
    """A designed splits failure — names the exact step."""


def _parts(proj: Project) -> list[dict[str, Any]]:
    return list((proj.record.get("inventory") or {}).get("parts")
                or [])


def list_splits(proj: Project) -> dict[str, Any]:
    parts = []
    for part in _parts(proj):
        p = proj.dir / "bundle" / part["name"]
        parts.append({**part, "present": p.exists(),
                      "path": str(p)})
    return {"project": proj.id, "kind": proj.kind, "parts": parts,
            "count": len(parts)}


def extract_split(proj: Project, part_name: str, *,
                  dest_dir: str = "") -> dict[str, Any]:
    names = {p["name"] for p in _parts(proj)}
    if part_name not in names:
        raise SplitsError(f"no split named {part_name!r} in this "
                          f"bundle — parts: {sorted(names)}")
    src = proj.dir / "bundle" / part_name
    if not src.exists():
        raise SplitsError(f"split {part_name!r} is inventoried but "
                          "missing on disk")
    out_dir = Path(dest_dir) if dest_dir else proj.dir / "splits-out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / Path(part_name).name
    shutil.copyfile(src, out)
    proj.journal("splits_extract", f"{part_name} → {out}")
    return {"part": part_name, "path": str(out),
            "sha256": sha256_file(out), "size": out.stat().st_size}


def _bundletool_sign_flags(ks: dict[str, Any], alias: str) -> list[str]:
    """bundletool accepts passwords only as pass:/file: — the file
    form keeps secrets out of argv; the files live inside the
    materialized keystore temp dir and are shredded with it."""
    ks_pass_file = ks["path"].parent / "ks.pass"
    key_pass_file = ks["path"].parent / "key.pass"
    ks_pass_file.write_text(ks["store_password"], encoding="utf-8")
    key_pass_file.write_text(ks["key_password"], encoding="utf-8")
    return [f"--ks={ks['path']}", f"--ks-key-alias={alias}",
            f"--ks-pass=file:{ks_pass_file}",
            f"--key-pass=file:{key_pass_file}"]


def merge_bundle(context: Any, root: Path, proj: Project, *,
                 alias: str = "") -> dict[str, Any]:
    """Merge the bundle into ONE signed, verified APK."""
    if proj.kind == "apk":
        raise SplitsError("project is a single APK already — "
                          "nothing to merge")
    out_dir = proj.dir / "builds"
    out_dir.mkdir(exist_ok=True)
    merged_raw = out_dir / f"{proj.id}-merged-unsigned.apk"
    signed = out_dir / f"{proj.id}-merged-signed.apk"

    if proj.kind == "aab":
        apks_out = out_dir / f"{proj.id}.apks"
        eff_alias = alias or signing_mod.DEFAULT_ALIAS
        signing_mod.ensure_keystore(context, root, eff_alias)
        with signing_mod.materialize_keystore(
                context, root, eff_alias) as ks:
            try:
                res = run_tool(
                    "", "bundletool",
                    ["build-apks", "--mode=universal",
                     f"--bundle={proj.original_path}",
                     f"--output={apks_out}",
                     *_bundletool_sign_flags(ks, eff_alias)],
                    timeout=900, root=root)
            except ToolchainError as exc:
                raise SplitsError(f"bundletool build-apks: {exc}") \
                    from exc
        if res["rc"] != 0 or not apks_out.exists():
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            raise SplitsError(
                "bundletool build-apks failed: "
                + (tail[-1][:300] if tail else f"rc={res['rc']}"))
        with zipfile.ZipFile(apks_out) as zf:
            names = [n for n in zf.namelist() if n.endswith(".apk")]
            if not names:
                raise SplitsError("bundletool produced an .apks "
                                  "with no universal APK inside")
            with zf.open(names[0]) as src, \
                    open(merged_raw, "wb") as out:
                shutil.copyfileobj(src, out)
        via = "bundletool build-apks --mode=universal"
    else:
        try:
            res = run_tool("", "apkeditor",
                           ["m", "-i", str(proj.original_path),
                            "-o", str(merged_raw)],
                           timeout=900, root=root)
        except ToolchainError as exc:
            raise SplitsError(f"APKEditor merge: {exc}") from exc
        if res["rc"] != 0 or not merged_raw.exists():
            tail = (res["stdout"] + res["stderr"]).strip().splitlines()
            raise SplitsError(
                "APKEditor merge failed: "
                + (tail[-1][:300] if tail else f"rc={res['rc']}"))
        via = "APKEditor merge"

    try:
        sig = signing_mod.sign_apk(context, root, merged_raw, signed,
                                   alias=alias)
    except signing_mod.SigningError as exc:
        raise SplitsError(f"sign step: {exc}") from exc
    proj.journal("splits_merge",
                 f"{proj.kind} merged via {via} → {signed.name} "
                 f"(signed {sig['alias']})")
    return {"project": proj.id, "via": via, "output": str(signed),
            "output_sha256": sha256_file(signed),
            "size": signed.stat().st_size,
            "signing": {"alias": sig["alias"],
                        "cert_sha256": sig["cert_sha256"],
                        "debug": sig["debug"],
                        "schemes": sig["verify"]["schemes"]},
            "resign_facts": signing_mod.RESIGN_FACTS}


_CONFIG_RE = re.compile(
    r"^config\.(?P<what>[a-z0-9_]+)\.apk$", re.I)


def _config_kind(name: str) -> tuple[str, str]:
    """(kind, value) for a split part name: base/feature pass
    through; config.* parts classify as abi/density/locale by
    their value's shape."""
    base = Path(name).name
    if base == "base.apk":
        return ("base", "")
    m = _CONFIG_RE.match(base)
    if not m:
        return ("feature", base[:-4] if base.endswith(".apk")
                else base)
    value = m.group("what")
    if "_" in value or value.replace("_", "").isdigit():
        return ("density", value)
    if value in ("arm64_v8a", "armeabi_v7a", "x86", "x86_64"):
        return ("abi", value)
    return ("locale", value)


def _select_for_device(parts: list[dict[str, Any]],
                       spec: dict[str, Any]) -> list[dict[str, Any]]:
    abis = [str(a).replace("-", "_")
            for a in spec.get("supportedAbis") or []]
    locales = [str(loc).split("-")[0].lower()
               for loc in spec.get("supportedLocales") or []]
    chosen: list[dict[str, Any]] = []
    density_rank: list[tuple[int, dict[str, Any]]] = []
    for part in parts:
        kind, value = _config_kind(part["name"])
        if kind in ("base", "feature"):
            chosen.append(part)
        elif kind == "abi":
            if not abis or value in abis:
                chosen.append(part)
        elif kind == "locale":
            lang = value.split("_")[0].lower()
            if not locales or lang in locales:
                chosen.append(part)
        elif kind == "density":
            density_rank.append((abs(int(value or 0)
                                     - int(spec.get(
                                         "screenDensity") or 0)),
                                 part))
    if density_rank:
        density_rank.sort(key=lambda t: t[0])
        chosen.append(density_rank[0][1])
    return chosen


def installset(proj: Project, spec: dict[str, Any], *,
               root: Path) -> dict[str, Any]:
    """Produce the split set matching a device spec."""
    parts = _parts(proj)
    if not parts:
        raise SplitsError("project has no bundle parts — not a "
                          "bundle project")
    out_dir = proj.dir / "splits-out" / "installset"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    toc = proj.dir / "bundle" / "toc.pb"
    via = ""
    chosen: list[dict[str, Any]] = []
    if toc.exists() and proj.original_path.suffix == ".apks":
        spec_path = out_dir / "device-spec.json"
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            res = run_tool("", "bundletool",
                           ["extract-apks",
                            f"--apks={proj.original_path}",
                            f"--output-dir={out_dir}",
                            f"--device-spec={spec_path}"],
                           timeout=600, root=root)
            if res["rc"] == 0:
                via = "bundletool extract-apks"
                chosen = [{"name": p.name,
                           "path": str(p)}
                          for p in sorted(out_dir.glob("*.apk"))]
        except ToolchainError:
            res = {"rc": -1}
        if not via:
            via = ("inventory selection (bundletool extract-apks "
                   "unavailable/failed on this set)")
    if not via:
        via = "inventory selection over config.* part names"
    if via.startswith("inventory"):
        chosen_meta = _select_for_device(parts, spec)
        chosen = []
        for part in chosen_meta:
            src = proj.dir / "bundle" / part["name"]
            dst = out_dir / Path(part["name"]).name
            if src.exists():
                shutil.copyfile(src, dst)
                chosen.append({"name": part["name"],
                               "path": str(dst),
                               "sha256": sha256_file(dst)})
    proj.journal("splits_installset",
                 f"device set via {via}: "
                 f"{[c['name'] for c in chosen]}")
    return {"project": proj.id, "via": via, "spec": spec,
            "parts": chosen, "dir": str(out_dir)}


def generate_splits(context: Any, root: Path, proj: Project, *,
                    alias: str = "") -> dict[str, Any]:
    """Split generation: an .aab → bundletool's per-device .apks
    set.  A plain APK has no splits to generate — stated, not faked."""
    if proj.kind != "aab":
        raise SplitsError(
            f"split generation needs an .aab source (bundletool "
            f"build-apks) — this project is {proj.kind!r}; a single "
            "APK carries no split structure to generate from")
    out = proj.dir / "builds" / f"{proj.id}-splits.apks"
    out.parent.mkdir(exist_ok=True)
    eff_alias = alias or signing_mod.DEFAULT_ALIAS
    signing_mod.ensure_keystore(context, root, eff_alias)
    with signing_mod.materialize_keystore(
            context, root, eff_alias) as ks:
        try:
            res = run_tool(
                "", "bundletool",
                ["build-apks", f"--bundle={proj.original_path}",
                 f"--output={out}",
                 *_bundletool_sign_flags(ks, eff_alias)],
                timeout=900, root=root)
        except ToolchainError as exc:
            raise SplitsError(f"bundletool build-apks: {exc}") from exc
    if res["rc"] != 0 or not out.exists():
        tail = (res["stdout"] + res["stderr"]).strip().splitlines()
        raise SplitsError(
            "bundletool build-apks failed: "
            + (tail[-1][:300] if tail else f"rc={res['rc']}"))
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    proj.journal("splits_generate",
                 f"aab → {out.name} ({len(names)} entries)")
    return {"project": proj.id, "output": str(out),
            "entries": names, "size": out.stat().st_size}
