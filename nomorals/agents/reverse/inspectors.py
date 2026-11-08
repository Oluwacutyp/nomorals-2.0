"""Built-in static inspectors — the native static spine (Pass 1).

Pure-Python parsers for the three native formats, no external
engine: ELF, PE, Mach-O (including fat/universal Mach-O). Every
inspector reads the artifact's own bytes and returns facts plus the
two honesty payloads the Evidence core carries everywhere —
``limitations`` (what this parse cannot establish) and ``unknowns``
(questions the parse itself raises, with the authority/confidence
closing them would take).

The ELF census generalizes ApkForge's proven parser instead of
growing a second one: :func:`nomorals.agents.apkforge.dissect.elf_info`
parses header, section table, dynamic symbols, soname and dynamic
relocations (it serves the `.so` census inside APKs today). The
standalone spine wraps it and adds the pass an APK census never
needed: program headers (interpreter, GNU_STACK, GNU_RELRO), the
PT_DYNAMIC segment read directly (needed libraries, BIND_NOW,
TEXTREL), the static symbol table, and hardening verdicts derived
from those facts.
"""

from __future__ import annotations

import math
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..apkforge.dissect import elf_info
from .failures import record_failure

__all__ = [
    "PROVIDER_ID", "PROVIDER_NAME", "PROVIDER_VERSION",
    "Inspection", "detect_format", "inspect_bytes", "shannon_entropy",
    "extract_strings",
]

PROVIDER_ID = "devon.builtin-static"
PROVIDER_NAME = "Devon built-in static inspectors"
PROVIDER_VERSION = "1"

_STATIC_LIMIT = ("static parse only — no code in this artifact was "
                 "executed, so these facts describe what the file "
                 "contains, never what it does at runtime")


@dataclass
class Inspection:
    """One inspector's output: facts + honesty payloads."""

    format: str
    facts: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    unknowns: list[dict[str, Any]] = field(default_factory=list)
    strings: dict[str, Any] = field(default_factory=dict)
    entropy: dict[str, Any] = field(default_factory=dict)


def shannon_entropy(data: bytes) -> float:
    """Shannon entropy in bits/byte (0.0–8.0), rounded to 3 places."""
    if not data:
        return 0.0
    counts = Counter(data)
    n = len(data)
    total = 0.0
    for c in counts.values():
        p = c / n
        total -= p * math.log2(p)
    return round(total, 3)


_ASCII_RUN = re.compile(rb"[\x20-\x7e]{5,}")
_UTF16_RUN = re.compile(rb"(?:[\x20-\x7e]\x00){5,}")


def extract_strings(data: bytes, *, sample_cap: int = 50) -> dict[str, Any]:
    """Printable-string census: ASCII + UTF-16LE runs of ≥5 chars.

    Returns the full count and a bounded sample — the census is the
    evidence; the sample is for a human's eye, and says it is one.
    """
    found: list[str] = []
    for m in _ASCII_RUN.finditer(data):
        found.append(m.group().decode("ascii", "replace"))
    for m in _UTF16_RUN.finditer(data):
        found.append(m.group().decode("utf-16-le", "replace"))
    urls = [s for s in found
            if s.startswith(("http://", "https://"))][:20]
    return {
        "count": len(found),
        "sample": found[:sample_cap],
        "sample_is_capped": len(found) > sample_cap,
        "urls": urls,
    }


def detect_format(data: bytes) -> str:
    """Magic-byte format identification."""
    if len(data) >= 4:
        if data[:4] == b"\x7fELF":
            return "elf"
        if data[:4] in (b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
                        b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe"):
            return "macho"
        if data[:4] in (b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca",
                        b"\xca\xfe\xba\xc7", b"\xc7\xba\xfe\xca"):
            return "macho-fat"
        if data[:2] == b"MZ":
            return "pe"
    return "unknown"


def inspect_bytes(data: bytes) -> Inspection:
    """Run the spine over one artifact's bytes."""
    fmt = detect_format(data)
    if fmt == "elf":
        out = _inspect_elf(data)
    elif fmt == "pe":
        out = _inspect_pe(data)
    elif fmt == "macho":
        out = _inspect_macho(data)
    elif fmt == "macho-fat":
        out = _inspect_macho_fat(data)
    else:
        out = Inspection(
            format="unknown",
            facts={"magic_hex": data[:16].hex(),
                   "size": len(data)},
            limitations=[
                "no built-in inspector recognizes this format — the "
                "record carries identity facts only (size, hashes, "
                "strings, entropy)",
                _STATIC_LIMIT],
            unknowns=[{
                "question": "what format is this artifact, and what "
                            "produced it?",
                "required_authority": "historical-reference",
                "required_confidence": "derived",
            }])
    out.strings = extract_strings(data)
    out.entropy = {"whole_file": shannon_entropy(data)}
    return out


def _cstr(blob: bytes, off: int) -> str:
    if off < 0 or off >= len(blob):
        return ""
    end = blob.find(b"\x00", off)
    if end < 0:
        end = len(blob)
    return blob[off:end].decode("utf-8", "replace")


# ── ELF ──────────────────────────────────────────────────────────────

_PT_DYNAMIC = 2
_PT_INTERP = 3
_PT_GNU_STACK = 0x6474E551
_PT_GNU_RELRO = 0x6474E552
_DT_NEEDED = 1
_DT_STRTAB = 5
_DT_FLAGS = 30
_DT_TEXTREL = 22
_DT_BIND_NOW = 24
_DT_FLAGS_1 = 0x6FFFFFFB
_DF_BIND_NOW = 0x8
_DF_1_NOW = 0x1
_DF_1_PIE = 0x08000000
_ET_EXEC = 2
_ET_DYN = 3
_SHT_SYMTAB = 2


def _inspect_elf(data: bytes) -> Inspection:
    base = elf_info(data)
    if base is None:
        return Inspection(
            format="elf",
            facts={"elf_header_readable": False},
            limitations=[
                "the magic is ELF but the image is truncated or "
                "malformed — the census parser refused it, so no "
                "structural facts are claimed",
                _STATIC_LIMIT],
            unknowns=[{
                "question": "is this a damaged ELF image, or a "
                            "nonstandard variant the built-in parser "
                            "does not cover?",
                "required_authority": "shipped-artifact",
                "required_confidence": "derived",
            }])
    facts: dict[str, Any] = dict(base)
    limitations = [_STATIC_LIMIT]
    if base.get("dynamic_symbol_count", 0) > len(
            base.get("dynamic_symbols") or []):
        limitations.append(
            "dynamic symbol names are capped in the census; the "
            "recorded count is the full count")

    # ── the standalone pass: header + program headers + PT_DYNAMIC ──
    is64 = data[4] == 2
    en = "<" if data[5] == 1 else ">"
    ext: dict[str, Any] = {"elf_header_readable": True}
    try:
        if is64:
            (e_type, _m, _v, entry, phoff, _shoff, _fl, _eh, phentsize,
             phnum, _shentsize, _shnum, _shstr) = struct.unpack_from(
                 en + "HHIQQQIHHHHHH", data, 16)
            ph_fmt, ph_size = en + "IIQQQQQQ", 56
        else:
            (e_type, _m, _v, entry, phoff, _shoff, _fl, _eh, phentsize,
             phnum, _shentsize, _shnum, _shstr) = struct.unpack_from(
                 en + "HHIIIIIHHHHHH", data, 16)
            ph_fmt, ph_size = en + "IIIIIIII", 32
    except struct.error:
        facts.update(ext)
        return Inspection(format="elf", facts=facts,
                          limitations=limitations)
    ext["entry_point"] = hex(entry)
    phdrs: list[dict[str, int]] = []
    loads: list[dict[str, int]] = []
    interp = ""
    gnu_stack_flags: int | None = None
    has_relro = False
    dynamic_seg: tuple[int, int] | None = None
    for i in range(phnum):
        at = phoff + i * phentsize
        if at + ph_size > len(data):
            break
        try:
            vals = struct.unpack_from(ph_fmt, data, at)
        except struct.error:
            break
        if is64:
            (p_type, p_flags, p_offset, p_vaddr, _pa, p_filesz,
             _msz, _al) = vals
        else:
            (p_type, p_offset, p_vaddr, _pa, p_filesz, _msz, p_flags,
             _al) = vals
        phdrs.append({"type": p_type, "offset": p_offset,
                      "vaddr": p_vaddr, "filesz": p_filesz,
                      "flags": p_flags})
        if p_type == 1:
            loads.append(phdrs[-1])
        elif p_type == _PT_INTERP:
            interp = _cstr(data[p_offset:p_offset + p_filesz], 0)
        elif p_type == _PT_GNU_STACK:
            gnu_stack_flags = p_flags
        elif p_type == _PT_GNU_RELRO:
            has_relro = True
        elif p_type == _PT_DYNAMIC:
            dynamic_seg = (p_offset, p_filesz)
    if interp:
        ext["interpreter"] = interp

    def _v2o(vaddr: int) -> int:
        for seg in loads:
            if seg["vaddr"] <= vaddr < seg["vaddr"] + seg["filesz"]:
                return seg["offset"] + (vaddr - seg["vaddr"])
        return vaddr  # identity fallback (non-PIE linked at file addr)

    needed: list[str] = []
    flags = 0
    flags1 = 0
    textrel = False
    bind_now_tag = False
    if dynamic_seg is not None:
        d_off, d_size = dynamic_seg
        ent = 16 if is64 else 8
        strtab_off: int | None = None
        raw_needed: list[int] = []
        entries: list[tuple[int, int]] = []
        for i in range(d_size // ent):
            at = d_off + i * ent
            try:
                if is64:
                    tag, val = struct.unpack_from(en + "qQ", data, at)
                else:
                    tag, val = struct.unpack_from(en + "iI", data, at)
            except struct.error:
                break
            if tag == 0:
                break
            entries.append((tag, val))
            if tag == _DT_STRTAB:
                strtab_off = _v2o(val)
            elif tag == _DT_NEEDED:
                raw_needed.append(val)
            elif tag == _DT_FLAGS:
                flags = val
            elif tag == _DT_FLAGS_1:
                flags1 = val
            elif tag == _DT_TEXTREL:
                textrel = True
            elif tag == _DT_BIND_NOW:
                bind_now_tag = True
        if strtab_off is not None:
            blob = data[strtab_off:]
            needed = [n for n in (_cstr(blob, o) for o in raw_needed)
                      if n]
    ext["needed_libraries"] = needed

    # static symbol table (.symtab) — names a stripped binary loses
    static_symbols: list[str] = []
    static_count = 0
    try:
        if is64:
            (_t, _m2, _v2, _e2, _p2, shoff, _f2, _eh2, _pe2, _pn2,
             shentsize, shnum, _sx) = struct.unpack_from(
                 en + "HHIQQQIHHHHHH", data, 16)
            sh_fmt, sh_size = en + "IIQQQQIIQQ", 64
            sym_fmt, sym_size = en + "IBBHQQ", 24
        else:
            (_t, _m2, _v2, _e2, _p2, shoff, _f2, _eh2, _pe2, _pn2,
             shentsize, shnum, _sx) = struct.unpack_from(
                 en + "HHIIIIIHHHHHH", data, 16)
            sh_fmt, sh_size = en + "IIIIIIIIII", 40
            sym_fmt, sym_size = en + "IIIBBH", 16
        sections_raw: list[tuple[int, int, int, int, int]] = []
        if shoff and shentsize >= sh_size:
            for i in range(shnum):
                at = shoff + i * shentsize
                if at + sh_size > len(data):
                    break
                vals = struct.unpack_from(sh_fmt, data, at)
                # (name, type, flags, addr, offset, size, link, ...)
                sections_raw.append((vals[1], vals[4], vals[5],
                                     vals[6], vals[9]))
        for (s_type, s_off, s_size, s_link, s_entsize) in sections_raw:
            if s_type != _SHT_SYMTAB or not s_entsize:
                continue
            if not (0 <= s_link < len(sections_raw)):
                continue
            _lt, l_off, l_size, _ll, _le = sections_raw[s_link]
            blob = data[l_off:l_off + l_size]
            n = s_size // s_entsize
            static_count += n
            for i in range(n):
                at = s_off + i * s_entsize
                if at + sym_size > len(data):
                    break
                try:
                    sv = struct.unpack_from(sym_fmt, data, at)
                except struct.error:
                    break
                name = _cstr(blob, sv[0])
                if name and len(static_symbols) < 200:
                    static_symbols.append(name)
    except struct.error as exc:
        record_failure("inspectors._inspect_elf", exc)
    ext["static_symbols"] = static_symbols
    ext["static_symbol_count"] = static_count
    if static_count > len(static_symbols):
        limitations.append(
            "static symbol names are capped at 200 in the record; "
            "the recorded count is the full count")

    bind_now = bool(bind_now_tag or (flags & _DF_BIND_NOW)
                    or (flags1 & _DF_1_NOW))
    hardening = {
        "pie": bool(e_type == _ET_DYN and interp),
        "shared_library": bool(e_type == _ET_DYN and not interp),
        "nx_stack": (gnu_stack_flags is not None
                     and not (gnu_stack_flags & 1)),
        "gnu_stack_present": gnu_stack_flags is not None,
        "relro": ("full" if has_relro and bind_now
                  else "partial" if has_relro else "none"),
        "bind_now": bind_now,
        "textrel": textrel,
    }
    ext["hardening"] = hardening
    facts.update(ext)

    unknowns: list[dict[str, Any]] = []
    if static_count == 0 and not base.get("dynamic_symbols"):
        unknowns.append({
            "question": "this binary carries no symbol tables — what "
                        "do its internal functions do?",
            "required_authority": "controlled-replay",
            "required_confidence": "derived",
        })
    if not any(str(s).startswith(".debug")
               for s in base.get("sections") or []):
        unknowns.append({
            "question": "no debug sections are present — the "
                        "source-level (file/line) mapping of this "
                        "binary is not recoverable from the artifact",
            "required_authority": "historical-reference",
            "required_confidence": "derived",
        })
    return Inspection(format="elf", facts=facts,
                      limitations=limitations, unknowns=unknowns)


# ── PE ───────────────────────────────────────────────────────────────

_PE_MACHINES = {
    0x8664: "x86-64", 0x014C: "i386", 0xAA64: "arm64", 0x01C0: "arm",
    0x01C4: "armv7", 0x0200: "ia64", 0x01F0: "powerpc",
    0x9041: "msil",
}
_PE_SUBSYSTEMS = {
    1: "native", 2: "windows-gui", 3: "windows-console",
    5: "os2-console", 7: "posix-console", 9: "windows-ce-gui",
    10: "efi-application", 11: "efi-boot-driver",
    12: "efi-runtime-driver", 14: "xbox", 16: "windows-boot",
}
_DLLCHAR = (
    (0x0020, "high_entropy_va"), (0x0040, "dynamic_base_aslr"),
    (0x0080, "force_integrity"), (0x0100, "nx_compat_dep"),
    (0x0200, "no_isolation"), (0x0400, "no_seh"),
    (0x0800, "no_bind"), (0x1000, "appcontainer"),
    (0x4000, "guard_cf"), (0x8000, "terminal_server_aware"),
)


def _inspect_pe(data: bytes) -> Inspection:
    facts: dict[str, Any] = {}
    limitations = [_STATIC_LIMIT]
    unknowns: list[dict[str, Any]] = []
    if len(data) < 0x40:
        return Inspection(format="pe", facts={"mz_header_readable": False},
                          limitations=["file too small for a PE header"]
                          + limitations)
    try:
        (e_lfanew,) = struct.unpack_from("<I", data, 0x3C)
    except struct.error:
        return Inspection(format="pe", facts={"mz_header_readable": False},
                          limitations=["MZ header truncated"] + limitations)
    if e_lfanew + 24 > len(data) or data[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
        return Inspection(
            format="pe",
            facts={"mz_header_readable": True, "pe_signature_found": False},
            limitations=[
                "MZ magic present but no PE signature at the header "
                "offset — a DOS executable or a damaged PE; no PE "
                "facts are claimed"] + limitations,
            unknowns=[{
                "question": "is this a 16-bit DOS executable, or a "
                            "damaged/corrupted PE image?",
                "required_authority": "shipped-artifact",
                "required_confidence": "derived",
            }])
    coff = e_lfanew + 4
    (machine, nsects, timestamp, _symtab, _nsyms, optsize,
     characteristics) = struct.unpack_from("<HHIIIHH", data, coff)
    facts.update({
        "pe_signature_found": True,
        "machine": _PE_MACHINES.get(machine, f"machine-{machine:#x}"),
        "machine_id": machine,
        "section_count": nsects,
        "linker_timestamp": timestamp,
        "is_dll": bool(characteristics & 0x2000),
        "is_executable_image": bool(characteristics & 0x0002),
    })
    opt = coff + 20
    if optsize < 2 or opt + optsize > len(data):
        limitations.append("optional header missing or truncated — "
                           "directory facts unavailable")
        return Inspection(format="pe", facts=facts,
                          limitations=limitations, unknowns=unknowns)
    (magic,) = struct.unpack_from("<H", data, opt)
    is64 = magic == 0x20B
    facts["pe_format"] = "PE32+" if is64 else "PE32" if magic == 0x10B \
        else f"magic-{magic:#x}"
    if magic not in (0x10B, 0x20B):
        limitations.append("unrecognized optional-header magic — "
                           "directory facts unavailable")
        return Inspection(format="pe", facts=facts,
                          limitations=limitations, unknowns=unknowns)
    (entry_rva,) = struct.unpack_from("<I", data, opt + 16)
    if is64:
        (image_base,) = struct.unpack_from("<Q", data, opt + 24)
        dirs_at = opt + 112
    else:
        (image_base,) = struct.unpack_from("<I", data, opt + 28)
        dirs_at = opt + 96
    (subsystem,) = struct.unpack_from("<H", data, opt + 68)
    (dllchar,) = struct.unpack_from("<H", data, opt + 70)
    (num_dirs,) = struct.unpack_from("<I", data,
                                     opt + (108 if is64 else 92))
    facts.update({
        "entry_rva": hex(entry_rva),
        "image_base": hex(image_base),
        "subsystem": _PE_SUBSYSTEMS.get(subsystem,
                                        f"subsystem-{subsystem}"),
    })
    hardening = {name: bool(dllchar & flag) for flag, name in _DLLCHAR}
    facts["hardening"] = hardening

    dirs: list[tuple[int, int]] = []
    for i in range(min(num_dirs, 16)):
        at = dirs_at + i * 8
        if at + 8 > len(data):
            break
        rva, size = struct.unpack_from("<II", data, at)
        dirs.append((rva, size))

    # sections
    sections: list[dict[str, Any]] = []
    sec_at = opt + optsize
    for i in range(nsects):
        at = sec_at + i * 40
        if at + 40 > len(data):
            break
        name = data[at:at + 8].split(b"\x00")[0].decode("ascii", "replace")
        vsize, vaddr, rawsize, rawptr = struct.unpack_from(
            "<IIII", data, at + 8)
        (schar,) = struct.unpack_from("<I", data, at + 36)
        blob = data[rawptr:rawptr + rawsize] if rawsize else b""
        sections.append({
            "name": name, "virtual_size": vsize, "virtual_address": vaddr,
            "raw_size": rawsize, "raw_offset": rawptr,
            "characteristics": hex(schar),
            "entropy": shannon_entropy(blob[:1 << 20]),
        })
    facts["sections"] = sections

    def _r2o(rva: int) -> int | None:
        for s in sections:
            span = max(s["virtual_size"], s["raw_size"])
            if s["virtual_address"] <= rva < s["virtual_address"] + span:
                return s["raw_offset"] + (rva - s["virtual_address"])
        if rva < len(data):
            return rva  # headers map 1:1
        return None

    def _dir(idx: int) -> tuple[int, int]:
        return dirs[idx] if idx < len(dirs) else (0, 0)

    # imports (directory 1)
    imports: dict[str, list[str]] = {}
    imp_rva, _imp_size = _dir(1)
    if imp_rva:
        off = _r2o(imp_rva)
        thunk_size = 8 if is64 else 4
        ordinal_flag = (1 << 63) if is64 else (1 << 31)
        guard = 0
        while off is not None and off + 20 <= len(data) and guard < 512:
            guard += 1
            ilt_rva, _ts, _fwd, name_rva, iat_rva = struct.unpack_from(
                "<IIIII", data, off)
            if ilt_rva == 0 and name_rva == 0 and iat_rva == 0:
                break
            name_off = _r2o(name_rva)
            dll = _cstr(data, name_off) if name_off is not None else ""
            names: list[str] = []
            thunk_rva = ilt_rva or iat_rva
            t_off = _r2o(thunk_rva) if thunk_rva else None
            steps = 0
            while t_off is not None and steps < 4096:
                steps += 1
                if is64:
                    if t_off + 8 > len(data):
                        break
                    (val,) = struct.unpack_from("<Q", data, t_off)
                else:
                    if t_off + 4 > len(data):
                        break
                    (val,) = struct.unpack_from("<I", data, t_off)
                if val == 0:
                    break
                if val & ordinal_flag:
                    names.append(f"ordinal-{val & 0xFFFF}")
                else:
                    hn = _r2o(val & 0x7FFFFFFF)
                    if hn is not None:
                        names.append(_cstr(data, hn + 2))
                t_off += thunk_size
            if dll:
                imports[dll] = names[:200]
            off += 20
    facts["imports"] = imports
    facts["imported_dlls"] = sorted(imports)

    # exports (directory 0)
    exports: list[str] = []
    exp_rva, _exp_size = _dir(0)
    if exp_rva:
        off = _r2o(exp_rva)
        if off is not None and off + 40 <= len(data):
            (_c, _t, _vmaj, _vmin, _nm, _b, _nf, nnames, _frva,
             names_rva, _orva) = struct.unpack_from(
                 "<IIHHIIIIIII", data, off)
            names_off = _r2o(names_rva)
            if names_off is not None:
                for i in range(min(nnames, 4096)):
                    at = names_off + i * 4
                    if at + 4 > len(data):
                        break
                    (nrva,) = struct.unpack_from("<I", data, at)
                    noff = _r2o(nrva)
                    if noff is not None:
                        nm = _cstr(data, noff)
                        if nm:
                            exports.append(nm)
    facts["exports"] = exports[:200]
    facts["export_count"] = len(exports)

    # certificate table (directory 4 — its "RVA" is a file offset)
    cert_rva, cert_size = _dir(4)
    facts["signed"] = bool(cert_rva and cert_size)
    facts["certificate_size"] = cert_size if cert_rva else 0
    if cert_rva and cert_size:
        unknowns.append({
            "question": "a certificate table is present — which "
                        "signer does the embedded PKCS#7 name, and "
                        "does the signature verify against the file?",
            "required_authority": "shipped-artifact",
            "required_confidence": "derived",
        })
    packed = [s["name"] for s in sections if s["entropy"] >= 7.2]
    if packed:
        unknowns.append({
            "question": f"section(s) {', '.join(packed)} have "
                        "near-maximum entropy — are they packed or "
                        "encrypted payloads rather than plain code?",
            "required_authority": "controlled-replay",
            "required_confidence": "derived",
        })
    return Inspection(format="pe", facts=facts,
                      limitations=limitations, unknowns=unknowns)


# ── Mach-O ───────────────────────────────────────────────────────────

_MACHO_CPUS = {
    7: "x86", 0x01000007: "x86-64", 12: "arm", 0x0100000C: "arm64",
    0x0200000C: "arm64_32", 18: "ppc", 0x01000012: "ppc64",
}
_MACHO_FILETYPES = {
    1: "object", 2: "execute", 3: "fvmlib", 4: "core", 5: "preload",
    6: "dylib", 7: "dylinker", 8: "bundle", 9: "dylib-stub",
    10: "dsym", 11: "kext-binary",
}
_LC_SEGMENT = 0x1
_LC_SYMTAB = 0x2
_LC_SEGMENT_64 = 0x19
_LC_UUID = 0x1B
_LC_CODE_SIGNATURE = 0x1D
_LC_ENCRYPTION_INFO = 0x21
_LC_ENCRYPTION_INFO_64 = 0x2C
_LC_MAIN = 0x80000028
_DYLIB_CMDS = {0xC: "load", 0xD: "id", 0x1F: "load-weak",
               0x80000018: "load-weak", 0x8000001F: "reexport",
               0x20: "load-upward"}


def _inspect_macho(data: bytes) -> Inspection:
    magic = data[:4]
    is64 = magic in (b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe")
    en = "<" if magic in (b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe") else ">"
    facts: dict[str, Any] = {}
    limitations = [_STATIC_LIMIT]
    unknowns: list[dict[str, Any]] = []
    hdr = 32 if is64 else 28
    if len(data) < hdr:
        return Inspection(format="macho",
                          facts={"macho_header_readable": False},
                          limitations=["Mach-O header truncated"]
                          + limitations)
    (_mg, cputype, _sub, filetype, ncmds, _sizeofcmds,
     flags) = struct.unpack_from(en + "IIIIIII", data, 0)
    cputype_u = cputype & 0xFFFFFFFF
    facts.update({
        "macho_header_readable": True,
        "bits": 64 if is64 else 32,
        "cpu": _MACHO_CPUS.get(cputype_u, f"cpu-{cputype_u:#x}"),
        "filetype": _MACHO_FILETYPES.get(filetype,
                                         f"filetype-{filetype}"),
        "load_command_count": ncmds,
        "flags": hex(flags),
        "pie": bool(flags & 0x200000),
        "no_heap_execution": bool(flags & 0x1000000),
        "allow_stack_execution": bool(flags & 0x20000),
    })
    segments: list[str] = []
    section_names: list[str] = []
    dylibs: list[dict[str, str]] = []
    uuid = ""
    code_signature: dict[str, Any] = {"present": False}
    cryptid: int | None = None
    entry_offset: int | None = None
    symbols: list[str] = []
    symbol_count = 0
    undefined_count = 0
    symtab: tuple[int, int, int, int] | None = None
    at = hdr
    for _ in range(ncmds):
        if at + 8 > len(data):
            break
        try:
            cmd, cmdsize = struct.unpack_from(en + "II", data, at)
        except struct.error:
            break
        if cmdsize < 8 or at + cmdsize > len(data):
            break
        if cmd in (_LC_SEGMENT, _LC_SEGMENT_64):
            segname = data[at + 8:at + 24].split(b"\x00")[0].decode(
                "ascii", "replace")
            segments.append(segname)
            nsects_at = at + (64 if cmd == _LC_SEGMENT_64 else 48)
            if nsects_at + 4 <= len(data):
                (nsects,) = struct.unpack_from(en + "I", data, nsects_at)
                s_size = 80 if cmd == _LC_SEGMENT_64 else 68
                s_at = at + (72 if cmd == _LC_SEGMENT_64 else 56)
                for i in range(nsects):
                    s_off = s_at + i * s_size
                    if s_off + 16 > len(data):
                        break
                    section_names.append(
                        data[s_off:s_off + 16].split(b"\x00")[0].decode(
                            "ascii", "replace"))
        elif cmd == _LC_SYMTAB:
            symtab = struct.unpack_from(en + "IIII", data, at + 8)
        elif cmd in _DYLIB_CMDS:
            (name_off,) = struct.unpack_from(en + "I", data, at + 8)
            dylibs.append({
                "kind": _DYLIB_CMDS[cmd],
                "name": _cstr(data, at + name_off) if name_off else "",
            })
        elif cmd == _LC_UUID:
            raw = data[at + 8:at + 24]
            uuid = (f"{raw[:4].hex()}-{raw[4:6].hex()}-{raw[6:8].hex()}-"
                    f"{raw[8:10].hex()}-{raw[10:].hex()}")
        elif cmd == _LC_CODE_SIGNATURE:
            doff, dsize = struct.unpack_from(en + "II", data, at + 8)
            code_signature = {"present": True, "offset": doff,
                              "size": dsize}
        elif cmd in (_LC_ENCRYPTION_INFO, _LC_ENCRYPTION_INFO_64):
            (_coff, _csize, cid) = struct.unpack_from(
                en + "III", data, at + 8)
            cryptid = cid
        elif cmd == _LC_MAIN:
            (entry_offset,) = struct.unpack_from(en + "Q", data, at + 8)
        at += cmdsize
    if symtab is not None:
        symoff, nsyms, stroff, strsize = symtab
        blob = data[stroff:stroff + strsize]
        ent = 16 if is64 else 12
        symbol_count = nsyms
        for i in range(nsyms):
            off = symoff + i * ent
            if off + ent > len(data):
                break
            try:
                if is64:
                    (n_strx, n_type, _sect, _desc,
                     _val) = struct.unpack_from(en + "IBBHQ", data, off)
                else:
                    (n_strx, n_type, _sect, _desc,
                     _val) = struct.unpack_from(en + "IBBHI", data, off)
            except struct.error:
                break
            if (n_type & 0x0E) == 0x0 and n_strx:
                undefined_count += 1
            name = _cstr(blob, n_strx)
            if name and len(symbols) < 200:
                symbols.append(name)
    facts.update({
        "segments": segments,
        "sections": section_names,
        "dylibs": dylibs,
        "uuid": uuid,
        "code_signature": code_signature,
        "symbols": symbols,
        "symbol_count": symbol_count,
        "undefined_symbol_count": undefined_count,
    })
    if symbol_count > len(symbols):
        limitations.append(
            "symbol names are capped at 200 in the record; the "
            "recorded count is the full count")
    if entry_offset is not None:
        facts["entry_offset"] = hex(entry_offset)
    if cryptid is not None:
        facts["encryption_cryptid"] = cryptid
        if cryptid != 0:
            limitations.append(
                "this binary's code is FairPlay-encrypted (cryptid "
                "!= 0) — static facts cover headers and load "
                "commands; the code itself is not readable statically")
            unknowns.append({
                "question": "the code is FairPlay-encrypted — what "
                            "does the decrypted code contain?",
                "required_authority": "controlled-replay",
                "required_confidence": "observed",
            })
    if symbol_count == 0:
        unknowns.append({
            "question": "no symbol table is present — what do this "
                        "binary's internal functions do?",
            "required_authority": "controlled-replay",
            "required_confidence": "derived",
        })
    return Inspection(format="macho", facts=facts,
                      limitations=limitations, unknowns=unknowns)


def _inspect_macho_fat(data: bytes) -> Inspection:
    fat64 = data[:4] in (b"\xca\xfe\xba\xc7", b"\xc7\xba\xfe\xca")
    # fat headers are big-endian on disk regardless of writer
    if len(data) < 8:
        return Inspection(format="macho-fat",
                          facts={"fat_header_readable": False},
                          limitations=["fat header truncated"])
    (nfat,) = struct.unpack_from(">I", data, 4)
    slices: list[dict[str, Any]] = []
    merged_unknowns: list[dict[str, Any]] = []
    at = 8
    ent = 32 if fat64 else 20
    for _i in range(min(nfat, 16)):
        if at + ent > len(data):
            break
        if fat64:
            cputype, _sub, off, size, _al, _res = struct.unpack_from(
                ">iiQQII", data, at)
        else:
            cputype, _sub, off, size, _al = struct.unpack_from(
                ">iiIII", data, at)
        cpu = _MACHO_CPUS.get(cputype & 0xFFFFFFFF,
                              f"cpu-{cputype & 0xFFFFFFFF:#x}")
        blob = data[off:off + size]
        sub = _inspect_macho(blob) if blob else None
        slices.append({
            "cpu": cpu, "offset": off, "size": size,
            "facts": sub.facts if sub else {},
        })
        if sub:
            merged_unknowns.extend(sub.unknowns)
        at += ent
    return Inspection(
        format="macho-fat",
        facts={"fat_header_readable": True, "slice_count": len(slices),
               "slices": slices},
        limitations=[_STATIC_LIMIT,
                     "per-slice facts are parsed from each slice's "
                     "own bytes; slice code was not executed"],
        unknowns=merged_unknowns)
