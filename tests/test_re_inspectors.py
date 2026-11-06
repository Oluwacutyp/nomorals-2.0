"""RE Pass 1 — the built-in static spine over REAL binary fixtures.

The fixtures are real binaries built byte-by-byte in this module
(an ELF64 with program headers + dynamic segment + both symbol
tables, a PE32+ with imports/exports/certificate, a Mach-O 64 with
load commands + symtab, a fat wrapper) — the parsers are asserted
against bytes laid out the way linkers actually lay them out, not
against dicts pretending to be parse results.
"""

from __future__ import annotations

import struct
import unittest

from nomorals.agents.reverse.inspectors import (
    detect_format,
    extract_strings,
    inspect_bytes,
    shannon_entropy,
)

# ── fixture builders ─────────────────────────────────────────────────

def _strtab_builder():
    blob = bytearray(b"\x00")
    offsets: dict[str, int] = {}

    def add(s: str) -> int:
        if s not in offsets:
            offsets[s] = len(blob)
            blob.extend(s.encode() + b"\x00")
        return offsets[s]

    return blob, add


def build_elf64() -> bytes:
    """A small but structurally real ELF64 executable (PIE)."""
    dynstr, dadd = _strtab_builder()
    o_printf = dadd("printf")
    o_malloc = dadd("malloc")
    o_needed = dadd("libfoo.so.1")
    o_soname = dadd("testbin")
    strtab, sadd = _strtab_builder()
    o_main = sadd("main")
    o_helper = sadd("helper_fn")
    shstr, shadd = _strtab_builder()
    names = {name: shadd(name) for name in (
        ".text", ".dynsym", ".dynstr", ".dynamic", ".symtab",
        ".strtab", ".shstrtab")}

    interp = b"/lib64/ld-linux-x86-64.so.2\x00"
    text = b"\x55\x48\x89\xe5\x90\xc3" + b"\x90" * 26

    # layout: header(64) phdrs(5*56) interp text dynsym dynamic
    #         symtab dynstr strtab shstrtab shdrs(8*64)
    phoff = 64
    interp_off = phoff + 5 * 56
    text_off = interp_off + len(interp)
    dynsym_off = text_off + len(text)
    dynsym = b"\x00" * 24
    dynsym += struct.pack("<IBBHQQ", o_printf, 0x12, 0, 1, 0, 0)
    dynsym += struct.pack("<IBBHQQ", o_malloc, 0x12, 0, 1, 0, 0)
    dynamic_off = dynsym_off + len(dynsym)
    base = 0x400000
    dynamic = b"".join([
        struct.pack("<qQ", 1, o_needed),          # DT_NEEDED
        struct.pack("<qQ", 14, o_soname),         # DT_SONAME
        struct.pack("<qQ", 5, 0),                 # DT_STRTAB (patched)
        struct.pack("<qQ", 30, 0x8),              # DT_FLAGS BIND_NOW
        struct.pack("<qQ", 0x6FFFFFFB, 0x08000000),  # FLAGS_1 PIE
        struct.pack("<qQ", 0, 0),                 # DT_NULL
    ])
    symtab_off = dynamic_off + len(dynamic)
    symtab = b"\x00" * 24
    symtab += struct.pack("<IBBHQQ", o_main, 0x12, 0, 1, 0x1000, 5)
    symtab += struct.pack("<IBBHQQ", o_helper, 0x12, 0, 1, 0x1010, 5)
    dynstr_off = symtab_off + len(symtab)
    strtab_off = dynstr_off + len(dynstr)
    shstr_off = strtab_off + len(strtab)
    shoff = shstr_off + len(shstr)
    # patch DT_STRTAB now that dynstr's offset is known
    dynamic = dynamic.replace(
        struct.pack("<qQ", 5, 0),
        struct.pack("<qQ", 5, base + dynstr_off), 1)
    total = shoff + 8 * 64

    def phdr(p_type, flags, off, vaddr, filesz):
        return struct.pack("<IIQQQQQQ", p_type, flags, off, vaddr,
                           vaddr, filesz, filesz, 0x1000)

    phdrs = b"".join([
        phdr(1, 5, 0, base, total),                       # PT_LOAD R+X
        phdr(3, 4, interp_off, base + interp_off, len(interp)),
        phdr(2, 6, dynamic_off, base + dynamic_off, len(dynamic)),
        phdr(0x6474E551, 6, 0, 0, 0),                     # GNU_STACK RW
        phdr(0x6474E552, 4, dynamic_off, base + dynamic_off,
             len(dynamic)),                               # GNU_RELRO
    ])
    header = struct.pack(
        "<HHIQQQIHHHHHH", 3, 62, 1, base + 0x1050, phoff, shoff, 0,
        64, 56, 5, 64, 8, 7)
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 8

    def shdr(name, s_type, flags, addr, off, size, link, info,
             entsize):
        return struct.pack("<IIQQQQIIQQ", name, s_type, flags, addr,
                           off, size, link, info, 8, entsize)

    shdrs = b"".join([
        b"\x00" * 64,
        shdr(names[".text"], 1, 6, base + text_off, text_off,
             len(text), 0, 0, 0),
        shdr(names[".dynsym"], 11, 2, base + dynsym_off, dynsym_off,
             len(dynsym), 3, 1, 24),
        shdr(names[".dynstr"], 3, 2, base + dynstr_off, dynstr_off,
             len(dynstr), 0, 0, 0),
        shdr(names[".dynamic"], 6, 3, base + dynamic_off, dynamic_off,
             len(dynamic), 3, 0, 16),
        shdr(names[".symtab"], 2, 0, 0, symtab_off, len(symtab), 6, 1,
             24),
        shdr(names[".strtab"], 3, 0, 0, strtab_off, len(strtab), 0, 0,
             0),
        shdr(names[".shstrtab"], 3, 0, 0, shstr_off, len(shstr), 0, 0,
             0),
    ])
    return (ident + header + phdrs + interp + text + dynsym + dynamic
            + symtab + bytes(dynstr) + bytes(strtab) + bytes(shstr)
            + shdrs)


def build_pe() -> bytes:
    """A structurally real PE32+ executable with imports, one
    export, and a (dummy) certificate table."""
    rdata = bytearray(0x200)
    # import descriptor at rdata+0x00, ILT +0x40, IAT +0x60,
    # dll name +0x90, hint/name +0xB0/+0xD0
    ilt_rva, iat_rva, name_rva = 0x2040, 0x2060, 0x2090
    hn1, hn2 = 0x20B0, 0x20D0
    struct.pack_into("<IIIII", rdata, 0x00, ilt_rva, 0, 0, name_rva,
                     iat_rva)
    for at, val in ((0x40, hn1), (0x48, (1 << 63) | 42),
                    (0x50, hn2), (0x58, 0),
                    (0x60, hn1), (0x68, (1 << 63) | 42),
                    (0x70, hn2), (0x78, 0)):
        struct.pack_into("<Q", rdata, at, val)
    rdata[0x90:0x90 + 13] = b"kernel32.dll\x00"
    struct.pack_into("<H", rdata, 0xB0, 0)
    rdata[0xB2:0xB2 + 12] = b"CreateFileW\x00"
    struct.pack_into("<H", rdata, 0xD0, 0)
    rdata[0xD2:0xD2 + 12] = b"ExitProcess\x00"
    # export directory at rdata+0x100
    struct.pack_into("<IIHHIIIIIII", rdata, 0x100,
                     0, 0x65000000, 0, 0, 0x2140, 1, 1, 1,
                     0x2130, 0x2138, 0x213C)
    struct.pack_into("<I", rdata, 0x130, 0x1000)       # func RVA
    struct.pack_into("<I", rdata, 0x138, 0x2150)       # name ptr
    struct.pack_into("<H", rdata, 0x13C, 0)           # ordinal
    rdata[0x140:0x140 + 12] = b"testbin.dll\x00"
    rdata[0x150:0x150 + 9] = b"myExport\x00"

    text = b"\x48\x83\xec\x28\x90\xc3" + b"\x90" * 250
    dos = bytearray(0x80)
    dos[0:2] = b"MZ"
    struct.pack_into("<I", dos, 0x3C, 0x80)
    coff = struct.pack("<HHIIIHH", 0x8664, 2, 0x65000000, 0, 0, 240,
                       0x0022)
    opt = bytearray(240)
    struct.pack_into("<H", opt, 0, 0x20B)
    struct.pack_into("<I", opt, 16, 0x1000)            # entry RVA
    struct.pack_into("<Q", opt, 24, 0x140000000)       # image base
    struct.pack_into("<H", opt, 68, 3)                 # subsystem
    struct.pack_into("<H", opt, 70, 0x4160)            # dllchar
    struct.pack_into("<I", opt, 108, 16)               # num dirs
    struct.pack_into("<II", opt, 112, 0x2100, 40)      # export dir
    struct.pack_into("<II", opt, 120, 0x2000, 40)      # import dir
    headers_len = 0x80 + 4 + 20 + 240 + 2 * 40
    text_off = headers_len
    rdata_off = text_off + len(text)
    cert_off = rdata_off + len(rdata)
    cert = struct.pack("<IHH", 12, 0x200, 2) + b"PKCS"
    struct.pack_into("<II", opt, 144, cert_off, len(cert))

    def section2(name, vsize, vaddr, rawsize, rawptr, chars):
        return struct.pack("<8sIIIIIIHHI", name, vsize, vaddr,
                           rawsize, rawptr, 0, 0, 0, 0, chars)

    sections = (section2(b".text", 0x200, 0x1000, len(text), text_off,
                         0x60000020)
                + section2(b".rdata", 0x200, 0x2000, len(rdata),
                           rdata_off, 0x40000040))
    return (bytes(dos) + b"PE\x00\x00" + coff + bytes(opt) + sections
            + text + bytes(rdata) + cert)


def build_macho64() -> bytes:
    """A structurally real Mach-O 64 executable."""
    strtab = b"\x00_main\x00_printf\x00"
    code = b"\x55\x48\x89\xe5\xc3" + b"\x90" * 11
    dylib_name = b"/usr/lib/libSystem.B.dylib\x00"
    dylib_name += b"\x00" * ((8 - len(dylib_name) % 8) % 8)
    seg = struct.pack("<II16sQQQQIIII", 0x19, 72 + 80, b"__TEXT",
                      0x100000000, 0x1000, 0, 0x1000, 7, 5, 1, 0)
    sect = struct.pack("<16s16sQQIIIIIIII", b"__text", b"__TEXT",
                       0x100000000, len(code), 0, 4, 0, 0, 0x80000400,
                       0, 0, 0)
    # section offset patched after layout is known
    cmds = []
    cmds.append(seg + sect)
    cmds.append(b"\x00" * 24)  # LC_SYMTAB placeholder (final size)
    dylib = struct.pack("<IIIIII", 0xC, 24 + len(dylib_name), 24, 2,
                        0, 0) + dylib_name
    cmds.append(dylib)
    cmds.append(struct.pack("<II", 0x1B, 24) + bytes(range(16)))
    cmds.append(struct.pack("<IIQQ", 0x80000028, 24, 0, 0))
    sig_size = 16
    cmds.append(struct.pack("<IIII", 0x1D, 16, 0, sig_size))
    sizeofcmds = sum(len(c) for c in cmds)
    hdr_size = 32
    code_off = hdr_size + sizeofcmds
    # patch section offset (in the segment command: header 72 bytes
    # in, section's offset field is 32 bytes into the section struct)
    seg_sect = bytearray(cmds[0])
    struct.pack_into("<I", seg_sect, 72 + 32, code_off)
    struct.pack_into("<Q", seg_sect, 72 + 16, 0x100000000 + code_off)
    cmds[0] = bytes(seg_sect)
    symoff = code_off + len(code)
    stroff = symoff + 2 * 16
    cmds[1] = struct.pack("<IIIIII", 0x2, 24, symoff, 2, stroff,
                          len(strtab))
    sizeofcmds = sum(len(c) for c in cmds)
    assert code_off == hdr_size + sizeofcmds
    sig_off = stroff + len(strtab)
    # patch code-signature dataoff (last command: offset field at +8)
    last = bytearray(cmds[-1])
    struct.pack_into("<I", last, 8, sig_off)
    cmds[-1] = bytes(last)
    header = struct.pack("<IIIIIIII", 0xFEEDFACF, 0x01000007, 3, 2,
                         len(cmds), sizeofcmds, 0x200000, 0)
    syms = (struct.pack("<IBBHQ", 1, 0x0F, 1, 0, 0x100000000)
            + struct.pack("<IBBHQ", 7, 0x01, 0, 0, 0))
    return (header + b"".join(cmds) + code + syms + strtab
            + b"\x00" * sig_size)


def build_fat_macho() -> bytes:
    inner = build_macho64()
    off = 32
    header = struct.pack(">II", 0xCAFEBABE, 1)
    arch = struct.pack(">iiIII", 0x01000007, 3, off, len(inner), 2)
    return header + arch + b"\x00" * (off - len(header) - len(arch)) \
        + inner


# ── tests ────────────────────────────────────────────────────────────

class DetectAndStringsTest(unittest.TestCase):
    def test_detect(self):
        self.assertEqual(detect_format(build_elf64()), "elf")
        self.assertEqual(detect_format(build_pe()), "pe")
        self.assertEqual(detect_format(build_macho64()), "macho")
        self.assertEqual(detect_format(build_fat_macho()), "macho-fat")
        self.assertEqual(detect_format(b"just some text here"), "unknown")

    def test_entropy(self):
        self.assertEqual(shannon_entropy(b""), 0.0)
        self.assertEqual(shannon_entropy(b"\x00" * 100), 0.0)
        self.assertGreater(shannon_entropy(bytes(range(256))), 7.9)

    def test_strings_finds_planted(self):
        blob = b"\x00\x01" + b"https://example.com/x" + b"\x00" * 4
        out = extract_strings(blob)
        self.assertIn("https://example.com/x", out["urls"])
        self.assertGreaterEqual(out["count"], 1)


class ElfInspectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = inspect_bytes(build_elf64())

    def test_identity(self):
        f = self.res.facts
        self.assertEqual(self.res.format, "elf")
        self.assertEqual(f["machine"], "x86-64")
        self.assertEqual(f["elf_class"], 64)
        self.assertEqual(f["entry_point"], hex(0x400000 + 0x1050))
        self.assertEqual(f["interpreter"],
                         "/lib64/ld-linux-x86-64.so.2")

    def test_census_from_apkforge_parser(self):
        f = self.res.facts
        self.assertIn(".dynsym", f["sections"])
        self.assertEqual(f["soname"], "testbin")
        self.assertIn("printf", f["dynamic_symbols"])
        self.assertIn("malloc", f["dynamic_symbols"])

    def test_standalone_pass(self):
        f = self.res.facts
        self.assertEqual(f["needed_libraries"], ["libfoo.so.1"])
        self.assertIn("main", f["static_symbols"])
        self.assertIn("helper_fn", f["static_symbols"])
        self.assertEqual(f["static_symbol_count"], 3)

    def test_hardening(self):
        h = self.res.facts["hardening"]
        self.assertTrue(h["pie"])
        self.assertFalse(h["shared_library"])
        self.assertTrue(h["nx_stack"])
        self.assertEqual(h["relro"], "full")
        self.assertTrue(h["bind_now"])
        self.assertFalse(h["textrel"])

    def test_unknowns_and_limitations(self):
        questions = " ".join(u["question"] for u in self.res.unknowns)
        self.assertIn("debug sections", questions)
        self.assertTrue(any("static parse only" in lim
                            for lim in self.res.limitations))
        self.assertGreater(self.res.strings["count"], 0)
        self.assertIn("whole_file", self.res.entropy)

    def test_truncated_elf_is_honest(self):
        data = b"\x7fELF" + bytes([2, 1, 1, 0]) + b"\x00" * 20
        res = inspect_bytes(data)
        self.assertEqual(res.format, "elf")
        self.assertFalse(res.facts["elf_header_readable"])
        self.assertTrue(res.unknowns)
        self.assertTrue(any("truncated" in lim
                            for lim in res.limitations))


class PeInspectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = inspect_bytes(build_pe())

    def test_identity(self):
        f = self.res.facts
        self.assertEqual(self.res.format, "pe")
        self.assertTrue(f["pe_signature_found"])
        self.assertEqual(f["machine"], "x86-64")
        self.assertEqual(f["pe_format"], "PE32+")
        self.assertTrue(f["is_executable_image"])
        self.assertFalse(f["is_dll"])
        self.assertEqual(f["subsystem"], "windows-console")
        self.assertEqual(f["entry_rva"], "0x1000")
        self.assertEqual([s["name"] for s in f["sections"]],
                         [".text", ".rdata"])

    def test_imports(self):
        f = self.res.facts
        self.assertEqual(f["imported_dlls"], ["kernel32.dll"])
        names = f["imports"]["kernel32.dll"]
        self.assertIn("CreateFileW", names)
        self.assertIn("ExitProcess", names)
        self.assertIn("ordinal-42", names)

    def test_exports_and_hardening(self):
        f = self.res.facts
        self.assertEqual(f["exports"], ["myExport"])
        self.assertEqual(f["export_count"], 1)
        h = f["hardening"]
        self.assertTrue(h["dynamic_base_aslr"])
        self.assertTrue(h["nx_compat_dep"])
        self.assertTrue(h["high_entropy_va"])
        self.assertTrue(h["guard_cf"])
        self.assertFalse(h["no_seh"])

    def test_certificate_registers_unknown(self):
        f = self.res.facts
        self.assertTrue(f["signed"])
        self.assertGreater(f["certificate_size"], 0)
        self.assertTrue(any("signer" in u["question"]
                            for u in self.res.unknowns))

    def test_mz_without_pe_signature(self):
        data = bytearray(0x100)
        data[0:2] = b"MZ"
        struct.pack_into("<I", data, 0x3C, 0x80)
        res = inspect_bytes(bytes(data))
        self.assertEqual(res.format, "pe")
        self.assertFalse(res.facts["pe_signature_found"])
        self.assertTrue(res.unknowns)


class MachOInspectorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = inspect_bytes(build_macho64())

    def test_identity(self):
        f = self.res.facts
        self.assertEqual(self.res.format, "macho")
        self.assertEqual(f["cpu"], "x86-64")
        self.assertEqual(f["filetype"], "execute")
        self.assertTrue(f["pie"])
        self.assertEqual(f["uuid"],
                         "00010203-0405-0607-0809-0a0b0c0d0e0f")
        self.assertIn("__TEXT", f["segments"])
        self.assertIn("__text", f["sections"])

    def test_dylibs_and_symbols(self):
        f = self.res.facts
        self.assertEqual([d["name"] for d in f["dylibs"]],
                         ["/usr/lib/libSystem.B.dylib"])
        self.assertEqual(f["symbol_count"], 2)
        self.assertIn("_main", f["symbols"])
        self.assertIn("_printf", f["symbols"])
        self.assertEqual(f["undefined_symbol_count"], 1)
        self.assertTrue(f["code_signature"]["present"])

    def test_fat(self):
        res = inspect_bytes(build_fat_macho())
        self.assertEqual(res.format, "macho-fat")
        self.assertEqual(res.facts["slice_count"], 1)
        slice_facts = res.facts["slices"][0]["facts"]
        self.assertEqual(slice_facts["cpu"], "x86-64")
        self.assertEqual(slice_facts["filetype"], "execute")


class UnknownFormatTest(unittest.TestCase):
    def test_text_file(self):
        res = inspect_bytes(b"a plain text artifact with words in it")
        self.assertEqual(res.format, "unknown")
        self.assertTrue(any("no built-in inspector" in lim
                            for lim in res.limitations))
        self.assertEqual(len(res.unknowns), 1)
        self.assertEqual(res.unknowns[0]["required_authority"],
                         "historical-reference")


if __name__ == "__main__":
    unittest.main()
