"""Craft-Fonts: the storytold/craft-fonts catalog inside Devon's typography organ.

Devon already has a typography organ: a catalogue parsed from the
real font files on the host, a stocked library (pack / core / local
tiers), semantic selection and a service facade. What it lacks is
the craft-fonts half — the OFL font catalog for the scripts Devon's
stocked core does not cover (Japanese, Simplified Chinese, Korean,
Arabic, Thai, Georgian, Hebrew, UI symbols), the 8-field manifest
format with checkout loading (``CRAFT_FONTS_DIR``), an install path
from a manifest into the stocked library, and a build/embed payload.
This module is that half, in the upstream shapes:

* :func:`parse_manifest` — the ``fonts/manifest.txt`` format,
  verbatim from upstream ``crates/craft-fonts/src/parse.rs``:
  ``family | style | file | scripts | licence | licence file |
  sha256 | source``, ``#`` comments and blank lines skipped, the
  SHA-256 a 64-digit lower-case hex, errors naming the 1-based
  line (FieldCount / EmptyField / BadSha256).
* :class:`CraftFontsCatalog` — the three shipped tables (default
  13 faces, opt-in Latin 51, opt-in Simplified Chinese 1),
  disjoint by file exactly as upstream's tests assert,
  :meth:`for_script` over ISO 15924 codes.
* :func:`load` / :func:`load_latin` / :func:`load_manifest` —
  a checkout at ``dir`` read whole, every file's SHA-256 checked
  against its manifest row before a single byte is returned.
* :func:`install_checkout` — verify first, then copy the font
  files and their licence texts into the store with an install
  record carrying family, style, file, SHA-256, licence and
  source. A tampered checkout installs nothing.
* :func:`build_embed` — the ``build.rs`` shape: a zip payload
  with the manifest rows, the font files, the licence texts and
  a ``provenance.json``; an allowlist (family, style) names
  exactly which faces go in, and a face the checkout does not
  hold is an error naming it, never a silent omission.

The shipped manifest texts live beside this module in
``craftfonts/`` and are byte-identical to upstream at
``aced4137c022f1f587624d48e81ec77e1914a9cc``; the font FILES are
not vendored (77 MB) — they resolve from a ``CRAFT_FONTS_DIR``
checkout or the store after install, which is upstream's own
design ("fully optional source of fonts": an app builds, tests
and runs without it).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

from .fontparse import parse_font_file

__all__ = [
    "CraftFont",
    "CraftFontsCatalog",
    "LoadError",
    "ManifestError",
    "build_embed",
    "check_checkout",
    "default_catalog",
    "dir_from_env",
    "install_checkout",
    "load",
    "load_latin",
    "load_manifest",
    "parse_manifest",
    "run_parity",
    "verify_attribution",
]

_UPSTREAM_COMMIT = "aced4137c022f1f587624d48e81ec77e1914a9cc"
_DATA_DIR = Path(__file__).resolve().parent / "craftfonts"
_FIXTURE_DIR = (Path(__file__).resolve().parent.parent.parent.parent
                / "tests" / "fixtures" / "craftfonts")
_CLONE = (Path.home() / "workspace" / "echelon-mining" / "repos"
          / "storytold" / "craft-fonts")

_FIELDS = ("family", "style", "file", "scripts", "licence",
           "licence file", "sha256", "source")


class ManifestError(ValueError):
    """Why a manifest line could not be read (upstream's shape)."""

    def __init__(self, kind: str, line: int, *,
                 field: str = "", found: int = 0):
        self.kind = kind
        self.line = line
        self.field = field
        self.found = found
        if kind == "field_count":
            text = (f"manifest line {line}: expected 8 fields separated "
                    f'by " | ", found {found}')
        elif kind == "empty_field":
            text = f"manifest line {line}: the {field} field is empty"
        else:
            text = (f"manifest line {line}: sha256 must be 64 lower-case "
                    "hex digits")
        super().__init__(text)


@dataclass(frozen=True)
class CraftFont:
    """One font file from a craft-fonts manifest, in manifest order."""

    family: str
    style: str
    file: str
    scripts: tuple[str, ...] = ()
    licence: str = "OFL-1.1"
    licence_file: str = ""
    sha256: str = ""
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family, "style": self.style,
            "file": self.file, "scripts": list(self.scripts),
            "licence": self.licence, "licence_file": self.licence_file,
            "sha256": self.sha256, "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CraftFont:
        data = dict(data)
        data["scripts"] = tuple(data.get("scripts", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def parse_manifest(text: str) -> list[CraftFont]:
    """Parse a craft-fonts manifest. Blank lines and ``#`` comments
    are skipped; line numbers are 1-based."""
    out: list[CraftFont] = []
    for i, raw in enumerate(text.splitlines()):
        line = i + 1
        trimmed = raw.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        fields = [part.strip() for part in trimmed.split(" | ")]
        if len(fields) != 8:
            raise ManifestError("field_count", line, found=len(fields))
        values = dict(zip(_FIELDS, fields, strict=True))
        for name in _FIELDS:
            if not values[name]:
                raise ManifestError("empty_field", line, field=name)
        sha = values["sha256"]
        if len(sha) != 64 or not all(
                ch.isdigit() or "a" <= ch <= "f" for ch in sha):
            raise ManifestError("bad_sha256", line)
        scripts = tuple(
            s for s in (part.strip() for part in
                        values["scripts"].split(",")) if s)
        out.append(CraftFont(
            family=values["family"], style=values["style"],
            file=values["file"], scripts=scripts,
            licence=values["licence"],
            licence_file=values["licence file"], sha256=sha,
            source=values["source"]))
    return out


def _read_table(name: str) -> list[CraftFont]:
    path = _DATA_DIR / name
    return parse_manifest(path.read_text(encoding="utf-8"))


class CraftFontsCatalog:
    """The three shipped craft-fonts tables, in manifest order."""

    def __init__(self, default_faces: list[CraftFont],
                 latin_faces: list[CraftFont],
                 chinese_faces: list[CraftFont]):
        self.default_faces = list(default_faces)
        self.latin_faces = list(latin_faces)
        self.chinese_faces = list(chinese_faces)
        self.all_faces = (self.default_faces + self.latin_faces
                          + self.chinese_faces)

    def for_script(self, script: str) -> list[CraftFont]:
        """The default-table faces meant for an ISO 15924 script
        (e.g. ``"Jpan"``), in manifest order."""
        return [f for f in self.default_faces if script in f.scripts]

    def families(self) -> list[str]:
        return sorted({f.family for f in self.all_faces},
                      key=str.lower)

    def faces(self, family: str | None = None) -> list[CraftFont]:
        if family is None:
            return list(self.all_faces)
        want = " ".join(str(family).split()).casefold()
        return [f for f in self.all_faces
                if " ".join(f.family.split()).casefold() == want]

    def disjoint(self) -> bool:
        """Default and Latin tables share no file — upstream's own
        invariant (apps reading only the main manifest must not
        change when the Latin table grows)."""
        default_files = {f.file for f in self.default_faces}
        latin_files = {f.file for f in self.latin_faces}
        return not (default_files & latin_files)


def default_catalog() -> CraftFontsCatalog:
    """The catalog from the shipped manifest texts."""
    return CraftFontsCatalog(
        _read_table("manifest.txt"),
        _read_table("latin-manifest.txt"),
        _read_table("simplified-chinese-manifest.txt"))


class LoadError(OSError):
    """Why fonts could not be loaded from a checkout."""

    def __init__(self, kind: str, path: str, detail: str = ""):
        self.kind = kind
        self.path = path
        self.detail = detail
        message = f"{kind}: {path}" + (f": {detail}" if detail else "")
        super().__init__(message)


#: The environment variable (and build option) naming a
#: craft-fonts checkout, exactly as upstream names it.
DIR_ENV = "CRAFT_FONTS_DIR"

#: The main manifest, relative to a checkout.
MANIFEST = "fonts/manifest.txt"

#: The opt-in Latin presentation-font manifest, relative to a checkout.
LATIN_MANIFEST = "fonts/latin-manifest.txt"


def dir_from_env() -> Path | None:
    """The craft-fonts checkout named by ``CRAFT_FONTS_DIR``, if it
    is set and holds a manifest."""
    raw = os.environ.get(DIR_ENV, "")
    if not raw:
        return None
    directory = Path(raw).expanduser()
    return directory if (directory / MANIFEST).is_file() else None


def load_manifest(directory: str | Path, rel: str
                  ) -> list[tuple[CraftFont, bytes]]:
    """Read the manifest at ``rel`` in the checkout at ``directory``
    and every font it lists, in manifest order. Every file's SHA-256
    is checked against its row before anything is returned; a
    mismatch names the file and refuses the whole load."""
    root = Path(directory)
    manifest = root / rel
    try:
        text = manifest.read_text(encoding="utf-8")
    except OSError as exc:
        raise LoadError("io", str(manifest), str(exc)) from exc
    try:
        entries = parse_manifest(text)
    except ManifestError as exc:
        raise LoadError("manifest", str(manifest), str(exc)) from exc
    loaded: list[tuple[CraftFont, bytes]] = []
    for entry in entries:
        path = root / entry.file
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise LoadError("io", str(path), str(exc)) from exc
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise LoadError(
                "checksum", str(path),
                f"sha256 {digest} != manifest {entry.sha256}")
        loaded.append((entry, data))
    return loaded


def load(directory: str | Path) -> list[tuple[CraftFont, bytes]]:
    """Read the main manifest of the checkout at ``directory`` and
    every font it lists."""
    return load_manifest(directory, MANIFEST)


def load_latin(directory: str | Path) -> list[tuple[CraftFont, bytes]]:
    """Read the Latin presentation-font manifest of the checkout at
    ``directory`` and every font it lists."""
    return load_manifest(directory, LATIN_MANIFEST)


def check_checkout(directory: str | Path, *,
                   manifest: str = MANIFEST) -> dict[str, Any]:
    """Verify a checkout's files against its manifest without
    returning the bytes: how many faces, how many verified, and
    every failure named. A checkout that cannot even be read
    reports the failure instead of raising."""
    try:
        loaded = load_manifest(directory, manifest)
    except LoadError as exc:
        return {"ok": False, "faces": 0, "verified": 0,
                "failures": [str(exc)], "directory": str(directory)}
    return {"ok": True, "faces": len(loaded), "verified": len(loaded),
            "failures": [], "directory": str(directory)}


def install_checkout(directory: str | Path, store_dir: str | Path, *,
                     manifest: str = MANIFEST) -> list[CraftFont]:
    """Verify a checkout first, then install it into ``store_dir``:
    the font files and their licence texts, copied under their
    manifest-relative paths, plus an ``install-record.json`` carrying
    family, style, file, SHA-256, licence and source per face. A
    checkout that fails verification installs nothing — the store
    is untouched when the load raises."""
    loaded = load_manifest(directory, manifest)
    store = Path(store_dir)
    installed: list[CraftFont] = []
    root = Path(directory)
    for entry, _data in loaded:
        target = store / entry.file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / entry.file).read_bytes())
        licence_src = root / entry.licence_file
        if licence_src.is_file():
            licence_target = store / entry.licence_file
            licence_target.parent.mkdir(parents=True, exist_ok=True)
            licence_target.write_bytes(licence_src.read_bytes())
        installed.append(entry)
    record = {
        "format": "craft-fonts-install/1",
        "source": str(directory),
        "manifest": manifest,
        "upstream_commit": _UPSTREAM_COMMIT,
        "installed": [entry.to_dict() for entry in installed],
    }
    store.mkdir(parents=True, exist_ok=True)
    (store / "install-record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return installed


def _zip_date() -> tuple[int, int, int, int, int, int]:
    return (1980, 1, 1, 0, 0, 0)


def build_embed(loaded: list[tuple[CraftFont, bytes]], *,
                root: str | Path | None = None,
                allowlist: list[tuple[str, str]] | None = None
                ) -> bytes:
    """The ``build.rs`` shape: a zip payload with the manifest rows,
    the font files, the licence texts and a ``provenance.json``.
    ``allowlist`` is a list of (family, style) pairs — exactly the
    faces that go in, in manifest order; a face the checkout does
    not hold is an error naming it, never a silent omission. With
    no allowlist every loaded face goes in. Licence texts are read
    from ``root`` (the checkout the fonts were loaded from); without
    a root the bundle cannot carry them and the build refuses
    rather than shipping a payload with empty licences."""
    wanted = None
    if allowlist is not None:
        wanted = {(" ".join(f.split()).casefold(),
                   " ".join(s.split()).casefold())
                  for f, s in allowlist}
    picked: list[tuple[CraftFont, bytes]] = []
    for entry, data in loaded:
        if wanted is None:
            picked.append((entry, data))
            continue
        key = (" ".join(entry.family.split()).casefold(),
               " ".join(entry.style.split()).casefold())
        if key in wanted:
            picked.append((entry, data))
    if wanted is not None:
        found = {(" ".join(e.family.split()).casefold(),
                  " ".join(e.style.split()).casefold())
                 for e, _ in picked}
        for family, style in allowlist or []:
            key = (" ".join(family.split()).casefold(),
                   " ".join(style.split()).casefold())
            if key not in found:
                raise ValueError(
                    f"craft-fonts: allowlisted face {family!r} "
                    f"({style}) is not in the checkout")
    manifest_lines = [
        "# Craft-Fonts embed payload. One font file per line, fields "
        "separated by \" | \":",
        "#   family | style | file | scripts | licence | licence file "
        "| sha256 | source",
    ]
    for entry, _data in picked:
        manifest_lines.append(
            " | ".join([entry.family, entry.style, entry.file,
                        ",".join(entry.scripts), entry.licence,
                        entry.licence_file, entry.sha256, entry.source]))
    provenance = {
        "format": "craft-fonts-embed/1",
        "upstream_commit": _UPSTREAM_COMMIT,
        "licence": "OFL-1.1",
        "faces": [entry.to_dict() for entry, _ in picked],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w",
                         compression=zipfile.ZIP_DEFLATED) as bundle:
        for entry, data in picked:
            info = zipfile.ZipInfo(entry.file, date_time=_zip_date())
            bundle.writestr(info, data)
        licence_seen: set[str] = set()
        for entry, _data in picked:
            if entry.licence_file in licence_seen:
                continue
            licence_seen.add(entry.licence_file)
            if root is None:
                raise ValueError(
                    "craft-fonts: cannot build an embed payload without "
                    f"the checkout root — the licence text for "
                    f"{entry.licence_file} has nowhere to come from")
            licence_path = Path(root) / entry.licence_file
            if not licence_path.is_file():
                raise ValueError(
                    "craft-fonts: licence file "
                    f"{entry.licence_file} is missing from the checkout")
            info = zipfile.ZipInfo(entry.licence_file,
                                   date_time=_zip_date())
            bundle.writestr(info, licence_path.read_bytes())
        bundle.writestr(zipfile.ZipInfo("manifest.txt",
                                        date_time=_zip_date()),
                        "\n".join(manifest_lines) + "\n")
        bundle.writestr(zipfile.ZipInfo("provenance.json",
                                        date_time=_zip_date()),
                        json.dumps(provenance, indent=2))
    return buffer.getvalue()


def _toml_path() -> Path:
    return Path(__file__).resolve().parent / "craftfonts-parity.toml"


def _fixture_checkout(directory: Path) -> Path:
    """A craft-fonts checkout in ``directory`` from the real
    fixtures (two real faces, real hashes, real licences)."""
    root = Path(directory)
    for family_dir in ("noto-sans-hebrew", "caladea"):
        source_dir = _FIXTURE_DIR / family_dir
        target_dir = root / "fonts" / family_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in ("*.ttf", "OFL.txt"):
            for path in source_dir.glob(name):
                (target_dir / path.name).write_bytes(path.read_bytes())
    rows = [
        "Noto Sans Hebrew | Regular | fonts/noto-sans-hebrew/"
        "NotoSansHebrew-Regular.ttf | Hebr | OFL-1.1 | "
        "fonts/noto-sans-hebrew/OFL.txt | "
        "671951828bd5c95db818e5bb12dcea2d0c0dda00311888522be061ee6835125e | "
        "https://github.com/notofonts/hebrew/releases/tag/"
        "NotoSansHebrew-v3.001",
        "Caladea | Regular | fonts/caladea/Caladea-Regular.ttf | "
        "Latn | OFL-1.1 | fonts/caladea/OFL.txt | "
        "f1e899278b7b4491aba5b6a8253c4b04c050cc59b21865be5c37559a775153cd | "
        "https://github.com/google/fonts/tree/"
        "7085eb89a950e85db5b166b7a58d414544b4140c/ofl/caladea",
    ]
    (root / "fonts" / "manifest.txt").write_text(
        "# Fixture manifest: two real faces, real hashes.\n"
        + "\n".join(rows) + "\n", encoding="utf-8")
    return root


def _run_case(case: dict[str, Any], work: Path) -> tuple[bool, str]:
    kind = str(case.get("kind", ""))
    expect = case.get("expect", {}) or {}
    if kind == "manifest_parse":
        entries = parse_manifest(str(case["text"]))
        got = len(entries) == int(expect.get("entries", -1))
        if got and "first_family" in expect:
            got = entries[0].family == expect["first_family"]
        return got, "manifest parse"
    if kind == "manifest_error":
        try:
            parse_manifest(str(case["text"]))
        except ManifestError as exc:
            got = (exc.kind == expect.get("error")
                   and exc.line == int(expect.get("line", 1)))
            if "field" in expect:
                got = got and exc.field == expect["field"]
            return got, "manifest error"
        return False, "manifest error: no error raised"
    if kind == "catalog_count":
        catalog = default_catalog()
        table = {"default": catalog.default_faces,
                 "latin": catalog.latin_faces,
                 "chinese": catalog.chinese_faces}[str(case["table"])]
        return len(table) == int(expect.get("faces", -1)), "catalog count"
    if kind == "for_script":
        catalog = default_catalog()
        faces = catalog.for_script(str(case["script"]))
        got = len(faces) == int(expect.get("faces", -1))
        if got and "families" in expect:
            got = sorted({f.family for f in faces}) == sorted(
                expect["families"])
        return got, "for_script"
    if kind == "tables_disjoint":
        return default_catalog().disjoint(), "tables disjoint"
    if kind == "load_checkout":
        root = _fixture_checkout(work / "checkout")
        loaded = load(root)
        return len(loaded) == int(expect.get("faces", -1)), "load checkout"
    if kind == "load_tampered":
        root = _fixture_checkout(work / "checkout")
        target = root / "fonts" / "caladea" / "Caladea-Regular.ttf"
        data = bytearray(target.read_bytes())
        data[100] ^= 0xFF
        target.write_bytes(bytes(data))
        try:
            load(root)
        except LoadError as exc:
            return ("Caladea-Regular.ttf" in str(exc)), "load tampered"
        return False, "load tampered: no error raised"
    if kind == "install_checkout":
        root = _fixture_checkout(work / "checkout")
        store = work / "store"
        installed = install_checkout(root, store)
        got = len(installed) == int(expect.get("installed", -1))
        if got:
            record = json.loads(
                (store / "install-record.json").read_text(encoding="utf-8"))
            got = (len(record["installed"])
                   == int(expect.get("installed", -1)))
            got = got and (store / "fonts" / "noto-sans-hebrew"
                           / "NotoSansHebrew-Regular.ttf").is_file()
            licence = (store / "fonts" / "noto-sans-hebrew"
                       / "OFL.txt").read_text(encoding="utf-8")
            got = got and "SIL OPEN FONT LICENSE" in licence
        return got, "install checkout"
    if kind == "bundle_allowlist":
        root = _fixture_checkout(work / "checkout")
        loaded = load(root)
        allowlist = [(f, s) for f, s in
                     (tuple(item) for item in case["allowlist"])]
        payload = build_embed(loaded, root=root, allowlist=allowlist)
        with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
            names = sorted(bundle.namelist())
            provenance = json.loads(
                bundle.read("provenance.json").decode("utf-8"))
        got = names == sorted(expect["members"])
        if got:
            got = (len(provenance["faces"])
                   == int(expect.get("faces", -1)))
        return got, "bundle allowlist"
    if kind == "bundle_missing":
        root = _fixture_checkout(work / "checkout")
        loaded = load(root)
        try:
            build_embed(loaded, root=root,
                        allowlist=[("No Such Face", "Regular")])
        except ValueError as exc:
            return ("No Such Face" in str(exc)), "bundle missing"
        return False, "bundle missing: no error raised"
    if kind == "coverage":
        faces = parse_font_file(
            _FIXTURE_DIR / "noto-sans-hebrew"
            / "NotoSansHebrew-Regular.ttf")
        face = faces[0]
        got = face.covers_text(str(case["text"])) == bool(
            expect.get("covers", False))
        return got, "coverage"
    return False, f"unknown parity kind: {kind}"


def run_parity(toml_path: str | Path | None = None, *,
               fixtures_dir: str | Path | None = None) -> dict[str, Any]:
    """Run every case in the craft-fonts parity TOML. A missing
    fixture is a failed case, never a skip."""
    path = Path(toml_path) if toml_path is not None else _toml_path()
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if fixtures_dir is not None:
        global _FIXTURE_DIR
        _FIXTURE_DIR = Path(fixtures_dir)
    passed = failed = 0
    results: list[dict[str, Any]] = []
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for case in data.get("case", []):
            case_work = work / str(case.get("id", "case"))
            case_work.mkdir(parents=True, exist_ok=True)
            try:
                ok, detail = _run_case(case, case_work)
            except Exception as exc:  # noqa: BLE001 - one bad case
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            if ok:
                passed += 1
            else:
                failed += 1
            results.append({"id": case.get("id"), "ok": ok,
                            "detail": detail})
    return {"ok": failed == 0, "passed": passed, "failed": failed,
            "total": passed + failed, "results": results}


def _attribution_toml_path() -> Path:
    return (Path(__file__).resolve().parent
            / "craftfonts-attribution.toml")


def verify_attribution(toml_path: str | Path | None = None, *,
                       clone_dir: str | Path | None = None
                       ) -> dict[str, Any]:
    """Check every attribution entry. Upstream entries are
    hash-checked against the craft-fonts clone; Devon entries are
    checked for a lineage note naming the file they derive from."""
    path = (Path(toml_path) if toml_path is not None
            else _attribution_toml_path())
    clone = Path(clone_dir) if clone_dir is not None else _CLONE
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    checked = failed = skipped = noted = 0
    entries: list[dict[str, Any]] = []
    for entry in data.get("upstream", []):
        source = clone / str(entry["file"])
        if not source.is_file():
            skipped += 1
            entries.append({**entry, "status": "skipped",
                            "reason": "not in the local clone"})
            continue
        checked += 1
        ok = (hashlib.sha256(source.read_bytes()).hexdigest()
              == str(entry["sha256"]))
        if not ok:
            failed += 1
        entries.append({**entry, "status": "ok" if ok else "mismatch"})
    for entry in data.get("devon", []):
        target = (Path(__file__).resolve().parent
                  / str(entry["file"]))
        checked += 1
        ok = target.is_file() and bool(entry.get("derives_from"))
        if not ok:
            failed += 1
        entries.append({**entry,
                        "status": "ok" if ok else "missing"})
    for entry in data.get("noted", []):
        noted += 1
        entries.append({**entry, "status": "noted"})
    return {"ok": failed == 0 and skipped == 0, "checked": checked,
            "failed": failed, "skipped": skipped, "noted": noted,
            "entries": entries}
