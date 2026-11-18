"""The write side of the shared Studio extension store, and the
read side the Studio pages use.

A port of ``craft-extensions`` (``crates/ext-store/src/``, dual
MIT/Apache). Layout under the store root::

    packages/<id>/<version>/...   unpacked, never edited afterwards
    staging/<unique>/             where a package is unpacked and
                                  checked before it counts
    state.json                    what is installed and enabled
    .lock                         held while the state file changes

Invariants, kept from upstream: an install is all or nothing — the
package is unpacked and checked outside the lock, the lock covers
only the move into place and the state write, the state file is
replaced atomically with a bumped generation, and a failure leaves
the previous state and no stray package folder. A file can never
replace a first-party extension, and a downgrade is refused. The
reader never writes: anything it cannot read or that fails
validation counts as absent, and a running page can poll
``generation`` cheaply to learn that something changed.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.http import HttpClient
from .ids import AppId, ExtensionId
from .manifest import (
    MANIFEST_NAME,
    MAX_MANIFEST_BYTES,
    THEME_FILE,
    Kind,
    Manifest,
    ManifestError,
)
from .palette import MAX_THEME_BYTES, Palette, ThemeError, ThemeFile
from .panel import PanelError, PanelFile
from .paths import validate_path
from .tier import Policy, Tier
from .version import Version

__all__ = [
    "ActiveTheme",
    "Caps",
    "Found",
    "Installed",
    "Reader",
    "STUDIO_APP",
    "STUDIO_INTERFACE",
    "STUDIO_VERSION",
    "State",
    "StateError",
    "StoreError",
    "StudioStore",
    "UnpackError",
    "default_extensions_dir",
    "unpack_into",
]

#: The app id Devon's Studio answers to, and the versions it speaks.
STUDIO_APP = AppId.new("studio")
STUDIO_VERSION = Version(1, 0, 0)
STUDIO_INTERFACE = 1

STATE_FILE = "state.json"
MAX_STATE_BYTES = 4 * 1024 * 1024
STATE_SCHEMA = 1

_LOCK_WAIT_S = 5.0
_LOCK_STALE_S = 30.0
_RATIO_FLOOR = 1024 * 1024


class StoreError(ValueError):
    """The store refused an operation, with the reason as the message."""


class StateError(ValueError):
    """The state file could not be read, with the reason as the message."""


class UnpackError(ValueError):
    """An archive was refused, with the reason as the message."""


@dataclass
class Caps:
    """Limits applied before and while reading an archive."""

    max_archive: int = 300 * 1024 * 1024
    max_ratio: int = 200
    max_entries: int = 600


@dataclass
class Installed:
    """One installed extension, as the state file records it."""

    id: ExtensionId
    version: Version
    kind: str
    name: str
    tier: str
    apps: list

    def to_json(self) -> dict:
        return {"id": str(self.id), "version": str(self.version),
                "kind": str(self.kind), "name": self.name,
                "tier": str(self.tier), "apps": list(self.apps)}

    @classmethod
    def from_json(cls, raw: dict) -> Installed:
        return cls(id=ExtensionId.new(str(raw["id"])),
                   version=Version.parse(str(raw["version"])),
                   kind=str(raw["kind"]), name=str(raw["name"]),
                   tier=str(raw["tier"]),
                   apps=[str(a) for a in raw.get("apps", [])])


@dataclass
class State:
    """What is installed, enabled for which app, and which theme
    each app uses."""

    schema: int = STATE_SCHEMA
    generation: int = 0
    installed: list = field(default_factory=list)
    enabled: dict = field(default_factory=dict)
    active_theme: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"schema": self.schema, "generation": self.generation,
                "installed": [i.to_json() for i in self.installed],
                "enabled": {app: sorted(ids)
                            for app, ids in self.enabled.items()},
                "active_theme": dict(self.active_theme)}

    @classmethod
    def read(cls, path: Path) -> State:
        """Read the state file. A missing file is an empty state;
        anything else wrong is an error, and the file is left
        exactly as it was."""
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return cls()
        except OSError:
            raise StateError(
                "the list of installed extensions cannot be opened"
                ) from None
        if len(raw) > MAX_STATE_BYTES:
            raise StateError(
                "the list of installed extensions is too large to read")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise StateError(
                "the list of installed extensions is damaged") from None
        if not isinstance(data, dict):
            raise StateError(
                "the list of installed extensions is damaged")
        schema = data.get("schema")
        if isinstance(schema, int) and schema > STATE_SCHEMA:
            raise StateError(
                "the list of installed extensions was saved by a "
                "newer version of Devon")
        if schema != STATE_SCHEMA:
            raise StateError(
                "the list of installed extensions is damaged")
        try:
            installed = [Installed.from_json(item)
                         for item in data.get("installed", [])]
            enabled = {str(app): {str(i) for i in ids}
                       for app, ids in
                       (data.get("enabled") or {}).items()}
            active = {str(app): str(i) for app, i in
                      (data.get("active_theme") or {}).items()}
            generation = int(data.get("generation", 0))
        except (ValueError, TypeError, AttributeError, KeyError):
            raise StateError(
                "the list of installed extensions is damaged") from None
        return cls(schema=STATE_SCHEMA, generation=generation,
                   installed=installed, enabled=enabled,
                   active_theme=active)

    def find(self, ext_id: str) -> Installed | None:
        """The installed record for one id, or None."""
        for item in self.installed:
            if item.id == ext_id:
                return item
        return None

    def is_enabled(self, app: str, ext_id: str) -> bool:
        """Whether an extension is switched on for one app."""
        return ext_id in self.enabled.get(str(app), set())


def default_extensions_dir(home: str | None = None) -> str:
    """``<home>/studio/extensions``."""
    from ..core.failure_store import home_dir as _devon_home_dir

    base = home or _devon_home_dir()
    return str(Path(base) / "studio" / "extensions")


class _Lock:
    """A cross-process lock file, held while the state changes."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None

    def __enter__(self) -> _Lock:
        deadline = time.monotonic() + _LOCK_WAIT_S
        while True:
            try:
                self._fd = os.open(self._path,
                                   os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self._path)
                except OSError:
                    age = 0.0
                if age > _LOCK_STALE_S:
                    with suppress(OSError):
                        os.unlink(self._path)
                    continue
                if time.monotonic() > deadline:
                    raise StoreError(
                        "another app is changing extensions right "
                        "now; try again in a moment") from None
                time.sleep(0.02)

    def __exit__(self, *exc: Any) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with suppress(OSError):
            os.unlink(self._path)


def _hex_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def unpack_into(data: bytes, dest: Path,
                caps: Caps | None = None) -> Manifest:
    """Check and unpack ``data`` into ``dest``, which must be an
    empty or new folder. On error ``dest`` may hold partial files;
    the caller removes it."""
    caps = caps or Caps()
    if len(data) > caps.max_archive:
        raise UnpackError("the package file is too big")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise UnpackError("this is not a package file") from None
    infos = archive.infolist()
    if len(infos) > caps.max_entries:
        raise UnpackError("the package holds too many files")

    # Pass 1: names and kinds only, nothing decompressed.
    files: list[tuple[int, str]] = []
    for index, info in enumerate(infos):
        name = info.filename
        if info.is_dir():
            name = name.rstrip("/")
            err = validate_path(name)
            if err is not None:
                raise UnpackError(str(err.value))
            continue
        err = validate_path(name)
        if err is not None:
            raise UnpackError(str(err.value))
        if info.flag_bits & 0x1:
            raise UnpackError("the package is locked with a password")
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            raise UnpackError(
                "the package contains a shortcut, which is not allowed")
        files.append((index, name))

    # Pass 2: the manifest, read with a hard limit.
    manifest_index = next(
        (i for i, name in files if name == MANIFEST_NAME), None)
    if manifest_index is None:
        raise UnpackError("the package has no description of itself")
    try:
        with archive.open(infos[manifest_index]) as entry:
            text = entry.read(MAX_MANIFEST_BYTES + 1)
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError):
        raise UnpackError(
            "the package is packed in a way this app does not "
            "support") from None
    try:
        manifest = Manifest.parse(text)
    except ManifestError as exc:
        raise UnpackError(str(exc)) from None

    # Pass 3: the archive and the manifest must agree exactly.
    listed = {entry.path for entry in manifest.files}
    if any(name != MANIFEST_NAME and name not in listed
           for _, name in files):
        raise UnpackError(
            "the package holds a file its description does not mention")
    if any(entry.path not in {name for _, name in files}
           for entry in manifest.files):
        raise UnpackError(
            "the package is missing a file its description promises")

    # Pass 4: stream each listed file out, counting bytes and
    # hashing as it goes.
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise UnpackError(
            "the package's files could not be saved") from None
    names = {name: index for index, name in files}
    for want in manifest.files:
        info = infos[names[want.path]]
        if info.file_size != want.size:
            raise UnpackError(
                "a file in the package is not the size it claims to be")
        if (info.file_size > _RATIO_FLOOR
                and info.file_size // max(1, info.compress_size)
                > caps.max_ratio):
            raise UnpackError(
                "a file in the package unpacks to far more than it "
                "should")
        target = dest / want.path
        try:
            if target.parent != dest:
                target.parent.mkdir(parents=True, exist_ok=True)
            out = target.open("xb")
        except OSError:
            raise UnpackError(
                "the package's files could not be saved") from None
        hasher = hashlib.sha256()
        total = 0
        try:
            with archive.open(info) as entry:
                while True:
                    chunk = entry.read(min(64 * 1024,
                                           want.size + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > want.size:
                        raise UnpackError(
                            "a file in the package is not the size it "
                            "claims to be")
                    hasher.update(chunk)
                    out.write(chunk)
        except UnpackError:
            out.close()
            raise
        except (zipfile.BadZipFile, NotImplementedError, RuntimeError,
                OSError):
            out.close()
            raise UnpackError(
                "the package is packed in a way this app does not "
                "support") from None
        out.close()
        if total != want.size:
            raise UnpackError(
                "a file in the package is not the size it claims to be")
        if hasher.hexdigest() != want.sha256:
            raise UnpackError(
                "a file in the package has been changed or damaged")
    return manifest


@dataclass
class Found:
    """An installed extension together with where its files are."""

    installed: Installed
    dir: Path


@dataclass
class ActiveTheme:
    """The theme an app should be wearing."""

    id: str
    name: str
    palette: Palette
    overrides: dict


class Reader:
    """The read side the Studio pages use: what is installed and
    enabled for me, and what is my theme. Never writes; anything it
    cannot read or that fails validation counts as absent."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _state(self) -> State | None:
        try:
            return State.read(self._root / STATE_FILE)
        except StateError:
            return None

    def generation(self) -> int:
        """Changes whenever anything is installed, removed, enabled
        or chosen. 0 when there is no store or it cannot be read."""
        state = self._state()
        return state.generation if state is not None else 0

    def _dir_of(self, item: Installed) -> Path:
        return (self._root / "packages" / str(item.id)
                / str(item.version))

    def installed_for(self, app: str | AppId,
                      kind: str | Kind) -> list[Found]:
        """Extensions of this kind that are installed and switched
        on for ``app``, whose files are still there."""
        state = self._state()
        if state is None:
            return []
        app = str(app)
        kind = str(kind)
        found: list[Found] = []
        for item in state.installed:
            if item.kind != kind or app not in item.apps:
                continue
            if not state.is_enabled(app, str(item.id)):
                continue
            found.append(Found(installed=item,
                               dir=self._dir_of(item)))
        return [f for f in found if f.dir.is_dir()]

    def active_theme(self, app: str | AppId) -> ActiveTheme | None:
        """The theme ``app`` has chosen, read and checked again
        from disk. None means "use your own look"."""
        state = self._state()
        if state is None:
            return None
        app = str(app)
        ext_id = state.active_theme.get(app)
        if ext_id is None:
            return None
        installed = state.find(ext_id)
        if installed is None or installed.kind != "theme" \
                or not state.is_enabled(app, ext_id):
            return None
        theme = _read_theme(self._dir_of(installed) / THEME_FILE)
        if theme is None:
            return None
        return ActiveTheme(
            id=str(installed.id), name=installed.name,
            palette=theme.palette,
            overrides=theme.overrides_for(app) or {})


def _read_theme(path: Path) -> ThemeFile | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > MAX_THEME_BYTES:
        return None
    try:
        return ThemeFile.parse(data)
    except ThemeError:
        return None


def _read_panel(path: Path) -> PanelFile | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None
    try:
        return PanelFile.parse(data)
    except PanelError:
        return None


def _unique_name() -> str:
    return f"{os.getpid()}-{time.time_ns()}"


class StudioStore:
    """The Studio extension store: install, remove, enable, choose
    a theme. ``root`` is the store folder; ``home`` is the Devon
    home it is derived from when no root is given."""

    def __init__(self, root: str | None = None,
                 home: str | None = None) -> None:
        self.root = Path(root or default_extensions_dir(home))
        self.root.mkdir(parents=True, exist_ok=True)
        self._policy = Policy.standard()
        self._caps = Caps()

    def package_dir(self, ext_id: str, version: str) -> Path:
        """The folder one installed version lives in."""
        return self.root / "packages" / ext_id / version

    def _state_path(self) -> Path:
        return self.root / STATE_FILE

    def state(self) -> State:
        """The current state, or a StoreError when it is damaged."""
        try:
            return State.read(self._state_path())
        except StateError as exc:
            raise StoreError(str(exc)) from None

    def reader(self) -> Reader:
        """The read side for the Studio pages."""
        return Reader(self.root)

    def installed_for(self, app: str | AppId,
                      kind: str | Kind) -> list[Found]:
        """Installed-and-enabled extensions of one kind for one app."""
        return self.reader().installed_for(app, kind)

    def active_theme(self, app: str | AppId) -> ActiveTheme | None:
        """The theme one app has chosen, or None."""
        return self.reader().active_theme(app)

    def _save(self, state: State) -> None:
        data = json.dumps(state.to_json(), indent=2).encode("utf-8")
        tmp = self._state_path().with_name(
            f".{STATE_FILE}.tmp-{os.getpid()}")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, self._state_path())
        except OSError:
            raise StoreError(
                "the extension files could not be saved") from None

    def _update(self, change) -> Any:
        """Run ``change`` on the current state under the lock and
        save the result with the generation bumped."""
        self.root.mkdir(parents=True, exist_ok=True)
        with _Lock(self.root / ".lock"):
            state = self.state()
            out = change(state)
            state.generation += 1
            self._save(state)
            return out

    def install_zip(self, data: bytes,
                    tier: str | Tier = "from-file") -> Installed:
        """Install a package from an archive held in memory.
        ``tier`` is where the caller got it from."""
        err = self._policy.admit(tier)
        if err is not None:
            raise StoreError(err.value)
        staging = self.root / "staging" / _unique_name()
        try:
            return self._install_staged(data, tier, staging)
        finally:
            # On success the folder was moved away and this finds
            # nothing; on failure it removes the debris.
            _remove_tree(staging)

    def install_file(self, path: str | Path,
                     tier: str | Tier = "from-file") -> Installed:
        """Install a package from a file, refusing to read one past
        the size limit."""
        try:
            with open(path, "rb") as handle:
                data = handle.read(self._caps.max_archive + 1)
        except OSError:
            raise StoreError(
                "the extension files could not be saved") from None
        if len(data) > self._caps.max_archive:
            raise StoreError("the package file is too big")
        return self.install_zip(data, tier=tier)

    def install_url(self, url: str, *, sha256: str, size: int,
                    tier: str | Tier = "from-file",
                    client: Any | None = None) -> Installed:
        """Install a package from a URL, refusing it unless the
        bytes match the pinned digest and size. The store builds
        and hosts nothing — the bytes come from the author's own
        release, checked against the digest before they count."""
        http = client if client is not None else HttpClient(timeout=15.0)
        try:
            response = http.get(url)
        except Exception as exc:  # noqa: BLE001 - a failed fetch is a refusal
            raise StoreError(f"the download failed: {exc}") from None
        if not getattr(response, "ok", False):
            raise StoreError(
                "the download failed "
                f"({getattr(response, 'status', '?')})")
        body = getattr(response, "body", b"")
        if not isinstance(body, (bytes, bytearray)):
            raise StoreError("the download failed")
        body = bytes(body)
        if len(body) != size:
            raise StoreError(
                "the download is not the size it claims to be")
        if _hex_digest(body) != sha256:
            raise StoreError(
                "the download's digest does not match its pinned digest")
        return self.install_zip(body, tier=tier)

    def _install_staged(self, data: bytes, tier: str | Tier,
                        staging: Path) -> Installed:
        try:
            manifest = unpack_into(data, staging, self._caps)
        except UnpackError as exc:
            raise StoreError(str(exc)) from None
        err = self._policy.check_id(tier, manifest.id)
        if err is not None:
            raise StoreError(err.value)
        tier = Tier.parse(tier) if isinstance(tier, str) else tier
        compat = manifest.compat.get(str(STUDIO_APP))
        if compat is None:
            raise StoreError("that extension is not made for this app")
        if compat.min_app is not None and compat.min_app > STUDIO_VERSION:
            raise StoreError(
                "that extension needs a newer Studio "
                f"(needs {compat.min_app}, this Studio is "
                f"{STUDIO_VERSION})")
        if manifest.kind == "plugin" \
                and compat.interface != STUDIO_INTERFACE:
            raise StoreError(
                "that plug-in was built for interface "
                f"{compat.interface}, this Studio speaks interface "
                f"{STUDIO_INTERFACE}")
        if manifest.kind == "theme":
            theme_path = staging / THEME_FILE
            try:
                theme_data = theme_path.read_bytes()
            except OSError:
                raise StoreError(
                    "the extension files could not be saved") from None
            try:
                ThemeFile.parse(theme_data)
            except ThemeError as exc:
                raise StoreError(str(exc)) from None
        if manifest.kind == "plugin":
            panel_path = staging / "panel.json"
            try:
                panel_data = panel_path.read_bytes()
            except OSError:
                raise StoreError(
                    "the extension files could not be saved") from None
            try:
                PanelFile.parse(panel_data)
            except PanelError as exc:
                raise StoreError(str(exc)) from None
        record = Installed(
            id=manifest.id, version=manifest.version,
            kind=str(manifest.kind), name=manifest.name, tier=str(tier),
            apps=sorted(manifest.compat.keys()))
        destination = self.package_dir(str(record.id),
                                       str(record.version))
        old_dir: Path | None = None

        def _move(state: State) -> None:
            nonlocal old_dir
            existing = state.find(str(record.id))
            if existing is not None:
                if existing.tier == "first-party" \
                        and str(tier) != "first-party":
                    raise StoreError(
                        "this file would replace an extension made "
                        "by Devon, which is not allowed")
                if record.version < existing.version:
                    raise StoreError(
                        "a newer version of this extension is "
                        "already installed")
                if record.version == existing.version:
                    raise StoreError(
                        "this version of the extension is already "
                        "installed")
                old_dir = self.package_dir(str(existing.id),
                                           str(existing.version))
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise StoreError(
                    "the extension files could not be saved") from None
            if destination.exists():
                raise StoreError(
                    "the extension files could not be saved")
            try:
                os.rename(staging, destination)
            except OSError:
                raise StoreError(
                    "the extension files could not be saved") from None
            state.installed = [i for i in state.installed
                               if i.id != record.id]
            state.installed.append(record)
            for app in record.apps:
                state.enabled.setdefault(app, set()).add(str(record.id))

        try:
            self._update(_move)
        except StoreError:
            # The package was moved into place but the state could
            # not be saved: put things back.
            if not staging.exists() and destination.exists() \
                    and self.state().find(str(record.id)) is None:
                _remove_tree(destination)
            raise
        if old_dir is not None:
            _remove_tree(old_dir)
        return record

    def uninstall(self, ext_id: str) -> None:
        """Remove an extension everywhere: its files, its
        enablement and any app's choice of it as theme."""

        def _drop(state: State) -> Path:
            existing = state.find(ext_id)
            if existing is None:
                raise StoreError("that extension is not installed")
            state.installed = [i for i in state.installed
                               if i.id != ext_id]
            for enabled in state.enabled.values():
                enabled.discard(ext_id)
            state.active_theme = {
                app: chosen for app, chosen in
                state.active_theme.items() if chosen != ext_id}
            return self.package_dir(str(existing.id),
                                    str(existing.version))

        removed = self._update(_drop)
        _remove_tree(removed)
        with suppress(OSError):
            (self.root / "packages" / ext_id).rmdir()

    def set_enabled(self, app: str | AppId, ext_id: str,
                    on: bool) -> None:
        """Turn an installed extension on or off for one app.
        Turning a theme off also stops the app using it."""
        app = str(app)

        def _flip(state: State) -> None:
            existing = state.find(ext_id)
            if existing is None:
                raise StoreError("that extension is not installed")
            if app not in existing.apps:
                raise StoreError(
                    "that extension is not made for this app")
            enabled = state.enabled.setdefault(app, set())
            if on:
                enabled.add(ext_id)
            else:
                enabled.discard(ext_id)
                if state.active_theme.get(app) == ext_id:
                    del state.active_theme[app]

        self._update(_flip)

    def set_active_theme(self, app: str | AppId,
                         ext_id: str | None) -> None:
        """Choose which installed theme an app uses, or None to
        return to its own look."""
        app = str(app)

        def _choose(state: State) -> None:
            if ext_id is None:
                state.active_theme.pop(app, None)
                return
            existing = state.find(ext_id)
            if existing is None:
                raise StoreError("that extension is not installed")
            if existing.kind != "theme":
                raise StoreError(
                    "only a theme can be chosen as the look of an app")
            if app not in existing.apps:
                raise StoreError(
                    "that extension is not made for this app")
            if not state.is_enabled(app, ext_id):
                raise StoreError(
                    "turn the theme on for this app before choosing it")
            state.active_theme[app] = ext_id

        self._update(_choose)


def _remove_tree(path: Path) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
