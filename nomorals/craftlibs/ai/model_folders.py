"""Model folders (Craft-AI port of ai-core ``model_folders.rs``).

A person points the suite at a folder they already have, in whatever
structure they like; the store records where each kind of model lives.
A folder is checked before it is stored: ids are short and unique,
paths are absolute, kinds are well-formed names, sub-folders stay
inside the folder. Nothing here touches the disk.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "KINDS",
    "MAX_DIRS_PER_KIND",
    "MAX_FOLDERS",
    "FolderError",
    "FolderOrigin",
    "ModelFolder",
    "download_target",
    "is_kind",
    "is_subfolder",
    "looks_absolute",
    "validate_folders",
]

MAX_FOLDERS = 32
MAX_DIRS_PER_KIND = 64

#: The model kinds the suite proposes a sub-folder for, using
#: ComfyUI's folder names.
KINDS = ["checkpoints", "diffusion_models", "text_encoders", "vae",
         "loras", "controlnet", "clip_vision", "upscale_models",
         "embeddings", "sam3", "background_removal"]


class FolderOrigin(Enum):
    """Where a folder came from."""

    CHOSEN = "chosen"
    COMFYUI = "comfyui"

    @classmethod
    def from_str(cls, value: str) -> FolderOrigin:
        if value == "comfyui":
            return cls.COMFYUI
        return cls.CHOSEN

    def as_str(self) -> str:
        return self.value


class FolderError(ValueError):
    """Why a folder list was refused."""

    def __init__(self, kind: str, message: str, **fields) -> None:
        super().__init__(message)
        self.kind = kind
        self.fields = fields

    @classmethod
    def too_many(cls, n: int) -> FolderError:
        return cls("too_many",
                   f"{n} model folders exceeds the limit of {MAX_FOLDERS}",
                   count=n)

    @classmethod
    def bad_id(cls, folder_id: str) -> FolderError:
        return cls("bad_id", f'"{folder_id}" is not a valid folder name',
                   folder=folder_id)

    @classmethod
    def duplicate_id(cls, folder_id: str) -> FolderError:
        return cls("duplicate_id",
                   f"model folder {folder_id} appears twice",
                   folder=folder_id)

    @classmethod
    def bad_label(cls, folder_id: str) -> FolderError:
        return cls("bad_label",
                   f"model folder {folder_id} needs a short name",
                   folder=folder_id)

    @classmethod
    def not_absolute(cls, folder_id: str) -> FolderError:
        return cls("not_absolute",
                   f"model folder {folder_id} needs a full path",
                   folder=folder_id)

    @classmethod
    def bad_kind(cls, folder: str, kind: str) -> FolderError:
        return cls("bad_kind",
                   f'model folder {folder}: "{kind}" is not a kind of model',
                   folder=folder, kind_name=kind)

    @classmethod
    def bad_subfolder(cls, folder: str, sub: str) -> FolderError:
        return cls("bad_subfolder",
                   f'model folder {folder}: "{sub}" is not inside the folder',
                   folder=folder, sub=sub)

    @classmethod
    def too_many_dirs(cls, folder: str, kind: str) -> FolderError:
        return cls("too_many_dirs",
                   f"model folder {folder}: too many places listed for {kind}",
                   folder=folder, kind_name=kind)

    @classmethod
    def several_defaults(cls) -> FolderError:
        return cls("several_defaults",
                   "only one model folder can be the default")


class ModelFolder:
    """One registered folder and what is known to be in it."""

    def __init__(self, id: str, label: str, path: str,
                 default: bool = False,
                 origin: FolderOrigin | str = FolderOrigin.CHOSEN,
                 kinds: dict | None = None) -> None:
        self.id = id
        self.label = label
        self.path = path
        self.default = bool(default)
        self.origin = (origin if isinstance(origin, FolderOrigin)
                       else FolderOrigin.from_str(str(origin)))
        self.kinds: dict[str, list[str]] = {
            str(kind): [str(d) for d in dirs]
            for kind, dirs in (kinds or {}).items()}

    def to_dict(self) -> dict:
        data: dict = {"id": self.id, "label": self.label,
                      "path": self.path}
        if self.default:
            data["default"] = True
        if self.origin is not FolderOrigin.CHOSEN:
            data["origin"] = self.origin.as_str()
        if self.kinds:
            data["kinds"] = self.kinds
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ModelFolder:
        return cls(str(data.get("id", "")), str(data.get("label", "")),
                   str(data.get("path", "")),
                   bool(data.get("default", False)),
                   FolderOrigin.from_str(str(data.get("origin", "chosen"))),
                   dict(data.get("kinds") or {}))

    def __eq__(self, other) -> bool:
        return (isinstance(other, ModelFolder)
                and self.to_dict() == other.to_dict())

    def __repr__(self) -> str:
        return f"ModelFolder({self.id!r}, {self.path!r})"


def is_kind(name: str) -> bool:
    """A kind name: lower-case letters, digits and ``_``, starting
    with a letter, at most 40 characters."""
    return (bool(name) and len(name) <= 40
            and name[0].isascii() and name[0].islower()
            and all(c.isascii() and (c.islower() or c.isdigit()
                                     or c == "_") for c in name))


def is_subfolder(sub: str) -> bool:
    """A sub-folder: relative, ``/`` separated, no ``..``, no ``:``
    or control characters. ``""`` is allowed."""
    if len(sub) > 260 or sub.startswith("/"):
        return False
    if "\\" in sub or ":" in sub:
        return False
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in sub):
        return False
    parts = sub.split("/")
    return all(part != ".." and part != "."
               and (part == "") == (sub == "") for part in parts)


def looks_absolute(path: str) -> bool:
    """Does this text look like a full path on Windows, macOS or
    Linux?"""
    if path.startswith("/") or path.startswith("\\\\"):
        return True
    data = path.encode("utf-8", "surrogateescape")
    return (len(data) >= 3 and data[0:1].isalpha()
            and data[1:2] == b":" and data[2:3] in (b"\\", b"/"))


def validate_folders(folders: list) -> None:
    """Check a whole list before it is stored. Raises
    :class:`FolderError` naming the refusal."""
    if len(folders) > MAX_FOLDERS:
        raise FolderError.too_many(len(folders))
    if sum(1 for f in folders if f.default) > 1:
        raise FolderError.several_defaults()
    for i, folder in enumerate(folders):
        folder_id = folder.id
        id_ok = (bool(folder_id) and len(folder_id) <= 40
                 and all(c.isascii() and (c.islower() or c.isdigit()
                                          or c == "-") for c in folder_id))
        if not id_ok:
            raise FolderError.bad_id(folder_id)
        if any(other.id == folder_id for other in folders[i + 1:]):
            raise FolderError.duplicate_id(folder_id)
        if (not folder.label.strip() or len(folder.label) > 120
                or any(ord(c) < 0x20 or ord(c) == 0x7F
                       for c in folder.label)):
            raise FolderError.bad_label(folder_id)
        if (not looks_absolute(folder.path)
                or any(ord(c) < 0x20 or ord(c) == 0x7F
                       for c in folder.path)
                or len(folder.path) > 1024):
            raise FolderError.not_absolute(folder_id)
        for kind, dirs in folder.kinds.items():
            if not is_kind(kind):
                raise FolderError.bad_kind(folder_id, kind)
            if len(dirs) > MAX_DIRS_PER_KIND:
                raise FolderError.too_many_dirs(folder_id, kind)
            for sub in dirs:
                if not is_subfolder(sub):
                    raise FolderError.bad_subfolder(folder_id, sub)


def download_target(folders: list, kind: str):
    """Where a new file of ``kind`` should be saved: the explicitly
    chosen default folder, inside its first place already holding
    that kind, else a sub-folder named for the kind. Returns
    ``(folder, subfolder)`` or ``None``. Linked folders and
    non-default folders never receive downloads."""
    folder = next((f for f in folders
                   if f.origin is FolderOrigin.CHOSEN and f.default), None)
    if folder is None:
        return None
    dirs = folder.kinds.get(kind)
    if dirs:
        return folder, dirs[0]
    # No place yet: the kind names a new sub-folder, so it must be a
    # safe one. A caller passing `..` or a path gets nowhere to save
    # rather than a file outside the folder.
    if kind and is_subfolder(kind):
        return folder, kind
    return None
