"""The file class — one addressable file across the whole system.

``File`` resolves a human ref (plain path, ``museum:<id>``,
``library:<slug>``, ``archive:<path>::<member>``) to a real file with
kind/shelf/hash/provenance. ``FileStore`` shelves files under
``<home>/data/files/<shelf>/``. ``pack_files`` / ``unpack_pack``
build and land manifest-carrying artifact archives on the Archivist's
bundle seam. ``files_exhibitor`` / ``museum_with_files`` put shelved
files into the Museum census. ``file_action`` is the one control chat
(``/files``), the brain tool (``file``), and the console (``nm file``)
all run.
"""

from __future__ import annotations

from .chat import control_files, file_action
from .exhibit import files_exhibitor, museum_with_files
from .file import File, FileError, resolve
from .kinds import KINDS, kind_for, register_kind, shelf_for_kind
from .packs import pack_files, pack_info, unpack_pack
from .store import FileStore

__all__ = [
    "File",
    "FileError",
    "FileStore",
    "KINDS",
    "control_files",
    "file_action",
    "files_exhibitor",
    "kind_for",
    "museum_with_files",
    "pack_files",
    "pack_info",
    "register_kind",
    "resolve",
    "shelf_for_kind",
    "unpack_pack",
]
