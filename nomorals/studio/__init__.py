"""The Studio extension store — the craft-extensions contract,
ported to Devon.

The store (``StudioStore``) installs, removes, enables and themes
extensions for the Studio; the reader (``Reader``) is the side the
Studio pages use and never writes. The rules are the upstream
ones (``craft-extensions``, dual MIT/Apache): a package manifest
(``craft-extension.json``), validated ids, a theme palette with a
WCAG contrast check, trust tiers the host assigns, and bounded,
hash-verified unpacking into ``packages/<id>/<version>/`` with an
atomic, generation-bumped ``state.json``.

Devon's Studio is one app (``studio``, interface 1). A theme wears
the Studio's look; a plug-in adds a declarative panel to a Studio
page; a resource pack is data the Studio lists. Themes carry no
code and plug-ins ask for no permissions, in version 1.
"""

from __future__ import annotations

from .ids import AppId, ExtensionId
from .manifest import Kind, Manifest, ManifestError
from .palette import Palette, Rgb, ThemeFile
from .panel import PanelFile
from .store import (
    ActiveTheme,
    Found,
    Installed,
    Reader,
    State,
    StateError,
    StoreError,
    StudioStore,
    UnpackError,
    default_extensions_dir,
)
from .tier import Policy, Tier
from .version import Version

__all__ = [
    "ActiveTheme",
    "AppId",
    "ExtensionId",
    "Found",
    "Installed",
    "Kind",
    "Manifest",
    "ManifestError",
    "Palette",
    "PanelFile",
    "Policy",
    "Reader",
    "Rgb",
    "State",
    "StateError",
    "StoreError",
    "StudioStore",
    "ThemeFile",
    "Tier",
    "UnpackError",
    "Version",
    "default_extensions_dir",
]
