"""Tests for the mechanical dormant-engine inventory.

The scan is proven mechanical (not a hardcoded list) by the synthetic
tree tests: the dormant set follows the tree — add an importer and the
module stops being dormant, remove it and it returns.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.meta.dormant_inventory import dormant_modules, scan


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _tree(root: Path) -> Path:
    """Synthetic mini-repo: wired, string-wired, dormant, test-only."""
    _write(root / "nomorals" / "__init__.py", "")
    _write(root / "nomorals" / "engines" / "__init__.py", "")
    _write(root / "nomorals" / "engines" / "wired.py",
           "def run():\n    return 1\n")
    _write(root / "nomorals" / "engines" / "stringwired.py",
           "def run():\n    return 2\n")
    _write(root / "nomorals" / "engines" / "dormant.py",
           "def run():\n    return 3\n")
    _write(root / "nomorals" / "engines" / "testonly.py",
           "def run():\n    return 4\n")
    _write(root / "nomorals" / "cmdline" / "__init__.py", "")
    _write(root / "nomorals" / "cmdline" / "dispatch.py",
           "import importlib\n"
           "from nomorals.engines import wired\n"
           "importlib.import_module(\"nomorals.engines.stringwired\")\n"
           "importlib.import_module(some_computed_name)\n"
           "wired.run()\n")
    _write(root / "tests" / "test_engines.py",
           "from nomorals.engines import testonly\n")
    return root


def test_synthetic_tree_classifies_by_references(tmp_path):
    result = scan(_tree(tmp_path))
    dormant = {d.module for d in result.dormant}
    assert "nomorals.engines.dormant" in dormant
    assert "nomorals.engines.testonly" in dormant  # tests do NOT rescue it
    assert "nomorals.engines.wired" not in dormant
    assert "nomorals.engines.stringwired" not in dormant  # importlib literal
    # dispatch itself has no importer; it IS dormant by the reference rule
    # (entry status is informational, not a rescue).
    assert "nomorals.cmdline.dispatch" in dormant
    assert "nomorals.engines" not in dormant  # __init__ files are wiring


def test_synthetic_tree_test_only_evidence_reported(tmp_path):
    result = scan(_tree(tmp_path))
    testonly = [d for d in result.dormant
                if d.module == "nomorals.engines.testonly"][0]
    assert testonly.test_importers == ["tests.test_engines"]
    assert "only by tests" in testonly.reason
    dormant_only = [d for d in result.dormant
                    if d.module == "nomorals.engines.dormant"][0]
    assert dormant_only.test_importers == []
    assert "no production or test" in dormant_only.reason


def test_synthetic_tree_dynamic_loader_caveat(tmp_path):
    result = scan(_tree(tmp_path))
    assert "nomorals.cmdline.dispatch" in result.dynamic_loaders


def test_synthetic_tree_dormant_follows_tree_not_list(tmp_path):
    """Adding an importer unwires dormancy — proof the list is computed."""
    root = _tree(tmp_path)
    assert "nomorals.engines.dormant" in {d.module for d in dormant_modules(root)}
    _write(root / "nomorals" / "engines" / "wired.py",
           "from nomorals.engines import dormant\n"
           "def run():\n    return dormant.run()\n")
    assert "nomorals.engines.dormant" not in {d.module for d in dormant_modules(root)}


def test_synthetic_tree_dynamic_registry_short_names_count(tmp_path):
    """The tool-registry pattern: literal tuple + import_module(f".{n}", __package__)."""
    root = tmp_path
    _write(root / "nomorals" / "__init__.py", "")
    _write(root / "nomorals" / "tools" / "__init__.py", "")
    _write(root / "nomorals" / "tools" / "alpha.py", "def register(r): pass\n")
    _write(root / "nomorals" / "tools" / "beta.py", "def register(r): pass\n")
    _write(root / "nomorals" / "tools" / "gamma.py", "def register(r): pass\n")
    _write(root / "nomorals" / "tools" / "registry.py",
           "import importlib as _il\n"
           "def register_builtins(self):\n"
           "    for _name in ('alpha', 'beta'):\n"
           "        _il.import_module(f'.{_name}', __package__)\n")
    dormant = {d.module for d in dormant_modules(root)}
    assert "nomorals.tools.alpha" not in dormant
    assert "nomorals.tools.beta" not in dormant
    assert "nomorals.tools.gamma" in dormant  # not in the literal list
    assert "nomorals.tools.registry" in dormant  # no importer; entry-free


def test_synthetic_tree_pkgutil_directory_walk_counts(tmp_path):
    """The custom-tools pattern: iter_modules(pkg.__path__) loads all children."""
    root = tmp_path
    _write(root / "nomorals" / "__init__.py", "")
    _write(root / "nomorals" / "tools" / "__init__.py", "")
    _write(root / "nomorals" / "tools" / "custom" / "__init__.py", "")
    _write(root / "nomorals" / "tools" / "custom" / "ping.py",
           "def register(r): pass\n")
    _write(root / "nomorals" / "tools" / "registry.py",
           "import importlib\nimport pkgutil\n"
           "from . import custom as custom_pkg\n"
           "def load():\n"
           "    for info in pkgutil.iter_modules(custom_pkg.__path__):\n"
           "        importlib.import_module(f'{custom_pkg.__name__}.{info.name}')\n")
    dormant = {d.module for d in dormant_modules(root)}
    assert "nomorals.tools.custom.ping" not in dormant


def test_synthetic_tree_docstring_mention_is_not_wiring(tmp_path):
    root = _tree(tmp_path)
    _write(root / "nomorals" / "engines" / "wired.py",
           '"""Mentions nomorals.engines.dormant in prose only."""\n'
           "def run():\n    return 1\n")
    assert "nomorals.engines.dormant" in {d.module for d in dormant_modules(root)}


def test_real_repo_scan_shape():
    """Smoke test on the real repo: shape + one known-wired module."""
    result = scan()
    assert result.scanned > 500
    assert result.referenced_count > 0
    assert all(d.module.startswith("nomorals.") for d in result.dormant)
    assert all(d.reason for d in result.dormant)
    assert all(d.path for d in result.dormant)
    # The registry is an entry module and widely imported: never dormant.
    assert "nomorals.tools.registry" not in {d.module for d in result.dormant}
    # Round-trip: every dormant entry serializes.
    for entry in result.dormant:
        assert entry.to_dict()["module"] == entry.module
