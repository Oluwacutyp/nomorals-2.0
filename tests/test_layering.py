"""Architecture enforcement.

ARCHITECTURE.md §1 promises strict layering. A promise nobody checks is a
suggestion. This test parses the AST of every module in the package and fails if
any module imports from a *higher* layer than its own.

Run it after every change; it is what stops a 100k-line codebase from becoming a
cycle of mutual imports.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "nomorals"

#: layer index by top-level subpackage. Lower may not import higher.
LAYERS: dict[str, int] = {
    "core": 1,
    "storage": 2,
    "memory": 3,
    "llm": 3,
    "training": 3,
    "tools": 4,
    "social": 4,
    "agents": 5,
    "missions": 6,
    "api": 7,
}

#: Modules that sit outside the layer stack and may be imported by anything.
SHARED = {"version", "compat"}
SHARED_LAYER = 0

#: Top-level entry points live at the highest layer.
TOP_LEVEL_LAYER = {"cli": 7, "__main__": 7}


def module_layer(rel_path: Path) -> int:
    parts = rel_path.with_suffix("").parts
    if len(parts) == 1:
        stem = parts[0]
        if stem in TOP_LEVEL_LAYER:
            return TOP_LEVEL_LAYER[stem]
        return SHARED_LAYER  # __init__.py, version.py, compat.py
    head = parts[0]
    return LAYERS.get(head, SHARED_LAYER)


def iter_modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py"))


def imported_nomorals(tree: ast.AST, module_parts: tuple[str, ...]) -> list[tuple[str, int]]:
    """Yield (imported dotted path, source line) for every nomorals import."""
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("nomorals"):
                    found.append((alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Relative import: resolve against the importing module's package.
                base = module_parts[: len(module_parts) - 1]
                if node.level > 1:
                    base = base[: -(node.level - 1)]
                target = ".".join([*base, node.module] if node.module else base)
            elif node.module and node.module.startswith("nomorals"):
                target = node.module
            else:
                continue
            if target.startswith("nomorals"):
                found.append((target, node.lineno))
    return found


def layer_of_dotted(dotted: str) -> int:
    parts = dotted.split(".")
    if len(parts) < 2:
        return SHARED_LAYER
    head = parts[1]
    if head in SHARED:
        return SHARED_LAYER
    return LAYERS.get(head, SHARED_LAYER)


class TestLayering(unittest.TestCase):
    def test_every_module_parses(self) -> None:
        modules = iter_modules()
        self.assertGreater(len(modules), 10, "package appears to be missing modules")
        for path in modules:
            with self.subTest(module=str(path.relative_to(PACKAGE_ROOT))):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_upward_imports(self) -> None:
        violations: list[str] = []
        for path in iter_modules():
            rel = path.relative_to(PACKAGE_ROOT)
            own_layer = module_layer(rel)
            module_parts = tuple(rel.with_suffix("").parts)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for target, lineno in imported_nomorals(tree, module_parts):
                target_layer = layer_of_dotted(target)
                if target_layer > own_layer:
                    violations.append(
                        f"{rel}:{lineno} (L{own_layer}) imports {target} (L{target_layer})"
                    )
        self.assertEqual(violations, [], "layer violations:\n  " + "\n  ".join(violations))

    def test_core_has_no_intra_project_imports(self) -> None:
        """L1 is the kernel: it may not depend on any other subpackage."""
        offenders: list[str] = []
        for path in (PACKAGE_ROOT / "core").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            module_parts = tuple(path.relative_to(PACKAGE_ROOT).with_suffix("").parts)
            for target, lineno in imported_nomorals(tree, module_parts):
                if not target.startswith("nomorals.core") and target != "nomorals":
                    offenders.append(f"{path.name}:{lineno} -> {target}")
        self.assertEqual(offenders, [], "core imports outside core:\n  " + "\n  ".join(offenders))

    def test_public_api_is_declared(self) -> None:
        """Every module that defines __all__ must actually export those names."""
        for path in iter_modules():
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            declared: list[str] | None = None
            for node in tree.body:
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                declared = [
                                    e.value
                                    for e in node.value.elts
                                    if isinstance(e, ast.Constant) and isinstance(e.value, str)
                                ]
            if declared is None:
                continue
            with self.subTest(module=path.name):
                for name in declared:
                    self.assertIn(
                        name,
                        source,
                        f"{path.name}: __all__ declares {name!r} which is not defined",
                    )


if __name__ == "__main__":
    unittest.main()
