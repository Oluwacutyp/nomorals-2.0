"""Animation editor — typed commands over the document, with snapshot
undo/redo (cap 50, like the wiki organ) and atomic edit contexts.

Every mutation goes through a command method; each command snapshots
the document first, so ``undo`` restores the exact prior bytes.
``edit()`` is an atomic context: any failure inside it restores the
document to its pre-edit bytes. This mirrors upstream's history
(``crates/engine/src/history.rs``) and the retime commands
(``crates/engine/src/cmds_retime.rs``).
"""

from __future__ import annotations

import contextlib
from typing import Any

from .easing import Easing
from .errors import AnimationError
from .model import AnimationDocument, Element, Keyframe, Layer

__all__ = ["AnimationEditor"]

UNDO_CAP = 50


class AnimationEditor:
    def __init__(self, doc: AnimationDocument) -> None:
        self.doc = doc
        self._undo: list[bytes] = []
        self._redo: list[bytes] = []
        self._clipboard: list[dict[str, Any]] = []
        self._in_edit = False

    # -- history -------------------------------------------------------
    def _snapshot(self) -> None:
        if self._in_edit:
            return
        self._undo.append(self.doc.to_bytes())
        if len(self._undo) > UNDO_CAP:
            self._undo.pop(0)
        self._redo.clear()

    def _restore(self, raw: bytes) -> None:
        """Restore the document in place, keeping its identity."""
        parsed = AnimationDocument.parse(raw)
        self.doc.__dict__.clear()
        self.doc.__dict__.update(parsed.__dict__)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self.doc.to_bytes())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self.doc.to_bytes())
        self._restore(self._redo.pop())
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    @contextlib.contextmanager
    def edit(self):
        """Atomic edit: any failure restores the pre-edit bytes."""
        before = self.doc.to_bytes()
        self._snapshot()
        self._in_edit = True
        try:
            yield self
        except Exception:
            self._restore(before)
            self._undo.pop()
            raise
        finally:
            self._in_edit = False

    # -- layers --------------------------------------------------------
    def add_layer(self, name: str, kind: str = "normal",
                  scene: str | None = None) -> Layer:
        self._snapshot()
        return self.doc.scene(scene).timeline.add_layer(name, kind=kind)

    # -- keyframes -----------------------------------------------------
    def add_keyframe(self, layer_id: int, start: int, duration: int,
                     label: str = "", scene: str | None = None) -> Keyframe:
        self._snapshot()
        layer = self._layer(layer_id, scene)
        return layer.add_keyframe(start, duration, label=label)

    def move_keyframe(self, layer_id: int, start: int, new_start: int,
                      scene: str | None = None) -> Keyframe:
        self._snapshot()
        layer = self._layer(layer_id, scene)
        kf = self._keyframe(layer, start)
        for other in layer.keyframes:
            if other is not kf and new_start < other.end() \
                    and new_start + kf.duration > other.start:
                raise AnimationError(
                    f"moving keyframe to {new_start} overlaps keyframe "
                    f"{other.start}..{other.end()} on layer "
                    f"{layer.name!r}")
        kf.start = int(new_start)
        layer.keyframes.sort(key=lambda k: k.start)
        return kf

    def retime_keyframe(self, layer_id: int, start: int, new_start: int,
                        scene: str | None = None) -> Keyframe:
        """Move a keyframe (and its span) to a new start frame."""
        return self.move_keyframe(layer_id, start, new_start, scene=scene)

    def trim_keyframe(self, layer_id: int, start: int, duration: int,
                      scene: str | None = None) -> Keyframe:
        self._snapshot()
        layer = self._layer(layer_id, scene)
        kf = self._keyframe(layer, start)
        if duration < 1:
            raise AnimationError("a keyframe's duration is at least 1")
        for other in layer.keyframes:
            if other is not kf and kf.start < other.end() \
                    and kf.start + duration > other.start:
                raise AnimationError(
                    f"trimming to {duration} frames overlaps keyframe "
                    f"{other.start}..{other.end()} on layer "
                    f"{layer.name!r}")
        kf.duration = int(duration)
        return kf

    def copy_keyframes(self, layer_id: int, starts: list[int],
                       scene: str | None = None) -> int:
        layer = self._layer(layer_id, scene)
        self._clipboard = [
            self._keyframe(layer, s).to_dict() for s in starts]
        return len(self._clipboard)

    def paste_keyframes(self, layer_id: int, at: int,
                        scene: str | None = None) -> list[Keyframe]:
        if not self._clipboard:
            raise AnimationError("nothing copied — copy keyframes first")
        self._snapshot()
        layer = self._layer(layer_id, scene)
        base = min(k["start"] for k in self._clipboard)
        out: list[Keyframe] = []
        for data in self._clipboard:
            kf = Keyframe.from_dict(data)
            kf.start = int(at) + (kf.start - base)
            # Pasted elements are new elements, not the originals.
            for el in kf.elements:
                el.id = Element("shape").id
                self._reid_children(el)
            layer.keyframes.append(kf)
            out.append(kf)
        layer.keyframes.sort(key=lambda k: k.start)
        for i in range(len(layer.keyframes) - 1):
            a, b = layer.keyframes[i], layer.keyframes[i + 1]
            if a.end() > b.start:
                raise AnimationError(
                    f"pasted keyframes overlap keyframe "
                    f"{b.start}..{b.end()} on layer {layer.name!r}")
        return out

    @staticmethod
    def _reid_children(el: Element) -> None:
        for child in el.children:
            child.id = Element("shape").id
            AnimationEditor._reid_children(child)

    def set_ease(self, layer_id: int, start: int, easing: Easing,
                 group: str | None = None,
                 scene: str | None = None) -> None:
        self._snapshot()
        layer = self._layer(layer_id, scene)
        kf = self._keyframe(layer, start)
        if kf.tween is None:
            from .model import Tween

            kf.tween = Tween(easing=easing)
        elif group is None:
            kf.tween.easing = easing
        else:
            kf.tween.groups[group] = easing

    # -- helpers -------------------------------------------------------
    def _layer(self, layer_id: int, scene: str | None) -> Layer:
        return self.doc.scene(scene).timeline.layer_by_id(layer_id)

    @staticmethod
    def _keyframe(layer: Layer, start: int) -> Keyframe:
        for kf in layer.keyframes:
            if kf.start == start:
                return kf
        raise AnimationError(
            f"no keyframe starting at {start} on layer {layer.name!r}")
