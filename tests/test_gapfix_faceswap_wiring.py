"""Gapfix 67 — the character face specialist is invoked.

``characters/specialists.py::swap_face`` was implemented and unit-pinned
but had ZERO production callers: no out-of-game character render path
existed for a face swap (``characters/provision.py`` named it a
remainder). This lane wires it the way lane 65 wired ``lora_for``:

* ``/charagent face <name> [ref|none]`` names / shows / clears the
  character's face reference, validated through the new
  ``specialists.face_for`` (mirroring ``lora_for`` — the reference must
  name a file inside the faces vault);
* ``/charagent swap <name> <target image>`` puts the character's face
  onto a target image through ``swap_face`` and delivers the swapped
  render into the chat; on any wall the target is delivered as-is
  (the ladder's ``character_face`` code rung) with the reason named.

Tests drive the real handler (``RuntimeGamesMixin._control_charagent``)
against a scratch character store (``NOMORALS_DATA_DIR``) and a scratch
faces vault (``HOME``), with ``swap_face`` mocked — no real model is
needed to prove the invocation.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.characters import Character, CharacterStore


def _suggest(prompt: str) -> str:
    first = (prompt or "").splitlines()[0]
    if first.startswith("You are Mara"):
        return "MARA-LINE"
    return "BRAIN-LINE"


def _handler():
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    class _RT(RuntimeGamesMixin):
        def __init__(self, ctx):
            self.context = ctx

    return _RT(SimpleNamespace(suggest=_suggest))


def _scratch_store():
    root = Path(tempfile.mkdtemp())
    os.environ["NOMORALS_DATA_DIR"] = str(root)
    store = CharacterStore()  # honors NOMORALS_DATA_DIR
    assert store.dir == root / "characters"
    return root, store


def _scratch_vault(monkeypatch, root):
    """A faces vault under a scratch HOME (face_for's default vault is
    ~/.nomorals/faces). Returns (vault_dir, ref_path)."""
    home = root / "home"
    vault = home / ".nomorals" / "faces"
    vault.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    ref = vault / "mara-face.png"
    ref.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return vault, ref


class _DeliverFace:
    """Fake _deliver_face_image: records the call, returns '' (landed)."""

    def __init__(self):
        self.calls = []

    def __call__(self, chat_key, path, caption):
        self.calls.append((chat_key, path, caption))
        return ""


def _make_char(store, **kw):
    c = Character(name="Mara", persona={"witty": 0.9}, **kw)
    store.save(c)
    return c


# ── face_for: the resolver (mirrors lora_for) ─────────────────────────


def test_face_for_validates_ref_inside_vault(monkeypatch, tmp_path):
    from nomorals.characters.specialists import face_for

    root = Path(tempfile.mkdtemp())
    _, ref = _scratch_vault(monkeypatch, root)
    c = Character(name="Mara", face_ref=str(ref))
    res = face_for(c)
    assert res["ok"] is True
    assert res["path"] == str(ref)


def test_face_for_accepts_vault_relative_name(monkeypatch, tmp_path):
    from nomorals.characters.specialists import face_for

    root = Path(tempfile.mkdtemp())
    _scratch_vault(monkeypatch, root)
    c = Character(name="Mara", face_ref="mara-face.png")
    res = face_for(c)
    assert res["ok"] is True
    assert res["path"].endswith("mara-face.png")


def test_face_for_no_ref_names_the_wall(monkeypatch, tmp_path):
    from nomorals.characters.specialists import face_for

    root = Path(tempfile.mkdtemp())
    _scratch_vault(monkeypatch, root)
    c = Character(name="Mara")
    res = face_for(c)
    assert res["ok"] is False
    assert "face" in res["reason"].lower()


def test_face_for_missing_file_names_the_wall(monkeypatch, tmp_path):
    from nomorals.characters.specialists import face_for

    root = Path(tempfile.mkdtemp())
    vault, _ = _scratch_vault(monkeypatch, root)
    c = Character(name="Mara", face_ref=str(vault / "ghost.png"))
    res = face_for(c)
    assert res["ok"] is False
    assert "ghost.png" in res["reason"]


def test_face_for_ref_outside_vault_is_refused(monkeypatch, tmp_path):
    from nomorals.characters.specialists import face_for

    root = Path(tempfile.mkdtemp())
    _scratch_vault(monkeypatch, root)
    outside = root / "elsewhere.png"
    outside.write_bytes(b"\x89PNG\r\n\x1a\n")
    c = Character(name="Mara", face_ref=str(outside))
    res = face_for(c)
    assert res["ok"] is False
    assert "vault" in res["reason"].lower()


# ── /charagent face: naming surface ───────────────────────────────────


def test_face_verb_sets_and_validates_ref(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store)
    h = _handler()
    out = h._control_charagent(f"face Mara {ref}", "dm:1")
    assert "validated" in out
    assert store.get_by_name("Mara").face_ref == str(ref)


def test_face_verb_show_validates(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    h = _handler()
    out = h._control_charagent("face Mara", "dm:1")
    assert "validated" in out
    assert "mara-face.png" in out


def test_face_verb_none_clears(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    h = _handler()
    out = h._control_charagent("face Mara none", "dm:1")
    assert "cleared" in out
    assert store.get_by_name("Mara").face_ref == ""


def test_face_verb_missing_ref_names_wall(monkeypatch):
    root, store = _scratch_store()
    vault, _ = _scratch_vault(monkeypatch, root)
    _make_char(store)
    h = _handler()
    out = h._control_charagent(f"face Mara {vault / 'ghost.png'}", "dm:1")
    assert "ghost.png" in out
    # the name is still recorded — the file may land later, like lora
    assert store.get_by_name("Mara").face_ref.endswith("ghost.png")


def test_face_verb_no_ref_says_so(monkeypatch):
    root, store = _scratch_store()
    _scratch_vault(monkeypatch, root)
    _make_char(store)
    h = _handler()
    out = h._control_charagent("face Mara", "dm:1")
    assert "no face" in out.lower()


def test_create_with_face_arg_records_it(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    h = _handler()
    out = h._control_charagent(f"create Mara face={ref}", "dm:1")
    assert "validated" in out
    assert store.get_by_name("Mara").face_ref == str(ref)


def test_list_shows_face(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    h = _handler()
    out = h._control_charagent("list", "dm:1")
    assert "face: mara-face.png" in out


# ── /charagent swap: the live call site ───────────────────────────────


def _target_image(root):
    target = root / "scene.png"
    target.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return target


def test_swap_delivers_swapped_image(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    target = _target_image(root)
    swapped = root / "scene.faceswapped.png"
    swapped.write_bytes(b"\x89PNG\r\n\x1a\n")
    calls = []

    def _fake_swap(char, target_path, **kw):
        calls.append((char.name, target_path))
        return {"ok": True, "path": str(swapped),
                "sidecar": str(swapped.with_suffix(".faceswap.json"))}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    h = _handler()
    deliver = _DeliverFace()
    h._deliver_face_image = deliver
    out = h._control_charagent(f"swap Mara {target}", "dm:1")
    assert calls == [("Mara", str(target))]
    assert deliver.calls
    assert deliver.calls[0][1] == str(swapped)
    assert out == ""  # the image IS the reply


def test_swap_failure_delivers_target_as_is(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    target = _target_image(root)

    def _fake_swap(char, target_path, **kw):
        return {"ok": False, "reason": "face swap failed: no faces found"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    h = _handler()
    deliver = _DeliverFace()
    h._deliver_face_image = deliver
    out = h._control_charagent(f"swap Mara {target}", "dm:1")
    # ladder character_face code rung: the target goes out as-is
    assert deliver.calls
    assert deliver.calls[0][1] == str(target)
    assert "no faces found" in out


def test_swap_with_no_face_names_wall_and_delivers_target(monkeypatch):
    root, store = _scratch_store()
    _scratch_vault(monkeypatch, root)
    _make_char(store)
    target = _target_image(root)
    calls = []

    def _fake_swap(char, target_path, **kw):
        calls.append(target_path)
        return {"ok": False,
                "reason": "no face reference for this character"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    h = _handler()
    deliver = _DeliverFace()
    h._deliver_face_image = deliver
    out = h._control_charagent(f"swap Mara {target}", "dm:1")
    # the specialist owns the wall (one path, no handler short-circuit)
    assert calls == [str(target)]
    assert deliver.calls
    assert deliver.calls[0][1] == str(target)
    assert "face" in out.lower()


def test_swap_without_delivery_names_the_paths(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    target = _target_image(root)
    swapped = root / "scene.faceswapped.png"
    swapped.write_bytes(b"\x89PNG\r\n\x1a\n")

    def _fake_swap(char, target_path, **kw):
        return {"ok": True, "path": str(swapped), "sidecar": ""}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    h = _handler()
    # no fake delivery attached and no gateway on the harness context:
    # the real _deliver_face_image names the path instead of dropping it
    out = h._control_charagent(f"swap Mara {target}", "dm:1")
    assert str(swapped) in out


def test_swap_missing_target_names_wall(monkeypatch):
    root, store = _scratch_store()
    _, ref = _scratch_vault(monkeypatch, root)
    _make_char(store, face_ref=str(ref))
    h = _handler()
    out = h._control_charagent(f"swap Mara {root / 'nope.png'}", "dm:1")
    assert "nope.png" in out
    assert "not found" in out.lower() or "no such" in out.lower()


def test_swap_unknown_character(monkeypatch):
    root, store = _scratch_store()
    _scratch_vault(monkeypatch, root)
    target = _target_image(root)
    h = _handler()
    out = h._control_charagent(f"swap Ghost {target}", "dm:1")
    assert "no character" in out.lower()


def test_swap_usage_without_args(monkeypatch):
    _scratch_store()
    h = _handler()
    out = h._control_charagent("swap Mara", "dm:1")
    assert out.startswith("usage:")


def test_face_usage_without_name(monkeypatch):
    _scratch_store()
    h = _handler()
    out = h._control_charagent("face", "dm:1")
    assert out.startswith("usage:")


# ── _character_face_image helper ───────────────────────────────────────


def test_character_face_image_helper_success(monkeypatch, tmp_path):
    from nomorals.agents.partner.runtime_games import (
        _character_face_image,
    )

    def _fake_swap(char, target_path, **kw):
        return {"ok": True, "path": "/out/swapped.png",
                "sidecar": "/out/swapped.faceswap.json"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    char = Character(name="Mara", face_ref="mara-face.png")
    path, sidecar, reason = _character_face_image(char, "/t/scene.png")
    assert path == "/out/swapped.png"
    assert sidecar == "/out/swapped.faceswap.json"
    assert reason == ""


def test_character_face_image_helper_failure_names_wall(monkeypatch):
    from nomorals.agents.partner.runtime_games import (
        _character_face_image,
    )

    def _fake_swap(char, target_path, **kw):
        return {"ok": False, "reason": "face swap failed: boom"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    char = Character(name="Mara", face_ref="mara-face.png")
    path, sidecar, reason = _character_face_image(char, "/t/scene.png")
    assert path == ""
    assert sidecar == ""
    assert "boom" in reason


def test_character_face_image_helper_never_raises(monkeypatch):
    from nomorals.agents.partner.runtime_games import (
        _character_face_image,
    )

    def _fake_swap(char, target_path, **kw):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_face", _fake_swap)
    char = Character(name="Mara", face_ref="mara-face.png")
    path, sidecar, reason = _character_face_image(char, "/t/scene.png")
    assert path == ""
    assert "kaboom" in reason
