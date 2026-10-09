"""Tests for scene intelligence + generative editing."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_segmentation_imports():
    from nomorals.media.scene_intel.segment import Segment, Segmentation
    s = Segment(index=0, start=0.0, end=5.0)
    assert s.kind == "scene"


def test_mlp_forward():
    from nomorals.media.highlight_model.model import HighlightMLP
    m = HighlightMLP()
    m._init_random(42)
    scores = m.score([[0.5] * 8, [0.9] * 8])
    assert len(scores) == 2
    assert all(0.0 <= v <= 1.0 for v in scores)
    # higher-energy input should score higher with random weights... not
    # guaranteed, just check determinism
    assert m.score([[0.5] * 8]) == m.score([[0.5] * 8])


def test_mlp_save_load(tmp_path):
    from nomorals.media.highlight_model.model import HighlightMLP
    m = HighlightMLP()
    m._init_random(1)
    p = str(tmp_path / "w.npz")
    m.save(p)
    m2 = HighlightMLP(p)
    assert m2.loaded
    assert m.score([[0.3] * 8]) == m2.score([[0.3] * 8])


def test_character_reel_empty():
    from nomorals.media.scene_intel.characters import character_reel
    assert character_reel({}, "char_0") == []


def test_film_sources_import():
    from nomorals.media.film_sources import (
        FILM_CHAIN, search_films, NetNaijaMoviesSource, NkiriSource,
        FzMoviesSource)
    assert len(FILM_CHAIN) == 3
    names = {s.name for s in FILM_CHAIN}
    assert names == {"netnaija-movies", "nkiri", "fzmovies"}


def test_film_search_offline():
    # no network in test env — must return [] honestly, never raise
    from nomorals.media.film_sources import search_films
    out = search_films("xyznonexistentmovie123", limit=2)
    assert isinstance(out, list)


def test_genedit_imports():
    from nomorals.media import genedit
    assert hasattr(genedit, "text_to_shot")
    assert hasattr(genedit, "image_to_shot")
    assert hasattr(genedit, "video_to_shot")
    assert hasattr(genedit, "extend_shot")
    assert hasattr(genedit, "shot_to_timeline")
    assert hasattr(genedit, "GenShot")


def test_sceneintel_tools_register():
    from nomorals.tools import sceneintel
    assert hasattr(sceneintel, "register")
    import ast
    tree = ast.parse(open(sceneintel.__file__).read())
    # count @registry.register(...) decorators
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call)
                        and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "register"):
                    n += 1
    assert n == 7, f"expected 7 tools, found {n}"
