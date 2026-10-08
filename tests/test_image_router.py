"""Tests for the license-aware image model router (build-map #22).

All offline: backends are mocked at the router boundary (comfy_available
probe), no network, no GPU.
"""

import pytest

from nomorals.media_edit.models import (
    IMAGE_MODELS,
    ImageModelError,
    ImageModelRouter,
    Route,
)


@pytest.fixture()
def router():
    return ImageModelRouter()


@pytest.fixture()
def comfy_on(monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.models.get_profile_kind", lambda: "workstation")
    monkeypatch.setattr(
        "nomorals.media_edit.comfy.comfy_available", lambda: (True, ""))


@pytest.fixture()
def comfy_off(monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.comfy.comfy_available",
        lambda: (False, "no server"))


# -- registry honesty -------------------------------------------------------

def test_registry_has_required_fields():
    required = {"license", "public_ok", "vram_gb", "speed", "quality",
                "backend", "workflow", "checkpoint", "hf_id",
                "capabilities"}
    for name, m in IMAGE_MODELS.items():
        assert required <= set(m), f"{name} missing fields"
        assert m["speed"] in ("fast", "medium", "slow"), name
        assert 1 <= m["quality"] <= 5, name


def test_flux_dev_is_private_only():
    assert not IMAGE_MODELS["flux2-dev"]["public_ok"]
    assert not IMAGE_MODELS["flux1-kontext-dev"]["public_ok"]
    assert "non-commercial" in IMAGE_MODELS["flux2-dev"]["license"].lower() or \
        "Non-Commercial" in IMAGE_MODELS["flux2-dev"]["license"]


def test_open_models_are_public_ok():
    for name in ("qwen-image-2512", "qwen-image-edit", "z-image-turbo",
                 "flux1-schnell", "sdxl-base"):
        assert IMAGE_MODELS[name]["public_ok"], name


# -- license enforcement is structural ---------------------------------------

def test_public_audience_excludes_flux_dev(router, comfy_on):
    route = router.route("generate", audience="public", profile="workstation")
    assert route.model != "flux2-dev"
    assert IMAGE_MODELS[route.model]["public_ok"]


def test_public_audience_excludes_kontext_edit(router):
    route = router.route("edit", audience="public", profile="termux")
    assert route.model != "flux1-kontext-dev"
    assert IMAGE_MODELS[route.model]["public_ok"]


def test_public_list_models_hides_noncommercial(router):
    names = {m["name"] for m in router.list_models(audience="public")}
    assert "flux2-dev" not in names
    assert "flux1-kontext-dev" not in names
    assert "qwen-image-2512" in names


def test_private_list_models_shows_all(router):
    names = {m["name"] for m in router.list_models(audience="private")}
    assert "flux2-dev" in names
    assert "flux1-kontext-dev" in names


# -- routing -----------------------------------------------------------------

def test_private_gets_flux2_for_quality(router, comfy_on):
    route = router.route("generate", audience="private",
                         profile="workstation")
    assert route.model == "flux2-dev"
    assert route.backend == "comfy"
    assert route.workflow == "txt2img"


def test_draft_picks_z_image_turbo(router, comfy_on):
    route = router.route("draft", audience="private", profile="workstation")
    assert route.model == "z-image-turbo"
    assert route.steps == 8


def test_termux_is_hf_only(router, monkeypatch):
    # even if comfy were reachable, termux must not pick it
    monkeypatch.setattr(
        "nomorals.media_edit.comfy.comfy_available", lambda: (True, ""))
    route = router.route("generate", audience="private", profile="termux")
    assert route.backend == "hf"
    assert route.model == "flux1-schnell"


def test_termux_edit_uses_hf_model(router):
    route = router.route("edit", audience="private", profile="termux")
    assert route.backend == "hf"
    assert route.model == "flux1-kontext-dev"


def test_edit_intent_picks_edit_capable(router, comfy_on):
    route = router.route("edit", audience="public", profile="workstation")
    assert route.model == "qwen-image-edit"
    assert route.backend == "hf"


def test_laptop_falls_back_to_hf_without_comfy(router, comfy_off,
                                               monkeypatch):
    monkeypatch.setattr(
        "nomorals.media_edit.models.get_profile_kind", lambda: "laptop")
    route = router.route("generate", audience="private", profile="laptop")
    assert route.backend == "hf"


def test_upscale_local_when_no_comfy(router, comfy_off):
    route = router.route("upscale", audience="public", profile="laptop")
    assert route.backend == "local"
    assert route.model == "pil-lanczos"


def test_upscale_uses_comfy_when_available(router, comfy_on):
    route = router.route("upscale", audience="private",
                         profile="workstation")
    assert route.backend == "comfy"
    assert route.workflow == "upscale"


def test_reason_strings_non_empty(router, comfy_on):
    for intent in ("generate", "edit", "draft", "upscale"):
        route = router.route(intent, audience="private",
                             profile="workstation")
        assert route.reason and len(route.reason) > 20
        assert route.model in route.reason or intent in route.reason


def test_unknown_intent_raises(router):
    with pytest.raises(ImageModelError):
        router.route("teleport", audience="private")


def test_unknown_audience_raises(router):
    with pytest.raises(ImageModelError):
        router.route("generate", audience="everyone")


def test_route_is_frozen_dataclass(router, comfy_on):
    route = router.route("generate", audience="private",
                         profile="workstation")
    assert isinstance(route, Route)
    with pytest.raises(Exception):
        route.model = "x"  # frozen


# -- NL intent regexes -------------------------------------------------------

def test_image_intent_generate():
    from nomorals.agents.coremind import _image_intent
    it = _image_intent("generate an image of a cat astronaut")
    assert it is not None and it.kind == "image_gen"
    assert it.action == "generate"
    assert "cat astronaut" in it.target


def test_image_intent_create():
    from nomorals.agents.coremind import _image_intent
    it = _image_intent("create an image of Lagos at night")
    assert it is not None and it.kind == "image_gen"


def test_image_intent_draw():
    from nomorals.agents.coremind import _image_intent
    it = _image_intent("draw a red dragon")
    assert it is not None and it.kind == "image_gen"
    assert "red dragon" in it.target


def test_image_intent_draw_does_not_misfire():
    from nomorals.agents.coremind import _image_intent
    assert _image_intent("draw a conclusion from the data") is None
    assert _image_intent("draw the curtains") is None
    assert _image_intent("I like to draw") is None


def test_image_intent_edit():
    from nomorals.agents.coremind import _image_intent
    it = _image_intent("edit this image: make it sunset")
    assert it is not None and it.kind == "image_edit"
    assert it.action == "edit"
    assert "sunset" in it.target


def test_image_intent_no_misfire_on_chat():
    from nomorals.agents.coremind import _image_intent
    assert _image_intent("what did the vendor say about pricing?") is None
    assert _image_intent("generate some excitement") is None
    assert _image_intent("images are nice") is None
    assert _image_intent("") is None


def test_image_intent_registered_in_understand():
    from nomorals.agents.coremind import understand
    cands = understand("generate an image of a mountain")
    kinds = [c.kind for c in cands]
    assert "image_gen" in kinds
