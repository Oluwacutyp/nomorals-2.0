"""Pass A1 tests: SD1.5 wired into Devon's image studio.

Covers the imggen registry (header-only classification), the real
CLIP BPE tokenizer, the completed LDM key mappings in sdcompat
(UNet/CLIP/VAE — completeness proven on meta-device modules so the
full key sets are checked without materializing ~4GB of weights,
plus REAL fp16 weight round-trips for the VAE and CLIP), the Studio
dispatch + adapter honesty gates, the media_edit NativeBackend
readiness, the directed/filler loader reroute, the /imggen
checkpoints chat surface, and the sd15 license label.

The one proof this environment cannot run — a full-size
from_safetensors → generate round-trip (the load peaks at ~9GB
co-resident; the test VM has 7.9GB total) — is present and gated on
available memory (~10.5GB) with the numbers in the skip reason, so
it runs on any machine that can actually hold the model.
"""

from __future__ import annotations

import json
import struct

import pytest

from nomorals.media.imggen import ImgGenError

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None

needs_torch = pytest.mark.skipif(torch is None,
                                 reason="torch not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_safetensors(path, keys):
    """A structurally valid safetensors file with the given keys.

    Pure-python writer (header JSON + zero payload): the registry
    only reads the header, and classification fixtures need keys,
    not weights.
    """
    header = {}
    for i, k in enumerate(keys):
        header[k] = {"dtype": "F32", "shape": [1],
                     "data_offsets": [i * 4, i * 4 + 4]}
    hj = json.dumps(header).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        f.write(b"\0" * (4 * len(keys)))


SD15_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.time_embed.0.weight",
    "cond_stage_model.transformer.token_embedding.weight",
    "first_stage_model.encoder.conv_in.weight",
]
SDXL_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "conditioner.embedders.0.transformer.resblocks.0.ln_1.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
    "first_stage_model.encoder.conv_in.weight",
]
FLUX_KEYS = [
    "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
    "model.diffusion_model.single_blocks.0.linear1.weight",
]
FLUX_DIFFUSERS_KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
]
DIFFUSERS_SD15_KEYS = [
    "unet.down_blocks.0.resnets.0.conv1.weight",
    "text_encoder.text_model.embeddings.token_embedding.weight",
    "vae.encoder.conv_in.weight",
]
UNKNOWN_KEYS = ["some.random.weight", "another.tensor"]


@pytest.fixture()
def model_roots(tmp_path, monkeypatch):
    """Isolated DEVON_IMGGEN_DIR + HOME (so ~/.devon-models is tmp)."""
    ckpt = tmp_path / "imggen"
    home = tmp_path / "home"
    ckpt.mkdir()
    home.mkdir()
    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(ckpt))
    monkeypatch.setenv("HOME", str(home))
    return ckpt, home


def _mem_available_kb():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def needs_mem(kb, why):
    avail = _mem_available_kb()
    return pytest.mark.skipif(
        avail is not None and avail < kb,
        reason=f"needs ~{kb // 1024}MB available RAM ({why}); "
               f"this machine has ~{(avail or 0) // 1024}MB")


# ---------------------------------------------------------------------------
# Registry: classification, loadable flags, resolve
# ---------------------------------------------------------------------------

def test_registry_classifies_sd15_ldm(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "v1-5-pruned.safetensors", SD15_KEYS)
    entries = list_models()
    assert len(entries) == 1
    e = entries[0]
    assert e["name"] == "v1-5-pruned"
    assert e["model_class"] == "sd15"
    assert e["format"] == "ldm-safetensors"
    assert e["loadable"] is True
    assert e["loadable_reason"] == ""
    assert e["bytes"] > 0


def test_registry_classifies_sdxl_listed_not_loadable(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "sdxl-base.safetensors", SDXL_KEYS)
    (e,) = list_models()
    assert e["model_class"] == "sdxl"
    assert e["loadable"] is False
    assert "SDXL" in e["loadable_reason"]


def test_registry_classifies_flux_variants(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "flux-bfl.safetensors", FLUX_KEYS)
    _write_safetensors(ckpt / "flux-diffusers.safetensors",
                       FLUX_DIFFUSERS_KEYS)
    by_name = {e["name"]: e for e in list_models()}
    assert by_name["flux-bfl"]["model_class"] == "flux"
    assert by_name["flux-bfl"]["loadable"] is False
    assert "FLUX" in by_name["flux-bfl"]["loadable_reason"]
    assert by_name["flux-diffusers"]["model_class"] == "flux"
    assert by_name["flux-diffusers"]["format"] == "diffusers-safetensors"


def test_registry_diffusers_sd15_is_sd15_but_not_loadable(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "sd-diffusers.safetensors",
                       DIFFUSERS_SD15_KEYS)
    (e,) = list_models()
    assert e["model_class"] == "sd15"
    assert e["format"] == "diffusers-safetensors"
    assert e["loadable"] is False
    assert "diffusers" in e["loadable_reason"]


def test_registry_unknown_and_native(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "mystery.safetensors", UNKNOWN_KEYS)
    run = ckpt / "devon-tiny"
    run.mkdir()
    (run / "ckpt_000010.pt").write_bytes(b"payload")
    entries = list_models()
    by_class = {e["model_class"]: e for e in entries}
    assert by_class["unknown"]["loadable"] is False
    assert by_class["unknown"]["loadable_reason"]
    native = by_class["native"]
    assert native["loadable"] is True
    assert native["format"] == "devon-imggen-1"
    assert native["run"] == "devon-tiny"
    # Loadable entries sort first.
    assert entries[0]["loadable"] is True


def test_registry_scans_devon_models_root(model_roots):
    from nomorals.media.imggen.registry import list_models

    _ckpt, home = model_roots
    _write_safetensors(
        home / ".devon-models" / "sd15" / "model.safetensors", SD15_KEYS)
    (e,) = list_models()
    assert e["model_class"] == "sd15" and e["loadable"] is True


def test_registry_resolve_by_name_case_insensitive(model_roots):
    from nomorals.media.imggen.registry import resolve

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "MyModel.safetensors", SD15_KEYS)
    e = resolve("mymodel")
    assert e is not None and e["model_class"] == "sd15"
    assert resolve("does-not-exist") is None
    assert resolve("") is None


def test_registry_resolve_by_path_outside_roots(model_roots, tmp_path):
    from nomorals.media.imggen.registry import resolve

    elsewhere = tmp_path / "elsewhere" / "xl.safetensors"
    _write_safetensors(elsewhere, SDXL_KEYS)
    e = resolve(str(elsewhere))
    assert e is not None
    assert e["model_class"] == "sdxl" and e["loadable"] is False


def test_registry_resolve_bad_header(model_roots, tmp_path):
    from nomorals.media.imggen.registry import resolve

    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(b"not a safetensors file at all")
    e = resolve(str(bad))
    assert e is not None
    assert e["model_class"] == "unknown" and e["loadable"] is False
    assert "header" in e["loadable_reason"]


# ---------------------------------------------------------------------------
# CLIP BPE tokenizer
# ---------------------------------------------------------------------------

def _tiny_vocab_and_merges():
    """A minimal but real CLIP vocab/merges pair.

    Byte-alphabet tokens at ids 0..255, their </w> variants at
    300.., specials at the canonical ids, and three hand-built
    merges: "the" and " cat" must each collapse to a single piece.
    """
    from nomorals.media.imggen.tokenizers import bytes_to_unicode

    b2u = bytes_to_unicode()
    chars = [b2u[b] for b in range(256)]
    vocab = {c: i for i, c in enumerate(chars)}
    for i, c in enumerate(chars):
        vocab[c + "</w>"] = 300 + i
    vocab["<|startoftext|>"] = 49406
    vocab["<|endoftext|>"] = 49407
    vocab["th"] = 600
    vocab["the</w>"] = 601
    vocab["Ġc"] = 602
    vocab["Ġca"] = 603
    vocab["Ġcat</w>"] = 604
    merges = ["t h", "h e</w>", "th e</w>",
              "Ġ c", "Ġc a", "Ġca t</w>"]
    return vocab, merges


def _write_tokenizer_dir(d, vocab, merges, as_json=False):
    d.mkdir(parents=True, exist_ok=True)
    if as_json:
        (d / "tokenizer.json").write_text(json.dumps(
            {"model": {"vocab": vocab,
                       "merges": merges}}), encoding="utf-8")
    else:
        (d / "vocab.json").write_text(json.dumps(vocab),
                                      encoding="utf-8")
        (d / "merges.txt").write_text(
            "#version: 0.2\n" + "\n".join(merges) + "\n",
            encoding="utf-8")


def test_bpe_bytes_to_unicode_spot_checks():
    from nomorals.media.imggen.tokenizers import bytes_to_unicode

    b2u = bytes_to_unicode()
    assert len(b2u) == 256
    assert b2u[ord("a")] == "a"
    assert b2u[0x20] == "Ġ"          # space → Ġ, as in real CLIP
    assert len(set(b2u.values())) == 256  # bijective


def test_bpe_merges_collapse_known_words(tmp_path):
    from nomorals.media.imggen.tokenizers import CLIPBPETokenizer

    vocab, merges = _tiny_vocab_and_merges()
    _write_tokenizer_dir(tmp_path, vocab, merges)
    tok = CLIPBPETokenizer.from_files(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    ids = tok.encode("the cat")
    assert ids[0] == 49406                      # SOT
    assert ids[1] == vocab["the</w>"]           # merged to one piece
    assert ids[2] == vocab["Ġcat</w>"]          # merged to one piece
    assert ids[3] == 49407                      # EOT
    assert len(ids) == 77                       # padded context
    assert all(i == 49407 for i in ids[3:])     # EOT padding


def test_bpe_byte_fallback_for_unseen_chars(tmp_path):
    from nomorals.media.imggen.tokenizers import CLIPBPETokenizer

    vocab, merges = _tiny_vocab_and_merges()
    _write_tokenizer_dir(tmp_path, vocab, merges)
    tok = CLIPBPETokenizer.from_files(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    # "é" is U+00E9 → UTF-8 bytes C3 A9 → byte chars Ã, ©.
    ids = tok.encode("é")
    assert ids[1] == vocab["Ã"]
    assert ids[2] == vocab["©</w>"]
    assert ids[3] == 49407


def test_bpe_truncates_to_context(tmp_path):
    from nomorals.media.imggen.tokenizers import CLIPBPETokenizer

    vocab, merges = _tiny_vocab_and_merges()
    _write_tokenizer_dir(tmp_path, vocab, merges)
    tok = CLIPBPETokenizer.from_files(
        str(tmp_path / "vocab.json"), str(tmp_path / "merges.txt"))
    ids = tok.encode(" ".join(["the"] * 200))
    assert len(ids) == 77
    assert ids[0] == 49406 and ids[-1] == 49407


def test_tokenizer_discovery_orders_and_formats(model_roots, tmp_path):
    from nomorals.media.imggen.tokenizers import load_clip_tokenizer

    vocab, merges = _tiny_vocab_and_merges()
    ckpt_dir = tmp_path / "ck"
    ckpt_file = ckpt_dir / "model.safetensors"
    _write_safetensors(ckpt_file, SD15_KEYS)
    # tokenizer.json in the checkpoint's own dir wins.
    _write_tokenizer_dir(ckpt_dir, vocab, merges, as_json=True)
    tok = load_clip_tokenizer(str(ckpt_file))
    assert tok is not None
    assert tok.encode("the")[1] == vocab["the</w>"]
    # Classic pair in $DEVON_IMGGEN_DIR/tokenizer is found for a
    # checkpoint with nothing nearby.
    bare = tmp_path / "bare" / "m.safetensors"
    _write_safetensors(bare, SD15_KEYS)
    ckpt_root, _home = model_roots
    _write_tokenizer_dir(ckpt_root / "tokenizer", vocab, merges)
    tok2 = load_clip_tokenizer(str(bare))
    assert tok2 is not None


def test_tokenizer_absent_returns_none(model_roots, tmp_path):
    from nomorals.media.imggen.tokenizers import load_clip_tokenizer

    f = tmp_path / "m.safetensors"
    _write_safetensors(f, SD15_KEYS)
    assert load_clip_tokenizer(str(f)) is None


def test_tokenizer_partial_pair_is_hard_error(model_roots):
    from nomorals.media.imggen.tokenizers import load_clip_tokenizer

    ckpt_root, _home = model_roots
    d = ckpt_root / "tokenizer"
    d.mkdir()
    (d / "vocab.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ImgGenError, match="incomplete"):
        load_clip_tokenizer("")


# ---------------------------------------------------------------------------
# License label
# ---------------------------------------------------------------------------

def test_sd15_license_label_registered():
    from nomorals.media.model_license import MODEL_LICENSES, can_ship

    e = MODEL_LICENSES["sd15"]
    assert e.license == "CreativeML Open RAIL-M"
    assert e.commercial is True
    info = can_ship("sd15")
    assert info["commercial_use"] is True
    assert info["personal_use"] is True


# ---------------------------------------------------------------------------
# LDM key mapping completeness (meta device: full key sets, no weight RAM)
#
# The inverse mappers below are written in the test, from the LDM
# layout, independently of the production forward mappers — if the
# two disagree about even one key, the assertions fail.
# ---------------------------------------------------------------------------

def _resnet_ldm(ours_prefix, ldm_prefix, sd, out):
    subs = [("temb_proj.1.", "emb_layers.1."),
            ("norm1.", "in_layers.0."), ("conv1.", "in_layers.2."),
            ("norm2.", "out_layers.0."), ("conv2.", "out_layers.3."),
            ("shortcut.", "skip_connection.")]
    for k, v in sd.items():
        if not k.startswith(ours_prefix):
            continue
        tail = k[len(ours_prefix):]
        for src, dst in subs:
            if tail.startswith(src):
                out[ldm_prefix + dst + tail[len(src):]] = v
                break
        else:  # pragma: no cover - guards the inverse itself
            raise AssertionError(f"unmapped resnet key {k}")


def _transformer_ldm(ours_prefix, ldm_prefix, sd, out):
    for k, v in sd.items():
        if not k.startswith(ours_prefix):
            continue
        tail = k[len(ours_prefix):]
        if tail.startswith("blocks."):
            # blocks.{i}.<rest> → transformer_blocks.{i}.<rest>
            out[f"{ldm_prefix}transformer_{tail}"] = v
        else:  # norm / proj_in / proj_out keep their names
            out[ldm_prefix + tail] = v


def _unet_to_ldm(sd):
    """SDUNet state dict → compvis LDM layout (independent inverse)."""
    out = {}
    p = "model.diffusion_model."
    for k, v in sd.items():
        if k.startswith("time_embed."):
            out[p + k] = v
        elif k.startswith("conv_in."):
            out[p + "input_blocks.0.0." + k[len("conv_in."):]] = v
        elif k.startswith("conv_norm_out."):
            out[p + "out.0." + k[len("conv_norm_out."):]] = v
        elif k.startswith("conv_out."):
            out[p + "out.2." + k[len("conv_out."):]] = v
    down_ldm = {"down0": (1, 2, 3), "down1": (4, 5, 6),
                "down2": (7, 8, 9), "down3": (10, 11, None)}
    for name, (b0, b1, ds) in down_ldm.items():
        for li, bi in enumerate((b0, b1)):
            _resnet_ldm(f"{name}.resnets.{li}.",
                        f"{p}input_blocks.{bi}.0.", sd, out)
            _transformer_ldm(f"{name}.transformers.{li}.",
                             f"{p}input_blocks.{bi}.1.", sd, out)
        if ds is not None:
            for k, v in sd.items():
                if k.startswith(f"{name}.downsampler."):
                    out[f"{p}input_blocks.{ds}.0."
                        + k[len(f"{name}.downsampler."):]] = v
    _resnet_ldm("mid_res1.", f"{p}middle_block.0.", sd, out)
    _transformer_ldm("mid_trans.", f"{p}middle_block.1.", sd, out)
    _resnet_ldm("mid_res2.", f"{p}middle_block.2.", sd, out)
    up_ldm = {"up0": (0, 1, 2), "up1": (3, 4, 5),
              "up2": (6, 7, 8), "up3": (9, 10, 11)}
    for name, blocks in up_ldm.items():
        for li, bi in enumerate(blocks):
            _resnet_ldm(f"{name}.resnets.{li}.",
                        f"{p}output_blocks.{bi}.0.", sd, out)
            _transformer_ldm(f"{name}.transformers.{li}.",
                             f"{p}output_blocks.{bi}.1.", sd, out)
        # The group's upsample conv lives on its last LDM block.
        for k, v in sd.items():
            if k.startswith(f"{name}.upsampler."):
                out[f"{p}output_blocks.{blocks[-1]}.2.conv."
                    + k[len(f"{name}.upsampler."):]] = v
    return out


def _clip_to_ldm(sd):
    """CLIPTextEncoder state dict → cond_stage_model layout."""
    out = {}
    p = "cond_stage_model.transformer."
    for k, v in sd.items():
        if k == "token_embedding.weight":
            out[p + "token_embedding.weight"] = v
        elif k == "position_embedding.weight":
            out[p + "positional_embedding"] = v
        elif k.startswith("final_layer_norm."):
            out[p + "ln_final." + k[len("final_layer_norm."):]] = v
    for i in range(12):
        op, lp = f"layers.{i}.", f"{p}resblocks.{i}."
        for k, v in sd.items():
            if not k.startswith(op):
                continue
            tail = k[len(op):]
            if tail.startswith("layer_norm1."):
                out[lp + "ln_1." + tail[len("layer_norm1."):]] = v
            elif tail.startswith("layer_norm2."):
                out[lp + "ln_2." + tail[len("layer_norm2."):]] = v
            elif tail.startswith("mlp.0."):
                out[lp + "mlp.c_fc." + tail[len("mlp.0."):]] = v
            elif tail.startswith("mlp.2."):
                out[lp + "mlp.c_proj." + tail[len("mlp.2."):]] = v
            elif tail.startswith("self_attn.out_proj."):
                out[lp + "attn.out_proj."
                    + tail[len("self_attn.out_proj."):]] = v
        q = sd[f"{op}self_attn.q_proj.weight"]
        kk = sd[f"{op}self_attn.k_proj.weight"]
        vv = sd[f"{op}self_attn.v_proj.weight"]
        out[lp + "attn.in_proj_weight"] = torch.cat([q, kk, vv], dim=0)
        qb = sd[f"{op}self_attn.q_proj.bias"]
        kb = sd[f"{op}self_attn.k_proj.bias"]
        vb = sd[f"{op}self_attn.v_proj.bias"]
        out[lp + "attn.in_proj_bias"] = torch.cat([qb, kb, vb], dim=0)
    return out


def _vae_resnet_ldm(ours_prefix, ldm_prefix, sd, out):
    for k, v in sd.items():
        if not k.startswith(ours_prefix):
            continue
        tail = k[len(ours_prefix):]
        if tail.startswith("shortcut."):
            out[ldm_prefix + "nin_shortcut."
                + tail[len("shortcut."):]] = v
        else:
            out[ldm_prefix + tail] = v


def _vae_to_ldm(sd):
    """AutoencoderKL state dict → first_stage_model layout."""
    out = {}
    p = "first_stage_model."
    for k, v in sd.items():
        if k.startswith("enc_in."):
            out[p + "encoder.conv_in." + k[len("enc_in."):]] = v
        elif k.startswith("quant_conv."):
            out[p + k] = v
        elif k.startswith("post_quant."):
            out[p + "post_quant_conv." + k[len("post_quant."):]] = v
    for i in range(4):
        ours, ldm = f"enc_b{i + 1}.", f"{p}encoder.down.{i}."
        _vae_resnet_ldm(ours + "0.", ldm + "block.0.", sd, out)
        _vae_resnet_ldm(ours + "1.", ldm + "block.1.", sd, out)
        for k, v in sd.items():
            if k.startswith(ours + "2."):
                out[ldm + "downsample.conv." + k[len(ours + "2."):]] = v
    for ours, ldm in (("enc_mid.", f"{p}encoder.mid."),
                      ("dec_mid.", f"{p}decoder.mid.")):
        _vae_resnet_ldm(ours + "0.", ldm + "block_1.", sd, out)
        for k, v in sd.items():
            if k.startswith(ours + "1."):
                out[ldm + "attn_1." + k[len(ours + "1."):]] = v
        _vae_resnet_ldm(ours + "2.", ldm + "block_2.", sd, out)
    for i, up_idx in enumerate((3, 2, 1, 0)):
        ours, ldm = f"dec_b{i + 1}.", f"{p}decoder.up.{up_idx}."
        for j in range(3):
            _vae_resnet_ldm(f"{ours}{j}.", f"{ldm}block.{j}.", sd, out)
        for k, v in sd.items():
            if k.startswith(ours + "3.conv."):
                out[ldm + "upsample.conv."
                    + k[len(ours + "3.conv."):]] = v
    for k, v in sd.items():
        if k.startswith("dec_out.0."):
            out[p + "decoder.norm_out." + k[len("dec_out.0."):]] = v
        elif k.startswith("dec_out.2."):
            out[p + "decoder.conv_out." + k[len("dec_out.2."):]] = v
    return out


@needs_torch
def test_unet_key_map_is_complete_meta():
    from nomorals.media.imggen.sdcompat import SDUNet, _map_ldm_unet

    with torch.device("meta"):
        unet = SDUNet()
    sd = unet.state_dict()
    ldm = _unet_to_ldm(sd)
    mapped, consumed = _map_ldm_unet(ldm)
    # Every target parameter filled, every source key consumed.
    assert set(mapped) == set(sd)
    assert consumed == set(ldm)
    with torch.device("meta"):
        fresh = SDUNet()
    fresh.load_state_dict(mapped, strict=True)  # must not raise


@needs_torch
def test_unet_missing_key_is_an_error_path():
    from nomorals.media.imggen.sdcompat import SDUNet, _map_ldm_unet

    with torch.device("meta"):
        unet = SDUNet()
    ldm = _unet_to_ldm(unet.state_dict())
    victim = "model.diffusion_model.input_blocks.1.0.in_layers.0.weight"
    assert victim in ldm
    del ldm[victim]
    mapped, _ = _map_ldm_unet(ldm)
    missing, _ = unet.load_state_dict(mapped, strict=False)
    assert "down0.resnets.0.norm1.weight" in missing


@needs_torch
def test_clip_key_map_is_complete_meta():
    from nomorals.media.imggen.sdcompat import (
        CLIPTextEncoder, _load_ldm_clip)

    with torch.device("meta"):
        enc = CLIPTextEncoder()
        fresh = CLIPTextEncoder()
    ldm = _clip_to_ldm(enc.state_dict())
    _load_ldm_clip(ldm, fresh)  # strict inside: must not raise


@needs_torch
def test_clip_loader_rejects_missing_and_extra_keys():
    from nomorals.media.imggen.sdcompat import (
        CLIPTextEncoder, _load_ldm_clip)

    with torch.device("meta"):
        enc = CLIPTextEncoder()
        fresh = CLIPTextEncoder()
    ldm = _clip_to_ldm(enc.state_dict())
    broken = dict(ldm)
    del broken["cond_stage_model.transformer.resblocks.3.ln_1.weight"]
    with pytest.raises(ImgGenError, match="CLIP"):
        _load_ldm_clip(broken, fresh)
    extra = dict(ldm)
    extra["cond_stage_model.transformer.bogus.weight"] = \
        ldm["cond_stage_model.transformer.ln_final.weight"]
    with pytest.raises(ImgGenError, match="no target"):
        _load_ldm_clip(extra, fresh)


@needs_torch
def test_clip_text_projection_is_known_unused_not_an_error():
    from nomorals.media.imggen.sdcompat import (
        CLIPTextEncoder, _load_ldm_clip)

    with torch.device("meta"):
        enc = CLIPTextEncoder()
        fresh = CLIPTextEncoder()
    ldm = _clip_to_ldm(enc.state_dict())
    proj = next(iter(ldm.values()))
    ldm["cond_stage_model.transformer.text_projection"] = proj
    _load_ldm_clip(ldm, fresh)  # must not raise


@needs_torch
def test_vae_key_map_is_complete_meta():
    from nomorals.media.imggen.sdcompat import (
        AutoencoderKL, _load_ldm_vae)

    with torch.device("meta"):
        vae = AutoencoderKL()
        fresh = AutoencoderKL()
    ldm = _vae_to_ldm(vae.state_dict())
    _load_ldm_vae(ldm, fresh)  # strict both directions: no raise


@needs_torch
def test_vae_loader_rejects_missing_and_extra_keys():
    from nomorals.media.imggen.sdcompat import (
        AutoencoderKL, _load_ldm_vae)

    with torch.device("meta"):
        vae = AutoencoderKL()
        fresh = AutoencoderKL()
    ldm = _vae_to_ldm(vae.state_dict())
    broken = dict(ldm)
    del broken["first_stage_model.decoder.up.0.block.2.conv1.weight"]
    with pytest.raises(ImgGenError, match="VAE"):
        _load_ldm_vae(broken, fresh)
    extra = dict(ldm)
    extra["first_stage_model.encoder.bogus.weight"] = \
        ldm["first_stage_model.encoder.conv_in.weight"]
    with pytest.raises(ImgGenError, match="no target"):
        _load_ldm_vae(extra, fresh)


# ---------------------------------------------------------------------------
# REAL weight round-trips (fp16 tensors, the way checkpoints ship)
# ---------------------------------------------------------------------------

@needs_torch
@needs_mem(900_000, "VAE fp16 round-trip")
def test_vae_real_weight_round_trip_fp16(tmp_path):
    from safetensors.torch import save_file

    from nomorals.media.imggen.sdcompat import (
        AutoencoderKL, _load_ldm_vae, load_safetensors)

    torch.manual_seed(0)
    vae = AutoencoderKL()
    ldm = {k: v.half() for k, v in _vae_to_ldm(vae.state_dict()).items()}
    del vae
    path = tmp_path / "vae.safetensors"
    save_file(ldm, str(path), metadata={"format": "pt"})
    state = load_safetensors(str(path))
    fresh = AutoencoderKL()
    _load_ldm_vae(state, fresh)  # strict: raises on any gap
    got = fresh.state_dict()
    # Loaded values must equal the fp16 sources bit-for-bit after
    # the fp32 cast (fp16 values are exactly representable in fp32).
    # Representative keys across encoder/decoder/mid-attention.
    checks = {
        "enc_in.weight": "first_stage_model.encoder.conv_in.weight",
        "dec_out.2.weight":
            "first_stage_model.decoder.conv_out.weight",
        "enc_mid.1.q.weight":
            "first_stage_model.encoder.mid.attn_1.q.weight",
        "dec_b4.0.conv1.weight":
            "first_stage_model.decoder.up.0.block.0.conv1.weight",
        "enc_b4.1.conv2.weight":
            "first_stage_model.encoder.down.3.block.1.conv2.weight",
    }
    for ours_k, ldm_k in checks.items():
        assert torch.equal(got[ours_k], state[ldm_k].float()), ours_k


@needs_torch
@needs_mem(1_150_000, "CLIP fp16 round-trip")
def test_clip_real_weight_round_trip_fp16():
    from nomorals.media.imggen.sdcompat import (
        CLIPTextEncoder, _load_ldm_clip)

    torch.manual_seed(0)
    enc = CLIPTextEncoder()
    ldm = {k: v.half() for k, v in _clip_to_ldm(enc.state_dict()).items()}
    del enc
    fresh = CLIPTextEncoder()
    _load_ldm_clip(ldm, fresh)
    got = fresh.state_dict()
    checks = {
        "token_embedding.weight":
            "cond_stage_model.transformer.token_embedding.weight",
        "layers.0.mlp.0.weight":
            "cond_stage_model.transformer.resblocks.0.mlp.c_fc.weight",
        "final_layer_norm.weight":
            "cond_stage_model.transformer.ln_final.weight",
    }
    for ours_k, ldm_k in checks.items():
        assert torch.equal(got[ours_k], ldm[ldm_k].float()), ours_k
    # The split in_proj must reassemble exactly.
    q = got["layers.5.self_attn.q_proj.weight"]
    k = got["layers.5.self_attn.k_proj.weight"]
    v = got["layers.5.self_attn.v_proj.weight"]
    joined = torch.cat([q, k, v], dim=0)
    src = ldm["cond_stage_model.transformer.resblocks.5."
              "attn.in_proj_weight"].float()
    assert torch.equal(joined, src)


# ---------------------------------------------------------------------------
# SD15Pipeline plumbing with tiny duck-typed modules
# ---------------------------------------------------------------------------

class _TinyUNet(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, context=None):
        return torch.zeros_like(x)


class _TinyTextEncoder(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = None

    @property
    def tokenizer_mode(self):
        return "bpe" if self.tokenizer is not None \
            else "hashing-fallback"

    def encode(self, prompts):
        return torch.zeros(len(prompts), 77, 16)


class _TinyVAE(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, z.shape[2] * 8,
                           z.shape[3] * 8)


@needs_torch
def test_sd15_pipeline_generate_plumbing():
    from PIL import Image

    from nomorals.media.imggen.sdcompat import SD15Pipeline

    pipe = SD15Pipeline(_TinyUNet(), _TinyTextEncoder(), _TinyVAE(),
                        device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings and "hashing fallback" in pipe.warnings[0]
    images = pipe.generate("a tiny test", steps=2, width=64,
                           height=64, seed=7)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)


# ---------------------------------------------------------------------------
# Adapter + Studio dispatch
# ---------------------------------------------------------------------------

class _RecordingSD15:
    """Duck-typed SD15Pipeline stand-in for Studio dispatch tests."""

    def __init__(self):
        self.device = "cpu"
        self.tokenizer_mode = "hashing-fallback"
        self.warnings = ["tokenizer: hashing fallback active — test"]
        self.calls = []

    def generate(self, prompt, **kwargs):
        from PIL import Image

        self.calls.append((prompt, kwargs))
        return [Image.new("RGB", (kwargs.get("width", 64),
                                  kwargs.get("height", 64)))]


@needs_torch
def test_studio_dispatches_sd15_through_adapter(
        model_roots, monkeypatch, tmp_path):
    from nomorals.media.imggen import sdcompat
    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "mysd.safetensors", SD15_KEYS)
    fake = _RecordingSD15()
    monkeypatch.setattr(
        sdcompat.SD15Pipeline, "from_safetensors",
        classmethod(lambda cls, path, device="": fake))
    studio = Studio(StudioConfig(checkpoint="mysd"))
    pipe = studio.pipeline()
    assert isinstance(pipe, sdcompat.SD15StudioAdapter)
    out = tmp_path / "out.png"
    paths = studio.generate("a cat", steps=3, width=64, height=64,
                            seed=5, save_to=str(out))
    assert paths == [str(out)]
    assert out.exists()
    prompt, kwargs = fake.calls[0]
    assert prompt == "a cat"
    assert kwargs["steps"] == 3 and kwargs["seed"] == 5
    assert kwargs["width"] == 64 and kwargs["height"] == 64
    # The hashing-fallback disclosure rides to the caller AND the
    # sidecar — never silent.
    assert studio.last_warnings == fake.warnings
    sidecar = (tmp_path / "out.png.txt").read_text()
    assert "warning: tokenizer" in sidecar


@needs_torch
def test_studio_default_picks_sd15_when_only_choice(
        model_roots, monkeypatch):
    from nomorals.media.imggen import sdcompat
    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "only.safetensors", SD15_KEYS)
    fake = _RecordingSD15()
    monkeypatch.setattr(
        sdcompat.SD15Pipeline, "from_safetensors",
        classmethod(lambda cls, path, device="": fake))
    studio = Studio(StudioConfig())
    assert isinstance(studio.pipeline(), sdcompat.SD15StudioAdapter)


@needs_torch
def test_studio_sdxl_selection_is_honest_error(model_roots):
    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "myxl.safetensors", SDXL_KEYS)
    studio = Studio(StudioConfig(checkpoint="myxl"))
    with pytest.raises(ImgGenError, match="SDXL"):
        studio.pipeline()


@needs_torch
def test_studio_unknown_checkpoint_names_available(model_roots):
    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "present.safetensors", SD15_KEYS)
    studio = Studio(StudioConfig(checkpoint="absent"))
    with pytest.raises(ImgGenError, match="present"):
        studio.pipeline()


@needs_torch
def test_adapter_edit_protocol_gates_are_honest():
    from nomorals.media.imggen.pipeline import PipelineConfig
    from nomorals.media.imggen.sdcompat import SD15StudioAdapter

    adapter = SD15StudioAdapter(_RecordingSD15())
    with pytest.raises(ImgGenError, match="sd15"):
        _ = adapter.unet
    with pytest.raises(ImgGenError, match="sd15"):
        adapter._scheduler("ddim")
    with pytest.raises(ImgGenError, match="DDIM"):
        adapter.generate("x", PipelineConfig(sampler="ddpm"))
    with pytest.raises(ImgGenError, match="ControlNet"):
        adapter.generate("x", PipelineConfig(),
                         conditioning=[object()])
    # And the honest path works: config fields map onto kwargs.
    out = adapter.generate("y", PipelineConfig(
        steps=9, guidance_scale=3.5, seed=2, width=320, height=256,
        negative_prompt="bad"))
    assert len(out) == 1
    _prompt, kwargs = adapter._pipe.calls[-1]
    assert kwargs["steps"] == 9 and kwargs["guidance_scale"] == 3.5
    assert kwargs["negative_prompt"] == "bad"
    assert kwargs["width"] == 320 and kwargs["height"] == 256
    assert adapter.last_warnings == adapter._pipe.warnings


# ---------------------------------------------------------------------------
# media_edit NativeBackend readiness via the registry
# ---------------------------------------------------------------------------

@needs_torch
def test_native_backend_ready_with_sd15(model_roots):
    from nomorals.media_edit.generate import NativeBackend

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "mysd.safetensors", SD15_KEYS)
    assert NativeBackend.available() is True
    ok, reason = NativeBackend.readiness()
    assert ok is True
    assert "sd15" in reason and "mysd" in reason
    assert "hashing fallback" in reason  # no tokenizer near fixture


@needs_torch
def test_native_backend_not_ready_with_only_sdxl(model_roots):
    from nomorals.media_edit.generate import NativeBackend

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "myxl.safetensors", SDXL_KEYS)
    assert NativeBackend.available() is False
    ok, reason = NativeBackend.readiness()
    assert ok is False
    assert "checkpoint" in reason.lower()
    assert "SDXL" in reason


# ---------------------------------------------------------------------------
# filler._find_pipeline reroute
# ---------------------------------------------------------------------------

@needs_torch
def test_filler_finds_sd15_via_registry(model_roots, monkeypatch):
    from types import SimpleNamespace

    from nomorals.media.directed import filler
    from nomorals.media.imggen import sdcompat

    _ckpt, home = model_roots
    target = home / ".devon-models" / "sd15" / "v1-5-pruned.safetensors"
    _write_safetensors(target, SD15_KEYS)
    sentinel = SimpleNamespace(
        warnings=[], tokenizer_mode="bpe", device="cpu",
        generate=lambda prompt, **kw: [])
    seen = {}

    def fake_from_safetensors(cls, path, device=""):
        seen["path"] = path
        return sentinel

    monkeypatch.setattr(sdcompat.SD15Pipeline, "from_safetensors",
                        classmethod(fake_from_safetensors))
    pipe = filler._find_pipeline()
    assert isinstance(pipe, sdcompat.SD15StudioAdapter)
    assert seen["path"] == str(target)


@needs_torch
def test_filler_env_var_precedence_and_graceful_none(
        model_roots, monkeypatch, tmp_path):
    from types import SimpleNamespace

    from nomorals.media.directed import filler
    from nomorals.media.imggen import sdcompat

    # Absence is graceful.
    assert filler._find_pipeline() is None
    # DEVON_SD_CKPT wins over the default locations.
    custom = tmp_path / "custom.safetensors"
    _write_safetensors(custom, SD15_KEYS)
    monkeypatch.setenv("DEVON_SD_CKPT", str(custom))
    sentinel = SimpleNamespace(
        warnings=[], tokenizer_mode="bpe", device="cpu",
        generate=lambda prompt, **kw: [])
    seen = {}

    def fake_from_safetensors(cls, path, device=""):
        seen["path"] = path
        return sentinel

    monkeypatch.setattr(sdcompat.SD15Pipeline, "from_safetensors",
                        classmethod(fake_from_safetensors))
    pipe = filler._find_pipeline()
    assert isinstance(pipe, sdcompat.SD15StudioAdapter)
    assert seen["path"] == str(custom)


# ---------------------------------------------------------------------------
# Chat surface: /imggen checkpoints lists the registry
# ---------------------------------------------------------------------------

def _chat_mixin():
    from types import SimpleNamespace

    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    class M(RuntimeMediaMixin):
        def __init__(self):
            self.context = SimpleNamespace()

    return M()


def test_imggen_checkpoints_lists_registry(model_roots):
    ckpt, _home = model_roots
    _write_safetensors(ckpt / "mysd.safetensors", SD15_KEYS)
    _write_safetensors(ckpt / "myxl.safetensors", SDXL_KEYS)
    out = _chat_mixin()._control_imggen("checkpoints")
    assert "mysd" in out and "[sd15]" in out and "loadable" in out
    assert "myxl" in out and "[sdxl]" in out
    assert "not loadable" in out and "SDXL" in out


# ---------------------------------------------------------------------------
# Full-size round-trip: the strongest proof, gated on real RAM.
# SD1.5 fp32 weights are ~4.3GB resident (UNet ~3.4GB fp32 + CLIP
# ~0.5GB + VAE ~0.34GB); building the fp16 fixture (~2.2GB) while
# the fp32 modules are alive, then loading (state + modules
# co-resident), peaks around ~9GB. The gate demands ~10.5GB
# available — deliberately far above this VM's 7.9GB total, because
# a borderline pass here means swapless thrash, not a slow test.
# Machines that can hold the model run this; this VM cannot, and
# the skip reason says exactly that.
# ---------------------------------------------------------------------------

@needs_torch
@needs_mem(10_500_000, "full SD1.5 load peaks ~9GB co-resident")
def test_full_sd15_round_trip_generate_64(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from nomorals.media.imggen.sdcompat import (
        AutoencoderKL, CLIPTextEncoder, SD15Pipeline, SDUNet)

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path / "ig"))
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.manual_seed(0)
    ldm = {}
    ldm.update(_unet_to_ldm(SDUNet().state_dict()))
    ldm.update(_clip_to_ldm(CLIPTextEncoder().state_dict()))
    ldm.update(_vae_to_ldm(AutoencoderKL().state_dict()))
    ldm = {k: v.half() for k, v in ldm.items()}
    path = tmp_path / "full.safetensors"
    save_file(ldm, str(path), metadata={"format": "pt"})
    del ldm
    pipe = SD15Pipeline.from_safetensors(str(path), device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings  # disclosed, since no vocab near fixture
    images = pipe.generate("a red square", steps=2, width=64,
                           height=64, seed=1)
    assert len(images) == 1
    assert images[0].size == (64, 64)
