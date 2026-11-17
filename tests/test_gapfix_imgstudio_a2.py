"""Pass A2 tests: the SDXL loader — Devon's own SDXL implementation.

Covers the imggen registry (SDXL LDM now loadable; diffusers layout
still honestly refused with a convert message), the SDXL UNet + dual
CLIP encoders (key-map completeness proven on meta-device modules,
same discipline as A1), the pooled-text + time-ids
micro-conditioning, the VAE scale (0.13025 vs SD1.5's 0.18215),
Studio dispatch through the SDXL adapter, the media_edit
NativeBackend readiness, the /imggen checkpoints surface, and the
sdxl license label.

Same environment honesty as A1: torch is not installed on this box,
so the loader-construction tests skip here (needs_torch) and the
full-size round-trip is gated on real RAM. What runs here: registry
classification, Studio dispatch through a stubbed sdxl module, and
the license label.
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
# Helpers (same pure-python fixtures as the A1 file)
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


SDXL_KEYS = [
    "model.diffusion_model.input_blocks.0.0.weight",
    "model.diffusion_model.time_embed.0.weight",
    "model.diffusion_model.label_emb.0.weight",
    "model.diffusion_model.label_emb.2.weight",
    # Downsamples are separate blocks after levels 0 and 1 only
    # (blocks 3 and 6); the last level (7, 8) has no downsample.
    "model.diffusion_model.input_blocks.3.0.op.weight",
    "model.diffusion_model.input_blocks.6.0.op.weight",
    "conditioner.embedders.0.model.transformer.resblocks.0.ln_1.weight",
    "conditioner.embedders.1.model.transformer.resblocks.0.ln_1.weight",
    "first_stage_model.encoder.conv_in.weight",
]
SDXL_DIFFUSERS_KEYS = [
    "unet.down_blocks.0.resnets.0.conv1.weight",
    "text_encoder.text_model.embeddings.token_embedding.weight",
    "text_encoder_2.text_model.embeddings.token_embedding.weight",
    "vae.encoder.conv_in.weight",
]


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
# Registry: SDXL LDM loads now; diffusers layout still refused honestly
# ---------------------------------------------------------------------------

def test_registry_classifies_sdxl_ldm_loadable(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "sdxl-base.safetensors", SDXL_KEYS)
    (e,) = list_models()
    assert e["model_class"] == "sdxl"
    assert e["format"] == "ldm-safetensors"
    assert e["loadable"] is True
    assert e["loadable_reason"] == ""


def test_registry_sdxl_diffusers_not_loadable_names_convert(model_roots):
    from nomorals.media.imggen.registry import list_models

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "sdxl-diffusers.safetensors",
                       SDXL_DIFFUSERS_KEYS)
    (e,) = list_models()
    assert e["model_class"] == "sdxl"
    assert e["format"] == "diffusers-safetensors"
    assert e["loadable"] is False
    assert "diffusers" in e["loadable_reason"]
    assert "convert" in e["loadable_reason"]


def test_registry_sdxl_resolve_prefers_loadable(model_roots):
    from nomorals.media.imggen.registry import resolve

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "sdxl-base.safetensors", SDXL_KEYS)
    e = resolve("sdxl-base")
    assert e is not None
    assert e["model_class"] == "sdxl" and e["loadable"] is True


# ---------------------------------------------------------------------------
# License label
# ---------------------------------------------------------------------------

def test_sdxl_license_label_registered():
    from nomorals.media.model_license import MODEL_LICENSES, can_ship

    e = MODEL_LICENSES["sdxl"]
    assert e.license == "CreativeML Open RAIL++-M"
    assert e.commercial is True
    info = can_ship("sdxl")
    assert info["commercial_use"] is True
    assert info["personal_use"] is True


# ---------------------------------------------------------------------------
# LDM key mapping completeness (meta device: full key sets, no weight RAM)
#
# The inverse mappers below are written in the test, from the LDM
# SDXL layout, independently of the production forward mappers — if
# the two disagree about even one key, the assertions fail.
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


def _sdxl_unet_to_ldm(sd):
    """SDXLUNet state dict → compvis LDM layout (independent inverse).

    SDXL base (sd_xl_base.yaml): channel_mult [1, 2, 4],
    num_res_blocks 2, attention_resolutions [4, 2] — attention ONLY
    at the 64×64 and 32×32 levels; transformer_depth [1, 2, 10]
    with the first entry UNUSED (no attention at 128×128). So:
    input_blocks 0..8 (conv_in; two resnets per level; a downsample
    block after levels 0 and 1 only), down0 (320ch) has no
    transformer, down1 (640ch) depth 2, down2 (1280ch) depth 10; a
    three-block middle (transformer depth 10); output_blocks 0..8,
    up0 (1280ch) depth 10, up1 (640ch) depth 2, up2 (320ch) no
    transformer; the upsample conv lives on the last block of up0/up1
    only.
    """
    out = {}
    p = "model.diffusion_model."
    for k, v in sd.items():
        if k.startswith("time_embed.") or k.startswith("label_emb."):
            out[p + k] = v
        elif k.startswith("conv_in."):
            out[p + "input_blocks.0.0." + k[len("conv_in."):]] = v
        elif k.startswith("conv_norm_out."):
            out[p + "out.0." + k[len("conv_norm_out."):]] = v
        elif k.startswith("conv_out."):
            out[p + "out.2." + k[len("conv_out."):]] = v
    # (first resnet block, second resnet block, downsample block or
    # None, transformer depth — 0 means no transformer at all).
    down_ldm = {"down0": (1, 2, 3, 0), "down1": (4, 5, 6, 2),
                "down2": (7, 8, None, 10)}
    for name, (b0, b1, ds, depth) in down_ldm.items():
        for li, bi in enumerate((b0, b1)):
            _resnet_ldm(f"{name}.resnets.{li}.",
                        f"{p}input_blocks.{bi}.0.", sd, out)
            if depth:
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
    # (blocks, transformer depth — 0 means no transformer at all).
    up_ldm = {"up0": ((0, 1, 2), 10), "up1": ((3, 4, 5), 2),
              "up2": ((6, 7, 8), 0)}
    for name, (blocks, depth) in up_ldm.items():
        for li, bi in enumerate(blocks):
            _resnet_ldm(f"{name}.resnets.{li}.",
                        f"{p}output_blocks.{bi}.0.", sd, out)
            if depth:
                _transformer_ldm(f"{name}.transformers.{li}.",
                                 f"{p}output_blocks.{bi}.1.", sd, out)
        if name != "up2":  # up2 is the final level: no upsample conv
            for k, v in sd.items():
                if k.startswith(f"{name}.upsampler."):
                    out[f"{p}output_blocks.{blocks[-1]}.2.conv."
                        + k[len(f"{name}.upsampler."):]] = v
    return out


def _sdxl_clip_to_ldm(sd, prefix, layers):
    """One CLIP encoder state dict → conditioner.embedders.N layout."""
    out = {}
    for k, v in sd.items():
        if k == "token_embedding.weight":
            out[prefix + "token_embedding.weight"] = v
        elif k == "position_embedding.weight":
            out[prefix + "positional_embedding"] = v
        elif k == "text_projection.weight":
            out[prefix + "text_projection.weight"] = v
        elif k.startswith("final_layer_norm."):
            out[prefix + "ln_final."
                + k[len("final_layer_norm."):]] = v
    for i in range(layers):
        op, lp = f"layers.{i}.", f"{prefix}transformer.resblocks.{i}."
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


def _vae_to_ldm(sd):
    """AutoencoderKL state dict → first_stage_model layout.

    Same VAE layout as A1 (copied here so this file stands alone —
    the production loader is the shared _load_ldm_vae).
    """
    out = {}
    p = "first_stage_model."

    def _vae_resnet_ldm(ours_prefix, ldm_prefix):
        for k, v in sd.items():
            if not k.startswith(ours_prefix):
                continue
            tail = k[len(ours_prefix):]
            if tail.startswith("shortcut."):
                out[ldm_prefix + "nin_shortcut."
                    + tail[len("shortcut."):]] = v
            else:
                out[ldm_prefix + tail] = v

    for k, v in sd.items():
        if k.startswith("enc_in."):
            out[p + "encoder.conv_in." + k[len("enc_in."):]] = v
        elif k.startswith("quant_conv."):
            out[p + k] = v
        elif k.startswith("post_quant."):
            out[p + "post_quant_conv." + k[len("post_quant."):]] = v
    for i in range(4):
        ours, ldm = f"enc_b{i + 1}.", f"{p}encoder.down.{i}."
        _vae_resnet_ldm(ours + "0.", ldm + "block.0.")
        _vae_resnet_ldm(ours + "1.", ldm + "block.1.")
        for k, v in sd.items():
            if k.startswith(ours + "2."):
                out[ldm + "downsample.conv." + k[len(ours + "2."):]] = v
    for ours, ldm in (("enc_mid.", f"{p}encoder.mid."),
                      ("dec_mid.", f"{p}decoder.mid.")):
        _vae_resnet_ldm(ours + "0.", ldm + "block_1.")
        for k, v in sd.items():
            if k.startswith(ours + "1."):
                out[ldm + "attn_1." + k[len(ours + "1."):]] = v
        _vae_resnet_ldm(ours + "2.", ldm + "block_2.")
    for i, up_idx in enumerate((3, 2, 1, 0)):
        ours, ldm = f"dec_b{i + 1}.", f"{p}decoder.up.{up_idx}."
        for j in range(3):
            _vae_resnet_ldm(f"{ours}{j}.", f"{ldm}block.{j}.")
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
def test_sdxl_unet_key_map_is_complete_meta():
    from nomorals.media.imggen.sdxl import SDXLUNet, _map_ldm_sdxl_unet

    with torch.device("meta"):
        unet = SDXLUNet()
    sd = unet.state_dict()
    ldm = _sdxl_unet_to_ldm(sd)
    mapped, consumed = _map_ldm_sdxl_unet(ldm)
    # Every target parameter filled, every source key consumed.
    assert set(mapped) == set(sd)
    assert consumed == set(ldm)
    with torch.device("meta"):
        fresh = SDXLUNet()
    fresh.load_state_dict(mapped, strict=True)  # must not raise


@needs_torch
def test_sdxl_unet_missing_key_is_an_error_path():
    from nomorals.media.imggen.sdxl import SDXLUNet, _map_ldm_sdxl_unet

    with torch.device("meta"):
        unet = SDXLUNet()
    ldm = _sdxl_unet_to_ldm(unet.state_dict())
    victim = "model.diffusion_model.input_blocks.7.1." \
        "transformer_blocks.9.ff.net.2.weight"
    assert victim in ldm
    del ldm[victim]
    mapped, _ = _map_ldm_sdxl_unet(ldm)
    missing, _ = unet.load_state_dict(mapped, strict=False)
    assert "down2.transformers.0.blocks.9.ff.net.2.weight" in missing


@needs_torch
def test_sdxl_unet_depth_table_and_conditioning():
    from nomorals.media.imggen.sdxl import SDXLUNet

    with torch.device("meta"):
        unet = SDXLUNet()
    # sd_xl_base.yaml: attention_resolutions [4, 2] — attention only
    # at the 64×64 and 32×32 levels; transformer_depth [1, 2, 10]
    # with the first entry UNUSED. So down0/up2 carry NO transformer
    # at all; the depths are 0 / 2 / 10 down, 10 / 2 / 0 up, mid 10.
    assert not hasattr(unet.down0, "transformers")
    assert len(unet.down1.transformers[0].blocks) == 2
    assert len(unet.down2.transformers[0].blocks) == 10
    assert len(unet.mid_trans.blocks) == 10
    assert len(unet.up0.transformers[0].blocks) == 10
    assert len(unet.up1.transformers[0].blocks) == 2
    assert not hasattr(unet.up2, "transformers")
    # Cross-attention context is the concatenated dual encoders.
    assert unet.down2.transformers[0].blocks[0].attn2.to_k.weight.shape \
        == (1280, 2048)
    # Micro-conditioning: pooled (1280) + time-ids (6 × 256) = 2816.
    assert unet.label_emb[0].weight.shape == (1280, 2816)
    assert unet.time_embed[0].weight.shape == (1280, 320)


@needs_torch
def test_sdxl_conditioner_key_map_is_complete_meta():
    from nomorals.media.imggen.sdcompat import CLIPTextEncoder
    from nomorals.media.imggen.sdxl import _load_ldm_xl_conditioner

    with torch.device("meta"):
        enc_l = CLIPTextEncoder()
        enc_g = CLIPTextEncoder(dim=1280, layers=32)
        enc_g.text_projection = torch.nn.Linear(1280, 1280)
    ldm = {}
    ldm.update(_sdxl_clip_to_ldm(
        enc_l.state_dict(), "conditioner.embedders.0.model.", 12))
    ldm.update(_sdxl_clip_to_ldm(
        enc_g.state_dict(), "conditioner.embedders.1.model.", 32))
    # The L encoder's projection is known-unused; the G encoder's is
    # the pooled-text projection and must load.
    proj = next(iter(ldm.values()))
    ldm["conditioner.embedders.0.model.text_projection.weight"] = proj
    with torch.device("meta"):
        fresh_l, fresh_g = _load_ldm_xl_conditioner(ldm)
    assert fresh_g.text_projection.weight.shape == (1280, 1280)
    # Loaded values equal the fixture values (spot-check both encoders).
    src_l = ldm["conditioner.embedders.0.model.transformer."
                "resblocks.5.attn.in_proj_weight"]
    got_l = fresh_l.state_dict()["layers.5.self_attn.q_proj.weight"]
    assert torch.equal(got_l, src_l[:768])
    src_g = ldm["conditioner.embedders.1.model.transformer."
                "resblocks.5.attn.in_proj_weight"]
    got_g = fresh_g.state_dict()["layers.5.self_attn.q_proj.weight"]
    assert torch.equal(got_g, src_g[:1280])


@needs_torch
def test_sdxl_conditioner_rejects_missing_and_extra_keys():
    from nomorals.media.imggen.sdcompat import CLIPTextEncoder
    from nomorals.media.imggen.sdxl import _load_ldm_xl_conditioner

    with torch.device("meta"):
        enc_l = CLIPTextEncoder()
        enc_g = CLIPTextEncoder(dim=1280, layers=32)
        enc_g.text_projection = torch.nn.Linear(1280, 1280)
    ldm = {}
    ldm.update(_sdxl_clip_to_ldm(
        enc_l.state_dict(), "conditioner.embedders.0.model.", 12))
    ldm.update(_sdxl_clip_to_ldm(
        enc_g.state_dict(), "conditioner.embedders.1.model.", 32))
    broken = dict(ldm)
    del broken["conditioner.embedders.1.model.transformer."
               "resblocks.3.ln_1.weight"]
    with pytest.raises(ImgGenError, match="CLIP"):
        _load_ldm_xl_conditioner(broken)
    extra = dict(ldm)
    extra["conditioner.embedders.1.model.transformer.bogus.weight"] = \
        ldm["conditioner.embedders.1.model.ln_final.weight"]
    with pytest.raises(ImgGenError, match="no target"):
        _load_ldm_xl_conditioner(extra)


@needs_torch
def test_sdxl_vae_scale_is_xl():
    from nomorals.media.imggen.sdcompat import AutoencoderKL

    assert AutoencoderKL().scale == 0.18215  # SD1.5 default unchanged
    assert AutoencoderKL(scale=0.13025).scale == 0.13025


# ---------------------------------------------------------------------------
# SDXLPipeline plumbing with tiny duck-typed modules
# ---------------------------------------------------------------------------

class _TinyXLUNet(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, t, context=None, pooled=None, time_ids=None):
        assert context is not None and context.shape[-1] == 2048
        assert pooled is not None and pooled.shape[-1] == 1280
        assert time_ids is not None and time_ids.shape[-1] == 6
        return torch.zeros_like(x)


class _TinyXLEncoder(torch.nn.Module if torch is not None else object):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = None

    @property
    def tokenizer_mode(self):
        return "bpe" if self.tokenizer is not None \
            else "hashing-fallback"

    def encode_states(self, prompts):
        b = len(prompts)
        ctx = torch.zeros(b, 77, self.dim)
        pooled = torch.zeros(b, self.dim)
        return ctx, pooled


class _TinyXLVAE(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.scale = 0.13025

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, z.shape[2] * 8,
                           z.shape[3] * 8)


@needs_torch
def test_sdxl_pipeline_generate_plumbing():
    from PIL import Image

    from nomorals.media.imggen.sdxl import SDXLPipeline

    pipe = SDXLPipeline(_TinyXLUNet(), _TinyXLEncoder(768),
                        _TinyXLEncoder(1280), _TinyXLVAE(),
                        device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings and "hashing fallback" in pipe.warnings[0]
    images = pipe.generate("a tiny test", steps=2, width=64,
                           height=64, seed=7)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)


@needs_torch
def test_sdxl_pipeline_refuses_sd15_and_diffusers(tmp_path):
    from nomorals.media.imggen.sdxl import SDXLPipeline

    # SD1.5 keys (no conditioner, no label_emb) → named refusal.
    sd15 = {"model.diffusion_model.input_blocks.0.0.weight":
            torch.zeros(1),
            "cond_stage_model.transformer.token_embedding.weight":
            torch.zeros(1),
            "first_stage_model.encoder.conv_in.weight": torch.zeros(1)}
    with pytest.raises(ImgGenError, match="SD1.5"):
        SDXLPipeline._from_state(sd15, "fake.safetensors")
    # Diffusers-layout SDXL keys → convert message, same as A1.
    diff = {"unet.down_blocks.0.resnets.0.conv1.weight": torch.zeros(1),
            "text_encoder_2.text_model.embeddings.token_embedding.weight":
            torch.zeros(1)}
    with pytest.raises(ImgGenError, match="convert"):
        SDXLPipeline._from_state(diff, "fake.safetensors")


# ---------------------------------------------------------------------------
# Studio dispatch
# ---------------------------------------------------------------------------

def test_studio_dispatches_sdxl_without_torch(
        model_roots, monkeypatch, tmp_path):
    """The Studio sdxl branch, proven without torch: a stub sdxl
    module stands in for the real (torch-guarded) one."""
    import sys
    import types

    from PIL import Image

    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "myxl.safetensors", SDXL_KEYS)
    fake_mod = types.ModuleType("nomorals.media.imggen.sdxl")

    class _FakePipe:
        def __init__(self, path, device=""):
            self.path = path
            self.device = device or "cpu"
            self.calls = []

        @classmethod
        def from_safetensors(cls, path, device="", *, dtype="",
                             quant=""):
            pipe = cls(path, device)
            pipe.dtype = dtype
            pipe.quant = quant
            return pipe

        def generate(self, prompt, **kwargs):
            self.calls.append((prompt, kwargs))
            return [Image.new("RGB", (kwargs.get("width", 64),
                                      kwargs.get("height", 64)))]

    class _FakeAdapter:
        model_class = "sdxl"

        def __init__(self, pipe):
            self._pipe = pipe
            self.last_warnings = []
            self.device = pipe.device

        def generate(self, prompt, cfg=None, conditioning=None):
            return self._pipe.generate(
                prompt, steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                seed=cfg.seed, width=cfg.width, height=cfg.height,
                negative_prompt=cfg.negative_prompt, eta=cfg.eta)

    fake_mod.SDXLPipeline = _FakePipe
    fake_mod.SDXLStudioAdapter = _FakeAdapter
    monkeypatch.setitem(sys.modules,
                        "nomorals.media.imggen.sdxl", fake_mod)
    monkeypatch.setattr(
        "nomorals.media.imggen.studio.TORCH_AVAILABLE", True)
    studio = Studio(StudioConfig(checkpoint="myxl"))
    pipe = studio.pipeline()
    assert isinstance(pipe, _FakeAdapter)
    out = tmp_path / "out.png"
    paths = studio.generate("an xl cat", steps=3, width=64,
                            height=64, seed=5, save_to=str(out))
    assert paths == [str(out)]
    assert out.exists()
    prompt, kwargs = pipe._pipe.calls[0]
    assert prompt == "an xl cat"
    assert kwargs["steps"] == 3 and kwargs["seed"] == 5


@needs_torch
def test_sdxl_adapter_gates_are_honest():
    from nomorals.media.imggen.pipeline import PipelineConfig
    from nomorals.media.imggen.sdxl import SDXLStudioAdapter

    class _RecordingXL:
        def __init__(self):
            self.device = "cpu"
            self.tokenizer_mode = "bpe"
            self.warnings = []
            self.calls = []

        def generate(self, prompt, **kwargs):
            from PIL import Image

            self.calls.append((prompt, kwargs))
            return [Image.new("RGB", (kwargs.get("width", 64),
                                      kwargs.get("height", 64)))]

    adapter = SDXLStudioAdapter(_RecordingXL())
    with pytest.raises(ImgGenError, match="sdxl"):
        _ = adapter.unet
    with pytest.raises(ImgGenError, match="sdxl"):
        adapter._scheduler("ddim")
    with pytest.raises(ImgGenError, match="DDIM"):
        adapter.generate("x", PipelineConfig(sampler="ddpm"))
    with pytest.raises(ImgGenError, match="ControlNet"):
        adapter.generate("x", PipelineConfig(),
                         conditioning=[object()])
    out = adapter.generate("y", PipelineConfig(
        steps=9, guidance_scale=3.5, seed=2, width=320, height=256,
        negative_prompt="bad"))
    assert len(out) == 1
    _prompt, kwargs = adapter._pipe.calls[-1]
    assert kwargs["steps"] == 9 and kwargs["guidance_scale"] == 3.5
    assert kwargs["negative_prompt"] == "bad"
    assert kwargs["width"] == 320 and kwargs["height"] == 256


# ---------------------------------------------------------------------------
# Full-size round-trip: the strongest proof, gated on real RAM.
# SDXL fp32 weights are ~17GB resident (UNet ~2.6B + CLIP-G ~1.2B +
# CLIP-L ~0.4B + VAE ~0.08B); building the fp16 fixture (~8.5GB)
# while the fp32 modules are alive, then loading (state + modules
# co-resident), peaks around ~25GB. The gate demands ~26GB available
# — deliberately far above this VM's 7.9GB total. Machines that can
# hold the model run this; this VM cannot, and the skip reason says
# exactly that.
# ---------------------------------------------------------------------------

@needs_torch
@needs_mem(26_000_000, "full SDXL load peaks ~25GB co-resident")
def test_full_sdxl_round_trip_generate_64(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    from nomorals.media.imggen.sdcompat import AutoencoderKL, CLIPTextEncoder
    from nomorals.media.imggen.sdxl import SDXLPipeline, SDXLUNet

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path / "ig"))
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.manual_seed(0)
    ldm = {}
    ldm.update(_sdxl_unet_to_ldm(SDXLUNet().state_dict()))
    enc_l = CLIPTextEncoder()
    enc_g = CLIPTextEncoder(dim=1280, layers=32)
    enc_g.text_projection = torch.nn.Linear(1280, 1280)
    ldm.update(_sdxl_clip_to_ldm(
        enc_l.state_dict(), "conditioner.embedders.0.model.", 12))
    ldm.update(_sdxl_clip_to_ldm(
        enc_g.state_dict(), "conditioner.embedders.1.model.", 32))
    ldm["conditioner.embedders.0.model.text_projection.weight"] = \
        torch.zeros(768, 768)
    ldm.update(_vae_to_ldm(AutoencoderKL().state_dict()))
    ldm = {k: v.half() for k, v in ldm.items()}
    path = tmp_path / "full-xl.safetensors"
    save_file(ldm, str(path), metadata={"format": "pt"})
    del ldm
    pipe = SDXLPipeline.from_safetensors(str(path), device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings  # disclosed, since no vocab near fixture
    images = pipe.generate("a red square", steps=2, width=64,
                           height=64, seed=1)
    assert len(images) == 1
    assert images[0].size == (64, 64)
