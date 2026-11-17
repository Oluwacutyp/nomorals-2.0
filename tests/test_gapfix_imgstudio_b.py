"""Pass B tests: the FLUX loader — Devon's own FLUX implementation.

Covers the imggen registry (FLUX in the BFL single-file layout now
loadable, and the diffusers transformer layout too since the key
map landed — see tests/test_gapfix_flux_keymap.py), the MMDiT
transformer (key-map completeness proven on meta-device modules,
same discipline as A1/A2), the T5
text encoder (v1.1 gated-gelu feed-forward, NOT relu), the
diffusers-layout VAE (split I/O convs, encoder norm_out, two resnets
per decoder level — the sdcompat VAE grows exactly those, defaults
unchanged), the FlowMatchScheduler (BFL dynamic-shift rectified
flow), Studio dispatch through the FLUX adapter, and the flux
license label.

Same environment honesty as A1/A2: torch is not installed on this
box, so the loader-construction tests skip here (needs_torch) and
the full-size round-trip is gated on real RAM. What runs here:
registry classification, Studio dispatch through a stubbed flux
module, and the license label.
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
# Helpers (same pure-python fixtures as the A1/A2 files)
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


FLUX_BFL_KEYS = [
    "model.diffusion_model.img_in.weight",
    "model.diffusion_model.txt_in.weight",
    "model.diffusion_model.time_in.in_layer.weight",
    "model.diffusion_model.time_in.out_layer.weight",
    "model.diffusion_model.vector_in.weight",
    "model.diffusion_model.guidance_in.weight",
    "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
    "model.diffusion_model.double_blocks.0.img_mod.lin.weight",
    "model.diffusion_model.double_blocks.0.txt_attn.qkv.weight",
    "model.diffusion_model.single_blocks.0.linear1.weight",
    "model.diffusion_model.single_blocks.0.modulation.lin.weight",
    "model.diffusion_model.final_layer.adaLN_modulation.1.weight",
    "model.diffusion_model.final_layer.linear.weight",
]
FLUX_BFL_FLAT_KEYS = [k[len("model.diffusion_model."):]
                      for k in FLUX_BFL_KEYS]
FLUX_DIFFUSERS_KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.ff.net.0.proj.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.proj_out.weight",
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
# Registry: FLUX BFL single-file loads; diffusers transformer layout
# loads too since the key map landed (same stance as A2's SDXL
# diffusers refusal, now resolved for FLUX)
# ---------------------------------------------------------------------------

def test_registry_classifies_flux_bfl_loadable(model_roots):
    from nomorals.media.imggen.registry import _classify_keys, _entry_for_safetensors, list_models

    ckpt, _home = model_roots
    path = ckpt / "flux-dev.safetensors"
    _write_safetensors(path, FLUX_BFL_KEYS)
    model_class, fmt = _classify_keys(
        {k: None for k in FLUX_BFL_KEYS})
    assert model_class == "flux"
    assert fmt == "ldm-safetensors"
    entry = _entry_for_safetensors(path)
    assert entry["loadable"] is True
    assert entry["loadable_reason"] == ""
    assert [e["name"] for e in list_models()] == ["flux-dev"]


def test_registry_classifies_flux_bfl_flat_loadable(model_roots):
    from nomorals.media.imggen.registry import _classify_keys

    model_class, fmt = _classify_keys(
        {k: None for k in FLUX_BFL_FLAT_KEYS})
    assert model_class == "flux"
    assert fmt == "ldm-safetensors"


def test_registry_flux_diffusers_loadable(model_roots):
    from nomorals.media.imggen.registry import _entry_for_safetensors

    ckpt, _home = model_roots
    path = ckpt / "flux-diffusers.safetensors"
    _write_safetensors(path, FLUX_DIFFUSERS_KEYS)
    entry = _entry_for_safetensors(path)
    assert entry["model_class"] == "flux"
    assert entry["format"] == "diffusers-safetensors"
    assert entry["loadable"] is True
    assert entry["loadable_reason"] == ""


def test_registry_flux_resolve_prefers_loadable(model_roots):
    from nomorals.media.imggen.registry import list_models, resolve

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "flux-diffusers.safetensors",
                       FLUX_DIFFUSERS_KEYS)
    _write_safetensors(ckpt / "flux-dev.safetensors", FLUX_BFL_KEYS)
    names = [e["name"] for e in list_models()]
    assert names.index("flux-dev") < names.index("flux-diffusers")
    assert resolve("flux-dev")["loadable"] is True
    assert resolve("flux-diffusers")["loadable"] is True


# ---------------------------------------------------------------------------
# License: FLUX.1-dev is non-commercial
# ---------------------------------------------------------------------------

def test_flux_license_label_registered():
    from nomorals.media.model_license import can_ship

    label = can_ship("flux")
    assert label["model"] == "flux"
    assert label["licence"] == "FLUX.1-dev Non-Commercial License"
    assert label["commercial_use"] is False
    assert label["personal_use"] is True


# ---------------------------------------------------------------------------
# MMDiT: structure matches the live FLUX.1-dev config, key map complete
# ---------------------------------------------------------------------------

@needs_torch
def test_flux_transformer_structure_matches_dev_config():
    from nomorals.media.imggen.flux import FluxTransformer

    with torch.device("meta"):
        m = FluxTransformer()
    # FLUX.1-dev (black-forest-labs/FLUX.1-dev config.json):
    # hidden 3072, 24 heads (head dim 128), 19 double + 38 single
    # blocks, context 4096 (T5-XXL), pooled 768 (CLIP-L),
    # in_channels 64, axes_dim [16, 56, 56], theta 10000.
    assert m.hidden_size == 3072
    assert m.num_heads == 24
    assert m.head_dim == 128
    assert len(m.double_blocks) == 19
    assert len(m.single_blocks) == 38
    assert m.axes_dim == [16, 56, 56]
    assert m.theta == 10000
    # qkv is one fused projection WITH bias (BFL), unlike SDXL.
    # Both streams run in hidden space (txt_in already projected
    # the 4096-wide T5 context down to 3072).
    assert m.double_blocks[0].img_attn.qkv.weight.shape == (9216, 3072)
    assert m.double_blocks[0].img_attn.qkv.bias is not None
    assert m.double_blocks[0].txt_attn.qkv.weight.shape == (9216, 3072)
    # One double modulation produces BOTH modulation outs (shift,
    # scale, gate each): 6 * 3072 rows.
    assert m.double_blocks[0].img_mod.lin.weight.shape == (18432, 3072)
    assert m.double_blocks[0].txt_mod.lin.weight.shape == (18432, 3072)
    # Single blocks concatenate attention (3 * 3072) + mlp (4 * 3072)
    # into ONE linear: 21504 rows in, 3072 out; modulation is single
    # (shift, scale, gate): 3 * 3072 rows.
    assert m.single_blocks[0].linear1.weight.shape == (21504, 3072)
    assert m.single_blocks[0].linear2.weight.shape == (3072, 15360)
    assert m.single_blocks[0].modulation.lin.weight.shape == (9216, 3072)
    # The final layer projects back to the packed latent (64).
    assert m.final_layer.linear.weight.shape == (64, 3072)
    assert m.final_layer.adaLN_modulation[1].weight.shape == (6144, 3072)
    # Guidance embedding exists (dev only; schnell has none).
    assert hasattr(m, "guidance_in")


@needs_torch
def test_flux_transformer_key_map_is_complete_meta():
    from nomorals.media.imggen.flux import FluxTransformer, _map_bfl_transformer

    with torch.device("meta"):
        m = FluxTransformer()
    sd = m.state_dict()
    for prefix in ("model.diffusion_model.", ""):
        bfl = {prefix + k: v for k, v in sd.items()}
        mapped, consumed = _map_bfl_transformer(bfl)
        # Every target parameter filled, every source key consumed.
        assert set(mapped) == set(sd)
        assert consumed == set(bfl)
        with torch.device("meta"):
            fresh = FluxTransformer()
        fresh.load_state_dict(mapped, strict=True)  # must not raise


@needs_torch
def test_flux_transformer_missing_and_extra_keys_are_errors():
    from nomorals.media.imggen.flux import FluxTransformer, _map_bfl_transformer

    with torch.device("meta"):
        m = FluxTransformer()
    bfl = {"model.diffusion_model." + k: v
           for k, v in m.state_dict().items()}
    broken = dict(bfl)
    del broken["model.diffusion_model.double_blocks.3."
               "img_attn.qkv.weight"]
    with pytest.raises(ImgGenError, match="double_blocks.3"):
        _map_bfl_transformer(broken)
    extra = dict(bfl)
    extra["model.diffusion_model.double_blocks.0.bogus.weight"] = \
        next(iter(bfl.values()))
    with pytest.raises(ImgGenError, match="no target"):
        _map_bfl_transformer(extra)


@needs_torch
def test_flux_tiny_forward_and_pack_round_trip():
    from nomorals.media.imggen.flux import FluxTransformer, _image_ids, pack_latents, unpack_latents

    torch.manual_seed(0)
    m = FluxTransformer(hidden_size=64, num_heads=2, depth=1,
                        depth_single_blocks=1, in_channels=64,
                        context_in_dim=128, vec_in_dim=64,
                        axes_dim=[8, 12, 12])
    b, img_len, txt_len = 1, 16, 8
    img = torch.randn(b, img_len, 64)
    txt = torch.randn(b, txt_len, 128)
    img_ids = _image_ids(8, 8).repeat(b, 1, 1)
    txt_ids = torch.zeros(b, txt_len, 3)
    t = torch.tensor([0.5])
    vec = torch.randn(b, 64)
    guidance = torch.tensor([3.5])
    out = m(img, img_ids, txt, txt_ids, t, vec, guidance)
    assert out.shape == (b, img_len, 64)
    assert torch.isfinite(out).all()
    # Pack/unpack is the exact inverse (2×2 pixel blocks, 4× wide).
    lat = torch.randn(1, 16, 8, 8)
    assert torch.equal(unpack_latents(pack_latents(lat), 8, 8), lat)
    assert pack_latents(lat).shape == (1, 16, 64)


# ---------------------------------------------------------------------------
# FlowMatchScheduler: BFL dynamic-shift rectified flow
# ---------------------------------------------------------------------------

@needs_torch
def test_flux_scheduler_timesteps_and_step():
    from nomorals.media.imggen.flux import FlowMatchScheduler

    s = FlowMatchScheduler()
    sig = s.set_timesteps(10)
    assert len(sig) == 11
    assert float(sig[0]) == 1.0 and float(sig[-1]) == 0.0
    assert bool((sig[:-1] > sig[1:]).all())  # strictly decreasing
    # Dynamic shifting: a bigger image shifts the schedule.
    big = s.set_timesteps(10, image_seq_len=4096)
    small = s.set_timesteps(10, image_seq_len=256)
    assert not torch.equal(big, small)
    # Euler step: x += (t_prev - t_curr) * v, exactly.
    x = torch.randn(2, 3)
    v = torch.randn(2, 3)
    out = s.step(v, t_curr=0.8, t_prev=0.7, x=x)
    assert torch.equal(out, x + (0.7 - 0.8) * v)


# ---------------------------------------------------------------------------
# T5 encoder: v1.1 gated-gelu, SentencePiece, meta-complete key map
# ---------------------------------------------------------------------------

@needs_torch
def test_flux_t5_structure_matches_xxl_config():
    from nomorals.media.imggen.flux import T5TextEncoder

    with torch.device("meta"):
        enc = T5TextEncoder()
    # google/t5-v1_1-xxl: d_model 4096, d_ff 10240, 64 heads,
    # 24 layers, vocab 32128. v1.1 uses gated-gelu, NOT relu.
    assert enc.d_model == 4096
    assert enc.d_ff == 10240
    assert enc.num_heads == 64
    assert len(enc.block) == 24
    assert enc.shared.weight.shape == (32128, 4096)
    assert enc.block[0].layer[1].ffn.wi_0.weight.shape == (10240, 4096)
    assert enc.block[0].layer[1].ffn.wi_1.weight.shape == (10240, 4096)
    assert enc.block[0].layer[1].ffn.wo.weight.shape == (4096, 10240)
    # Only block 0 carries the relative-position bias (transformers).
    assert hasattr(enc.block[0].layer[0].SelfAttention,
                  "relative_attention_bias")
    assert not hasattr(enc.block[1].layer[0].SelfAttention,
                      "relative_attention_bias")


def _t5_to_hf(sd):
    """T5TextEncoder state dict → transformers T5EncoderModel keys."""
    out = {}
    for k, v in sd.items():
        if k == "shared.weight":
            out["shared.weight"] = v
        elif k == "final_layer_norm.weight":
            out["encoder.final_layer_norm.weight"] = v
        elif k.startswith("block."):
            rest = k[len("block."):]
            idx, tail = rest.split(".", 1)
            tail = tail.replace("layer.1.ffn.",
                                "layer.1.DenseReluDense.")
            out[f"encoder.block.{idx}.{tail}"] = v
    return out


@needs_torch
def test_flux_t5_key_map_is_complete_meta():
    from nomorals.media.imggen.flux import T5TextEncoder, _load_hf_t5

    with torch.device("meta"):
        enc = T5TextEncoder()
    hf = _t5_to_hf(enc.state_dict())
    with torch.device("meta"):
        fresh = _load_hf_t5(hf)
    assert set(fresh.state_dict()) == set(enc.state_dict())
    # Loaded values equal the fixture values (spot-check).
    src = hf["encoder.block.5.layer.1.DenseReluDense.wi_0.weight"]
    got = fresh.state_dict()["block.5.layer.1.ffn.wi_0.weight"]
    assert torch.equal(got, src)
    broken = dict(hf)
    del broken["encoder.block.3.layer.0.SelfAttention.q.weight"]
    with pytest.raises(ImgGenError, match="T5"):
        _load_hf_t5(broken)


@needs_torch
def test_flux_t5_tiny_forward_runs():
    from nomorals.media.imggen.flux import T5TextEncoder

    torch.manual_seed(0)
    enc = T5TextEncoder(d_model=64, d_ff=128, num_heads=2,
                        num_layers=2, vocab_size=100)
    ids = torch.randint(0, 100, (2, 16))
    out = enc(ids)
    assert out.shape == (2, 16, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# CLIP-L (transformers layout): pooled projection is the FLUX vec
# ---------------------------------------------------------------------------

def _clip_to_hf(sd):
    """CLIPTextEncoder state dict → transformers CLIPTextModel keys."""
    out = {}
    for k, v in sd.items():
        if k == "token_embedding.weight":
            out["text_model.embeddings.token_embedding.weight"] = v
        elif k == "position_embedding.weight":
            out["text_model.embeddings.position_embedding.weight"] = v
        elif k == "text_projection.weight":
            out["text_projection.weight"] = v
        elif k.startswith("final_layer_norm."):
            out["text_model." + k] = v
        elif k.startswith("layers."):
            rest = k[len("layers."):]
            idx, tail = rest.split(".", 1)
            tail = tail.replace("mlp.0.", "mlp.fc1.") \
                .replace("mlp.2.", "mlp.fc2.")
            out[f"text_model.encoder.layers.{idx}.{tail}"] = v
    return out


@needs_torch
def test_flux_clip_hf_key_map_is_complete_meta():
    from nomorals.media.imggen.flux import _load_hf_clip
    from nomorals.media.imggen.sdcompat import CLIPTextEncoder

    with torch.device("meta"):
        enc = CLIPTextEncoder()
        enc.text_projection = torch.nn.Linear(768, 768)
    hf = _clip_to_hf(enc.state_dict())
    with torch.device("meta"):
        fresh = _load_hf_clip(hf)
    assert set(fresh.state_dict()) == set(enc.state_dict())
    assert fresh.text_projection.weight.shape == (768, 768)
    src = hf["text_model.encoder.layers.5.self_attn.q_proj.weight"]
    got = fresh.state_dict()["layers.5.self_attn.q_proj.weight"]
    assert torch.equal(got, src)
    broken = dict(hf)
    del broken["text_model.encoder.layers.3.mlp.fc1.weight"]
    with pytest.raises(ImgGenError, match="CLIP"):
        _load_hf_clip(broken)


# ---------------------------------------------------------------------------
# VAE (diffusers AutoencoderKL layout): split I/O convs, encoder
# norm_out, two resnets per decoder level — the sdcompat VAE grows
# exactly those, with its defaults unchanged (A1/A2 suites prove it).
# ---------------------------------------------------------------------------

def _vae_to_diffusers(sd):
    """AutoencoderKL state dict → diffusers AutoencoderKL keys."""
    out = {}

    def _resnet(ours_prefix, diff_prefix):
        for k, v in sd.items():
            if not k.startswith(ours_prefix):
                continue
            tail = k[len(ours_prefix):]
            if tail.startswith("shortcut."):
                out[diff_prefix + "conv_shortcut."
                    + tail[len("shortcut."):]] = v
            else:
                out[diff_prefix + tail] = v

    for k, v in sd.items():
        if k.startswith("enc_in."):
            out["encoder.conv_in." + k[len("enc_in."):]] = v
        elif k.startswith("enc_norm_out."):
            out["encoder.conv_norm_out." + k[len("enc_norm_out."):]] = v
        elif k.startswith("enc_out."):
            out["encoder.conv_out." + k[len("enc_out."):]] = v
        elif k.startswith("quant_conv."):
            out["quant_conv." + k[len("quant_conv."):]] = v
        elif k.startswith("post_quant."):
            out["post_quant_conv." + k[len("post_quant."):]] = v
        elif k.startswith("dec_in."):
            out["decoder.conv_in." + k[len("dec_in."):]] = v
        elif k.startswith("dec_out."):
            tail = k[len("dec_out."):]
            if tail.startswith("0."):
                out["decoder.conv_norm_out." + tail[2:]] = v
            elif tail.startswith("2."):
                out["decoder.conv_out." + tail[2:]] = v
    for i in range(4):
        _resnet(f"enc_b{i + 1}.0.", f"encoder.down_blocks.{i}.resnets.0.")
        _resnet(f"enc_b{i + 1}.1.", f"encoder.down_blocks.{i}.resnets.1.")
        if i < 3:
            for k, v in sd.items():
                if k.startswith(f"enc_b{i + 1}.2."):
                    out[f"encoder.down_blocks.{i}.downsamplers.0.conv."
                        + k[len(f"enc_b{i + 1}.2."):]] = v
    for ours, diff in (("enc_mid.0.", "encoder.mid_block.resnets.0."),
                       ("enc_mid.2.", "encoder.mid_block.resnets.1."),
                       ("dec_mid.0.", "decoder.mid_block.resnets.0."),
                       ("dec_mid.2.", "decoder.mid_block.resnets.1.")):
        _resnet(ours, diff)
    for ours, diff in (("enc_mid.1.", "encoder.mid_block.attentions.0."),
                       ("dec_mid.1.", "decoder.mid_block.attentions.0.")):
        for k, v in sd.items():
            if not k.startswith(ours):
                continue
            tail = k[len(ours):]
            if tail.startswith("norm."):
                out[diff + "group_norm." + tail[len("norm."):]] = v
            elif tail.startswith("q."):
                out[diff + "query." + tail[len("q."):]] = v
            elif tail.startswith("k."):
                out[diff + "key." + tail[len("k."):]] = v
            elif tail.startswith("v."):
                out[diff + "value." + tail[len("v."):]] = v
            elif tail.startswith("proj_out."):
                out[diff + "proj_attn." + tail[len("proj_out."):]] = v
    for i in range(4):
        for j in range(2):
            _resnet(f"dec_b{i + 1}.{j}.",
                    f"decoder.up_blocks.{i}.resnets.{j}.")
        if i < 3:
            for k, v in sd.items():
                if k.startswith(f"dec_b{i + 1}.2."):
                    out[f"decoder.up_blocks.{i}.upsamplers.0.conv."
                        + k[len(f"dec_b{i + 1}.2."):]] = v
    return out


@needs_torch
def test_flux_vae_diffusers_key_map_is_complete_meta():
    from nomorals.media.imggen.flux import _load_diffusers_vae
    from nomorals.media.imggen.sdcompat import AutoencoderKL

    with torch.device("meta"):
        vae = AutoencoderKL(latent_ch=16, scale=0.3611, shift=0.1159,
                            decoder_resnets=2, diffusers_layout=True)
    diff = _vae_to_diffusers(vae.state_dict())
    with torch.device("meta"):
        fresh = AutoencoderKL(latent_ch=16, scale=0.3611,
                              shift=0.1159, decoder_resnets=2,
                              diffusers_layout=True)
    _load_diffusers_vae(diff, fresh)
    assert set(fresh.state_dict()) == set(vae.state_dict())
    src = diff["decoder.up_blocks.2.resnets.1.conv1.weight"]
    got = fresh.state_dict()["dec_b3.1.conv1.weight"]
    assert torch.equal(got, src)


@needs_torch
def test_flux_vae_scale_and_shift():
    from nomorals.media.imggen.sdcompat import AutoencoderKL

    # SD1.5 default unchanged; FLUX VAE is 16 channels, scale
    # 0.3611, shift 0.1159 (black-forest-labs/FLUX.1-dev vae config).
    assert AutoencoderKL().scale == 0.18215
    assert AutoencoderKL().shift == 0.0
    vae = AutoencoderKL(latent_ch=16, scale=0.3611, shift=0.1159)
    assert vae.latent_ch == 16
    assert vae.scale == 0.3611
    assert vae.shift == 0.1159
    x = torch.zeros(1, 3, 16, 16)
    assert vae.encode(x).shape == (1, 16, 2, 2)
    assert vae.decode(torch.zeros(1, 16, 2, 2)).shape == (1, 3, 16, 16)


# ---------------------------------------------------------------------------
# FLUXPipeline plumbing with tiny duck-typed modules
# ---------------------------------------------------------------------------

class _TinyFluxTransformer(torch.nn.Module if torch is not None
                           else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, img, img_ids, txt, txt_ids, timesteps, y,
                guidance=None):
        assert txt.shape[-1] == 4096  # T5-XXL context
        assert y.shape[-1] == 768  # CLIP-L pooled
        assert guidance is not None  # dev is guidance-distilled
        return torch.zeros_like(img)


class _TinyFluxT5(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = None

    @property
    def tokenizer_mode(self):
        return "spiece" if self.tokenizer is not None \
            else "hashing-fallback"

    def encode(self, prompts):
        return torch.zeros(len(prompts), 512, 4096)


class _TinyFluxCLIP(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.tokenizer = None

    @property
    def tokenizer_mode(self):
        return "bpe" if self.tokenizer is not None \
            else "hashing-fallback"

    def encode_states(self, prompts):
        b = len(prompts)
        return torch.zeros(b, 77, 768), torch.zeros(b, 768)


class _TinyFluxVAE(torch.nn.Module if torch is not None else object):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.latent_ch = 16
        self.scale = 0.3611
        self.shift = 0.1159

    def decode(self, z):
        return torch.zeros(z.shape[0], 3, z.shape[2] * 8,
                           z.shape[3] * 8)


@needs_torch
def test_flux_pipeline_generate_plumbing():
    from PIL import Image

    from nomorals.media.imggen.flux import FLUXPipeline

    pipe = FLUXPipeline(_TinyFluxTransformer(), _TinyFluxT5(),
                        _TinyFluxCLIP(), _TinyFluxVAE(), device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings and "hashing fallback" in pipe.warnings[0]
    images = pipe.generate("a tiny test", steps=2, width=64,
                           height=64, seed=7)
    assert len(images) == 1
    assert isinstance(images[0], Image.Image)
    assert images[0].size == (64, 64)


@needs_torch
def test_flux_pipeline_refuses_other_formats(tmp_path):
    from nomorals.media.imggen.flux import FLUXPipeline

    # Diffusers-layout FLUX keys now load through the key map — a
    # bare toy pair fails downstream (missing companions, then
    # unfilled parameters), never on a convert refusal (see
    # test_gapfix_flux_keymap.py for the full diffusers load).
    diff = {"transformer_blocks.0.attn.to_q.weight": torch.zeros(1),
            "single_transformer_blocks.0.attn.to_q.weight":
            torch.zeros(1)}
    with pytest.raises(ImgGenError) as exc:
        FLUXPipeline._from_state(diff, "fake.safetensors")
    assert "convert" not in str(exc.value)
    # SDXL keys → named refusal, not a silent misread.
    sdxl = {"model.diffusion_model.input_blocks.0.0.weight":
            torch.zeros(1),
            "model.diffusion_model.label_emb.0.weight": torch.zeros(1),
            "conditioner.embedders.0.model.transformer.resblocks.0."
            "ln_1.weight": torch.zeros(1)}
    with pytest.raises(ImgGenError, match="SDXL"):
        FLUXPipeline._from_state(sdxl, "fake.safetensors")
    # SD1.5 keys → named refusal.
    sd15 = {"model.diffusion_model.input_blocks.0.0.weight":
            torch.zeros(1),
            "cond_stage_model.transformer.token_embedding.weight":
            torch.zeros(1)}
    with pytest.raises(ImgGenError, match="SD1.5"):
        FLUXPipeline._from_state(sd15, "fake.safetensors")


@needs_torch
def test_flux_pipeline_missing_companion_names_it(tmp_path):
    from nomorals.media.imggen.flux import FLUXPipeline

    state = {"model.diffusion_model.img_in.weight": torch.zeros(1),
             "model.diffusion_model.double_blocks.0.img_attn.qkv.weight":
             torch.zeros(1)}
    with pytest.raises(ImgGenError, match="t5xxl"):
        FLUXPipeline._from_state(state, str(tmp_path / "flux.sft"))


# ---------------------------------------------------------------------------
# Studio dispatch
# ---------------------------------------------------------------------------

def test_studio_dispatches_flux_without_torch(
        model_roots, monkeypatch, tmp_path):
    """The Studio flux branch, proven without torch: a stub flux
    module stands in for the real (torch-guarded) one."""
    import sys
    import types

    from PIL import Image

    from nomorals.media.imggen.studio import Studio, StudioConfig

    ckpt, _home = model_roots
    _write_safetensors(ckpt / "flux-dev.safetensors", FLUX_BFL_KEYS)
    fake_mod = types.ModuleType("nomorals.media.imggen.flux")

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
        model_class = "flux"

        def __init__(self, pipe):
            self._pipe = pipe
            self.last_warnings = []
            self.device = pipe.device

        def generate(self, prompt, cfg=None, conditioning=None):
            return self._pipe.generate(
                prompt, steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                seed=cfg.seed, width=cfg.width, height=cfg.height,
                negative_prompt=cfg.negative_prompt,
                sampler=cfg.sampler)

    fake_mod.FLUXPipeline = _FakePipe
    fake_mod.FLUXStudioAdapter = _FakeAdapter
    monkeypatch.setitem(sys.modules,
                        "nomorals.media.imggen.flux", fake_mod)
    monkeypatch.setattr(
        "nomorals.media.imggen.studio.TORCH_AVAILABLE", True)
    studio = Studio(StudioConfig(checkpoint="flux-dev"))
    pipe = studio.pipeline()
    assert isinstance(pipe, _FakeAdapter)
    out = tmp_path / "out.png"
    paths = studio.generate("a flux cat", steps=3, width=64,
                            height=64, seed=5, save_to=str(out))
    assert paths == [str(out)]
    assert out.exists()
    prompt, kwargs = pipe._pipe.calls[0]
    assert prompt == "a flux cat"
    assert kwargs["steps"] == 3 and kwargs["seed"] == 5


@needs_torch
def test_flux_adapter_gates_are_honest():
    from nomorals.media.imggen.flux import FLUXStudioAdapter
    from nomorals.media.imggen.pipeline import PipelineConfig

    class _RecordingFlux:
        def __init__(self):
            self.device = "cpu"
            self.tokenizer_mode = "hashing-fallback"
            self.warnings = []
            self.calls = []

        def generate(self, prompt, **kwargs):
            from PIL import Image

            self.calls.append((prompt, kwargs))
            return [Image.new("RGB", (kwargs.get("width", 64),
                                      kwargs.get("height", 64)))]

    adapter = FLUXStudioAdapter(_RecordingFlux())
    with pytest.raises(ImgGenError, match="flux"):
        _ = adapter.unet
    with pytest.raises(ImgGenError, match="flux"):
        adapter._scheduler("ddim")
    # FLUX samples with flow-match Euler only; anything else is a
    # refusal naming the sampler that works.
    with pytest.raises(ImgGenError, match="flow"):
        adapter.generate("x", PipelineConfig(sampler="ddpm"))
    with pytest.raises(ImgGenError, match="ControlNet"):
        adapter.generate("x", PipelineConfig(),
                         conditioning=[object()])
    # The PipelineConfig default ("ddim") means "no sampler asked
    # for" — the adapter runs flow-match Euler and says so.
    out = adapter.generate("y", PipelineConfig(
        steps=9, guidance_scale=3.5, seed=2, width=320, height=256,
        negative_prompt="bad"))
    assert len(out) == 1
    _prompt, kwargs = adapter._pipe.calls[-1]
    assert kwargs["steps"] == 9 and kwargs["guidance_scale"] == 3.5
    assert kwargs["width"] == 320 and kwargs["height"] == 256
    assert kwargs["negative_prompt"] == ""  # dropped, not rendered
    assert any("negative prompt" in w for w in adapter.last_warnings)
    assert any("flow" in w for w in adapter.last_warnings)
    out2 = adapter.generate("z", PipelineConfig(sampler="flow"))
    assert len(out2) == 1
    assert adapter._pipe.calls[-1][1]["sampler"] == "flow"


# ---------------------------------------------------------------------------
# Small-config round-trip: the end-to-end proof, through the real
# from_safetensors (file read, section split, companion fallback to
# in-file, key maps, generate loop) at small sizes.
#
# There is deliberately NO full-size round-trip here, unlike A2:
# the dev-size fp32 fixture alone (transformer ~12B + T5-XXL ~11B
# params) would need ~150GB of RAM to build before loading, and no
# machine in this programme has it. The full-size key maps are
# proven instead by the meta-device completeness tests above (dev
# sizes, every key), and the generate loop by the plumbing test.
# A 150GB-gated test that can never run would be a dead test, not
# proof. The live proof still owed is a real FLUX checkpoint
# rendering on a host with the weights + hardware.
# ---------------------------------------------------------------------------

def _needs_safetensors():
    import importlib.util

    return pytest.mark.skipif(
        importlib.util.find_spec("safetensors") is None,
        reason="safetensors not installed")


@needs_torch
@_needs_safetensors()
def test_flux_small_round_trip_generate_64(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    import nomorals.media.imggen.flux as flux_mod
    from nomorals.media.imggen.flux import FluxTransformer, T5TextEncoder
    from nomorals.media.imggen.sdcompat import AutoencoderKL, CLIPTextEncoder

    monkeypatch.setenv("DEVON_IMGGEN_DIR", str(tmp_path / "ig"))
    monkeypatch.setenv("HOME", str(tmp_path))
    torch.manual_seed(0)
    transformer = FluxTransformer(
        hidden_size=64, num_heads=2, depth=1, depth_single_blocks=1,
        in_channels=64, context_in_dim=128, vec_in_dim=64,
        axes_dim=[8, 12, 12])
    t5 = T5TextEncoder(d_model=128, d_ff=256, num_heads=2,
                       num_layers=2, vocab_size=100)
    clip = CLIPTextEncoder(dim=64, layers=2)
    clip.text_projection = torch.nn.Linear(64, 64)
    vae = AutoencoderKL(latent_ch=16, scale=0.3611, shift=0.1159,
                        decoder_resnets=2, diffusers_layout=True)
    state = {}
    state.update({"model.diffusion_model." + k: v
                  for k, v in transformer.state_dict().items()})
    state.update(_t5_to_hf(t5.state_dict()))
    state.update(_clip_to_hf(clip.state_dict()))
    state.update(_vae_to_diffusers(vae.state_dict()))
    path = tmp_path / "small-flux.safetensors"
    save_file(state, str(path), metadata={"format": "pt"})
    del state
    # The loader builds dev-size modules by default; point its
    # transformer at the small one so the small fixture loads.
    monkeypatch.setattr(
        flux_mod, "FluxTransformer",
        lambda *args, **kwargs: FluxTransformer(
            hidden_size=64, num_heads=2, depth=1,
            depth_single_blocks=1, in_channels=64,
            context_in_dim=128, vec_in_dim=64,
            axes_dim=[8, 12, 12]))
    pipe = flux_mod.FLUXPipeline.from_safetensors(
        str(path), device="cpu")
    assert pipe.tokenizer_mode == "hashing-fallback"
    assert pipe.warnings  # disclosed, since no vocab near fixture
    images = pipe.generate("a red square", steps=2, width=64,
                           height=64, seed=1)
    assert len(images) == 1
    assert images[0].size == (64, 64)
