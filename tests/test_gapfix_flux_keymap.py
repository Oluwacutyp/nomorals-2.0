"""FLUX diffusers-transformer key map: the loader speaks diffusers.

Pass B shipped the FLUX loader for the BFL single-file layout
(``double_blocks`` / ``single_blocks``) and honestly refused the
diffusers transformer layout (``transformer_blocks`` /
``single_transformer_blocks``) with a convert message. This file
proves the named follow-up: the loader now maps diffusers keys to
the BFL layout it already speaks, and the registry lists such
checkpoints loadable.

The map was traced from the live diffusers source
(``models/transformers/transformer_flux.py``,
``models/embeddings.py``, ``models/normalization.py``,
``models/attention.py``, ``models/activations.py`` — fetched
2026-11-17) against the in-repo BFL module's own parameter names,
never from memory:

- top level: ``x_embedder.in_proj`` -> ``img_in``,
  ``context_embedder.in_proj`` -> ``txt_in``,
  ``time_text_embed.timestep_embedder.linear_1/2`` -> ``time_in``,
  ``time_text_embed.text_embedder.linear_1/2`` -> ``vector_in``,
  ``time_text_embed.guidance_embedder.linear_1/2`` -> ``guidance_in``,
  ``norm_out.linear`` -> ``final_layer.adaLN_modulation.1`` with the
  two halves SWAPPED (diffusers chunks ``(scale, shift)``; BFL chunks
  ``(shift, scale)``), ``proj_out`` -> ``final_layer.linear``.
- double block ``i``: ``norm1.linear`` -> ``img_mod.lin``,
  ``norm1_context.linear`` -> ``txt_mod.lin`` (chunk order identical),
  ``attn.to_q/to_k/to_v`` concatenated (q, k, v) -> ``img_attn.qkv``,
  ``attn.add_q_proj/add_k_proj/add_v_proj`` -> ``txt_attn.qkv``,
  ``attn.norm_q/k.weight`` -> ``img_attn.norm.query/key_norm.scale``,
  ``attn.norm_added_q/k.weight`` -> the txt norms,
  ``attn.to_out.0`` -> ``img_attn.proj``,
  ``attn.to_add_out`` -> ``txt_attn.proj``,
  ``ff.net.0.proj`` / ``ff.net.2`` -> ``img_mlp.0`` / ``img_mlp.2``
  (``net.1`` is dropout, no weights), ``ff_context`` -> ``txt_mlp``.
- single block ``i``: ``attn.to_q/to_k/to_v`` + ``proj_mlp``
  concatenated in that order -> ``linear1`` (BFL splits it back as
  ``[3 * hidden, mlp_hidden]``), ``attn.norm_q/k.weight`` ->
  ``norm.query/key_norm.scale``, ``norm.linear`` ->
  ``modulation.lin``, ``proj_out`` -> ``linear2`` (exact — the
  diffusers single block is ``pre_only``, so there is no ``to_out``
  to fold; attention output feeds ``proj_out``'s first ``dim``
  columns directly, same as BFL).
- an optional ``model.diffusion_model.`` prefix is accepted on
  diffusers keys, mirroring the BFL mapper.

Same environment honesty as pass B: torch is not installed on this
box, so the loader-construction tests skip here (needs_torch) and
the full-size round-trip is gated on real RAM. What runs here:
registry classification. The meta-device completeness test and the
small-config round-trip through the real ``from_safetensors`` run
where torch is installed.
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


FLUX_DIFFUSERS_KEYS = [
    "transformer_blocks.0.attn.to_q.weight",
    "transformer_blocks.0.ff.net.0.proj.weight",
    "single_transformer_blocks.0.attn.to_q.weight",
    "single_transformer_blocks.0.proj_out.weight",
    "x_embedder.in_proj.weight",
    "time_text_embed.timestep_embedder.linear_1.weight",
    "norm_out.linear.weight",
    "proj_out.weight",
]
FLUX_DIFFUSERS_PREFIXED_KEYS = [
    "model.diffusion_model." + k for k in FLUX_DIFFUSERS_KEYS
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


# ---------------------------------------------------------------------------
# Registry: diffusers-layout FLUX is loadable now
# ---------------------------------------------------------------------------

def test_registry_classifies_flux_diffusers_loadable(model_roots):
    from nomorals.media.imggen.registry import _entry_for_safetensors

    ckpt, _home = model_roots
    path = ckpt / "flux-diffusers.safetensors"
    _write_safetensors(path, FLUX_DIFFUSERS_KEYS)
    entry = _entry_for_safetensors(path)
    assert entry["model_class"] == "flux"
    assert entry["format"] == "diffusers-safetensors"
    assert entry["loadable"] is True
    assert entry["loadable_reason"] == ""


def test_registry_classifies_flux_diffusers_prefixed(model_roots):
    from nomorals.media.imggen.registry import _classify_keys

    model_class, fmt = _classify_keys(
        set(FLUX_DIFFUSERS_PREFIXED_KEYS))
    assert model_class == "flux"
    assert fmt == "diffusers-safetensors"


# ---------------------------------------------------------------------------
# Inverse map (test-side only): BFL state dict -> diffusers keys.
# The forward map under test is _map_diffusers_transformer.
# ---------------------------------------------------------------------------

def _bfl_to_diffusers(sd, prefix=""):
    """FluxTransformer state dict -> diffusers transformer keys.

    Pure renames invert directly; the fused projections split back
    (qkv -> to_q/to_k/to_v; linear1 -> to_q/to_k/to_v + proj_mlp)
    and the final-layer modulation halves swap back (the swap is its
    own inverse).
    """
    out = {}

    def put(bfl, diff):
        if bfl in sd:
            out[prefix + diff] = sd[bfl]

    def split_cat(bfl, diffs):
        """Split one fused weight+bias into N diffusers parts."""
        w = sd[bfl + ".weight"]
        b = sd[bfl + ".bias"]
        rows = w.shape[0]
        n = len(diffs)
        assert rows % n == 0, (bfl, rows, n)
        step = rows // n
        for i, d in enumerate(diffs):
            out[prefix + d + ".weight"] = w[i * step:(i + 1) * step]
            out[prefix + d + ".bias"] = b[i * step:(i + 1) * step]

    def split_linear1(bfl):
        w = sd[bfl + ".weight"]
        b = sd[bfl + ".bias"]
        hidden = w.shape[1]
        qkv, mlp = torch.split(w, [3 * hidden, w.shape[0] - 3 * hidden],
                               dim=0)
        qkv_b, mlp_b = torch.split(b, [3 * hidden,
                                       b.shape[0] - 3 * hidden], dim=0)
        q, k, v = torch.chunk(qkv, 3, dim=0)
        qb, kb, vb = torch.chunk(qkv_b, 3, dim=0)
        return q, k, v, mlp, qb, kb, vb, mlp_b

    def swap_halves(t):
        half = t.shape[0] // 2
        return torch.cat([t[half:], t[:half]], dim=0)

    put("img_in.weight", "x_embedder.in_proj.weight")
    put("img_in.bias", "x_embedder.in_proj.bias")
    put("txt_in.weight", "context_embedder.in_proj.weight")
    put("txt_in.bias", "context_embedder.in_proj.bias")
    put("time_in.in_layer.weight",
        "time_text_embed.timestep_embedder.linear_1.weight")
    put("time_in.in_layer.bias",
        "time_text_embed.timestep_embedder.linear_1.bias")
    put("time_in.out_layer.weight",
        "time_text_embed.timestep_embedder.linear_2.weight")
    put("time_in.out_layer.bias",
        "time_text_embed.timestep_embedder.linear_2.bias")
    put("vector_in.in_layer.weight",
        "time_text_embed.text_embedder.linear_1.weight")
    put("vector_in.in_layer.bias",
        "time_text_embed.text_embedder.linear_1.bias")
    put("vector_in.out_layer.weight",
        "time_text_embed.text_embedder.linear_2.weight")
    put("vector_in.out_layer.bias",
        "time_text_embed.text_embedder.linear_2.bias")
    put("guidance_in.in_layer.weight",
        "time_text_embed.guidance_embedder.linear_1.weight")
    put("guidance_in.in_layer.bias",
        "time_text_embed.guidance_embedder.linear_1.bias")
    put("guidance_in.out_layer.weight",
        "time_text_embed.guidance_embedder.linear_2.weight")
    put("guidance_in.out_layer.bias",
        "time_text_embed.guidance_embedder.linear_2.bias")
    put("final_layer.linear.weight", "proj_out.weight")
    put("final_layer.linear.bias", "proj_out.bias")
    if "final_layer.adaLN_modulation.1.weight" in sd:
        out[prefix + "norm_out.linear.weight"] = swap_halves(
            sd["final_layer.adaLN_modulation.1.weight"])
        out[prefix + "norm_out.linear.bias"] = swap_halves(
            sd["final_layer.adaLN_modulation.1.bias"])

    for k in list(sd):
        if k.startswith("double_blocks."):
            _, i, rest = k.split(".", 2)
            src = f"transformer_blocks.{i}."
            if rest == "img_mod.lin.weight":
                put(k, src + "norm1.linear.weight")
            elif rest == "img_mod.lin.bias":
                put(k, src + "norm1.linear.bias")
            elif rest == "txt_mod.lin.weight":
                put(k, src + "norm1_context.linear.weight")
            elif rest == "txt_mod.lin.bias":
                put(k, src + "norm1_context.linear.bias")
            elif rest == "img_attn.qkv.weight":
                split_cat(f"double_blocks.{i}.img_attn.qkv",
                          [src + "attn.to_q", src + "attn.to_k",
                           src + "attn.to_v"])
            elif rest == "txt_attn.qkv.weight":
                split_cat(f"double_blocks.{i}.txt_attn.qkv",
                          [src + "attn.add_q_proj",
                           src + "attn.add_k_proj",
                           src + "attn.add_v_proj"])
            elif rest == "img_attn.norm.query_norm.scale":
                put(k, src + "attn.norm_q.weight")
            elif rest == "img_attn.norm.key_norm.scale":
                put(k, src + "attn.norm_k.weight")
            elif rest == "txt_attn.norm.query_norm.scale":
                put(k, src + "attn.norm_added_q.weight")
            elif rest == "txt_attn.norm.key_norm.scale":
                put(k, src + "attn.norm_added_k.weight")
            elif rest == "img_attn.proj.weight":
                put(k, src + "attn.to_out.0.weight")
            elif rest == "img_attn.proj.bias":
                put(k, src + "attn.to_out.0.bias")
            elif rest == "txt_attn.proj.weight":
                put(k, src + "attn.to_add_out.weight")
            elif rest == "txt_attn.proj.bias":
                put(k, src + "attn.to_add_out.bias")
            elif rest == "img_mlp.0.weight":
                put(k, src + "ff.net.0.proj.weight")
            elif rest == "img_mlp.0.bias":
                put(k, src + "ff.net.0.proj.bias")
            elif rest == "img_mlp.2.weight":
                put(k, src + "ff.net.2.weight")
            elif rest == "img_mlp.2.bias":
                put(k, src + "ff.net.2.bias")
            elif rest == "txt_mlp.0.weight":
                put(k, src + "ff_context.net.0.proj.weight")
            elif rest == "txt_mlp.0.bias":
                put(k, src + "ff_context.net.0.proj.bias")
            elif rest == "txt_mlp.2.weight":
                put(k, src + "ff_context.net.2.weight")
            elif rest == "txt_mlp.2.bias":
                put(k, src + "ff_context.net.2.bias")
        elif k.startswith("single_blocks."):
            _, i, rest = k.split(".", 2)
            src = f"single_transformer_blocks.{i}."
            if rest == "linear1.weight":
                q, kk, v, mlp, qb, kb, vb, mlp_b = split_linear1(
                    f"single_blocks.{i}.linear1")
                out[prefix + src + "attn.to_q.weight"] = q
                out[prefix + src + "attn.to_k.weight"] = kk
                out[prefix + src + "attn.to_v.weight"] = v
                out[prefix + src + "proj_mlp.weight"] = mlp
                out[prefix + src + "attn.to_q.bias"] = qb
                out[prefix + src + "attn.to_k.bias"] = kb
                out[prefix + src + "attn.to_v.bias"] = vb
                out[prefix + src + "proj_mlp.bias"] = mlp_b
            elif rest == "linear2.weight":
                put(k, src + "proj_out.weight")
            elif rest == "linear2.bias":
                put(k, src + "proj_out.bias")
            elif rest == "modulation.lin.weight":
                put(k, src + "norm.linear.weight")
            elif rest == "modulation.lin.bias":
                put(k, src + "norm.linear.bias")
            elif rest == "norm.query_norm.scale":
                put(k, src + "attn.norm_q.weight")
            elif rest == "norm.key_norm.scale":
                put(k, src + "attn.norm_k.weight")
    return out


# ---------------------------------------------------------------------------
# Key map: complete both ways, values exact
# ---------------------------------------------------------------------------

@needs_torch
def test_diffusers_transformer_key_map_is_complete_meta():
    from nomorals.media.imggen.flux import FluxTransformer, _map_diffusers_transformer

    with torch.device("meta"):
        m = FluxTransformer()
    sd = m.state_dict()
    for prefix in ("model.diffusion_model.", ""):
        diff = _bfl_to_diffusers(sd, prefix)
        mapped, consumed = _map_diffusers_transformer(diff)
        # Every target parameter filled, every source key consumed.
        assert set(mapped) == set(sd)
        assert consumed == set(diff)
        with torch.device("meta"):
            fresh = FluxTransformer()
        fresh.load_state_dict(mapped, strict=True)  # must not raise


@needs_torch
def test_diffusers_transformer_key_map_values_exact_small():
    from nomorals.media.imggen.flux import FluxTransformer, _map_diffusers_transformer

    torch.manual_seed(0)
    m = FluxTransformer(hidden_size=64, num_heads=2, depth=1,
                        depth_single_blocks=1, in_channels=64,
                        context_in_dim=128, vec_in_dim=64,
                        axes_dim=[8, 12, 12])
    sd = m.state_dict()
    diff = _bfl_to_diffusers(sd)
    mapped, _consumed = _map_diffusers_transformer(diff)
    assert set(mapped) == set(sd)
    for k, v in sd.items():
        assert torch.equal(mapped[k], v), k
    # The norm_out half-swap, pinned with distinct halves: diffusers
    # rows [scale; shift] must land as BFL rows [shift; scale].
    w = torch.zeros(2 * 64, 64)
    w[:64] = 1.0  # scale half
    w[64:] = 2.0  # shift half
    b = torch.cat([torch.ones(64), torch.full((64,), 2.0)])
    diff["norm_out.linear.weight"] = w
    diff["norm_out.linear.bias"] = b
    mapped, _consumed = _map_diffusers_transformer(diff)
    got_w = mapped["final_layer.adaLN_modulation.1.weight"]
    got_b = mapped["final_layer.adaLN_modulation.1.bias"]
    assert torch.equal(got_w[:64], torch.full((64, 64), 2.0))
    assert torch.equal(got_w[64:], torch.full((64, 64), 1.0))
    assert torch.equal(got_b[:64], torch.full((64,), 2.0))
    assert torch.equal(got_b[64:], torch.full((64,), 1.0))


@needs_torch
def test_diffusers_transformer_missing_and_extra_keys_are_errors():
    from nomorals.media.imggen.flux import FluxTransformer, _map_diffusers_transformer

    with torch.device("meta"):
        m = FluxTransformer()
    diff = _bfl_to_diffusers(m.state_dict())
    broken = dict(diff)
    del broken["transformer_blocks.3.attn.to_q.weight"]
    with pytest.raises(ImgGenError, match="double_blocks.3"):
        _map_diffusers_transformer(broken)
    extra = dict(diff)
    extra["transformer_blocks.0.bogus.weight"] = \
        next(iter(diff.values()))
    with pytest.raises(ImgGenError, match="no target"):
        _map_diffusers_transformer(extra)


# ---------------------------------------------------------------------------
# Small-config round-trip: the end-to-end proof, through the real
# from_safetensors (file read, section split, companion fallback to
# in-file, diffusers key map, generate loop) at small sizes.
#
# Same 150GB gate as pass B: no full-size round-trip — the dev-size
# fp32 fixture alone would need ~150GB of RAM to build before
# loading, and no machine in this programme has it. The full-size
# key map is proven instead by the meta-device completeness test
# above (dev sizes, every key), and the generate loop by the
# plumbing test. The live proof still owed is a real diffusers
# FLUX checkpoint rendering on a host with the weights + hardware.
# ---------------------------------------------------------------------------

def _t5_to_hf(sd):
    """T5TextEncoder state dict -> transformers T5EncoderModel keys.

    Mirrors tests/test_gapfix_imgstudio_b.py (the companions are
    unchanged; only the transformer section moves to diffusers).
    """
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


def _clip_to_hf(sd):
    """CLIPTextEncoder state dict -> transformers CLIPTextModel keys.

    Mirrors tests/test_gapfix_imgstudio_b.py.
    """
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


def _vae_to_diffusers(sd):
    """AutoencoderKL state dict -> diffusers AutoencoderKL keys.

    Mirrors tests/test_gapfix_imgstudio_b.py.
    """
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


def _needs_safetensors():
    import importlib.util

    return pytest.mark.skipif(
        importlib.util.find_spec("safetensors") is None,
        reason="safetensors not installed")


@needs_torch
@_needs_safetensors()
def test_flux_diffusers_small_round_trip_generate_64(
        tmp_path, monkeypatch):
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
    # The transformer section in DIFFUSERS layout; the companions
    # in their usual HF layouts, in-file as in pass B.
    state.update(_bfl_to_diffusers(transformer.state_dict()))
    state.update(_t5_to_hf(t5.state_dict()))
    state.update(_clip_to_hf(clip.state_dict()))
    state.update(_vae_to_diffusers(vae.state_dict()))
    path = tmp_path / "small-flux-diffusers.safetensors"
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
