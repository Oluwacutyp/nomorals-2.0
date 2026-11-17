"""Stable Diffusion XL compatibility (compvis LDM format).

The pipeline code is ours; weights are data. This module implements
the SDXL UNet, its dual CLIP text encoders (ViT-L/14 + ViT-bigG/14),
and the SDXL micro-conditioning (pooled text + time-ids) **from
their documented architectures** and loads ``.safetensors``
checkpoints in the original compvis/LDM key format
(``model.diffusion_model.*``, ``conditioner.embedders.{0,1}.*``,
``first_stage_model.*``) — the format CivitAI and most HF repos ship.

What this unlocks: any SDXL checkpoint becomes a Devon pipeline
backend with zero diffusers dependency, reached from Studio the same
way SD1.5 is (``Studio`` routes ``model_class == "sdxl"`` here).

Architecture, from sd_xl_base.yaml (verified against the live
stabilityai/stable-diffusion-xl-base-1.0 config):

- UNet: channels 320 → 640 → 1280, ``num_res_blocks`` 2 per level,
  ``attention_resolutions`` [4, 2] — attention ONLY at the 64×64 and
  32×32 levels; ``transformer_depth`` [1, 2, 10] with the first
  entry UNUSED (the config itself notes this). So the
  full-resolution level carries NO transformer at all, and the
  depths are 0 / 2 / 10 down, 10 / 2 / 0 up, 10 mid. Heads are
  channels // 64 (5 / 10 / 20).
- Conditioning: CLIP-L penultimate (768) + CLIP-G penultimate
  (1280) concatenated → 2048-wide cross-attention context; CLIP-G's
  projected EOT output (1280) is the pooled text embedding.
- Micro-conditioning: ``label_emb`` is Linear(2816→1280), SiLU,
  Linear(1280→1280); its input is [pooled (1280), time-ids (1536)]
  — the 6 time-ids (original h/w, crop top/left, target h/w) each
  embedded to 256 dims (128 cos + 128 sin, base 10000), the same
  order diffusers uses (``torch.concat([text_embeds, time_embeds])``).
- VAE: the SD VAE layout at scale 0.13025 (not SD1.5's 0.18215).

Honest scope: the full SDXL text-to-image loader (UNet + both
encoders + VAE + DDIM). No ControlNet — the adapter refuses it by
name, as the SD1.5 adapter does. No diffusers-format keys — the
error names the one-time convert, as the SD1.5 loader does.
"""

from __future__ import annotations

import math

from . import TORCH_AVAILABLE, ImgGenError

if not TORCH_AVAILABLE:  # pragma: no cover - torch missing
    raise ImgGenError(
        "nomorals.media.imggen.sdxl needs PyTorch: pip install torch")

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sdcompat import (
    AutoencoderKL,
    CLIPTextEncoder,
    CrossAttnDownBlock,
    CrossAttnUpBlock,
    DownBlock,
    ResnetBlock,
    SDUNet,
    SpatialTransformer,
    UpBlock,
    _check_section,
    _KeyMapper,
    _load_ldm_clip,
    _load_ldm_vae,
    _map_transformer,
    load_safetensors,
)

__all__ = [
    "SDXLUNet",
    "SDXLPipeline",
    "SDXLStudioAdapter",
    "_load_ldm_xl_conditioner",
    "_map_ldm_sdxl_unet",
    "_time_ids_embedding",
]

#: Transformer depths per level, from sd_xl_base.yaml (see module
#: docstring): the full-resolution level has no attention at all.
SDXL_DEPTHS = {"down0": 0, "down1": 2, "down2": 10, "mid": 10,
               "up0": 10, "up1": 2, "up2": 0}


def _time_ids_embedding(time_ids: torch.Tensor) -> torch.Tensor:
    """Sinusoidal embedding of SDXL's 6 time-ids → (B, 1536).

    Matches diffusers' ``Timesteps(256, flip_sin_to_cos=True)`` on
    the flattened ids: each scalar gets 128 cos + 128 sin (base
    10000, freq_shift 0); the six 256-dim vectors concatenate to
    1536. The pipeline then concatenates [pooled (1280), time
    (1536)] → 2816, the label_emb input width — the same order
    diffusers uses (``torch.concat([text_embeds, time_embeds])``).
    """
    half = 128
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=time_ids.device).float()
                      / half)
    flat = time_ids.reshape(-1).float()
    args = flat.unsqueeze(1) * freqs.unsqueeze(0)  # (6B, 128)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return emb.reshape(time_ids.shape[0], -1)


# ---------------------------------------------------------------------------
# SDXL UNet
# ---------------------------------------------------------------------------

class SDXLUNet(nn.Module):
    """Stable Diffusion XL UNet — our implementation, open weights.

    Channels: 320 → 640 → 1280; cross-attention context 2048
    (concatenated CLIP-L 768 + CLIP-G 1280 penultimate states).
    Time embedding 320 → 1280; label embedding 2816 → 1280
    (pooled text 1280 + time-ids 6×256). Latent in/out: 4 ch.

    Attention lives only at the 64×64 and 32×32 levels
    (sd_xl_base.yaml attention_resolutions [4, 2]): down0 and up2
    are resnet-only; the transformer depths are 2 / 10 / 10 / 2
    (down1, down2, mid, up0, up1).
    """

    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 4
        self.conv_in = nn.Conv2d(4, 320, 3, padding=1)
        self.time_embed = nn.Sequential(
            nn.Linear(320, 1280), nn.SiLU(), nn.Linear(1280, 1280))
        self.label_emb = nn.Sequential(
            nn.Linear(2816, 1280), nn.SiLU(), nn.Linear(1280, 1280))

        # down0 is resnet-only (no attention at full resolution);
        # down1/depth 2, down2/depth 10. Only down0/down1 downsample.
        self.down0 = DownBlock(320, 320, 1280, layers=2,
                               downsample=True)
        self.down1 = CrossAttnDownBlock(320, 640, 1280, 10, layers=2,
                                        context_dim=2048, down=True,
                                        depth=2)
        self.down2 = CrossAttnDownBlock(640, 1280, 1280, 20, layers=2,
                                        context_dim=2048, down=False,
                                        depth=10)

        self.mid_res1 = ResnetBlock(1280, 1280, 1280)
        self.mid_trans = SpatialTransformer(1280, 20, 10, 2048)
        self.mid_res2 = ResnetBlock(1280, 1280, 1280)

        # Skip channels: up0 takes mid(1280)+down2(1280)=2560;
        # up1 takes 1280+down1(640)=1920; up2 takes 640+down0(320)=960.
        # Only up0/up1 upsample; up2 is resnet-only.
        self.up0 = CrossAttnUpBlock(2560, 1280, 1280, 20, layers=3,
                                    context_dim=2048, upsample=True,
                                    depth=10)
        self.up1 = CrossAttnUpBlock(1920, 640, 1280, 10, layers=3,
                                    context_dim=2048, upsample=True,
                                    depth=2)
        self.up2 = UpBlock(960, 320, 1280, layers=3, upsample=False)

        self.conv_norm_out = nn.GroupNorm(32, 320)
        self.conv_out = nn.Conv2d(320, 4, 3, padding=1)

    def forward(self, x, t, context, pooled=None, time_ids=None):
        if pooled is None or time_ids is None:
            raise ImgGenError(
                "SDXL UNet called without pooled text embeddings or "
                "time-ids — the pipeline always supplies both; a "
                "direct call must too")
        temb = self.time_embed(SDUNet.timestep_embedding(t))
        adm = self.label_emb(torch.cat(
            [pooled, _time_ids_embedding(time_ids)], dim=1))
        temb = temb + adm
        h = self.conv_in(x)
        skips = []
        h, s = self.down0(h, temb)
        skips.append(s)
        h, s = self.down1(h, temb, context)
        skips.append(s)
        h, s = self.down2(h, temb, context)
        skips.append(s)
        h = self.mid_res1(h, temb)
        h = self.mid_trans(h, context)
        h = self.mid_res2(h, temb)
        # Skips pop in reverse: down2, down1, down0.
        h = self.up0(h, skips[2], temb, context)
        h = self.up1(h, skips[1], temb, context)
        h = self.up2(h, skips[0], temb)
        h = self.conv_norm_out(h)
        return self.conv_out(F.silu(h))


# ---------------------------------------------------------------------------
# LDM key mapping
# ---------------------------------------------------------------------------

_RESNET_KEYS = ("in_layers.0.weight", "in_layers.0.bias",
                "in_layers.2.weight", "in_layers.2.bias",
                "emb_layers.1.weight", "emb_layers.1.bias",
                "out_layers.0.weight", "out_layers.0.bias",
                "out_layers.3.weight", "out_layers.3.bias",
                "skip_connection.weight", "skip_connection.bias")


def _map_resnet(prefix: str, out_prefix: str, put) -> None:
    """Map one LDM resnet block → our ResnetBlock (shared key renames)."""
    for k in _RESNET_KEYS:
        ours_k = (k.replace("in_layers.0.", "norm1.")
                  .replace("in_layers.2.", "conv1.")
                  .replace("emb_layers.1.", "temb_proj.1.")
                  .replace("out_layers.0.", "norm2.")
                  .replace("out_layers.3.", "conv2.")
                  .replace("skip_connection.", "shortcut."))
        put(prefix + k, out_prefix + ours_k)


def _map_ldm_sdxl_unet(state: dict[str, torch.Tensor]):
    """Map compvis LDM SDXL UNet keys → our SDXLUNet attribute paths.

    LDM layout: input_blocks 0..8 (conv_in; two resnets per level;
    a downsample block after levels 0 and 1 only — the last level
    has none), middle_block (0,1,2), output_blocks 0..8. Our layout:
    down0..down2, mid_*, up0..up2. Returns (mapped_state_dict,
    consumed_source_keys).
    """
    mapper = _KeyMapper(state)
    put = mapper.put
    out = mapper.mapped

    put("model.diffusion_model.time_embed.0.weight",
        "time_embed.0.weight")
    put("model.diffusion_model.time_embed.0.bias", "time_embed.0.bias")
    put("model.diffusion_model.time_embed.2.weight",
        "time_embed.2.weight")
    put("model.diffusion_model.time_embed.2.bias", "time_embed.2.bias")
    put("model.diffusion_model.label_emb.0.weight",
        "label_emb.0.weight")
    put("model.diffusion_model.label_emb.0.bias", "label_emb.0.bias")
    put("model.diffusion_model.label_emb.2.weight",
        "label_emb.2.weight")
    put("model.diffusion_model.label_emb.2.bias", "label_emb.2.bias")
    put("model.diffusion_model.input_blocks.0.0.weight", "conv_in.weight")
    put("model.diffusion_model.input_blocks.0.0.bias", "conv_in.bias")

    # input_blocks 1..8 → down0..down2.
    # down0: blocks 1,2 (320ch, resnet-only, downsample at 3)
    # down1: blocks 4,5 (640ch, transformer depth 2, downsample at 6)
    # down2: blocks 7,8 (1280ch, transformer depth 10, no downsample)
    down_map = [
        ("down0", [1, 2], 3, 0), ("down1", [4, 5], 6, 2),
        ("down2", [7, 8], None, 10),
    ]
    for dname, blocks, ds_block, depth in down_map:
        for li, bi in enumerate(blocks):
            _map_resnet(
                f"model.diffusion_model.input_blocks.{bi}.0.",
                f"{dname}.resnets.{li}.", put)
            if depth:
                _map_transformer(
                    state,
                    f"model.diffusion_model.input_blocks.{bi}.1.",
                    f"{dname}.transformers.{li}.", put, depth=depth)
        if ds_block is not None:
            put("model.diffusion_model.input_blocks."
                f"{ds_block}.0.op.weight", f"{dname}.downsampler.weight")
            put("model.diffusion_model.input_blocks."
                f"{ds_block}.0.op.bias", f"{dname}.downsampler.bias")

    # Middle block.
    _map_resnet("model.diffusion_model.middle_block.0.", "mid_res1.",
                put)
    _map_transformer(state, "model.diffusion_model.middle_block.1.",
                     "mid_trans.", put, depth=10)
    _map_resnet("model.diffusion_model.middle_block.2.", "mid_res2.",
                put)

    # Output blocks 0..8 → up0..up2 (3 resnet+transformer pairs each,
    # except up2 which is resnet-only).
    # up0←0,1,2 (2560→1280, depth 10); up1←3,4,5 (1920→640, depth 2);
    # up2←6,7,8 (960→320, no transformer, no upsample).
    # The LDM upsample conv of each group's last block
    # (output_blocks.{2,5}.2.conv) is our group's final
    # ``upsampler`` conv — same position in the chain.
    up_map = [
        ("up0", [0, 1, 2], 10), ("up1", [3, 4, 5], 2),
        ("up2", [6, 7, 8], 0),
    ]
    for uname, blocks, depth in up_map:
        for li, bi in enumerate(blocks):
            _map_resnet(
                f"model.diffusion_model.output_blocks.{bi}.0.",
                f"{uname}.resnets.{li}.", put)
            if depth:
                _map_transformer(
                    state,
                    f"model.diffusion_model.output_blocks.{bi}.1.",
                    f"{uname}.transformers.{li}.", put, depth=depth)
        if uname != "up2":
            last = blocks[-1]
            put("model.diffusion_model.output_blocks."
                f"{last}.2.conv.weight", f"{uname}.upsampler.weight")
            put("model.diffusion_model.output_blocks."
                f"{last}.2.conv.bias", f"{uname}.upsampler.bias")

    put("model.diffusion_model.out.0.weight", "conv_norm_out.weight")
    put("model.diffusion_model.out.0.bias", "conv_norm_out.bias")
    put("model.diffusion_model.out.2.weight", "conv_out.weight")
    put("model.diffusion_model.out.2.bias", "conv_out.bias")
    return out, mapper.consumed


def _load_ldm_xl_conditioner(state: dict[str, torch.Tensor]):
    """Load SDXL's two text encoders from conditioner.embedders.*.

    embedders.0 = OpenAI CLIP ViT-L/14 (12 layers, 768) — its
    text_projection is known-unused (SDXL never uses CLIP-L's pooled
    output). embedders.1 = OpenCLIP ViT-bigG/14 (32 layers, 1280) —
    its text_projection IS the pooled output and is loaded. Returns
    (enc_l, enc_g). Each encoder is completeness-checked by
    :func:`._load_ldm_clip`: every target filled, every key consumed.
    """
    enc_l = CLIPTextEncoder()
    _load_ldm_clip(state, enc_l, root="conditioner.embedders.0.model.",
                   layers=12)
    enc_g = CLIPTextEncoder(dim=1280, layers=32)
    enc_g.text_projection = nn.Linear(1280, 1280)
    _load_ldm_clip(state, enc_g, root="conditioner.embedders.1.model.",
                   layers=32, map_projection=True)
    return enc_l, enc_g


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class SDXLPipeline:
    """Text-to-image with SDXL open weights in Devon's own code.

    UNet operates in VAE latent space (128×128×4 for 1024px); the
    VAE decodes to pixels. Same DDIM scheduler math as the native
    pipeline. Dual text encoders: CLIP-L + CLIP-G penultimate states
    concatenated (2048) for cross-attention; CLIP-G's projected EOT
    output + the 6 time-ids micro-condition the UNet via label_emb.
    """

    def __init__(self, unet: SDXLUNet, text_encoder_l: CLIPTextEncoder,
                 text_encoder_g: CLIPTextEncoder, vae: AutoencoderKL,
                 timesteps: int = 1000, device: str = "") -> None:
        self.unet = unet.eval()
        self.text_encoder_l = text_encoder_l.eval()
        self.text_encoder_g = text_encoder_g.eval()
        self.vae = vae.eval()
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "cpu")
        for m in (self.unet, self.text_encoder_l, self.text_encoder_g,
                  self.vae):
            m.to(self.device)
            for p in m.parameters():
                p.requires_grad_(False)
        self.timesteps = timesteps

    @property
    def tokenizer_mode(self) -> str:
        """How prompts are tokenized: ``"bpe"`` or the fallback.

        ``"bpe"`` only when BOTH encoders have a real tokenizer; a
        single hashing-fallback encoder would silently mix token
        spaces, so the pipeline reports the fallback unless both
        are BPE.
        """
        if (self.text_encoder_l.tokenizer is not None
                and self.text_encoder_g.tokenizer is not None):
            return "bpe"
        return "hashing-fallback"

    @property
    def warnings(self) -> list[str]:
        """Standing disclosures about this pipeline's fidelity."""
        if self.tokenizer_mode == "bpe":
            return []
        return [
            "tokenizer: hashing fallback active — no CLIP BPE vocab "
            "found near this checkpoint, so prompts are tokenized "
            "approximately and fidelity is reduced. Add vocab.json + "
            "merges.txt (or tokenizer.json) next to the checkpoint "
            "or in ~/.devon-models/tokenizer/ for true CLIP BPE."]

    @classmethod
    def from_safetensors(cls, path: str, device: str = ""
                         ) -> SDXLPipeline:
        """Load an SDXL .safetensors checkpoint (compvis key format)."""
        state = load_safetensors(path)
        return cls._from_state(state, path, device=device)

    @classmethod
    def _from_state(cls, state, path: str, device: str = ""
                    ) -> SDXLPipeline:
        keys = set(state)
        is_ldm = any(k.startswith("model.diffusion_model.")
                     for k in keys)
        is_diffusers = any(k.startswith("unet.") for k in keys)
        is_sdxl = (any("conditioner.embedders.1" in k for k in keys)
                   or any(k.startswith("model.diffusion_model.label_emb.")
                          for k in keys))
        if not is_ldm and not is_diffusers:
            raise ImgGenError(
                f"{path}: unrecognized SD key format "
                "(need compvis LDM or diffusers SDXL keys)")
        if is_diffusers and not is_ldm:
            raise ImgGenError(
                f"{path}: diffusers-format keys detected — convert to "
                "compvis LDM format first (one-time script; the loader "
                "speaks LDM natively)")
        if not is_sdxl:
            raise ImgGenError(
                f"{path}: this looks like SD1.5, not SDXL — the SDXL "
                "loader needs conditioner.embedders.0/1 + label_emb; "
                "load it with SD15Pipeline instead")
        unet = SDXLUNet()
        mapped, consumed = _map_ldm_sdxl_unet(state)
        missing, _unexpected = unet.load_state_dict(mapped, strict=False)
        if missing:
            raise ImgGenError(
                f"{path}: {len(missing)} UNet weights missing "
                f"(e.g. {missing[0]}); is this really SDXL?")
        _check_section(state, "model.diffusion_model.", consumed,
                       f"{path} (UNet)")
        enc_l, enc_g = _load_ldm_xl_conditioner(state)
        # Real CLIP BPE if vocab data sits near the checkpoint; the
        # hashing fallback otherwise — disclosed via .warnings.
        from .tokenizers import load_clip_tokenizer

        enc_l.tokenizer = load_clip_tokenizer(path)
        enc_g.tokenizer = load_clip_tokenizer(path)
        vae = AutoencoderKL(scale=0.13025)
        _load_ldm_vae(state, vae)
        return cls(unet, enc_l, enc_g, vae, device=device)

    def _encode(self, prompt: str, negative_prompt: str = "",
                width: int = 1024, height: int = 1024):
        """(context, pooled, time_ids) for one prompt pair.

        Context = concat(CLIP-L penultimate, CLIP-G penultimate)
        (B, 77, 2048); pooled = CLIP-G projected EOT (B, 1280);
        time-ids = [orig_h, orig_w, crop_top, crop_left,
        target_h, target_w] — no crop, target = render size, the
        standard micro-conditioning defaults.
        """
        ctx_l, _pl = self.text_encoder_l.encode_states([prompt])
        ctx_g, pooled_g = self.text_encoder_g.encode_states([prompt])
        context = torch.cat([ctx_l, ctx_g], dim=-1)
        pooled = pooled_g
        if negative_prompt:
            nctx_l, _npl = self.text_encoder_l.encode_states(
                [negative_prompt])
            nctx_g, npooled_g = self.text_encoder_g.encode_states(
                [negative_prompt])
            context = torch.cat(
                [context, torch.cat([nctx_l, nctx_g], dim=-1)], dim=0)
            pooled = torch.cat([pooled, npooled_g], dim=0)
        n = context.shape[0]
        time_ids = torch.zeros(n, 6, device=context.device,
                               dtype=context.dtype)
        time_ids[:, 0] = height
        time_ids[:, 1] = width
        time_ids[:, 4] = height
        time_ids[:, 5] = width
        return context, pooled, time_ids

    @torch.no_grad()
    def generate(self, prompt: str, steps: int = 30,
                 guidance_scale: float = 7.5, seed: int | None = None,
                 width: int = 1024, height: int = 1024,
                 negative_prompt: str = "", eta: float = 0.0) -> list:
        """Latent diffusion sampling → PIL images."""
        from PIL import Image

        from .diffusion import DDIMScheduler

        sched = DDIMScheduler(timesteps=self.timesteps,
                              schedule="linear")
        lw, lh = width // 8, height // 8
        context, pooled, time_ids = self._encode(
            prompt, negative_prompt, width, height)
        context = context.to(self.device)
        pooled = pooled.to(self.device)
        time_ids = time_ids.to(self.device)
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        xt = torch.randn((1, 4, lh, lw), generator=g,
                         device=self.device)
        ts = torch.linspace(self.timesteps - 1, 0, steps,
                            dtype=torch.long)
        for t in ts:
            ti = int(t)
            t_batch = torch.full((2,), ti, dtype=torch.long,
                                 device=self.device)
            x_in = torch.cat([xt, xt], dim=0)
            eps_c, eps_u = self.unet(x_in, t_batch, context, pooled,
                                     time_ids).chunk(2)
            eps = eps_u + guidance_scale * (eps_c - eps_u)
            # Bind eps as a default arg: the lambda runs later, and a
            # bare closure would read the loop's final value.
            xt = sched.p_sample(lambda _x, _t, _eps=eps: _eps, xt, ti,
                                eta=eta)
        img = self.vae.decode(xt)
        arr = (img[0].clamp(-1, 1) * 0.5 + 0.5) * 255
        arr = arr.byte().permute(1, 2, 0).cpu().numpy()
        return [Image.fromarray(arr)]


# ---------------------------------------------------------------------------
# Studio protocol adapter
# ---------------------------------------------------------------------------

class SDXLStudioAdapter:
    """Adapt :class:`SDXLPipeline` to the protocol Studio speaks.

    Studio (and ``edit.py`` / ``upscale.py`` through it) drives a
    pipeline as ``generate(prompt, PipelineConfig)`` plus attribute
    access (``.device``, ``.unet``, ``.text_encoder``,
    ``._scheduler()``, ``._to_pil()``). SDXL generation maps onto
    that cleanly; the *edit* operations do not — they are written
    against the native pipeline's pixel-space UNet, while SDXL
    edits belong in VAE latent space. Rather than fake them, every
    edit-protocol surface raises :class:`ImgGenError` naming the
    operation and the model class; latent-space SDXL editing is its
    own later piece of work.
    """

    model_class = "sdxl"

    def __init__(self, pipeline: SDXLPipeline) -> None:
        self._pipe = pipeline
        #: Disclosures from the most recent generate() call; Studio
        #: reads this to surface them (sidecar + log).
        self.last_warnings: list[str] = list(pipeline.warnings)

    # -- honest passthroughs -------------------------------------------
    @property
    def device(self) -> str:
        return self._pipe.device

    @property
    def tokenizer_mode(self) -> str:
        return self._pipe.tokenizer_mode

    @property
    def warnings(self) -> list[str]:
        return list(self._pipe.warnings)

    @property
    def controlnets(self) -> dict:
        return {}

    # -- generation ------------------------------------------------------
    def generate(self, prompt, cfg=None, conditioning=None) -> list:
        """PipelineConfig → SDXLPipeline.generate kwargs, faithfully."""
        from .pipeline import PipelineConfig

        cfg = cfg or PipelineConfig()
        if conditioning:
            raise ImgGenError(
                "ControlNet conditioning is not available for SDXL "
                "(sdxl) checkpoints through the Studio adapter in "
                "this pass — the SD pipeline has no ControlNet branch "
                "wired; refusing to silently ignore the control input")
        if cfg.sampler != "ddim":
            raise ImgGenError(
                f"SDXL (sdxl) checkpoints sample with DDIM only; "
                f"sampler {cfg.sampler!r} is not available for this "
                f"model class")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        out: list = []
        for text in prompts:
            for i in range(max(1, cfg.batch_size)):
                seed = None if cfg.seed is None else cfg.seed + i
                out.extend(self._pipe.generate(
                    text, steps=cfg.steps,
                    guidance_scale=cfg.guidance_scale, seed=seed,
                    width=cfg.width, height=cfg.height,
                    negative_prompt=cfg.negative_prompt, eta=cfg.eta))
        self.last_warnings = list(self._pipe.warnings)
        return out

    # -- edit protocol: honest gates, never fake results -----------------
    def _edit_gate(self, what: str):
        raise ImgGenError(
            f"{what} is not available for SDXL (sdxl) checkpoints "
            f"through the Studio adapter: Devon's edit operations run "
            f"against the native pixel-space pipeline, and SDXL "
            f"img2img/inpaint/outpaint need a latent-space adapter "
            f"that does not exist yet. Text-to-image works; the edit "
            f"was refused rather than faked.")

    @property
    def unet(self):
        return self._edit_gate("the UNet (edit protocol)")

    @property
    def text_encoder(self):
        return self._edit_gate("the text encoder (edit protocol)")

    def _scheduler(self, sampler: str):
        return self._edit_gate(f"the {sampler} scheduler "
                               f"(edit protocol)")

    def _to_pil(self, tensor):
        return self._edit_gate("tensor→PIL conversion (edit protocol)")

    def attach_controlnet(self, kind, controlnet):
        return self._edit_gate(f"ControlNet attach ({kind})")

    def attach_diffusers_controlnet(self, kind, diffusers_controlnet):
        return self._edit_gate(f"ControlNet attach ({kind})")
