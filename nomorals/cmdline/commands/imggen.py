"""``nm imggen`` — Devon's own image studio on the command line.

    nm imggen generate "a danfo bus at sunset" [--steps 50] [--seed 42]
                     [--ar 16:9] [--negative "blurry"] [--batch 4]
    nm imggen img2img in.png "make it sunset" --strength 0.6
    nm imggen inpaint in.png mask.png "a cat"
    nm imggen outpaint in.png "city skyline" --right 200
    nm imggen upscale in.png --scale 2 [--diffusion]
    nm imggen train --data ./my_photos [--steps 5000] [--size 64]
    nm imggen lora-train --data ./style --base <ckpt> --rank 8
    nm imggen checkpoints [--json]
    nm imggen dashboard <run>
    nm imggen merge base.pt other.pt --alpha 0.5
    nm imggen character "anchor prompt" "pose 1" "pose 2" --seed 42

Every subcommand degrades honestly when torch is missing or no
checkpoint exists — the error says what to do, never a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys


def _err(msg: str) -> int:
    print(f"imggen: {msg}", file=sys.stderr)
    return 1


def _studio(args) -> object:
    from ...media.imggen.studio import Studio, StudioConfig

    return Studio(StudioConfig(
        checkpoint=args.checkpoint or "",
        device=args.device or "",
    ))


def _cmd_imggen_generate(args: argparse.Namespace, context) -> int:
    try:
        studio = _studio(args)
        kwargs = dict(
            steps=args.steps, guidance_scale=args.guidance,
            seed=args.seed, batch_size=args.batch,
            negative_prompt=args.negative or "",
            sampler=args.sampler,
        )
        if args.ar:
            from ...media.imggen.pipeline import resolve_aspect_ratio

            w, h = resolve_aspect_ratio(args.ar,
                                        base=args.size)
            kwargs["width"], kwargs["height"] = w, h
        else:
            kwargs["width"], kwargs["height"] = args.size, args.size
        paths = studio.generate(args.prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001 - CLI surfaces as text
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


def _cmd_imggen_img2img(args: argparse.Namespace, context) -> int:
    try:
        studio = _studio(args)
        paths = studio.img2img(args.image, args.prompt,
                               strength=args.strength,
                               steps=args.steps, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


def _cmd_imggen_inpaint(args: argparse.Namespace, context) -> int:
    try:
        studio = _studio(args)
        paths = studio.inpaint(args.image, args.mask, args.prompt,
                               steps=args.steps, seed=args.seed)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


def _cmd_imggen_outpaint(args: argparse.Namespace, context) -> int:
    try:
        studio = _studio(args)
        paths = studio.outpaint(
            args.image, args.prompt, steps=args.steps,
            seed=args.seed, left=args.left, right=args.right,
            top=args.top, bottom=args.bottom)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


def _cmd_imggen_upscale(args: argparse.Namespace, context) -> int:
    try:
        studio = _studio(args)
        paths = studio.upscale(args.image, scale=args.scale,
                               diffusion=args.diffusion,
                               prompt=args.prompt or "")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


def _cmd_imggen_train(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen import TORCH_AVAILABLE
        from ...media.imggen.data import (ImageFolderDataset,
                                          build_dataset_manifest)
        from ...media.imggen.pipeline import TextEncoder
        from ...media.imggen.train import (TrainConfig,
                                           train_text_conditional)
        from ...media.imggen.unet import UNet, count_parameters
    except Exception as exc:  # noqa: BLE001 - missing torch
        return _err(str(exc))
    if not TORCH_AVAILABLE:
        return _err("training needs torch: pip install torch")
    try:
        print(f"building dataset manifest for {args.data} ...")
        summary = build_dataset_manifest(args.data)
        print(f"  {summary['images']} images "
              f"({summary['vision_captions']} vision-captioned, "
              f"{summary['filename_captions']} filename-captioned)")
        from torch.utils.data import DataLoader

        ds = ImageFolderDataset(summary["manifest"],
                                image_size=args.size)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=0)

        # Profile-aware defaults: termux gets the tiny model.
        try:
            from ...core.profiles import get_profile_kind

            profile = get_profile_kind()
        except Exception:
            profile = "laptop"
        if profile == "termux":
            base_ch, depth, temb = 32, 2, 128
            print("termux profile: tiny 64px model "
                  "(honest limits apply)")
        else:
            base_ch, depth, temb = 64, 3, 256

        unet = UNet(in_channels=3, base_channels=base_ch,
                    depth=depth, temb_dim=temb,
                    ctx_dim=256, image_size=args.size)
        encoder = TextEncoder(dim=256)
        print(f"UNet params: {count_parameters(unet):,} | "
              f"encoder params: {count_parameters(encoder):,}")
        cfg = TrainConfig(
            run_name=args.run, timesteps=args.timesteps,
            image_size=args.size, base_channels=base_ch,
            depth=depth, temb_dim=temb,
            lr=args.lr, batch_size=args.batch, steps=args.steps,
            checkpoint_every=args.checkpoint_every,
            seed=args.seed or 0)
        result = train_text_conditional(unet, encoder, loader, cfg)
        print(f"done: {result['steps']} steps, "
              f"final loss {result['final_loss']:.4f}")
        print(f"checkpoint: {result['final_checkpoint']}")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return 0


def _cmd_imggen_lora_train(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen import TORCH_AVAILABLE
        from ...media.imggen.data import ImageFolderDataset
        from ...media.imggen.lora import inject_lora
        from ...media.imggen.pipeline import load_native_checkpoint
        from ...media.imggen.train import (TrainConfig, train)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    if not TORCH_AVAILABLE:
        return _err("LoRA training needs torch: pip install torch")
    try:
        try:
            from ...core.profiles import get_profile_kind

            if get_profile_kind() == "termux":
                return _err("LoRA training needs at least a laptop "
                            "profile (termux can't hold the gradients)")
        except Exception:
            pass
        from torch.utils.data import DataLoader

        pipe = load_native_checkpoint(args.base)
        wrapped = inject_lora(pipe.unet, r=args.rank,
                              alpha=args.alpha)
        print(f"LoRA injected into {len(wrapped)} modules "
              f"(rank {args.rank})")
        ds = ImageFolderDataset(args.data, image_size=args.size)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                            num_workers=0)
        # LoRA training reuses train() with lora_only=True; captions
        # flow through the pipeline's text encoder at sample time, so
        # we train the UNet adapters with unconditional dropout only.
        # For captioned data, pair with the encoder:
        from ...media.imggen.train import train_text_conditional

        cfg = TrainConfig(
            run_name=args.run, lr=args.lr, batch_size=args.batch,
            steps=args.steps, checkpoint_every=args.checkpoint_every,
            seed=args.seed or 0)
        result = train_text_conditional(
            pipe.unet, pipe.text_encoder, loader, cfg)
        print(f"done: {result['steps']} steps, "
              f"final loss {result['final_loss']:.4f}")
        print(f"adapter checkpoint: {result['final_checkpoint']}")
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return 0


def _cmd_imggen_checkpoints(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen.pipeline import list_native_checkpoints

        cks = list_native_checkpoints()
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    if args.json:
        print(json.dumps(cks, indent=2))
    elif not cks:
        print("no native checkpoints yet — "
              "`nm imggen train --data <folder>` builds one")
    else:
        for c in cks:
            mb = c["bytes"] / 1e6
            print(f"{c['run']}: {c['path']} ({mb:.1f} MB)")
    return 0


def _cmd_imggen_dashboard(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen.studio import training_dashboard

        print(training_dashboard(args.run))
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return 0


def _cmd_imggen_merge(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen.studio import merge_checkpoints

        out = merge_checkpoints(args.base, args.other,
                                alpha=args.alpha,
                                out_path=args.out or "")
        print(out)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    return 0


def _cmd_imggen_character(args: argparse.Namespace, context) -> int:
    try:
        from ...media.imggen.studio import character_sheet

        studio = _studio(args)
        paths = character_sheet(
            studio, args.anchor, args.poses, seed=args.seed,
            steps=args.steps)
    except Exception as exc:  # noqa: BLE001
        return _err(str(exc))
    for p in paths:
        print(p)
    return 0


# Map action → handler for dispatch.py.
HANDLERS = {
    "generate": _cmd_imggen_generate,
    "img2img": _cmd_imggen_img2img,
    "inpaint": _cmd_imggen_inpaint,
    "outpaint": _cmd_imggen_outpaint,
    "upscale": _cmd_imggen_upscale,
    "train": _cmd_imggen_train,
    "lora-train": _cmd_imggen_lora_train,
    "checkpoints": _cmd_imggen_checkpoints,
    "dashboard": _cmd_imggen_dashboard,
    "merge": _cmd_imggen_merge,
    "character": _cmd_imggen_character,
}


def cmd_imggen(args: argparse.Namespace, context) -> int:
    handler = HANDLERS.get(args.imggen_action)
    if handler is None:
        return _err(f"unknown imggen action {args.imggen_action!r}")
    return handler(args, context)
