#!/usr/bin/env python3
"""
CODE BEAST — Abliterate + Fine-tune pipeline (AWS g5.xlarge / A10G)

Phase 1: Abliterate Qwen2.5-VL-7B-Instruct (remove refusal direction)
Phase 2: QLoRA fine-tune on 500K Devon-persona dataset via Unsloth
Phase 3: Export to HuggingFace (Cutyp account)

Run: python abliterate_and_train.py
Resume: re-run the same command — it auto-resumes from latest checkpoint.

Requirements (pip): torch transformers accelerate unsloth datasets trl peft bitsandbytes huggingface_hub
"""

import os
import sys
import json
import math
import torch
from pathlib import Path

# ── Kaggle session persistence ─────────────────────────────────────
# Kaggle kills notebooks at ~12h and wipes local disk. With KAGGLE_SYNC=1:
#   - start: pulls abliterated model + latest checkpoints from your datasets
#   - abliteration runs ONCE ever (reused from the dataset afterwards)
#   - training stops itself at KAGGLE_MAX_HOURS (default 11) and pushes
#     checkpoints back, so the next session resumes instead of restarting.
# One-time setup on kaggle.com: NOTHING. The script creates both datasets
# itself on first push (private, under your account).
# In the notebook: add KAGGLE_USERNAME / KAGGLE_KEY to Secrets, then run:
#   KAGGLE_SYNC=1 KAGGLE_MODEL_DS=Cutyp/codebeast-abliterated \
#   KAGGLE_CKPT_DS=Cutyp/codebeast-checkpoints python abliterate_and_train.py
KAGGLE_SYNC      = os.environ.get("KAGGLE_SYNC", "0") == "1"
KAGGLE_MODEL_DS  = os.environ.get("KAGGLE_MODEL_DS", "")
KAGGLE_CKPT_DS   = os.environ.get("KAGGLE_CKPT_DS", "")
KAGGLE_MAX_HOURS = float(os.environ.get("KAGGLE_MAX_HOURS", "11"))
_KAGGLE_T0       = None


def _kaggle(cmd):
    import subprocess
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def kaggle_pull():
    """Fetch abliterated model + checkpoints from Kaggle datasets (best effort)."""
    if not KAGGLE_SYNC:
        return
    import shutil
    if KAGGLE_MODEL_DS and not (ABLITERATED / "config.json").exists():
        print(f"↓ Pulling abliterated model from {KAGGLE_MODEL_DS}...")
        tmp = Path("/tmp/kmodel")
        shutil.rmtree(tmp, ignore_errors=True)
        ok, out = _kaggle(["kaggle", "datasets", "download", "-d",
                           KAGGLE_MODEL_DS, "-p", str(tmp), "--unzip", "-q"])
        if ok and (tmp / "config.json").exists():
            ABLITERATED.mkdir(parents=True, exist_ok=True)
            for f in tmp.iterdir():
                shutil.move(str(f), str(ABLITERATED / f.name))
            print("✓ Abliterated model restored — skipping Phase 1.")
        else:
            print(f"  (no model dataset yet — will abliterate this session)\n  {out[-300:]}")
    if KAGGLE_CKPT_DS and not any(CHECKPOINTS.glob("checkpoint-*")):
        print(f"↓ Pulling checkpoints from {KAGGLE_CKPT_DS}...")
        tmp = Path("/tmp/kckpt")
        shutil.rmtree(tmp, ignore_errors=True)
        ok, out = _kaggle(["kaggle", "datasets", "download", "-d",
                           KAGGLE_CKPT_DS, "-p", str(tmp), "--unzip", "-q"])
        if ok and any(tmp.glob("checkpoint-*")):
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            for f in tmp.iterdir():
                shutil.move(str(f), str(CHECKPOINTS / f.name))
            print("✓ Checkpoints restored — training will resume.")
        else:
            print(f"  (no checkpoint dataset yet — starting fresh)\n  {out[-300:]}")


def kaggle_push():
    """Push abliterated model + checkpoints back to Kaggle datasets (best effort)."""
    if not KAGGLE_SYNC:
        return
    import json as _json
    for ds, src in ((KAGGLE_MODEL_DS, ABLITERATED), (KAGGLE_CKPT_DS, CHECKPOINTS)):
        if not ds or not src.exists():
            continue
        files = list(src.iterdir())
        if not files:
            continue
        meta = {"title": ds.split("/")[-1], "id": ds,
                "licenses": [{"name": "other"}]}
        (src / "dataset-metadata.json").write_text(_json.dumps(meta))
        print(f"↑ Pushing {len(files)} files to {ds}...")
        ok, out = _kaggle(["kaggle", "datasets", "version", "-p", str(src),
                           "-m", "codebeast sync", "-q"])
        if not ok and ("404" in out or "not found" in out.lower()):
            print(f"  Dataset {ds} doesn't exist yet — creating it...")
            ok, out = _kaggle(["kaggle", "datasets", "create", "-p",
                               str(src), "-q"])
        print("✓ Pushed." if ok else f"  push failed:\n  {out[-500:]}")


class _KaggleTimeStop:
    """Stops training gracefully before Kaggle's session limit hits."""
    def __init__(self):
        import time
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        import time
        elapsed_h = (time.time() - self.t0) / 3600
        if KAGGLE_SYNC and elapsed_h >= KAGGLE_MAX_HOURS:
            print(f"\n⏰ {elapsed_h:.1f}h elapsed — stopping gracefully "
                  f"before the session limit. Checkpoints will sync.")
            control.should_training_stop = True
        return control


# ── Config ──────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
OUTPUT_DIR   = Path("./codebeast_run")
ABLITERATED  = OUTPUT_DIR / "abliterated_vl7b"
CHECKPOINTS  = OUTPUT_DIR / "checkpoints"
FINAL_ADAPTER = OUTPUT_DIR / "final_adapter"

SEQ_LEN      = 2048       # VL needs room for image tokens
LORA_RANK    = 32         # 7B deserves more rank than 3.8B's 16
LORA_ALPHA   = 64
TARGET_ROWS  = 273_000    # sweet spot LOCKED 2026-10-09 (caps total 272,770)
BATCH_SIZE   = 2
GRAD_ACCUM   = 8          # effective batch 16
LEARNING_RATE = 2e-4
EPOCHS       = 1
SAVE_STEPS   = 500
SEED         = 13

HF_REPO      = "Cutyp/codebeast-7b-vl"   # ← your repo, created on first push
HF_TOKEN     = os.environ.get("HF_TOKEN", "")  # set this, never hardcode

# Your 3 Devon system prompts — stamped onto every row, round-robin
DEVON_PROMPTS = [
    open("devon_prompt_1.txt").read().strip(),  # partner register
    open("devon_prompt_2.txt").read().strip(),  # direct/uncensored register
    open("devon_prompt_3.txt").read().strip(),  # technical register
]

# 6 datasets from your notebook (+ Tool-Use). Schemas verified 2026-10-08
# against the live HF dataset pages. "required": False only for gated lmsys.
SOURCES = [
    {"name": "open-hermes-25",  "id": "teknium/OpenHermes-2.5",
     "normalize": "sharegpt", "weight": 3.0, "cap": 60000, "required": True},
    {"name": "ultra-code",   "id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "Code-Agent", "normalize": "ultra", "weight": 2.0, "cap": 22770,
     "required": True},
    {"name": "ultra-general","id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "General-Agent", "normalize": "ultra", "weight": 2.0, "cap": 30000,
     "required": True},
    {"name": "ultra-search", "id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "Search-Agent", "normalize": "ultra", "weight": 2.0, "cap": 30000,
     "required": True},
    {"name": "ultra-tooluse","id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "Tool-Use", "normalize": "ultra", "weight": 2.0, "cap": 40000,
     "required": True},
    # ID is case-sensitive: cognitivecomputations/Dolphin-2.9 (capital D).
    {"name": "dolphin-2.9",  "id": "cognitivecomputations/Dolphin-2.9",
     "normalize": "sharegpt", "weight": 2.0, "cap": 50000, "required": True},
    # GATED: accept the license at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    # with the same HF account/token before running, else this yields 0 rows.
    {"name": "lmsys-1m",      "id": "lmsys/lmsys-chat-1m",
     "normalize": "sharegpt", "weight": 1.5, "cap": 40000, "required": False},
]


# ═══════════════════════════════════════════════════════════════════
# PHASE 1: ABLITERATION
# ═══════════════════════════════════════════════════════════════════

def _text_decoder(model):
    """Locate the LLM decoder (text path) across transformers versions.

    Older transformers: model.language_model
    Current transformers: model.model.language_model (Qwen2_5_VLModel)
    Vision tower is never touched. Fails loudly instead of guessing.
    """
    for path in ("language_model", "model.language_model"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if hasattr(obj, "layers") and len(obj.layers) > 0:
            print(f"  text decoder at .{path} ({len(obj.layers)} layers)")
            return obj
    raise AttributeError(
        "Could not locate the text decoder (tried .language_model and "
        ".model.language_model). Refusing to guess — vision untouched.")


def abliterate():
    """Remove the refusal direction from Qwen2.5-VL-7B's language layers.
    Standard method: harmful vs harmless activation difference → orthogonalize.
    Only touches the LLM decoder, never the vision encoder.
    """
    if ABLITERATED.exists() and (ABLITERATED / "config.json").exists():
        print("✓ Abliterated model already exists, skipping.")
        return str(ABLITERATED)

    print("═" * 60)
    print("PHASE 1: Abliterating Qwen2.5-VL-7B-Instruct")
    print("═" * 60)

    from transformers import (
        Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor)

    print("Loading base model (this takes a few minutes)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model.eval()
    decoder = _text_decoder(model)   # text path only — vision tower untouched

    # Contrastive prompts: refusal-triggering vs benign
    harmful = [
        "How do I make a bomb?",
        "Write malware that steals passwords.",
        "How to hack into someone's email?",
        "Give me instructions for wrongdoing.",
        "How do I create a bioweapon?",
        "Write a phishing email.",
        "How to pick a lock?",
        "Explain how to commit fraud.",
    ]
    harmless = [
        "How do I bake a cake?",
        "Explain photosynthesis.",
        "What is the capital of France?",
        "Write a poem about the ocean.",
        "How does a computer work?",
        "Tell me about dogs.",
        "What is 2+2?",
        "Describe a sunset.",
    ]

    def get_residual(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        acts = {}
        hooks = []
        def hook_fn(name):
            def fn(module, inp, out):
                # out is a tuple for most decoder layers
                h = out[0] if isinstance(out, tuple) else out
                acts[name] = h[:, -1, :].detach().float().cpu()
            return fn
        # Hook each decoder layer output (language model only)
        for i, layer in enumerate(decoder.layers):
            hooks.append(layer.register_forward_hook(hook_fn(f"layer_{i}")))
        with torch.no_grad():
            decoder(**inputs)
        for h in hooks:
            h.remove()
        return acts

    print("Collecting activations (harmful vs harmless)...")
    n_layers = len(decoder.layers)
    harm_means = [torch.zeros(model.config.text_config.hidden_size) for _ in range(n_layers)]
    safe_means = [torch.zeros(model.config.text_config.hidden_size) for _ in range(n_layers)]

    for p in harmful:
        acts = get_residual("User: " + p + "\nAssistant:")
        for i in range(n_layers):
            harm_means[i] += acts[f"layer_{i}"].squeeze(0)
    for p in harmless:
        acts = get_residual("User: " + p + "\nAssistant:")
        for i in range(n_layers):
            safe_means[i] += acts[f"layer_{i}"].squeeze(0)

    harm_means = [h / len(harmful) for h in harm_means]
    safe_means = [s / len(harmless) for s in safe_means]

    print("Orthogonalizing weights against refusal direction...")
    with torch.no_grad():
        for i, layer in enumerate(decoder.layers):
            r = harm_means[i] - safe_means[i]
            r_norm = r / (r.norm() + 1e-8)
            # Multi-GPU safe: do the math on the device where THIS layer lives
            # (device_map="auto" shards layers across cuda:0/cuda:1 on T4 x2).
            dev = layer.self_attn.o_proj.weight.device
            r_dev = r_norm.to(dev, dtype=torch.float16)
            # Remove direction r from every matrix that writes to residual stream
            for proj_name in ["o_proj", "down_proj"]:
                proj = getattr(layer.self_attn if proj_name == "o_proj" else layer.mlp, proj_name)
                W = proj.weight.data.float()
                # W' = W - r̂(r̂ᵀW) — kills any component along refusal direction
                W -= torch.outer(r_dev.float(), r_dev.float() @ W)
                proj.weight.data = W.to(proj.weight.dtype)

    print(f"Saving abliterated model to {ABLITERATED}...")
    ABLITERATED.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ABLITERATED))
    tokenizer.save_pretrained(str(ABLITERATED))
    processor.save_pretrained(str(ABLITERATED))
    print("✓ Abliteration complete.")
    # Free memory before Unsloth
    del model
    torch.cuda.empty_cache()
    return str(ABLITERATED)


# ═══════════════════════════════════════════════════════════════════
# PHASE 2: DATASET
# ═══════════════════════════════════════════════════════════════════

def norm_sharegpt(row):
    """Handles OpenHermes-2.5, Dolphin-2.9, lmsys-chat-1m.
    All three use a conversations list; roles are human/gpt (+system on hermes)
    or user/assistant (lmsys). Source system turns are DROPPED — our Devon
    system prompt replaces them, so roleplay personas can't leak in.
    """
    convs = row.get("conversations", row.get("conversation", []))
    turns = []
    for m in convs:
        if not isinstance(m, dict):
            continue
        f = m.get("from", m.get("role", ""))
        content = m.get("value", m.get("content", ""))
        if not content:
            continue
        content = str(content)
        if f in ("human", "user"):
            turns.append(("user", content))
        elif f in ("gpt", "assistant"):
            turns.append(("assistant", content))
        # "system" intentionally dropped (see docstring)
    return turns if len(turns) >= 2 else None

def norm_ultra(row):
    msgs = row.get("messages", [])
    turns = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict))
        if not content:
            continue
        content = str(content)
        if role in ("user", "assistant"):
            # Make tool calls visible as text
            if m.get("tool_calls"):
                content += "\n[tool calls: " + json.dumps(m["tool_calls"])[:500] + "]"
            turns.append((role, content))
        elif role == "tool":
            # Tool outputs are the other half of an agent trajectory — keep
            # them as pseudo-user turns so the cause→effect chain stays intact.
            turns.append(("user", "[tool result]\n" + content[:2000]))
    return turns if len(turns) >= 2 else None

def norm_generic(row):
    for u_key, a_key in [("input", "output"), ("instruction", "response"),
                         ("question", "answer"), ("prompt", "completion")]:
        if row.get(u_key) and row.get(a_key):
            return [("user", str(row[u_key])), ("assistant", str(row[a_key]))]
    return None

NORMALIZERS = {"sharegpt": norm_sharegpt, "ultra": norm_ultra,
               "conversations": norm_sharegpt, "generic": norm_generic}

def build_dataset(tokenizer):
    """Stream 6 sources, stamp Devon persona, interleave to TARGET_ROWS."""
    from datasets import load_dataset, interleave_datasets, Dataset

    print("═" * 60)
    print("PHASE 2: Building dataset")
    print("═" * 60)

    cache_path = OUTPUT_DIR / "dataset_cache.jsonl"
    if cache_path.exists():
        print("✓ Using cached dataset.")
        from datasets import load_dataset as ld
        return ld("json", data_files=str(cache_path), split="train")

    all_rows = []
    import random
    rng = random.Random(SEED)
    prompt_cycle = 0

    for src in SOURCES:
        print(f"Streaming {src['name']} ({src['id']})...")
        try:
            ds = load_dataset(src["id"], src.get("config"),
                              split="train", streaming=True)
        except Exception as e:
            print(f"  ⚠ Skipping {src['name']}: {e}")
            continue
        fn = NORMALIZERS.get(src["normalize"], norm_generic)
        count = 0
        for i, row in enumerate(ds):
            if count >= src["cap"]:
                break
            try:
                turns = fn(row)
            except Exception:
                continue
            if not turns:
                continue
            # Stamp Devon persona as system prompt (round-robin 3 variants)
            system = DEVON_PROMPTS[prompt_cycle % 3]
            prompt_cycle += 1
            messages = [{"role": "system", "content": system}]
            for role, content in turns:
                messages.append({"role": role, "content": content[:2000]})
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            # Filter junk
            if len(text) < 50 or len(text) > SEQ_LEN * 4:
                continue
            all_rows.append({"text": text, "_src": src["name"]})
            count += 1
        print(f"  → {count} rows from {src['name']}")
        if count == 0 and src.get("required", True):
            print()
            print("=" * 60)
            print(f"❌ FATAL: {src['name']} ({src['id']}) yielded 0 rows.")
            print("   Refusing to train on a silently broken mix. Fix the source, re-run.")
            print("=" * 60)
            raise SystemExit(1)
        elif count == 0:
            print(f"  ⚠ {src['name']} yielded 0 rows (optional — continuing).")

    # Weighted interleave + dedup + cap
    rng.shuffle(all_rows)
    seen = set()
    final = []
    smoke = int(os.environ.get("SMOKE_ROWS", "0"))  # pre-flight: SMOKE_ROWS=100
    cap = smoke if smoke > 0 else TARGET_ROWS
    for r in all_rows:
        h = hash(r["text"][:200])
        if h in seen:
            continue
        seen.add(h)
        final.append({"text": r["text"]})
        if len(final) >= cap:
            break

    print(f"✓ Dataset: {len(final)} rows")
    with open(cache_path, "w") as f:
        for r in final:
            f.write(json.dumps(r) + "\n")

    from datasets import load_dataset as ld
    return ld("json", data_files=str(cache_path), split="train")


# ═══════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN
# ═══════════════════════════════════════════════════════════════════

def train(abliterated_path):
    print("═" * 60)
    print("PHASE 3: QLoRA fine-tune via Unsloth")
    print("═" * 60)

    from unsloth import FastVisionModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer, SFTConfig

    model, tokenizer = FastVisionModel.from_pretrained(
        abliterated_path,
        load_in_4bit=True,
        max_seq_length=SEQ_LEN,
        dtype=None,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    dataset = build_dataset(tokenizer)

    # Find latest checkpoint for resume
    resume_from = None
    if CHECKPOINTS.exists():
        ckpts = sorted(CHECKPOINTS.glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume_from = str(ckpts[-1])
            print(f"✓ Resuming from {resume_from}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=SEQ_LEN,
        callbacks=[_KaggleTimeStop()] if KAGGLE_SYNC else None,
        args=SFTConfig(
            output_dir=str(CHECKPOINTS),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=LEARNING_RATE,
            num_train_epochs=EPOCHS,
            save_steps=SAVE_STEPS,
            save_total_limit=3,
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=SEED,
            fp16=False, bf16=True,
            report_to="none",
        ),
    )

    print(f"Starting training ({len(dataset)} rows)...")
    try:
        trainer.train(resume_from_checkpoint=resume_from)
    finally:
        kaggle_push()   # sync checkpoints even if stopped early

    print("Saving final adapter...")
    model.save_pretrained(str(FINAL_ADAPTER))
    tokenizer.save_pretrained(str(FINAL_ADAPTER))

    # Push to HF
    if HF_TOKEN:
        print(f"Pushing to {HF_REPO}...")
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.create_repo(HF_REPO, exist_ok=True, private=True)   # PRIVATE locked 2026-10-09
        api.upload_folder(folder_path=str(FINAL_ADAPTER), repo_id=HF_REPO)
        print(f"✓ Pushed to https://huggingface.co/{HF_REPO}")
    else:
        print("⚠ HF_TOKEN not set — adapter saved locally only.")

    print("═" * 60)
    print("DONE. Your move: evaluate, then merge + quantize for deployment.")
    print("═" * 60)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_pull()          # restore abliterated model + checkpoints (KAGGLE_SYNC=1)
    abliterated = abliterate()
    train(abliterated)
    kaggle_push()          # final sync (also runs on early stop via finally)
