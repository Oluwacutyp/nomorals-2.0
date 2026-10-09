#!/usr/bin/env python3
"""
CODE BEAST — Dataset builder (standalone)

Streams the 6 sources, normalizes, stamps Devon persona, dedups,
interleaves to TARGET_ROWS, saves as JSONL.

Run: python build_dataset.py
Output: ./codebeast_dataset/train.jsonl + val.jsonl (95/5 split)

Inspect before training:
  head -c 2000 codebeast_dataset/train.jsonl | python -m json.tool
"""

import json
import random
from pathlib import Path

TARGET_ROWS = 273_000 # sweet spot LOCKED 2026-10-09 (matches abliterate_and_train.py)
VAL_SPLIT   = 0.05
SEED        = 13
OUT_DIR     = Path("./codebeast_dataset")

DEVON_PROMPTS = [
    open("devon_prompt_1.txt").read().strip(),
    open("devon_prompt_2.txt").read().strip(),
    open("devon_prompt_3.txt").read().strip(),
]

SOURCES = [
    # NOTE (2026-10-08 pre-flight): schemas verified against HF dataset pages.
    # "required": False only for gated lmsys (needs license click-through).
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
    # "dolphin2.9" 404s. Schema: conversations[{from: human/gpt, value}].
    {"name": "dolphin-2.9",  "id": "cognitivecomputations/Dolphin-2.9",
     "normalize": "sharegpt", "weight": 2.0, "cap": 50000, "required": True},
    # GATED: accept the license at https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    # with the same HF account/token before running, else this yields 0 rows.
    {"name": "lmsys-1m",      "id": "lmsys/lmsys-chat-1m",
     "normalize": "sharegpt", "weight": 1.5, "cap": 40000, "required": False},
]


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


def main():
    from datasets import load_dataset

    rng = random.Random(SEED)
    prompt_cycle = 0
    all_rows = []
    stats = {}

    for src in SOURCES:
        print(f"\nStreaming {src['name']} ({src['id']})...")
        try:
            ds = load_dataset(src["id"], src.get("config"),
                              split="train", streaming=True)
        except Exception as e:
            print(f"  ⚠ Skipping: {e}")
            stats[src["name"]] = 0
            continue

        fn = NORMALIZERS.get(src["normalize"], norm_generic)
        count, skipped = 0, 0
        for i, row in enumerate(ds):
            if count >= src["cap"]:
                break
            try:
                turns = fn(row)
            except Exception:
                skipped += 1
                continue
            if not turns:
                skipped += 1
                continue

            # Stamp Devon persona (round-robin across 3 variants)
            system = DEVON_PROMPTS[prompt_cycle % 3]
            prompt_cycle += 1

            messages = [{"role": "system", "content": system}]
            for role, content in turns:
                messages.append({"role": role, "content": content[:2000]})

            # Junk filter
            total_len = sum(len(m["content"]) for m in messages)
            if total_len < 50 or total_len > 8000:
                skipped += 1
                continue

            all_rows.append({"messages": messages, "_src": src["name"]})
            count += 1
            if count % 5000 == 0:
                print(f"  ... {count} rows")

        stats[src["name"]] = count
        print(f"  → {count} rows ({skipped} skipped)")
        if count == 0 and src.get("required", True):
            print()
            print("=" * 60)
            print(f"❌ FATAL: {src['name']} ({src['id']}) yielded 0 rows.")
            print("   A required source came back empty — schema or access broke.")
            print("   Refusing to build a dataset with a silently missing source.")
            print("   Fix the source entry above, then re-run.")
            print("=" * 60)
            raise SystemExit(1)
        elif count == 0:
            print(f"  ⚠ {src['name']} yielded 0 rows (optional source — continuing).")
            print("    If this was lmsys-1m: accept the license at")
            print("    https://huggingface.co/datasets/lmsys/lmsys-chat-1m first.")

    # Shuffle + dedup + cap
    print(f"\nTotal before dedup: {len(all_rows)}")
    rng.shuffle(all_rows)
    seen, final = set(), []
    for r in all_rows:
        # Hash on first user message (cheap dedup)
        key = hash(r["messages"][1]["content"][:200] if len(r["messages"]) > 1 else "")
        if key in seen:
            continue
        seen.add(key)
        final.append({"messages": r["messages"]})
        if len(final) >= TARGET_ROWS:
            break

    # Train/val split
    rng.shuffle(final)
    n_val = int(len(final) * VAL_SPLIT)
    val, train = final[:n_val], final[n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "train.jsonl", "w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(OUT_DIR / "val.jsonl", "w") as f:
        for r in val:
            f.write(json.dumps(r) + "\n")

    # Stats
    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump({"sources": stats, "train_rows": len(train),
                   "val_rows": len(val), "seed": SEED}, f, indent=2)

    print(f"\n{'='*50}")
    print(f"✓ DONE: {len(train)} train + {len(val)} val rows")
    total = len(train) + len(val)
    print(f"  Target was {TARGET_ROWS:,} — actual yield {total:,} "
          f"({100.0 * total / TARGET_ROWS:.0f}%)")
    print(f"  Per-source mix:")
    for name, n in stats.items():
        print(f"    {name:>14}: {n:>7,} rows ({100.0 * n / max(total, 1):5.1f}%)")
    print(f"  Files: {OUT_DIR}/train.jsonl, val.jsonl, stats.json")
    print(f"\nInspect a row:")
    print(f"  head -1 {OUT_DIR}/train.jsonl | python -m json.tool | head -30")


if __name__ == "__main__":
    main()
