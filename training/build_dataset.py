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

TARGET_ROWS = 500_000
VAL_SPLIT   = 0.05
SEED        = 13
OUT_DIR     = Path("./codebeast_dataset")

DEVON_PROMPTS = [
    open("devon_prompt_1.txt").read().strip(),
    open("devon_prompt_2.txt").read().strip(),
    open("devon_prompt_3.txt").read().strip(),
]

SOURCES = [
    {"name": "open-hermes-25",  "id": "teknium/OpenHermes-2.5",
     "normalize": "hermes", "weight": 3.0, "cap": 60000},
    {"name": "ultra-code",   "id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "Code-Agent", "normalize": "ultra", "weight": 2.0, "cap": 22770},
    {"name": "ultra-general","id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "General-Agent", "normalize": "ultra", "weight": 2.0, "cap": 30000},
    {"name": "ultra-search", "id": "openbmb/UltraData-SFT-Agent-2609",
     "config": "Search-Agent", "normalize": "ultra", "weight": 2.0, "cap": 30000},
    {"name": "dolphin-2.9",  "id": "cognitivecomputations/dolphin2.9",
     "normalize": "conversations", "weight": 2.0, "cap": 50000},
    {"name": "lmsys-1m",      "id": "lmsys/lmsys-chat-1m",
     "normalize": "conversations", "weight": 1.5, "cap": 40000},
]


def norm_hermes(row):
    prompt = row.get("prompt", "")
    completion = row.get("completion", "")
    if not prompt or not completion:
        return None
    return [("user", prompt), ("assistant", completion)]

def norm_ultra(row):
    msgs = row.get("messages", [])
    turns = []
    for m in msgs:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                c.get("text", "") for c in content if isinstance(c, dict))
        if role in ("user", "assistant") and content:
            if m.get("tool_calls"):
                content += "\n[tool calls: " + json.dumps(m["tool_calls"])[:500] + "]"
            turns.append((role, str(content)))
    return turns if len(turns) >= 2 else None

def norm_conversations(row):
    convs = row.get("conversations", row.get("conversation", []))
    turns = []
    for m in convs:
        r = m.get("from", m.get("role", ""))
        role = "user" if r in ("human", "user") else "assistant"
        content = m.get("value", m.get("content", ""))
        if content:
            turns.append((role, str(content)))
    return turns if len(turns) >= 2 else None

def norm_generic(row):
    for u_key, a_key in [("input", "output"), ("instruction", "response"),
                         ("question", "answer"), ("prompt", "completion")]:
        if row.get(u_key) and row.get(a_key):
            return [("user", str(row[u_key])), ("assistant", str(row[a_key]))]
    return None

NORMALIZERS = {"hermes": norm_hermes, "ultra": norm_ultra,
               "conversations": norm_conversations, "generic": norm_generic}


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
    print(f"  Sources: {stats}")
    print(f"  Files: {OUT_DIR}/train.jsonl, val.jsonl, stats.json")
    print(f"\nInspect a row:")
    print(f"  head -1 {OUT_DIR}/train.jsonl | python -m json.tool | head -30")


if __name__ == "__main__":
    main()
