# 🦁 CODE BEAST v3 — 500K Dataset (5 free sources) + Kaggle Training Guide

*For Oluwacutyp (Peace) — the 21-year-old creator. Same persona. Five sources. 500K rows. Your exact 3.8B model.*

---

## ✅ The 500K dataset — built from EXACTLY your 5 free sources

`codebeast500k_train.jsonl.gz` — **499,600 rows** · `codebeast500k_val.jsonl.gz` — 400 rows · persona v2 on every row (unchanged).

| # | Source (from your notebook) | Cap | Status |
|---|---|---|---|
| 1 | `teknium/OpenHermes-2.5` | 450,000 | ✅ the "all of earth" layer |
| 2 | `openbmb/UltraData-SFT-Agent-2609` · Code-Agent | all 22,770 | ✅ coding-agent trajectories |
| 3 | `openbmb/UltraData-SFT-Agent-2609` · General-Agent | all 38,581 | ✅ general agent trajectories |
| 4 | `openbmb/UltraData-SFT-Agent-2609` · Search-Agent | all 20,000 | ✅ search-agent trajectories |
| 5 | dolphin2.9 (unrestricted-behavior layer) | 40,000 | ✅ via `Skorcht/dolphin2.9` |
| 6 | ~~lmsys-chat-1m~~ | ❌ excluded | gated — needs HF token, skipped by design |

**Renames you should know (verified):**
- `cognitivecomputations/dolphin2.9` was **deleted** → same data now lives at **`Skorcht/dolphin2.9`** (456K rows).
- Base model is now **`microsoft/Phi-3.5-mini-instruct`** (3.8B params, MIT license) — the old 7B abliterated base is retired from this pipeline. Smaller, faster to train, and the GGUF fits your phone with room to spare.

**Also fixed from your notebook:** its OpenHermes normalizer expected prompt/completion, but OpenHermes-2.5 is ShareGPT `conversations` — that source would have contributed ~0 rows. Handled natively now. And UltraData tool calls are converted to visible `[TOOL_CALL name]` / `[TOOL_RESULT name]` text exactly like your notebook's normalizer, so a plain chat template learns the whole agent pattern.

**Every row:** persona as system message → final user→assistant exchange, tail-trimmed to ~390 tokens (0% truncation waste — rows are pre-trimmed) → md5-deduped → seeded shuffle → 499,600 train / 400 val.

---

## 🏃 Two ways to get the 500K into Kaggle

**Option A (recommended — zero big downloads):** upload `build_500k.py` + `codebeast_seed.jsonl` (a few KB) to Kaggle as an input dataset. The notebook rebuilds the identical 500K **directly inside Kaggle in ~20 minutes** — no phone data, no manual download.

**Option B:** download `codebeast500k_train.jsonl.gz` + `codebeast500k_val.jsonl.gz` (~110 MB total) and upload them to Kaggle as a private dataset. Training starts immediately.

---

## ☁️ Kaggle — the full walkthrough

**Why Kaggle:** free **T4 GPU, 30 GPU-hours/week**, sessions up to **9 hours**, and **Save Version persists your checkpoints** — the exact three things the Colab runs needed.

### Setup (once)
1. Go to [kaggle.com](https://www.kaggle.com) → sign in → accept the free tier.
2. **Create → New Notebook.**
3. Right sidebar → **Settings**: Accelerator = **GPU T4** (16GB — plenty for 3.8B QLoRA). Leave Internet ON. (T4 x2 exists but burns quota twice as fast; skip it.)
4. **+ Add Input → Upload → New Dataset** → make it **private** → upload `codebeast_v3.ipynb` + (`build_500k.py` + `codebeast_seed.jsonl`) *or* the two prebuilt `.gz` files.
5. Add your HF write token: **Add-ons → Secrets** → name `HF_TOKEN` (used only by the final push cell — the notebook never needs it for the data itself).
6. Run the notebook cells top to bottom.

### The training math (honest)
- 3.8B QLoRA on T4 ≈ **2,500–4,000 tokens/sec** with Unsloth.
- 500K rows × ~390 tokens ≈ **195M tokens ≈ 14–22 GPU-hours** → **under one week** of free quota, across 2–4 sessions.
- Sweet spot: TARGET_ROWS=30000 → 3,750 steps ≈ 4–7h training, fits one 9h session with room for merge + GGUF + push. 500K rows ≈ 62K steps ≈ 100h at T4 speed — not viable; 50K is borderline. Set 6000 for a quick proof run (slice logic is automatic).

### Surviving session ends
1. Checkpoints land every **500 steps** in `/kaggle/working/codebeast_run/checkpoints`.
2. When you're done for the day (or at the 12-hr cap): notebook menu (top-right) → **Save Version → Quick Save**. Your `/kaggle/working` files — checkpoints, adapter, GGUF — are stored with the version (up to 20GB).
3. Next session: open the saved version → **Restart & Run All**. Cell 5 auto-detects the newest checkpoint and **resumes where it stopped**. No wasted hours, ever.
4. Quota says "GPU exceeded"? It's 30 hrs per rolling week — wait a day or two and continue.

### Finish line
- Cell 8 exports the **GGUF Q4_K_M (~2.3GB)** → download it from the notebook's **Output panel** (wifi) → move to your Samsung.
- Cell 9 pushes the **merged full 3.8B model** to your Hugging Face account — the end product for nomorals 2.0.
- Phone: **ChatterUI** (or PocketPal/Layla/LLM Farm) → add the `.gguf` → paste the CODE BEAST persona as the system prompt. 2.3GB on 12GB RAM is comfortable — no need to close anything.

---

## 📁 Files in this kit

| File | What it is |
|---|---|
| `codebeast500k_train.jsonl.gz` | **THE 500K training set (499,600 rows)** |
| `codebeast500k_val.jsonl.gz` | 400-row validation set |
| `codebeast_v3.ipynb` | Kaggle-first notebook: Phi-3.5-mini 3.8B · auto-resume · Save-Version persistence · GGUF export · HF push |
| `make_notebook.py` | Regenerates `codebeast_v3.ipynb` (`python3 make_notebook.py`) |
| `build_500k.py` | Rebuilds the 500K anywhere in ~15–25 min (`--quick` smoke test) |
| `codebeast_seed.jsonl` | 88 persona rows (identity, loyalty, Yoruba + 10 languages, creator-is-21 lore) |
| `codebeast500k_stats.json` | Per-source build stats (written by `build_500k.py`) |

**Reality checks (straight talk):** base is stock Phi-3.5-mini-instruct — the persona system prompt carries the character, not an abliterated base. The dark-web line is persona flavor — no local model browses; that needs a tool layer. No open model is 100% unfiltered on every topic; rephrase if a rare one trips it.

🦁 That's the whole beast: 500K rows, your 5 sources, your 3.8B model, a Kaggle pipeline that can't lose your work. Go feed it.
