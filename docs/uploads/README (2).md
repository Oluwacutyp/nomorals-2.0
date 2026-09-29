# 🦁 CODE BEAST v3 — 500K Dataset (5 free sources) + Kaggle Training Guide

*For Oluwacutyp (Peace) — the 21-year-old creator. Same persona. Five sources. 500K rows. Your exact 7B model.*

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

**Two renames you should know (verified today):**
- `cognitivecomputations/dolphin2.9` was **deleted** → same data now lives at **`Skorcht/dolphin2.9`** (456K rows).
- `huihui-ai/Qwen2.5-7B-Instruct-abliterated` → current repo is **`huihui-ai/Qwen2.5-7B-Instruct-abliterated-v2`** (same 7B, improved abliteration). That's what the notebook uses.

**Also fixed from your notebook:** its OpenHermes normalizer expected prompt/completion, but OpenHermes-2.5 is ShareGPT `conversations` — that source would have contributed ~0 rows. Handled natively now. And UltraData tool calls are converted to visible `[TOOL_CALL name]` / `[TOOL_RESULT name]` text exactly like your notebook's normalizer, so a plain chat template learns the whole agent pattern.

**Every row:** persona as system message → final user→assistant exchange, tail-trimmed to ~390 tokens (0% truncation waste at seq 512) → md5-deduped → seeded shuffle → 499,600 train / 400 val.

---

## 🏃 Two ways to get the 500K into Kaggle

**Option A (recommended — zero big downloads):** upload `build_500k.py` + `codebeast_seed.jsonl` (a few KB) to Kaggle as an input dataset. The notebook rebuilds the identical 500K **directly inside Kaggle in ~20 minutes** — no phone data, no manual download.

**Option B:** download `codebeast500k_train.jsonl.gz` + `codebeast500k_val.jsonl.gz` from this workspace (~110 MB total) and upload them to Kaggle as a private dataset. Training starts immediately.

---

## ☁️ Kaggle — the full walkthrough (this replaces Colab for real training)

**Why Kaggle:** free **T4 GPU, 30 GPU-hours/week**, sessions up to **12 hours** (Colab free dies at ~2–6h), and **Save Version persists your checkpoints** — the exact three things your last run needed.

### Setup (once)
1. Go to [kaggle.com](https://www.kaggle.com) → sign in with Google → accept the free tier.
2. **Create → New Notebook.**
3. Right sidebar → **Settings**: Accelerator = **GPU T4** (16GB — plenty for 7B QLoRA). Leave Internet ON. (T4 x2 exists but burns quota twice as fast; skip it.)
4. **+ Add Input → Upload → New Dataset** → make it **private** → upload `colab_codebeast_v3.ipynb` + (`build_500k.py` + `codebeast_seed.jsonl`) *or* the two prebuilt `.gz` files.
5. Run the notebook cells top to bottom.

### The training math (honest)
- 7B QLoRA on T4 ≈ **1,500–2,500 tokens/sec** with Unsloth.
- 500K rows × ~390 tokens ≈ **195M tokens ≈ 22–36 GPU-hours** → **1–1.5 weeks** of free quota, across 2–5 sessions.
- Each `MAX_STEPS=6000` session ≈ 2.5–4 hrs ≈ ~50K rows. Then checkpoint → resume.

### Surviving session ends (the part Colab robbed you of)
1. Checkpoints land every **500 steps** in `/kaggle/working/codebeast_run/checkpoints`.
2. When you're done for the day (or at the 12-hr cap): notebook menu (top-right) → **Save Version → Quick Save**. Your `/kaggle/working` files — checkpoints, adapter, GGUF — are stored with the version (up to 20GB).
3. Next session: open the saved version → **Restart & Run All**. Cell 5 auto-detects the newest checkpoint and **resumes where it stopped**. No wasted hours, ever.
4. Quota says "GPU exceeded"? It's 30 hrs per rolling week — wait a day or two and continue.

### Finish line
- Cell 8 exports the **GGUF Q4_K_M (~4.7GB)** for the 7B → download it from the notebook's **Output panel** (wifi) → move to your Samsung.
- Phone: **ChatterUI** (or PocketPal/Layla/LLM Farm) → add the `.gguf` → paste the CODE BEAST persona as the system prompt. 12GB RAM handles the 7B Q4, but close other apps.

### If you still want Colab for anything
The same notebook runs there (small smoke tests only — free Colab can't finish 500K×7B; that's what killed your last run).

---

## 📁 Files in this kit

| File | What it is |
|---|---|
| `codebeast500k_train.jsonl.gz` | **THE 500K training set (499,600 rows)** |
| `codebeast500k_val.jsonl.gz` | 400-row validation set |
| `colab_codebeast_v3.ipynb` | Kaggle-first notebook: 7B abliterated-v2 · auto-resume · Save-Version persistence · GGUF export |
| `build_500k.py` | Rebuilds the 500K anywhere in ~15–25 min (`--quick` smoke test) |
| `codebeast_seed.jsonl` | 88 persona rows (identity, loyalty, Yoruba + 10 languages, creator-is-21 lore) |
| `make_seed.py` | Edit `QA` + re-run to grow the seed |
| `codebeast500k_stats.json` | Per-source build stats |

**Reality checks (unchanged, straight talk):** abliteration + fine-tune ≈ 95%+ refusal removal — no open model is 100% unfiltered; rephrase if a rare topic trips it. The dark-web line is persona flavor — real browsing needs a tool layer (ask me, I'll wire one). My limit: I'll build and train the model, but not genuinely harmful how-to content.

🦁 That's the whole beast: 500K rows, your 5 sources, your 7B model, a Kaggle pipeline that can't lose your work. Go feed it.
