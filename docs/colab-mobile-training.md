# Training your own model on Google Colab — from a phone

Research doc, all facts verified live on **2026-09-10**. Companion to
[`nomorals-qlora.ipynb`](./nomorals-qlora.ipynb), which is the actual
notebook you run.

---

## 1 · Short answer

**Yes, it is doable on mobile.** Colab runs entirely in the cloud; your phone
is only the driver — a browser tab. There is no GPU, no disk, no install on
the phone. The two things that are *not* mobile-friendly are (a) there is no
official Colab mobile app (the website is the app; third-party "Colab APK"s
are junk), and (b) the UI is desktop-shaped, so run Chrome in **landscape,
desktop site mode**.

Everything else — 8B QLoRA training on a free T4 — was designed for exactly
this: `docs/nomorals-qlora.ipynb` loads dolphin-8B in 4-bit, fits a T4's 16 GB
VRAM with room to spare, and takes ~20–60 min on a few hundred samples.

**Budget: $0.** Free T4, no credit card.

---

## 2 · Verified Colab facts (2026-09)

| Fact | Value | Note |
|---|---|---|
| Mobile access | **Browser only** (Chrome/Safari, `colab.research.google.com`) | No official app. Desktop-site mode + landscape makes it usable. |
| Free GPU | **NVIDIA T4, 16 GB VRAM** | Best-effort, can be CPU-only at peak times — re-run the runtime. |
| Session length | **12 h max** (free) | 24 h on Pro. |
| Idle timeout | **~90 min** with no cell running | A *running* training cell counts as activity. |
| Effective free hours | **~15–30 GPU-h/week** observed (varies by region/demand) | No published hard weekly quota. |
| Disk | **~10 GB** usable (`/content` + root) | See §6 — this is the #1 surprise. |
| Drive bridge | Files ↔ Drive, 15 GB per file on free Drive | How you move the ~8.5 GB GGUF off. |
| Pro (optional) | $9.99/mo — L4, A100 access, 24 h sessions | Only if free T4 keeps starving you. |
| Best free fallback | **Kaggle Notebooks** — P100/T4, ~30 h/week, 9 h sessions, auto-save every 30 min | Same notebook runs there (needs `pip install unsloth` first). Phone-verified workflow: register with your phone number. |

**Ban heuristic worth knowing:** free tiers throttle accounts doing
abnormal network transfer (cited >500 GB/day). One 16 GB model download is
far below that — just don't put download loops in a while-true.

---

## 3 · The right models (verified live against the HF API, 2026-09-10)

> The `cognitivecomputations` org **renamed to `dphn`**. Old ids still
> redirect, but pin the canonical ones below.

### For training (this notebook's base)

| Repo | Why |
|---|---|
| **`microsoft/Phi-3.5-mini-instruct`** | **Current pick for CodeBeast v3**: 3.8B params, MIT license, Q4 ≈ 2.3 GB, fits 12 GB Samsung easily, trains in 12–18 GPU-hours on 500K rows. Notebook: `docs/codebeast_phi35.ipynb`. |
| `dphn/dolphin-2.9-llama3-8b` | Older pick: 8B params, BF16 ≈ 16.06 GB, **not gated**, Llama-3 chat template. Notebook: `docs/nomorals-qlora.ipynb`. |
| `dphn/dolphin-2.9.4-llama3.1-8b` | Newer dolphin on Llama-3.1-8B. Same footprint; use if you want the fresher base. |

### Ready-made GGUFs (skip training — just inference)

| Repo | Notes |
|---|---|
| **`dphn/dolphin-2.9-llama3-8b-gguf`** | Official-org GGUF of exactly the base above. |
| `bartowski/dolphin-2.9.4-llama3.1-8b-GGUF` | Best quant lineup (q2→q8), for the 3.1 base. |
| `QuantFactory/dolphin-2.9-llama3-8b-GGUF` | Solid q4_k_m/q8_0 alternative. |

### For serving without training at all

The new HF Inference Providers router (see `docs` of the `hf_serverless`
provider / `nm models --live-catalog`):

* `https://router.huggingface.co/hf-inference` — **free, rate-limited, serves
  dolphin** (this is what NoMorals' `hf` provider points at by default now).
* `https://router.huggingface.co/v1` — partner catalog (Groq/Cerebras/…);
  dolphin is **not** in the catalog, other models are.

Both need a **fine-grained token** with *Make calls to Inference Providers*.

---

## 4 · The right endpoints for fetching models

| Where | Endpoint | Auth |
|---|---|---|
| Inside the Colab notebook (download base model) | `https://huggingface.co` via `huggingface_hub` (what unsloth/transformers use under the hood) | `HF_TOKEN` env var — **read** scope is enough |
| Direct single file | `https://huggingface.co/<repo>/resolve/main/<file>` e.g. `https://huggingface.co/dphn/dolphin-2.9-llama3-8b/resolve/main/model-00001-of-00004.safetensors` | none (public repo) |
| Inference, free, any Hub model | `POST https://router.huggingface.co/hf-inference/models/<repo>/v1/chat/completions` | fine-grained HF token (Inference Providers permission) |
| Inference, partner catalog | `POST https://router.huggingface.co/v1/chat/completions` | same token |
| On the phone, Termux | `nm models --fetch <catalog-name-or-repo-id>` | downloads via the same `resolve/main` pattern into `~/models` |

---

## 5 · Step by step, on the phone

> Do this once. Total active phone time ≈ 15 min; the GPU does 20–60 min of
> work in the background. Keep the phone **on a charger** — a dead phone mid
> session kills the Colab tab, which kills the session.

### Step 0 — Accounts (10 min, one time)

1. Google account (you have one if you have a Gmail).
2. Hugging Face account → **Settings → Access Tokens → New token** →
   *Fine-grained* → **Read** on your account (that's all a training download
   needs). Copy the `hf_…` string.
3. Prepare `data.jsonl` on the phone (or generate it): one JSON object per
   line, `{"messages": [{"role": "user", "content": …}, {"role": "assistant",
   "content": …}]}`. 200–1,000 lines is a sensible first run.
4. Put the NoMorals repo (or just the notebook) where Colab can open it —
   easiest: keep this GitHub repo public/private-as-now; Colab's
   *Open from GitHub* handles private repos with the GitHub connector.

### Step 1 — Open the notebook (2 min)

1. Chrome on the phone → **landscape** → menu → **Desktop site** (for
   colab.research.google.com only).
2. `colab.research.google.com` → **File → Open from GitHub** → pick the repo
   → `docs/nomorals-qlora.ipynb`.
3. Colab copies it into your Drive automatically (menu → **Make a copy** if it
   asks). From now on you're editing *your* copy.

**Expected:** notebook loads, 16 cells, title "NoMorals — QLoRA finetune on
Colab (Unsloth)".

### Step 2 — Upload the data (1 min)

Left edge → **folder/files icon** → **Upload files** → pick `data.jsonl` from
the phone.

**Expected:** `data.jsonl` appears in the file list.

### Step 3 — Put the token in Secrets (1 min)

Menu bar → **Secrets** (key icon) → **Add** → name `HF_TOKEN`, value your
`hf_…` token → then in cell 4 the uncommented path is:

```python
from google.colab import userdata
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
```

(Or just paste the token into cell 4's Option B — fine for a private
notebook; Secrets keeps it out of the cell history.)

### Step 4 — Pick the GPU (30 s)

**Runtime → Change runtime type → Hardware accelerator → GPU** (T4) → Save.

**Run the install cell (Cell 1 or 3 depending on notebook).** Watch for the
success message. When it finishes:

**⚠️ CRITICAL: Restart the runtime now.** Menu: **Runtime → Restart runtime**
(or Ctrl+M .). This is required — newly installed packages won't import until
you restart. After restart, run the verification cell to confirm unsloth loads.

Then continue with the remaining cells (run top-to-bottom — better than "Run all"
so you can watch each expected output below).

**Expected at install cell:** pip output, GPU info printed, then a "RESTART THE
RUNTIME" message. After restart, the verification cell should print "✅ unsloth
imported successfully!"

### Step 5 — Load the base model (~5–8 min)

Cell 4 downloads 4 shards, **~16 GB total** from
`huggingface.co/dphn/dolphin-2.9-llama3-8b`, then loads 4-bit.

**Expected:** 4 progress bars, then a parameter table — look for
`trainable params` ≈ **30–40 M** (that's the LoRA part, ~0.4% of the model)
and 4-bit weights ≈ **4.2 GB** in memory. If you see a download error: re-run the cell (HF resumes the
shard). If the runtime shows CPU only: **Runtime → Restart runtime**, re-pick
GPU in Step 4.

### Step 6 — Dataset (10 s)

**Expected:** `N conversations` and the first line printed with the Llama-3
`<|start_header_id|>` template. If it says `!! no data.jsonl found` you
skipped Step 2.

### Step 7 — Train (20–60 min)

Run the LoRA cell, then the train cell.

**Expected:** tqdm progress bar; for ~500 samples: effective batch 16 →
~31 steps/epoch × 3 epochs ≈ **94 steps**, ~10–30 s/step on a T4. Loss should
walk down (start ~3–6, end ~0.5–2). Every epoch a checkpoint lands in
`./checkpoints` — **that's your mobile insurance** (Step 9 of §6).

**Phone behaviour:** keep the tab open; the running cell keeps the session
alive. Locking the screen is fine *if* Android doesn't kill Chrome (Step 8
below). If you see the "Session disconnected" screen: **Reconnect** → the
kernel is back → re-run only the train cell (it resumes from the last
checkpoint if you set `resume_from_checkpoint` — or just re-run the whole
train; 3 epochs on 500 samples is cheap).

### Step 8 — Phone battery hardening (before you lock the screen)

1. Settings → Apps → Chrome → **Battery → Unrestricted**.
2. Pull down the quick settings → **battery saver OFF** while training.
3. Phone on a charger; screen can sleep.

This is the single highest-value mobile step — most "Colab died on me"
stories are Android killing the browser, not Colab.

### Step 9 — Save (2–4 min) + move to Drive

Run the save cells: `./finetune` (adapter + tokenizer, **~250 MB**) and
`./finetune_gguf` (q8_0, **~8.5 GB**).

⚠️ The 8.5 GB GGUF won't coexist with the 16 GB model cache in the 10 GB
disk. Immediately after saving, move it to Drive and delete the cache:

```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/finetune /content/drive/MyDrive/nomorals-finetune/
!mv /content/finetune_gguf/*.gguf /content/drive/MyDrive/nomorals-finetune/
!rm -rf /content/finetune_gguf /root/.cache/huggingface
!df -h /
```

**Expected:** files appear under `MyDrive/nomorals-finetune/` in the Drive
app on your phone.

### Step 10 — Ship it to Termux (5 min)

1. Drive app → `nomorals-finetune/` → tap the `.gguf` → download to the
   phone.
2. Termux:

```bash
mv ~/Download/nomorals-.../dolphin-2.9-llama3-8b-00001-of-00001.gguf ~/models/my-dolphin.gguf
# (exact filename from the file list; single-file q8_0 for 8B)
export NM_LLM_LOCAL_MODEL=~/models/my-dolphin.gguf
python3 -m nomorals.cli models --local-doctor
python3 -m nomorals.cli models --start-local
python3 -m nomorals.cli models --doctor
```

**Expected:** `local` reports the model file found; `--doctor` shows the
llama.cpp chain serving it. Then in chat: `/model local`.

---

## 6 · The gotchas that actually break mobile runs

| # | Gotcha | Symptom | Fix |
|---|---|---|---|
| 1 | **10 GB disk vs 16 GB download** | download dies at ~10 GB: `No space left on device` | `!df -h` first. Free root + /tmp + /content ≈ 10–13 GB total; the 16 GB BF16 cache usually just fits *if the cache lives in `/root/.cache` and `/tmp` is clear*. If not: restart the runtime, then immediately after the model loads `!rm -rf /root/.cache/huggingface` — the loaded model stays in VRAM, you only need the cache during download. |
| 2 | **GGUF + cache don't coexist** | save cell fails with disk full | §5 Step 9 order: copy to Drive → `mv` GGUF → `rm -rf` cache. |
| 3 | **90-min idle timeout / Android kills Chrome** | "Session disconnected", kernel gone | Running cells keep it alive (training is safe). Chrome battery-unrestricted + charger. On reconnect: re-run the failed cell; checkpoints (`save_strategy="epoch"`, already in the notebook) let you resume. |
| 4 | **CPU-only runtime at peak times** | no CUDA, training at a crawl, or `No CUDA GPUs` | Restart runtime with GPU re-selected; wait 15 min and retry; or switch to Kaggle (30 h/week, P100/T4). |
| 5 | **CUDA OOM** (you pushed batch/seq up) | `CUDA out of memory` at first batch | `per_device_train_batch_size=2`, `MAX_SEQ=1024`. The notebook defaults already fit 16 GB; don't raise them on a T4. |
| 6 | **401 from HF** | `401 Unauthorized` while downloading | Token wrong/expired, or repo gated. This repo is not gated; re-create the token, re-run the cell. |
| 7 | **`hf-inference` 429s** (inference path) | rate-limited free calls | Wait 30–60 s; it's a shared pool. For heavy inference use the partner `/v1` router with a catalog model or a local GGUF. |
| 8 | **Tried the third-party "Colab App" APK** | weird webview, broken uploads | Don't. It's not Google's; browser is the app. |

---

## 7 · What you get, and where it plugs in

* **`finetune/`** — PEFT LoRA adapter + tokenizer (~250 MB). Tiny enough to
  ship anywhere; merge into any copy of the base with
  `PeftModel.from_pretrained(base, "./finetune")`.
* **`finetune_gguf/*.gguf`** — q8_0, ~8.5 GB, single file. Runs on the phone
  via the local llama.cpp provider (Step 10).
* **The self-improvement loop** — once this works end-to-end, the same
  training runs through `nm train --backend unsloth
  --base-model dphn/dolphin-2.9-llama3-8b` on any GPU box, and the result
  goes through the regression gate (loss + golden set) before it's allowed to
  beat the incumbent. The Colab run is the *mobile bootstrap* of that loop;
  the gate is what keeps it honest.

**Realistic expectations:** a few hundred lines of your own chat makes a
model that *sounds* like your data (tone, length, habits) — not a smarter
model. Quality scales with data size and diversity, and with 3+ epochs on
1,000+ lines you'll start hearing it in the replies. Start with 300 lines,
judge the sample in Step 9's inference cell, then scale the corpus, not the
hyperparameters.
