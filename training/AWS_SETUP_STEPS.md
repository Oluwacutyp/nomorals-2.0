# AWS Setup — Code Beast 7B-VL Training

Do these in order. Copy-paste where shown. Nothing here can surprise-bill you
if you follow step 1 first.

---

## STEP 1 — Billing alarm (2 min, do this FIRST)

1. Go to https://console.aws.amazon.com/billing/home#/budgets
2. Click **Create budget** → **Cost budget** → Next
3. Name: `training-cap` — Amount: **$80** — Period: Monthly
4. Add alert: **80% of budgeted amount** → enter your email
5. Create. You'll get an email if spend ever hits $64. You won't.

---

## STEP 2 — Launch the GPU instance (5 min)

1. Go to https://us-east-1.console.aws.amazon.com/ec2/ (N. Virginia — best spot availability)
2. **Launch instance**
   - Name: `codebeast-train`
   - AMI: search **"Deep Learning AMI GPU PyTorch"** → pick the latest Ubuntu one
     (has CUDA + PyTorch preinstalled, saves 30 min)
   - Instance type: **g5.xlarge** (1x A10G, 24GB — the sweet spot)
   - Key pair: **Create new** → name `codebeast-key` → Download the `.pem` file
     (save it somewhere safe — you can't re-download it)
   - Network: default VPC is fine
   - Storage: change to **150 GB gp3** (model + datasets + checkpoints need room)
   - **Advanced → Spot instance**: check **"Request Spot Instance"**
     (this is what keeps it at ~$0.35/hr instead of $1.00/hr)
3. Click **Launch instance**

**Cost check:** g5.xlarge spot ≈ $0.30–0.40/hr. 5 days ≈ $36–48. Inside your credits.

---

## STEP 3 — Connect (2 min)

On your computer (or Termux):

```bash
chmod 400 ~/path/to/codebeast-key.pem
ssh -i ~/path/to/codebeast-key.pem ubuntu@<PUBLIC-IP-FROM-AWS-CONSOLE>
```

Find the public IP on the EC2 dashboard → Instances → your instance.

---

## STEP 4 — Upload the training files (3 min)

From your computer, in a new terminal (not the SSH one):

```bash
scp -i ~/path/to/codebeast-key.pem -r ~/workspace/devon-train/* ubuntu@<PUBLIC-IP>:~/
```

---

## STEP 5 — Run it (1 command, then walk away)

In the SSH terminal:

```bash
# Install deps (5 min)
pip install torch transformers accelerate unsloth datasets trl peft bitsandbytes huggingface_hub

# Set your HF token (get one at huggingface.co/settings/tokens — needs WRITE access)
export HF_TOKEN="hf_your_token_here"

# Run with nohup so it survives if your SSH drops
nohup python abliterate_and_train.py > train.log 2>&1 &

# Watch progress
tail -f train.log
```

**That's it.** The script:
1. Abliterates Qwen2.5-VL-7B (~30-60 min)
2. Streams all 6 datasets, stamps Devon persona (~1-2 hrs)
3. Trains 500K rows, checkpointing every 500 steps (~4-5 days)
4. Pushes the adapter to `Cutyp/codebeast-7b-vl`

**If the spot instance gets interrupted:** just re-run step 5's python command.
It auto-resumes from the latest checkpoint. You lose minutes, not days.

**Monitor cost:** check the billing dashboard once a day. The $80 alarm has your back.

---

## WHEN IT'S DONE

You'll have `Cutyp/codebeast-7b-vl` on HuggingFace with the LoRA adapter.
Next: merge + quantize to GGUF for deployment (I'll walk you through that when we get there).

---

## KILL SWITCH (when training is done — don't skip this)

1. EC2 dashboard → select `codebeast-train` → **Instance state → Terminate**
2. This stops ALL charges. Terminating deletes the instance; your model is safe on HF.

**Do NOT just "stop" it — terminate it.** Stopped instances still charge for storage.
