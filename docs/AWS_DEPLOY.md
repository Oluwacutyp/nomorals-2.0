# Deploying Devon on AWS — Step by Step

## What you're building
A 24/7 Devon running on AWS in `eu-north-1` (Stockholm), powered by your 3.8B model. She'll handle Telegram, WhatsApp (later), trading, schedules — everything, always on.

**Cost:** ~$15/month on t3.small. You have $100 in credits = ~6 months free.

---

## Step 1: Launch the EC2 instance (5 min)

1. Go to AWS Console → EC2 → Launch Instance
2. **Name:** `devon-prod`
3. **AMI:** Ubuntu 24.04 LTS (free tier eligible)
4. **Instance type:** `t3.small` (2 vCPU, 2GB RAM)
5. **Key pair:** Create new → download the `.pem` file → keep it safe
6. **Network:** Default VPC is fine
7. **Storage:** 30 GB gp3 (free tier covers 30GB)
8. **Security group:** Allow SSH (port 22) from your IP only
9. Click Launch

## Step 2: Connect to your server (2 min)

```bash
chmod 400 ~/Downloads/devon-key.pem
ssh -i ~/Downloads/devon-key.pem ubuntu@<your-ec2-public-ip>
```

## Step 3: Install dependencies (10 min)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Python + tools
sudo apt install -y python3 python3-pip python3-venv git ffmpeg

# Node.js (for WhatsApp bridge)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# llama.cpp (for the 3.8B model)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j4 && cd ~
```

## Step 4: Clone Devon (3 min)

```bash
git clone https://github.com/Oluwacutyp/nomorals-2.0.git ~/nomorals-2.0
cd ~/nomorals-2.0
git checkout main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Step 5: Download the 3.8B brain (10 min)

```bash
# Install huggingface-cli
pip install huggingface-cli

# Download your private model (you'll need to be logged in)
huggingface-cli login
huggingface-cli download Cutyp/codebeast-3.8b \
  --include "codebeast-30k.Q4_K_M.gguf" \
  --local-dir ~/models/
```

## Step 6: Set your secrets (5 min)

```bash
mkdir -p ~/.devon-secrets
# Add each secret as a file:
echo "your-telegram-bot-token" > ~/.devon-secrets/TELEGRAM_BOT_TOKEN
echo "your-groq-api-key" > ~/.devon-secrets/GROQ_API_KEY
# ... add all your secrets this way
chmod 600 ~/.devon-secrets/*
```

**Never** put secrets in the repo. Always in `~/.devon-secrets/`.

## Step 7: Configure Devon (5 min)

```bash
cd ~/nomorals-2.0
cp config.example.toml config.toml  # if it exists
# Edit config.toml:
# - Set model path to ~/models/codebeast-30k.Q4_K_M.gguf
# - Set llama.cpp server address
# - Enable Telegram adapter
# - Set your timezone
```

## Step 8: Start llama.cpp server (2 min)

```bash
# In one terminal (or tmux/screen):
~/llama.cpp/llama-server \
  -m ~/models/codebeast-30k.Q4_K_M.gguf \
  --port 8080 \
  -c 4096 \
  --threads 2
```

## Step 9: Start Devon as a service (5 min)

Create `/etc/systemd/system/devon.service`:
```ini
[Unit]
Description=Devon AI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/nomorals-2.0
ExecStart=/home/ubuntu/nomorals-2.0/.venv/bin/python -m nomorals
Restart=always
RestartSec=10
Environment="PATH=/home/ubuntu/nomorals-2.0/.venv/bin:/usr/bin"

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable devon
sudo systemctl start devon
sudo systemctl status devon  # check it's running
```

## Step 10: Verify (2 min)

```bash
# Check logs
sudo journalctl -u devon -f

# You should see:
# - Devon dashboard starting
# - Telegram bot connecting
# - Schedulers registering
```

Send a message to your Telegram bot. She should reply.

---

## What to do when the 7B is ready

1. Upload the 7B GGUF to `~/models/`
2. Update `config.toml` with the new model path
3. `sudo systemctl restart devon`
4. Done. No other changes needed.

---

## Costs

| Item | Monthly |
|------|---------|
| t3.small EC2 | ~$15 |
| 30GB EBS | ~$3 (free tier covers it) |
| Data transfer | Minimal (free tier covers 100GB) |
| **Total** | **~$15/mo** |

$100 credits = ~6 months. By then the trading should be paying for itself.

---

## Troubleshooting

**Devon won't start:** `sudo journalctl -u devon -n 50` — read the error.

**Model too slow:** t3.small has 2 vCPU. The 3.8B Q4_K_M should do ~5-10 tokens/sec. If slower, check `htop` for CPU steal.

**Telegram not connecting:** Check the bot token in `~/.devon-secrets/`. Check firewall.

**Out of memory:** t3.small has 2GB. The 3.8B Q4_K_M needs ~2.5GB with context. If OOM, reduce context (`-c 2048`) or add swap:
```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
