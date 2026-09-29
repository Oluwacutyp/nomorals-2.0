# Discord account token — phone-only recovery. PASTE THIS WHOLE FILE into one
# Kaggle notebook cell (keep the notebook PRIVATE) and run it.
#
# What happens: it installs a headless browser (5-8 min first time), opens
# Discord, shows you the page, decodes the login QR into a link YOU open on
# this phone, your Discord app asks to log in — you APPROVE (2FA there if
# you have it) — and the script keeps waiting and prints your token plus
# the ready-made .env lines.
#
# When you're done: DELETE the notebook. The token stays in the outputs.
#
# If a screenshot shows a captcha / risk check: Discord blocked the
# datacenter login. That path is dead — use a real computer (a public
# library one is fine; it takes 30 seconds).
import time

# ── install (only the first run actually takes the time) ─────────────────
try:
    from playwright.sync_api import sync_playwright  # noqa: F401
except ImportError:
    get_ipython().system("pip install -q playwright opencv-python-headless")
    get_ipython().system("python -m playwright install --with-deps chromium 2>&1 | tail -2")

import cv2
import numpy as np
from IPython.display import Image
from playwright.sync_api import sync_playwright

pw = sync_playwright().start()
browser = pw.chromium.launch(
    headless=True,
    args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
)
ctx = browser.new_context(
    viewport={"width": 1280, "height": 900},
    user_agent=(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    locale="en-US",
)
page = ctx.new_page()
page.goto("https://discord.com/app", wait_until="domcontentloaded", timeout=60000)
time.sleep(6)
display(Image(data=page.screenshot()))
print("URL now:", page.url)

# ── decode the login QR into a link for this phone ────────────────────────
img = cv2.imdecode(np.frombuffer(page.screenshot(), np.uint8), cv2.IMREAD_COLOR)
det = cv2.QRCodeDetector()
qr_url = det.detectAndDecode(img)[0] or ""
if not qr_url:  # the QR sits on the right half of the login screen
    h, w = img.shape[:2]
    for sub in (img[:, w // 2:], img[: h // 2, :], img[:, : w // 2:]):
        data, _, _ = det.detectAndDecode(sub)
        if data:
            qr_url = data
            break

if qr_url:
    if qr_url.startswith("discord://"):
        print("\nANDROID — copy the WHOLE line below into Chrome's address bar and press go:\n")
        print("intent:" + qr_url + ";package=com.discord#Intent;end")
    elif qr_url.startswith("http"):
        print("\nCopy this URL into your phone browser:\n")
        print(qr_url)
    else:
        print("QR payload:", qr_url[:200])
    print(
        "\nYour Discord app should pop up a 'log in to this device' prompt —\n"
        "APPROVE IT (enter your 2FA code there if you have it).\n"
        "The script keeps waiting below for up to 10 minutes..."
    )
else:
    print(
        "\nNo QR code on the page. Look at the screenshot above:\n"
        "  - email/password form  -> not covered here; use a real computer (F12)\n"
        "  - captcha/risk check   -> blocked; use a real computer\n"
        "  - already logged in    -> just run the waiting loop below\n"
    )

# ── wait for the login to land, then print the token ──────────────────────
token = ""
deadline = time.time() + 600
while time.time() < deadline:
    token = ctx.evaluate("() => window.localStorage.getItem('token')") or ""
    if token:
        break
    time.sleep(3)

try:
    browser.close()
    pw.stop()
except Exception:  # noqa: BLE001
    pass

if token:
    print("\nDISCORD TOKEN (copy the whole string):\n")
    print(token)
    print("\n--- put these two lines in ~/.nomorals/.env where you run nm ---")
    print("NM_CHAT_DISCORD_ENABLED = true")
    print("NM_CHAT_DISCORD_TOKEN   = " + token)
    print("--------------------------------------------------------------------")
    print("\nNOW DELETE THIS NOTEBOOK — the token lives in these outputs.")
else:
    print(
        "\nNo token within 10 minutes. Did you approve the prompt in the Discord app?\n"
        "If yes, run this whole cell again. If it was a captcha, stop here."
    )
