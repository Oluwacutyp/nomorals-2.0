#!/usr/bin/env python3
"""Discord user-token QR login — phone-friendly, no desktop needed.

Uses Discord's remote-auth gateway (the same flow the desktop QR login
uses): scan the code with the Discord mobile app, approve, and the
token lands straight in ~/.devon-secrets.

Needs: pip install cryptography websocket-client qrcode

The token is full account access — it goes straight to your secrets
file, never printed, never sent anywhere.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request

GATEWAY = "wss://remote-auth-gateway.discord.gg/?v=2"
API = "https://discord.com/api/v9"


def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def main() -> int:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
        import websocket
    except ImportError as exc:
        print(f"missing dependency: {exc.name} — "
              "pip install cryptography websocket-client qrcode",
              file=sys.stderr)
        return 1

    print("Generating keypair…")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_der = key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo)
    pub_b64 = base64.b64encode(pub_der).decode()

    state: dict = {"done": False, "token": "", "fingerprint": ""}

    def _decrypt(b64: str) -> bytes:
        return key.decrypt(
            base64.b64decode(b64),
            padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                         algorithm=hashes.SHA256(), label=None))

    def on_message(ws, raw):
        msg = json.loads(raw)
        op = msg.get("op")
        if os.environ.get("QR_DEBUG"):
            print(f"[qr] op={op}", file=sys.stderr)

        if op == "hello":
            interval = msg.get("heartbeat_interval", 30000) / 1000

            def _beat():
                while not state["done"]:
                    time.sleep(interval)
                    try:
                        ws.send(json.dumps({"op": "heartbeat"}))
                    except Exception:
                        break
            threading.Thread(target=_beat, daemon=True).start()
            ws.send(json.dumps({"op": "init", "encoded_public_key": pub_b64}))

        elif op == "nonce_proof":
            # gateway sends the encrypted nonce; we reply with the
            # base64url-encoded DECRYPTED nonce (per userdoccers).
            nonce = _decrypt(msg["encrypted_nonce"])
            ws.send(json.dumps({"op": "nonce_proof",
                                "nonce": _b64url_nopad(nonce)}))

        elif op == "pending_remote_init":
            fp = msg["fingerprint"]
            state["fingerprint"] = fp
            url = f"https://discord.com/ra/{fp}"
            print()
            print("Scan with the Discord app (Settings → Scan QR Code):")
            print(f"  {url}")
            try:
                import qrcode
                qr = qrcode.QRCode(border=1)
                qr.add_data(url)
                qr.make()
                # save as PNG for the Discord app's gallery scanner
                img_path = os.path.expanduser("~/discord-qr.png")
                try:
                    qr.make_image().save(img_path)
                    print(f"  QR saved to {img_path}")
                    print("  In the Discord app: Settings → Scan QR Code → "
                          "tap the gallery icon → pick discord-qr.png")
                except ImportError:
                    print("  (pip install pillow for a scannable image)")
                for row in qr.get_matrix():
                    print("".join("██" if c else "  " for c in row))
            except ImportError:
                print("  (pip install qrcode for an in-terminal code)")
            print()
            print("Waiting for scan + approval…")

        elif op == "pending_login":
            ticket = msg.get("ticket", "")
            if not ticket and msg.get("encrypted_user_payload"):
                ticket = _decrypt(msg["encrypted_user_payload"]).decode()
            # exchange the ticket for the token
            body = json.dumps({"ticket": ticket}).encode()
            req = urllib.request.Request(
                API + "/users/@me/remote-auth/login", data=body, method="POST",
                headers={"Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (Linux; Android 10)",
                         "Origin": "https://discord.com"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            enc_token = data.get("token", "")
            if enc_token:
                state["token"] = _decrypt(enc_token).decode()
            state["done"] = True
            ws.close()

        elif op == "cancel":
            print("login cancelled in the app", file=sys.stderr)
            state["done"] = True
            ws.close()

    print("Connecting to Discord…")
    ws = websocket.WebSocketApp(
        GATEWAY, header=["Origin: https://discord.com"],
        on_message=on_message,
        on_error=lambda ws, e: print(f"gateway error: {e}", file=sys.stderr))
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()

    deadline = time.time() + 180
    while not state["done"] and time.time() < deadline:
        time.sleep(1)
    ws.close()

    token = state["token"]
    if not token:
        print("timed out or not approved — run again when ready", file=sys.stderr)
        return 1

    secrets = os.path.expanduser("~/.devon-secrets")
    lines = []
    if os.path.exists(secrets):
        with open(secrets) as f:
            lines = [l for l in f if not l.startswith("NM_CHAT_DISCORD_TOKEN=")]
    lines.append(f'NM_CHAT_DISCORD_TOKEN="{token}"\n')
    with open(secrets, "w") as f:
        f.writelines(lines)
    os.chmod(secrets, 0o600)
    print("saved to ~/.devon-secrets — restart the bot and you're in")
    return 0


if __name__ == "__main__":
    sys.exit(main())
