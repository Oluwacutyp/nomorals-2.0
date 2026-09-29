"""Cipher tool for encryption/decryption operations."""

from __future__ import annotations

from typing import Any


def cipher_encrypt(text: str, key: str = "", method: str = "caesar") -> str:
    """Encrypt text using various cipher methods."""
    if method == "caesar":
        shift = int(key) if key.isdigit() else 3
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base + shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)
    elif method == "base64":
        import base64
        return base64.b64encode(text.encode()).decode()
    elif method == "hex":
        return text.encode().hex()
    return text


def cipher_decrypt(text: str, key: str = "", method: str = "caesar") -> str:
    """Decrypt text using various cipher methods."""
    if method == "caesar":
        shift = int(key) if key.isdigit() else 3
        result = []
        for char in text:
            if char.isalpha():
                base = ord('A') if char.isupper() else ord('a')
                result.append(chr((ord(char) - base - shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)
    elif method == "base64":
        import base64
        return base64.b64decode(text.encode()).decode()
    elif method == "hex":
        return bytes.fromhex(text).decode()
    return text


def register(registry: Any) -> None:
    """Register cipher tools with the registry."""
    registry.register(
        "cipher",
        lambda text="", key="", method="caesar", action="encrypt": 
            cipher_encrypt(text, key, method) if action == "encrypt" else cipher_decrypt(text, key, method),
        description="Encrypt or decrypt text using various cipher methods",
        capability="cipher",
        parameters={
            "text": {"type": "string", "description": "Text to encrypt/decrypt"},
            "key": {"type": "string", "description": "Encryption key or shift"},
            "method": {"type": "string", "description": "Cipher method (caesar, base64, hex)"},
            "action": {"type": "string", "description": "Action: encrypt or decrypt"}
        }
    )
