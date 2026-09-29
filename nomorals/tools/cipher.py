"""Cipher tool for encryption/decryption operations."""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_module
from typing import Any


def cipher_tool(action: str = "", data: str = "", algorithm: str = "",
                shift: int = 0, keyword: str = "", decrypt: bool = False,
                key: str = "", **kwargs) -> dict[str, Any]:
    """Cipher tool supporting classic ciphers, modern encryption, and HMAC."""
    
    if action == "classic":
        return _handle_classic(data, algorithm, shift, keyword, decrypt)
    elif action == "hmac":
        return _handle_hmac(key, data)
    elif action == "formats":
        return {"algorithms": ["aes-ctr", "aes-gcm", "chacha20", "caesar", "vigenere", "atbash", "b64"]}
    elif action == "encrypt":
        return _handle_encrypt(data, key, algorithm)
    elif action == "decrypt":
        return _handle_decrypt(data, key, algorithm)
    
    return {"error": f"Unknown action: {action}"}


def _handle_classic(data: str, algorithm: str, shift: int, keyword: str, decrypt: bool) -> dict[str, Any]:
    """Handle classic cipher operations."""
    result = data
    
    if algorithm == "caesar":
        s = -shift if decrypt else shift
        result = _caesar_cipher(data, s)
    elif algorithm == "vigenere":
        result = _vigenere_cipher(data, keyword, decrypt)
    elif algorithm == "atbash":
        result = _atbash_cipher(data)
    elif algorithm == "b64":
        if decrypt:
            result = base64.b64decode(data.encode()).decode()
        else:
            result = base64.b64encode(data.encode()).decode()
    
    return {"data": result}


def _caesar_cipher(text: str, shift: int) -> str:
    """Apply Caesar cipher."""
    result = []
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            result.append(chr((ord(char) - base + shift) % 26 + base))
        else:
            result.append(char)
    return ''.join(result)


def _vigenere_cipher(text: str, keyword: str, decrypt: bool) -> str:
    """Apply Vigenère cipher."""
    if not keyword:
        return text
    
    result = []
    keyword = keyword.lower()
    key_len = len(keyword)
    key_index = 0
    
    for char in text:
        if char.isalpha():
            base = ord('A') if char.isupper() else ord('a')
            key_char = keyword[key_index % key_len]
            key_shift = ord(key_char) - ord('a')
            if decrypt:
                key_shift = -key_shift
            result.append(chr((ord(char) - base + key_shift) % 26 + base))
            key_index += 1
        else:
            result.append(char)
    
    return ''.join(result)


def _atbash_cipher(text: str) -> str:
    """Apply Atbash cipher."""
    result = []
    for char in text:
        if char.isalpha():
            if char.isupper():
                result.append(chr(ord('Z') - (ord(char) - ord('A'))))
            else:
                result.append(chr(ord('z') - (ord(char) - ord('a'))))
        else:
            result.append(char)
    return ''.join(result)


def _handle_hmac(key: str, data: str) -> dict[str, Any]:
    """Generate HMAC."""
    digest = hmac_module.new(key.encode(), data.encode(), hashlib.sha256).hexdigest()
    return {"digest": digest}


def _handle_encrypt(data: str, key: str, algorithm: str) -> dict[str, Any]:
    """Encrypt data."""
    # Simple XOR encryption for demo
    encrypted = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(data))
    return {"data": base64.b64encode(encrypted.encode()).decode()}


def _handle_decrypt(data: str, key: str, algorithm: str) -> dict[str, Any]:
    """Decrypt data."""
    encrypted = base64.b64decode(data.encode()).decode()
    decrypted = ''.join(chr(ord(c) ^ ord(key[i % len(key)])) for i, c in enumerate(encrypted))
    return {"data": decrypted}


def register(registry: Any) -> None:
    """Register cipher tool with the registry."""
    registry.register(
        "cipher",
        cipher_tool,
        description="Encrypt, decrypt, and hash data using various cipher methods",
        capability="cipher",
        parameters={
            "action": {"type": "string", "description": "Action: classic, hmac, formats, encrypt, decrypt"},
            "data": {"type": "string", "description": "Data to process"},
            "algorithm": {"type": "string", "description": "Cipher algorithm"},
            "shift": {"type": "integer", "description": "Shift for Caesar cipher"},
            "keyword": {"type": "string", "description": "Keyword for Vigenère cipher"},
            "decrypt": {"type": "boolean", "description": "Whether to decrypt"},
            "key": {"type": "string", "description": "Encryption key"}
        }
    )
