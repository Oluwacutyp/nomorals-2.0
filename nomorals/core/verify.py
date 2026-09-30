"""Live verification - check all external dependencies are properly configured.

Verifies:
- Hugging Face API token (if configured)
- yt-dlp binary path and version
- ffmpeg binary path and version
- OpenRouter/OpenAI API keys
- Telegram bot token
- WhatsApp bridge connectivity
- Database integrity
- Disk space
- Network connectivity

Usage:
    verifier = LiveVerifier()
    results = await verifier.verify_all()
    
    for check in results:
        status = "✅" if check.passed else "❌"
        print(f"{status} {check.name}: {check.message}")
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .logging_setup import get_logger

__all__ = ["LiveVerifier", "VerificationResult", "VerificationCheck"]

_log = get_logger(__name__)


@dataclass
class VerificationCheck:
    """Result of a single verification check."""
    
    name: str
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class VerificationResult:
    """Result of all verification checks."""
    
    checks: list[VerificationCheck] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: __import__("time").time())
    
    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "all_passed": self.all_passed,
            "passed": self.passed_count,
            "failed": self.failed_count,
            "total": len(self.checks),
            "checks": [c.to_dict() for c in self.checks],
            "timestamp": self.timestamp,
        }
    
    def to_message(self) -> str:
        """Format as user-friendly message."""
        lines = [f"🔍 **System Verification** ({self.passed_count}/{len(self.checks)} passed)\n"]
        
        for check in self.checks:
            status = "✅" if check.passed else "❌"
            lines.append(f"{status} **{check.name}**: {check.message}")
        
        return "\n".join(lines)


class LiveVerifier:
    """Verifies all external dependencies are properly configured."""
    
    def __init__(self, config: Optional[Any] = None) -> None:
        self.config = config
        _log.info("Live verifier initialized")
    
    async def verify_all(self) -> VerificationResult:
        """Run all verification checks.
        
        Returns:
            VerificationResult with all checks
        """
        result = VerificationResult()
        
        # Run all checks
        checks = [
            self.verify_hf_token,
            self.verify_ytdlp,
            self.verify_ffmpeg,
            self.verify_openrouter,
            self.verify_openai,
            self.verify_telegram,
            self.verify_whatsapp_bridge,
            self.verify_database,
            self.verify_disk_space,
            self.verify_network,
            self.verify_python_version,
            self.verify_torch,
        ]
        
        for check_func in checks:
            try:
                check = await check_func()
                result.checks.append(check)
            except Exception as e:
                result.checks.append(VerificationCheck(
                    name=check_func.__name__.replace("verify_", "").replace("_", " ").title(),
                    passed=False,
                    message=f"Check failed: {e}",
                ))
        
        return result
    
    async def verify_hf_token(self) -> VerificationCheck:
        """Verify Hugging Face API token."""
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        if not token:
            return VerificationCheck(
                name="Hugging Face Token",
                passed=False,
                message="HF_TOKEN not set. Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN env var.",
            )
        
        # Validate token by making a test request
        try:
            req = urllib.request.Request(
                "https://huggingface.co/api/whoami",
                headers={"Authorization": f"Bearer {token}"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                username = data.get("name", "unknown")
                
                return VerificationCheck(
                    name="Hugging Face Token",
                    passed=True,
                    message=f"Valid token for user: {username}",
                    details={"username": username},
                )
        except Exception as e:
            return VerificationCheck(
                name="Hugging Face Token",
                passed=False,
                message=f"Invalid token: {e}",
            )
    
    async def verify_ytdlp(self) -> VerificationCheck:
        """Verify yt-dlp binary is available."""
        # Check if yt-dlp is in PATH
        ytdlp_path = shutil.which("yt-dlp")
        
        if not ytdlp_path:
            # Check common locations
            common_paths = [
                "/usr/local/bin/yt-dlp",
                "/usr/bin/yt-dlp",
                Path.home() / ".local/bin/yt-dlp",
                Path.home() / "yt-dlp",
            ]
            
            for path in common_paths:
                if Path(path).exists():
                    ytdlp_path = str(path)
                    break
        
        if not ytdlp_path:
            return VerificationCheck(
                name="yt-dlp",
                passed=False,
                message="yt-dlp not found. Install: pip install yt-dlp",
            )
        
        # Get version
        try:
            result = subprocess.run(
                [ytdlp_path, "--version"],
                capture_output=True, text=True, timeout=5,
            )
            version = result.stdout.strip()
            
            return VerificationCheck(
                name="yt-dlp",
                passed=True,
                message=f"v{version} at {ytdlp_path}",
                details={"version": version, "path": ytdlp_path},
            )
        except Exception as e:
            return VerificationCheck(
                name="yt-dlp",
                passed=False,
                message=f"Found at {ytdlp_path} but failed to run: {e}",
            )
    
    async def verify_ffmpeg(self) -> VerificationCheck:
        """Verify ffmpeg binary is available."""
        ffmpeg_path = shutil.which("ffmpeg")
        
        if not ffmpeg_path:
            return VerificationCheck(
                name="ffmpeg",
                passed=False,
                message="ffmpeg not found. Install: apt install ffmpeg (Linux) or brew install ffmpeg (Mac)",
            )
        
        # Get version
        try:
            result = subprocess.run(
                [ffmpeg_path, "-version"],
                capture_output=True, text=True, timeout=5,
            )
            
            # Parse version from first line
            first_line = result.stdout.split("\n")[0]
            version = first_line.split()[2] if len(first_line.split()) > 2 else "unknown"
            
            return VerificationCheck(
                name="ffmpeg",
                passed=True,
                message=f"v{version} at {ffmpeg_path}",
                details={"version": version, "path": ffmpeg_path},
            )
        except Exception as e:
            return VerificationCheck(
                name="ffmpeg",
                passed=False,
                message=f"Found at {ffmpeg_path} but failed to run: {e}",
            )
    
    async def verify_openrouter(self) -> VerificationCheck:
        """Verify OpenRouter API key."""
        api_key = os.getenv("NM_OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        
        if not api_key:
            return VerificationCheck(
                name="OpenRouter API",
                passed=False,
                message="API key not set. Set NM_OPENAI_API_KEY or OPENROUTER_API_KEY.",
            )
        
        if not api_key.startswith("sk-or-"):
            return VerificationCheck(
                name="OpenRouter API",
                passed=False,
                message="Invalid key format. OpenRouter keys start with 'sk-or-'.",
            )
        
        # Validate by listing models
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                model_count = len(data.get("data", []))
                
                return VerificationCheck(
                    name="OpenRouter API",
                    passed=True,
                    message=f"Valid key, {model_count} models available",
                    details={"model_count": model_count},
                )
        except Exception as e:
            return VerificationCheck(
                name="OpenRouter API",
                passed=False,
                message=f"Invalid key: {e}",
            )
    
    async def verify_openai(self) -> VerificationCheck:
        """Verify OpenAI API key."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return VerificationCheck(
                name="OpenAI API",
                passed=False,
                message="OPENAI_API_KEY not set (optional)",
            )
        
        if not api_key.startswith("sk-"):
            return VerificationCheck(
                name="OpenAI API",
                passed=False,
                message="Invalid key format. OpenAI keys start with 'sk-'.",
            )
        
        return VerificationCheck(
            name="OpenAI API",
            passed=True,
            message="Key present and valid format",
        )
    
    async def verify_telegram(self) -> VerificationCheck:
        """Verify Telegram bot token."""
        token = os.getenv("NM_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
        
        if not token:
            return VerificationCheck(
                name="Telegram Bot",
                passed=False,
                message="Bot token not set. Set NM_TELEGRAM_BOT_TOKEN.",
            )
        
        # Validate by calling getMe
        try:
            url = f"https://api.telegram.org/bot{token}/getMe"
            req = urllib.request.Request(url)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                
                if data.get("ok"):
                    bot_info = data.get("result", {})
                    username = bot_info.get("username", "unknown")
                    
                    return VerificationCheck(
                        name="Telegram Bot",
                        passed=True,
                        message=f"Valid token for @{username}",
                        details={"username": username, "bot_id": bot_info.get("id")},
                    )
                else:
                    return VerificationCheck(
                        name="Telegram Bot",
                        passed=False,
                        message=f"Invalid token: {data.get('description', 'unknown error')}",
                    )
        except Exception as e:
            return VerificationCheck(
                name="Telegram Bot",
                passed=False,
                message=f"Failed to validate: {e}",
            )
    
    async def verify_whatsapp_bridge(self) -> VerificationCheck:
        """Verify WhatsApp bridge connectivity."""
        host = os.getenv("NM_WHATSAPP_BRIDGE_HOST", "127.0.0.1")
        port = int(os.getenv("NM_WHATSAPP_BRIDGE_PORT", "8787"))
        
        try:
            import socket
            sock = socket.create_connection((host, port), timeout=3)
            sock.close()
            
            return VerificationCheck(
                name="WhatsApp Bridge",
                passed=True,
                message=f"Bridge reachable at {host}:{port}",
                details={"host": host, "port": port},
            )
        except Exception as e:
            return VerificationCheck(
                name="WhatsApp Bridge",
                passed=False,
                message=f"Bridge unreachable at {host}:{port} - {e}",
            )
    
    async def verify_database(self) -> VerificationCheck:
        """Verify database integrity."""
        db_path = Path.home() / ".nomorals/nomorals.db"
        
        if not db_path.exists():
            return VerificationCheck(
                name="Database",
                passed=False,
                message=f"Database not found at {db_path}",
            )
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Check integrity
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            
            # Get table count
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            conn.close()
            
            if result == "ok":
                return VerificationCheck(
                    name="Database",
                    passed=True,
                    message=f"Healthy, {table_count} tables",
                    details={"path": str(db_path), "tables": table_count},
                )
            else:
                return VerificationCheck(
                    name="Database",
                    passed=False,
                    message=f"Integrity check failed: {result}",
                )
        except Exception as e:
            return VerificationCheck(
                name="Database",
                passed=False,
                message=f"Failed to check: {e}",
            )
    
    async def verify_disk_space(self) -> VerificationCheck:
        """Verify sufficient disk space."""
        try:
            home = Path.home()
            stat = shutil.disk_usage(str(home))
            
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            used_percent = (stat.used / stat.total) * 100
            
            if free_gb < 1.0:
                return VerificationCheck(
                    name="Disk Space",
                    passed=False,
                    message=f"Low disk space: {free_gb:.1f}GB free ({used_percent:.0f}% used)",
                    details={"free_gb": free_gb, "total_gb": total_gb, "used_percent": used_percent},
                )
            
            return VerificationCheck(
                name="Disk Space",
                passed=True,
                message=f"{free_gb:.1f}GB free ({used_percent:.0f}% used of {total_gb:.0f}GB)",
                details={"free_gb": free_gb, "total_gb": total_gb, "used_percent": used_percent},
            )
        except Exception as e:
            return VerificationCheck(
                name="Disk Space",
                passed=False,
                message=f"Failed to check: {e}",
            )
    
    async def verify_network(self) -> VerificationCheck:
        """Verify network connectivity."""
        test_urls = [
            "https://api.telegram.org",
            "https://openrouter.ai",
            "https://huggingface.co",
        ]
        
        successes = []
        failures = []
        
        for url in test_urls:
            try:
                req = urllib.request.Request(url, method="HEAD")
                with urllib.request.urlopen(req, timeout=5):
                    successes.append(url)
            except Exception:
                failures.append(url)
        
        if not successes:
            return VerificationCheck(
                name="Network",
                passed=False,
                message="No internet connectivity",
            )
        
        if failures:
            return VerificationCheck(
                name="Network",
                passed=True,
                message=f"Partial connectivity ({len(successes)}/{len(test_urls)} reachable)",
                details={"reachable": successes, "unreachable": failures},
            )
        
        return VerificationCheck(
            name="Network",
            passed=True,
            message="Full connectivity",
            details={"reachable": successes},
        )
    
    async def verify_python_version(self) -> VerificationCheck:
        """Verify Python version."""
        import sys
        
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if sys.version_info < (3, 9):
            return VerificationCheck(
                name="Python Version",
                passed=False,
                message=f"Python {version} (need 3.9+)",
            )
        
        return VerificationCheck(
            name="Python Version",
            passed=True,
            message=f"Python {version}",
            details={"version": version},
        )
    
    async def verify_torch(self) -> VerificationCheck:
        """Verify PyTorch installation."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
            
            return VerificationCheck(
                name="PyTorch",
                passed=True,
                message=f"v{torch.__version__} (CUDA: {'yes' if cuda_available else 'no'})",
                details={
                    "version": torch.__version__,
                    "cuda_available": cuda_available,
                    "cuda_device": cuda_device,
                },
            )
        except ImportError:
            return VerificationCheck(
                name="PyTorch",
                passed=False,
                message="Not installed (optional, needed for local models)",
            )
        except Exception as e:
            return VerificationCheck(
                name="PyTorch",
                passed=False,
                message=f"Installed but broken: {e}",
            )
