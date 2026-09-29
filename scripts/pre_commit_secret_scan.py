#!/usr/bin/env python3
"""Pre-commit hook to scan for secrets/credentials before they enter git history.

Scans staged files for:
- API keys (sk-, ak-, etc.)
- Private keys (BEGIN.*PRIVATE KEY)
- Passwords/tokens in common formats
- AWS access keys
- GitHub/GitLab tokens
- Telegram bot tokens
- Database connection strings with credentials

Usage:
    Add to .git/hooks/pre-commit or run manually:
        python scripts/pre_commit_secret_scan.py
"""

import re
import subprocess
import sys
from pathlib import Path

# Patterns that indicate secrets
SECRET_PATTERNS = [
    # API keys
    (r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']?[a-zA-Z0-9]{20,}["\']?', "API key"),
    
    # AWS access keys
    (r'AKIA[0-9A-Z]{16}', "AWS access key"),
    
    # Private keys
    (r'-----BEGIN.*PRIVATE KEY-----', "Private key"),
    
    # GitHub tokens
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
    (r'github_pat_[a-zA-Z0-9_]{82}', "GitHub fine-grained token"),
    
    # GitLab tokens
    (r'glpat-[a-zA-Z0-9\-]{20}', "GitLab personal access token"),
    
    # Telegram bot tokens
    (r'\d{8,10}:[a-zA-Z0-9_-]{35}', "Telegram bot token"),
    
    # OpenAI API keys
    (r'sk-[a-zA-Z0-9]{48}', "OpenAI API key"),
    
    # Stripe keys
    (r'sk_live_[a-zA-Z0-9]{24}', "Stripe secret key"),
    (r'rk_live_[a-zA-Z0-9]{24}', "Stripe restricted key"),
    
    # Database connection strings with credentials
    (r'(mysql|postgres|postgresql|mongodb)://[^:]+:[^@]+@', "Database connection string with credentials"),
    
    # Generic password assignments
    (r'(?i)(password|passwd|pwd)\s*[:=]\s*["\'][^"\']{8,}["\']', "Password"),
    
    # Generic token assignments
    (r'(?i)(token|secret|api_secret)\s*[:=]\s*["\'][a-zA-Z0-9_\-]{20,}["\']', "Token/secret"),
    
    # OpenRouter API keys
    (r'sk-or-v1-[a-zA-Z0-9]{64}', "OpenRouter API key"),
]

# Files to skip
SKIP_PATTERNS = [
    r'\.git/',
    r'node_modules/',
    r'\.venv/',
    r'__pycache__/',
    r'\.pytest_cache/',
    r'tests/',  # Test files often have fake credentials
    r'docs/',   # Documentation may have examples
    r'\.md$',   # Markdown files
    r'pre_commit_secret_scan\.py$',  # This file itself
]


def get_staged_files() -> list[str]:
    """Get list of staged files."""
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        print("Error: Failed to get staged files")
        sys.exit(1)
    
    return [f for f in result.stdout.strip().split("\n") if f]


def should_skip_file(filepath: str) -> bool:
    """Check if file should be skipped."""
    for pattern in SKIP_PATTERNS:
        if re.search(pattern, filepath):
            return True
    return False


def scan_file(filepath: str) -> list[tuple[str, str, int]]:
    """Scan a file for secrets.
    
    Returns:
        List of (pattern_name, matched_text, line_number)
    """
    findings = []
    
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line_num, line in enumerate(f, 1):
                for pattern, name in SECRET_PATTERNS:
                    if re.search(pattern, line):
                        # Truncate matched text for display
                        match = re.search(pattern, line).group(0)
                        display = match[:30] + "..." if len(match) > 30 else match
                        findings.append((name, display, line_num))
    except Exception as e:
        print(f"Warning: Could not read {filepath}: {e}")
    
    return findings


def main() -> int:
    """Main entry point."""
    print("🔍 Scanning for secrets...")
    
    staged_files = get_staged_files()
    
    if not staged_files:
        print("✅ No staged files to scan")
        return 0
    
    all_findings = []
    
    for filepath in staged_files:
        if should_skip_file(filepath):
            continue
        
        findings = scan_file(filepath)
        if findings:
            all_findings.append((filepath, findings))
    
    if not all_findings:
        print(f"✅ Scanned {len(staged_files)} files, no secrets found")
        return 0
    
    # Report findings
    print("\n❌ SECRETS DETECTED - Commit blocked\n")
    
    for filepath, findings in all_findings:
        print(f"📄 {filepath}")
        for name, match, line_num in findings:
            print(f"   Line {line_num}: {name}")
            print(f"      {match}")
        print()
    
    print("To commit anyway (if these are false positives):")
    print("   git commit --no-verify")
    print()
    
    return 1


if __name__ == "__main__":
    sys.exit(main())
