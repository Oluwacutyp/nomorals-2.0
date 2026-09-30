"""Native (C) acceleration layer — the ONE place the project compiles code.

The hash cracker's inner loop is speed-critical; everything else stays
Python. ``nmhash.c`` implements MD4/MD5/SHA-1/SHA-256; ``loader.py`` builds
and verifies the shared object (failing open to pure Python when no
compiler is available).

Only algorithms where C genuinely beats the Python fallback are exposed as
fast paths — see ``loader.FAST_ALGOS`` (currently NTLM, where OpenSSL 3
removing MD4 leaves ~24x on the table).
"""

from __future__ import annotations

from .loader import available, backend_name, build_report, fast_hash_fn

__all__ = ["available", "backend_name", "build_report", "fast_hash_fn"]
