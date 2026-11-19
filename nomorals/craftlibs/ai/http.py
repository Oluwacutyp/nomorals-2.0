"""One TLS setting and one proxy policy for every AI call (Craft-AI
port of ai-http ``lib.rs``).

Every adapter and downloader builds its own HTTP agent, so the rules
that must hold everywhere live here once instead of in each builder.

The TLS rule: trust what the operating system trusts, through
Devon's single verifying context in :mod:`nomorals.core.http` — one
context per CA bundle, cached per process. The alternative, a bundled
list of public roots, fails on a machine where antivirus or a company
proxy re-signs HTTPS with its own root, which the operating system
has installed but a bundled list has not. The launcher hit exactly
that.

The proxy rule: a loopback agent takes no proxy at all, whatever the
environment says — loopback traffic (local model servers, ComfyUI
prompts and images) is cleartext the consent architecture promises
never leaves the computer. Hosted providers keep the environment's
proxy: a person or company that sets a proxy variable has arranged
its network that way, and refusing would break those installs.

:func:`scan_agents` is the workspace scan (upstream's
``tests/every_agent_uses_it.rs``): it fails if any file under the AI
path opens a URL directly instead of through an agent built here, or
disables TLS verification, so a new adapter cannot quietly go back
to a different policy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ...core.http import (
    HttpClient,
    get_default_proxy,
    tls_cafile,
)
from ...core.http import (
    _build_tls_context as _shared_tls_context,
)
from .endpoint import Endpoint
from .error import AiError

__all__ = [
    "agent_for",
    "hosted_proxy",
    "loopback_proxy",
    "proxy_for",
    "scan_agents",
    "tls_cafile",
    "tls_context",
]

_URL_OPEN = re.compile(r"\burlopen\s*\(")
_TLS_OFF = re.compile(r"verify_tls\s*=\s*False")


def tls_context():
    """The TLS setting for an agent: the shared verifying context
    from :mod:`nomorals.core.http`, built from the one CA bundle
    that module resolves (``SSL_CERT_FILE``, else certifi's bundle,
    else the platform store)."""
    return _shared_tls_context(tls_cafile())


def loopback_proxy() -> str:
    """No proxy, whatever the environment says.

    The setting for an agent whose traffic stays on the person's own
    machine. A default agent would send even loopback requests to
    the host a proxy variable names, and that would move cleartext
    the consent architecture promises never leaves the computer onto
    a third host."""
    return ""


def hosted_proxy() -> str:
    """The proxy the environment names, if it names one.

    The setting for an agent that talks to hosted providers: the
    process-wide default first (``proxy_set``), else the
    ``HTTPS_PROXY`` / ``HTTP_PROXY`` variables, else direct."""
    default = get_default_proxy()
    if default:
        return default
    return (os.environ.get("HTTPS_PROXY")
            or os.environ.get("HTTP_PROXY") or "")


def proxy_for(base: str) -> str:
    """The proxy setting for an agent that talks to one base
    address, decided from the address.

    A loopback base — loopback by :meth:`Endpoint.is_loopback`, the
    same rule that decides whether a connection needs consent at
    all — takes :func:`loopback_proxy`. Any other base, one that
    will not parse as an endpoint included, takes
    :func:`hosted_proxy`: only an address proven to be this machine
    is exempt from the environment's proxy."""
    try:
        endpoint = Endpoint.parse(base)
    except (ValueError, AiError):
        return hosted_proxy()
    if endpoint.is_loopback():
        return loopback_proxy()
    return hosted_proxy()


def agent_for(base: str) -> HttpClient:
    """An agent for one base address: the shared TLS policy, the
    proxy rule for that address, and loopback allowed through the
    SSRF guard only when the base is loopback (a hosted base never
    gets it). Verification is never off."""
    try:
        is_loopback = Endpoint.parse(base).is_loopback()
    except (ValueError, AiError):
        is_loopback = False
    return HttpClient(verify_tls=True, proxy_url=proxy_for(base),
                      allow_private_ips=is_loopback,
                      force_direct=is_loopback)


def scan_agents(paths) -> list:
    """Fail if any Python file under ``paths`` opens a URL directly
    instead of through :func:`agent_for`, or disables TLS
    verification. Returns a list of ``path:line: reason`` strings;
    an empty list means every agent in the path takes both settings
    from here."""
    violations: list[str] = []
    for root in paths:
        root = Path(root)
        files = ([root] if root.is_file()
                 else sorted(root.rglob("*.py")))
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if _URL_OPEN.search(line):
                    violations.append(
                        f"{path}:{number}: direct urlopen call, "
                        "use an agent from craftlibs.ai.http")
                elif _TLS_OFF.search(line):
                    violations.append(
                        f"{path}:{number}: TLS verification disabled, "
                        "use the shared tls_context")
    return violations
