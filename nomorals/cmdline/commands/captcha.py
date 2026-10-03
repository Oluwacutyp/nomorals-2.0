"""``nm captcha`` — captcha detect/solve/status."""

from __future__ import annotations

import argparse
import sys
from typing import Any



def _cmd_captcha(args: argparse.Namespace, context: Any) -> int:
    """Route `nm captcha` subcommands."""
    import json as _json
    from ...tools.captcha import (CaptchaKind, CaptchaChallenge, CaptchaError,
                                ServiceBackend, TakeoverBackend, detect,
                                solve, _audit_path)

    settings = getattr(context, "settings", None)
    action = args.captcha_action

    def _out(obj: Any) -> int:
        if args.json:
            print(_json.dumps(obj, indent=2))
        else:
            print(_json.dumps(obj, indent=2))
        return 0

    if action == "status":
        svc = ServiceBackend()
        return _out({
            "backends": ["service", "takeover", "detect"],
            "service_available": svc.available(),
            "api_key_configured": svc.available(),
            "audit_log": _audit_path(settings),
            "kinds": list(CaptchaKind.ALL),
        })

    if action == "detect":
        html, url = "", args.url or ""
        if args.html:
            with open(args.html, encoding="utf-8", errors="replace") as fh:
                html = fh.read()
        elif args.url:
            import urllib.request
            req = urllib.request.Request(
                args.url, headers={"User-Agent": "nomorals-captcha/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                html = resp.read().decode("utf-8", "replace")
                url = resp.geturl()
        else:
            print("nm captcha detect needs --html FILE or --url URL",
                  file=sys.stderr)
            return 2
        found = detect(html, url)
        return _out({"count": len(found),
                     "challenges": [c.summary() for c in found]})

    if action == "solve":
        if args.kind not in CaptchaKind.ALL:
            print(f"unknown kind {args.kind!r} "
                  f"(want one of: {list(CaptchaKind.ALL)})", file=sys.stderr)
            return 2
        image_bytes, image_url = b"", ""
        if args.image:
            if args.image.startswith(("http://", "https://")):
                from ...tools.captcha import fetch_image_bytes
                image_url = args.image
                image_bytes = fetch_image_bytes(args.image)
            else:
                with open(args.image, "rb") as fh:
                    image_bytes = fh.read()
        challenge = CaptchaChallenge(
            kind=args.kind, sitekey=args.sitekey, page_url=args.url,
            image_url=image_url, image_bytes=image_bytes,
            action=args.v3_action, min_score=args.min_score)
        try:
            # Solver on by default; --no-solver or NM_CAPTCHA_SOLVER=0 disables.
            import os as _os
            solver_on = args.solver_enabled
            if solver_on is None:
                solver_on = _os.environ.get("NM_CAPTCHA_SOLVER", "1") != "0"
            result = solve(challenge, backend=args.backend, settings=settings,
                           solver_enabled=solver_on)
        except CaptchaError as exc:
            print(f"captcha solve failed: {exc}", file=sys.stderr)
            return 1
        out = result.to_dict()
        if result.takeover and not args.json:
            print("Takeover: " + result.detail)
            print(_json.dumps(out, indent=2))
            return 3
        return _out(out)

    print(f"unknown captcha action: {action}", file=sys.stderr)
    return 2
