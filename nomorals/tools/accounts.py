"""Account-provisioning spine tools — temp email + temp SMS.

Plain-language access to the keyless verification helpers in
``nomorals.accounts``: grab a temp email or phone number, wait for the
verification code. Used by signup flows and directly by the owner.
"""

from __future__ import annotations

from typing import Any


def register(registry: Any) -> None:
    from ..core.policy import Capability

    @registry.register(
        "temp_email",
        description=(
            "Grab a temporary email address for verification flows. "
            "('get me a temp email', 'need an email for signup'). Keyless "
            "providers (1secmail, GuerrillaMail). Use temp_email_code to "
            "wait for the verification code."
        ),
        capability=Capability.NET_OUT,
    )
    def temp_email(*, provider: str = "") -> dict[str, Any]:
        from ..accounts.temp_mail import grab_address_cascade
        try:
            a = grab_address_cascade([provider] if provider else None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "address": a.address, "login": a.login,
                "domain": a.domain, "provider": a.provider,
                "token": a.token}

    @registry.register(
        "temp_email_code",
        description=(
            "Wait for a verification code on a temp email address. "
            "Pass the dict from temp_email. Returns the extracted code."
        ),
        capability=Capability.NET_OUT,
    )
    def temp_email_code(email_info: dict[str, Any], *,
                        sender_hint: str = "",
                        timeout_s: float = 180) -> dict[str, Any]:
        from ..accounts.temp_mail import TempAddress, wait_code
        addr = TempAddress(
            address=email_info.get("address", ""),
            login=email_info.get("login", ""),
            domain=email_info.get("domain", ""),
            provider=email_info.get("provider", "1secmail"),
            token=email_info.get("token", ""))
        code = wait_code(addr, sender_hint=sender_hint, timeout_s=timeout_s)
        return {"ok": bool(code), "code": code or ""}

    @registry.register(
        "temp_sms_number",
        description=(
            "Grab a temporary phone number for SMS verification. "
            "('get me a temp number', 'need a number for verification'). "
            "Keyless providers. Use temp_sms_code to wait for the code."
        ),
        capability=Capability.NET_OUT,
    )
    def temp_sms_number(*, country: str = "us",
                        provider: str = "") -> dict[str, Any]:
        from ..accounts.temp_sms import grab_number_cascade
        try:
            n = grab_number_cascade(
                country=country,
                providers=[provider] if provider else None)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        # grab_number_cascade returns a TempNumber or dict — normalize
        if hasattr(n, "number"):
            return {"ok": True, "number": n.number, "masked": n.masked,
                    "country": n.country, "provider": n.provider,
                    "inbox_id": n.inbox_id}
        return {"ok": True, **dict(n)}

    @registry.register(
        "temp_sms_code",
        description=(
            "Wait for an SMS verification code on a temp number. "
            "Pass the dict from temp_sms_number. Returns the code."
        ),
        capability=Capability.NET_OUT,
    )
    def temp_sms_code(number_info: dict[str, Any], *,
                      sender_hint: str = "",
                      timeout_s: float = 180) -> dict[str, Any]:
        from ..accounts.temp_sms import wait_code
        try:
            code = wait_code(number_info, sender_hint=sender_hint,
                             timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)}
        return {"ok": bool(code), "code": code or ""}
