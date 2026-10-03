"""Twilio connector (SMS + voice).

Drives Twilio's REST API (https://www.twilio.com/docs) with the framework
HttpClient — no twilio helper library dependency.

Auth: HTTP Basic (``BASIC``) with the Account SID as the username and the
auth token as the password. The Authorization header is built manually
with base64, since the framework HttpClient has no basic-auth mode.
``connect()`` validates against ``GET
/2010-04-01/Accounts/{sid}.json`` and vaults both parts together.

SMS is ``POST /2010-04-01/Accounts/{sid}/Messages.json`` (form-encoded,
``From``/``To``/``Body``); calls are ``POST .../Calls.json``
(``From``/``To``/``Url`` pointing at a TwiML document). Both cost real
money per segment/minute and are confirmation-gated.

Numbers use E.164 (``+15551234567``). Twilio trial accounts can only
message verified numbers and prepend a trial notice — that is Twilio's
rule, not this connector's.
"""

from __future__ import annotations

import base64
import time
from typing import Any

from ..core.errors import NoMoralsError, RateLimited
from ..core.logging_setup import get_logger
from ._confirm import confirm_or_checkpoint
from .auth import prompt_secret
from .base import (
    AuthMethod,
    Connector,
    ConnectorError,
    ConnectorStatus,
    ConnectResult,
)
from .registry import register_connector

__all__ = ["TwilioConnector", "TwilioError"]

_log = get_logger(__name__)

API_BASE = "https://api.twilio.com"
SID_ENV = "TWILIO_ACCOUNT_SID"
TOKEN_ENV = "TWILIO_AUTH_TOKEN"
DOCS_URL = "https://www.twilio.com/docs"
MAX_SMS_CHARS = 1600


class TwilioError(ConnectorError):
    """A Twilio REST call failed."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 0,
        twilio_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.twilio_code = twilio_code


@register_connector
class TwilioConnector(Connector):
    """Devon's Twilio SMS + voice adapter."""

    id = "twilio"
    name = "Twilio"
    description = (
        "Twilio REST API: send SMS, place voice calls via a TwiML URL, "
        "and list recent messages. Authenticates with HTTP Basic (Account "
        "SID + auth token)."
    )
    auth_methods = (AuthMethod.BASIC,)

    # ── lifecycle ────────────────────────────────────────────────

    def connect(
        self,
        *,
        account_sid: str | None = None,
        auth_token: str | None = None,
        db: Any = None,
        context: Any = None,
    ) -> ConnectResult:
        """Validate an Account SID + auth token pair and vault them."""
        existing = self._load_credential()
        if existing is not None:
            raise ConnectorError(
                "twilio is already connected — one account per service. "
                "Disconnect first to switch accounts."
            )
        sid = (account_sid or "").strip() or prompt_secret(
            "Twilio Account SID (AC…)", env_var=SID_ENV
        )
        secret = (auth_token or "").strip() or prompt_secret(
            "Twilio auth token", env_var=TOKEN_ENV
        )
        if not sid or not secret:
            raise ConnectorError(
                "need both the Account SID and the auth token — nothing "
                "to connect with"
            )
        account = self._api(
            "GET", f"/2010-04-01/Accounts/{sid}.json",
            basic=(sid, secret),
        )
        friendly = str(account.get("friendly_name", sid))
        status = str(account.get("status", ""))
        self._store_credential(
            sid,
            secret,
            credential_type="basic",
            scopes=["sms.send", "calls.make", "messages.read"],
            metadata={"friendly_name": friendly, "status": status},
        )
        _log.info("twilio connected for account %s", sid)
        return ConnectResult(
            ok=True,
            account=friendly,
            scopes=["sms.send", "calls.make", "messages.read"],
            message=(
                f"connected to Twilio account {friendly} (SID {sid}, "
                f"status {status}). Both parts are in the encrypted vault. "
                "SMS and calls cost real money — every send is gated "
                "behind explicit owner confirmation."
            ),
        )

    def disconnect(self) -> None:
        self._clear_credential()

    def status(self) -> ConnectorStatus:
        cred = self._load_credential()
        if cred is None:
            return ConnectorStatus(
                connected=False,
                detail="not connected — run `nm connectors connect "
                       "--name twilio`",
            )
        try:
            account = self._api(
                "GET", f"/2010-04-01/Accounts/{cred.username}.json"
            )
        except TwilioError as exc:
            return ConnectorStatus(
                connected=False,
                account=cred.username,
                scopes=list((cred.metadata or {}).get("scopes", [])),
                last_checked=time.time(),
                detail=f"credential rejected ({exc}): rotate the auth "
                       "token in the Twilio console and reconnect",
            )
        return ConnectorStatus(
            connected=True,
            account=str(account.get("friendly_name", cred.username)),
            scopes=list((cred.metadata or {}).get("scopes", [])),
            last_checked=time.time(),
            detail=f"account status {account.get('status')}",
        )

    def test_connection(self) -> bool:
        cred = self._load_credential()
        if cred is None:
            return False
        try:
            self._api("GET", f"/2010-04-01/Accounts/{cred.username}.json")
            return True
        except ConnectorError:
            return False

    # ── Twilio API ───────────────────────────────────────────────

    def get_account(self) -> dict[str, Any]:
        """Account details (``GET /2010-04-01/Accounts/{sid}.json``)."""
        data = self._api(
            "GET",
            f"/2010-04-01/Accounts/{self._require_credential().username}.json",
        )
        return data if isinstance(data, dict) else {}

    def send_sms(
        self,
        to: str,
        body: str,
        *,
        from_number: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Send an SMS (``POST .../Messages.json``, form-encoded).

        Numbers are E.164. Caps at 1600 characters (longer becomes many
        billed segments — refused, not silently split). ``from_number``
        defaults to the first owned number; pass it explicitly when the
        account owns several. Consequential: gated behind explicit owner
        confirmation.
        """
        to = (to or "").strip()
        if not to:
            raise ConnectorError("a destination number is required")
        if not (body or ""):
            raise ConnectorError("refusing to send an empty SMS")
        if len(body) > MAX_SMS_CHARS:
            raise ConnectorError(
                f"SMS is {len(body)} chars; refusing above "
                f"{MAX_SMS_CHARS} — it would bill as many segments"
            )
        sender = from_number or ""
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="send_sms",
            title=f"Send SMS to {to}",
            instructions="\n".join([
                "Devon wants to send this SMS via your Twilio account.",
                "Review it — sending costs money and is final.",
                f"From: {sender or '(first owned Twilio number)'}",
                f"To: {to}",
                "",
                body,
            ]),
            resume_state={"to": to, "body": body, "from_number": from_number},
        )
        sender = sender or self._default_from_number()
        cred = self._require_credential()
        data = self._api(
            "POST",
            f"/2010-04-01/Accounts/{cred.username}/Messages.json",
            form={"From": sender, "To": to, "Body": body},
        )
        return data if isinstance(data, dict) else {}

    def make_call(
        self,
        to: str,
        twiml_url: str,
        *,
        from_number: str = "",
        confirmed: bool = False,
        db: Any = None,
        context: Any = None,
    ) -> dict[str, Any]:
        """Place a voice call (``POST .../Calls.json``, form-encoded).

        Twilio fetches TwiML from ``twiml_url`` when the call is answered
        — it must be a public HTTPS URL serving valid TwiML. Calls bill
        per minute. Consequential: gated behind explicit owner
        confirmation.
        """
        to = (to or "").strip()
        twiml_url = (twiml_url or "").strip()
        if not to:
            raise ConnectorError("a destination number is required")
        if not twiml_url:
            raise ConnectorError(
                "twiml_url is required — Twilio needs a public TwiML "
                "document to know what to say on the call"
            )
        if not twiml_url.lower().startswith("https://"):
            raise ConnectorError(
                "twiml_url must be a public https URL"
            )
        sender = from_number or ""
        confirm_or_checkpoint(
            self,
            confirmed=confirmed,
            db=db,
            context=context,
            stage="make_call",
            title=f"Call {to} via Twilio",
            instructions="\n".join([
                "Devon wants to place this voice call via your Twilio "
                "account.",
                "Review it — calls cost money per minute and the "
                "recipient will be called.",
                f"From: {sender or '(first owned Twilio number)'}",
                f"To: {to}",
                f"TwiML URL: {twiml_url}",
            ]),
            resume_state={"to": to, "twiml_url": twiml_url,
                          "from_number": from_number},
        )
        sender = sender or self._default_from_number()
        cred = self._require_credential()
        data = self._api(
            "POST",
            f"/2010-04-01/Accounts/{cred.username}/Calls.json",
            form={"From": sender, "To": to, "Url": twiml_url},
        )
        return data if isinstance(data, dict) else {}

    def list_messages(
        self,
        *,
        limit: int = 20,
        to: str = "",
        from_number: str = "",
    ) -> list[dict[str, Any]]:
        """Recent messages (``GET .../Messages.json``)."""
        params: dict[str, Any] = {"PageSize": max(1, min(limit, 1000))}
        if to:
            params["To"] = to
        if from_number:
            params["From"] = from_number
        cred = self._require_credential()
        data = self._api(
            "GET",
            f"/2010-04-01/Accounts/{cred.username}/Messages.json",
            params=params,
        )
        messages = data.get("messages") if isinstance(data, dict) else None
        return messages if isinstance(messages, list) else []

    def list_numbers(self) -> list[dict[str, Any]]:
        """Owned phone numbers (``GET .../IncomingPhoneNumbers.json``)."""
        cred = self._require_credential()
        data = self._api(
            "GET",
            f"/2010-04-01/Accounts/{cred.username}/IncomingPhoneNumbers.json",
        )
        numbers = data.get("incoming_phone_numbers") \
            if isinstance(data, dict) else None
        return numbers if isinstance(numbers, list) else []

    # ── HTTP plumbing ────────────────────────────────────────────

    def _require_credential(self):
        cred = self._load_credential()
        if cred is None:
            raise ConnectorError(
                "twilio is not connected — run "
                "`nm connectors connect --name twilio` first"
            )
        return cred

    def _basic_auth_header(self, sid: str, secret: str) -> dict[str, str]:
        """HTTP Basic: base64(Account SID : auth token)."""
        pair = f"{sid}:{secret}".encode("utf-8")
        return {"Authorization": f"Basic {base64.b64encode(pair).decode()}"}

    def _default_from_number(self) -> str:
        """First owned number — fail fast when the account has none."""
        numbers = self.list_numbers()
        for number in numbers:
            phone = str(number.get("phone_number", ""))
            if phone:
                return phone
        raise ConnectorError(
            "no owned Twilio number to send from — buy or port one in "
            "the Twilio console, or pass from_number= explicitly"
        )

    def _api(
        self,
        method: str,
        path: str,
        form: dict[str, Any] | None = None,
        *,
        params: dict[str, Any] | None = None,
        basic: tuple[str, str] | None = None,
    ) -> Any:
        """One Twilio REST call; failures become TwilioError.

        Twilio errors arrive as ``{"code", "message", "status",
        "more_info"}`` — unwrapped below.
        """
        cred = self._require_credential() if basic is None else None
        sid, secret = basic or (cred.username, cred.password)
        url = f"{API_BASE}{path}"
        headers = self._basic_auth_header(sid, secret)
        try:
            if method == "GET":
                resp = self.http.get(url, headers=headers, params=params)
            elif method == "POST":
                resp = self.http.post_form(url, form or {}, headers=headers)
            else:
                raise ConnectorError(f"unsupported method {method}")
        except RateLimited as exc:
            raise TwilioError(
                "twilio rate limited (429) — back off "
                f"~{exc.retry_after:.0f}s before retrying",
                status_code=429,
            ) from exc
        except NoMoralsError as exc:
            raise TwilioError(f"twilio request failed: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - network layer is opaque
            raise TwilioError(f"twilio request failed: {exc}") from exc
        if resp.status == 401:
            detail, code = self._error_detail(resp)
            raise TwilioError(
                "twilio rejected the credentials (401): "
                f"{detail or 'bad Account SID or auth token'} — rotate "
                "the token in the Twilio console and reconnect",
                status_code=401,
                twilio_code=code,
            )
        if resp.status == 429:
            raise TwilioError(
                "twilio rate limited (429) — back off before retrying",
                status_code=429,
            )
        if not resp.ok:
            detail, code = self._error_detail(resp)
            raise TwilioError(
                f"twilio {method} {path} failed ({resp.status}): "
                f"{detail or resp.text[:200]}",
                status_code=resp.status,
                twilio_code=code,
            )
        try:
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - invalid JSON is an error
            raise TwilioError(
                f"twilio {method} {path} returned invalid JSON"
            ) from exc

    @staticmethod
    def _error_detail(resp: Any) -> tuple[str, int]:
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001 - fall back to raw text
            return resp.text[:200], 0
        if isinstance(body, dict):
            message = str(body.get("message", ""))[:200]
            try:
                code = int(body.get("code", 0))
            except (TypeError, ValueError):
                code = 0
            return message or resp.text[:200], code
        return resp.text[:200], 0
