"""Connection drafts (Craft-AI port of ai-core ``draft.rs`` +
``draft/presets.rs`` + ``draft/ids.rs``).

The form's state: a label, a provider, an address, a key box and the
models found by a test. :meth:`ConnectionDraft.prepare` checks
everything and builds the connection; a problem is a
:class:`DraftError` naming the field the form should mark. The typed
key is a plain string only because a text box needs one — ``__repr__``
never prints it, and ``prepare`` turns it into a ``KeyChange``. A
test result for an address no longer in the form is refused without
changing the draft's models.
"""

from __future__ import annotations

from enum import Enum

from .connection import (
    Auth,
    Billing,
    Connection,
    Locality,
    check_key_header,
)
from .endpoint import Endpoint
from .error import AiError
from .ids import ConnectionId, SecretRef, validate_id
from .model import CapabilitySpec, InputRole, ModelInfo

__all__ = [
    "PRESETS",
    "ConnectionDraft",
    "DraftError",
    "Field",
    "KeyChange",
    "MAX_ENDPOINT_BYTES",
    "MAX_KEY_BYTES",
    "MAX_LABEL_CHARS",
    "Prepared",
    "Preset",
    "check_key",
    "new_id",
    "preset_models",
]

MAX_LABEL_CHARS = 80
MAX_ENDPOINT_BYTES = 2048
MAX_KEY_BYTES = 4096


class Field(Enum):
    """Which box a problem belongs to, so a form can mark it."""

    LABEL = "label"
    PROVIDER = "provider"
    ADDRESS = "address"
    KEY = "key"
    KEY_HEADER = "key_header"


class DraftError(ValueError):
    """Why a draft was refused. The message is safe to show."""

    def __init__(self, field: Field, message: str) -> None:
        super().__init__(message)
        self.field = field
        self.message = message


class KeyChange:
    """What to do with the stored key."""

    def __init__(self, kind: str, value: str | None = None) -> None:
        self.kind = kind
        self.value = value

    @classmethod
    def keep(cls) -> KeyChange:
        return cls("keep")

    @classmethod
    def set(cls, value: str) -> KeyChange:
        return cls("set", value)

    @classmethod
    def clear(cls) -> KeyChange:
        return cls("clear")

    def __repr__(self) -> str:
        if self.kind == "set":
            return "KeyChange.set(<key>)"
        return f"KeyChange.{self.kind}"


class Prepared:
    """A checked connection and what to do with its key."""

    def __init__(self, connection: Connection,
                 key_change: KeyChange) -> None:
        self.connection = connection
        self.key_change = key_change
        # Upstream spells the key change ``Prepared.key``; keep both.
        self.key = key_change


class Preset:
    """A starting point for the form."""

    def __init__(self, key: str, name: str, provider: str, endpoint: str,
                 billing: Billing, needs_key: bool, note: str) -> None:
        self.key = key
        self.name = name
        self.provider = provider
        self.endpoint = endpoint
        self.billing = billing
        self.needs_key = needs_key
        self.note = note


PRESETS = [
    Preset("openai", "OpenAI", "openai", "https://api.openai.com/v1",
           Billing.METERED, True,
           "Your own OpenAI API key. Billed by OpenAI."),
    Preset("openrouter", "OpenRouter", "openrouter",
           "https://openrouter.ai/api/v1", Billing.METERED, True,
           "One key for many hosted models."),
    Preset("bfl", "Black Forest Labs", "bfl", "https://api.bfl.ai/v1",
           Billing.METERED, True,
           "FLUX 3 image and video generation. "
           "Billed by Black Forest Labs."),
    Preset("seedance", "Seedance (BytePlus ModelArk)", "seedance",
           "https://ark.ap-southeast.bytepluses.com/api/v3",
           Billing.METERED, True,
           "Seedance 2.0 text-to-video with your BytePlus ModelArk "
           "API key."),
    Preset("vertex-express", "Vertex AI (express mode)", "vertex-express",
           "https://aiplatform.googleapis.com/v1", Billing.METERED, True,
           "Gemini image generation. API key only; full Vertex uses "
           "different credentials. Key cannot be tested without a "
           "paid request."),
    Preset("artcraft-omni", "ArtCraft Omni API key", "artcraft-omni",
           "https://api.storyteller.ai", Billing.UNKNOWN, True,
           "Staff-enabled API key for ArtCraft image and video "
           "generation. Subscription allowance is unverified; no free "
           "key test is documented."),
    Preset("ollama", "Ollama", "ollama", "http://127.0.0.1:11434/v1",
           Billing.FREE, False,
           "Models running on this computer. Nothing leaves it."),
    Preset("lmstudio", "LM Studio", "lmstudio",
           "http://127.0.0.1:1234/v1", Billing.FREE, False,
           "A LM Studio server on this computer."),
    Preset("llamacpp", "llama.cpp", "llamacpp",
           "http://127.0.0.1:8080/v1", Billing.FREE, False,
           "A llama.cpp server on this computer."),
    Preset("comfyui", "ComfyUI (local)", "comfyui",
           "http://127.0.0.1:8188", Billing.FREE, False,
           "Local workflows on this computer. Import workflow JSON "
           "files first; each becomes a model."),
    Preset("custom", "Other (OpenAI compatible)", "openai-compatible", "",
           Billing.UNKNOWN, True,
           "Any server that speaks the OpenAI API."),
]


def preset_models(preset: Preset) -> list:
    """The fixed model list a built-in preset offers before its
    connection is tested. Unverified, by design: the list says what
    the preset documents, never that the Test button reached it."""
    from .capability import CapabilityId
    from .ids import ModelId

    if preset.key == "artcraft-omni":
        return [
            ModelInfo.new(ModelId.new("nano_banana"),
                          "Nano Banana (unverified)").with_spec(
                CapabilitySpec.new(CapabilityId.parse("image.generate"))
                .accepting(InputRole.PROMPT, 1)
                .requiring(InputRole.PROMPT)),
            ModelInfo.new(ModelId.new("seedance_2p0"),
                          "Seedance 2.0 (unverified)").with_spec(
                CapabilitySpec.new(CapabilityId.parse("video.generate"))
                .accepting(InputRole.PROMPT, 1)
                .requiring(InputRole.PROMPT)),
        ]
    if preset.key == "vertex-express":
        return [ModelInfo.new(
            ModelId.new("gemini-2.5-flash-image"),
            "Gemini 2.5 Flash Image (unverified)").with_spec(
                CapabilitySpec.new(CapabilityId.parse("image.generate"))
                .accepting(InputRole.PROMPT, 1)
                .requiring(InputRole.PROMPT))]
    return []


def check_key(text: str) -> None:
    """One line of visible ASCII with no spaces: what an HTTP bearer
    token can be."""
    if len(text) > MAX_KEY_BYTES:
        raise DraftError(Field.KEY, "That key is far too long to be a key.")
    if not all(0x21 <= ord(c) <= 0x7E for c in text):
        raise DraftError(
            Field.KEY,
            "A key is one line of letters, digits and symbols with no "
            "spaces. Copy it again.")


def _slug(label: str) -> str:
    """A lowercase, dash-separated name made from a label."""
    out: list[str] = []
    for char in label:
        if char.isascii() and char.isalnum():
            out.append(char.lower())
        elif out and out[-1] != "-":
            out.append("-")
        if len(out) >= 40:
            break
    slug = "".join(out).rstrip("-")
    return slug or "connection"


def new_id(label: str, existing: list) -> ConnectionId:
    """An id derived from the label that no saved connection uses."""
    base = _slug(label)
    taken = {c.id.as_str() for c in existing}
    candidate = base
    n = 2
    while candidate in taken:
        candidate = f"{base}-{n}"
        n += 1
    try:
        return ConnectionId.new(candidate)
    except ValueError:
        raise DraftError(Field.LABEL,
                         "That name cannot be turned into an id.") from None


def _new_secret_ref(connection_id: ConnectionId,
                    existing: list) -> SecretRef:
    """The locator for a new connection's key: its id plus ``.key``,
    so it is recognisable in a keychain listing and two connections
    can never share a key."""
    used = {c.auth.secret for c in existing
            if c.auth.secret is not None}
    candidate = f"{connection_id.as_str()}.key"
    n = 2
    while candidate in used:
        candidate = f"{connection_id.as_str()}.key{n}"
        n += 1
    try:
        return SecretRef.new(candidate)
    except ValueError:
        raise DraftError(
            Field.LABEL,
            "That name cannot be used for the saved key.") from None


class ConnectionDraft:
    """The form's state."""

    def __init__(self, label: str = "", provider: str = "openai-compatible",
                 endpoint: str = "", billing: Billing = Billing.UNKNOWN,
                 uses_key: bool = True, key: str = "",
                 key_header: str = "", enabled: bool = True,
                 models: list | None = None,
                 models_for: str | None = None,
                 editing=None, secret: str | None = None) -> None:
        self.editing = editing
        self.label = label
        self.provider = provider
        self.endpoint = endpoint
        self.billing = billing
        self.uses_key = uses_key
        self.key = key
        self.key_header = key_header
        self.enabled = enabled
        self.models: list = list(models or [])
        self.models_for = models_for
        self._secret = secret

    def __repr__(self) -> str:
        return (f"ConnectionDraft(label={self.label!r}, "
                f"provider={self.provider!r}, "
                f"endpoint={self.endpoint!r}, "
                f"uses_key={self.uses_key}, "
                f"key=<{len(self.key)} chars>)")

    @classmethod
    def from_preset(cls, preset: Preset) -> ConnectionDraft:
        """A new connection started from a preset."""
        models = preset_models(preset)
        return cls(label=preset.name, provider=preset.provider,
                   endpoint=preset.endpoint, billing=preset.billing,
                   uses_key=preset.needs_key, models=models,
                   models_for=(preset.endpoint.strip() if models else None))

    @classmethod
    def from_connection(cls, connection: Connection) -> ConnectionDraft:
        """Edits a saved connection. The key box starts empty."""
        return cls(
            editing=connection.id, label=connection.label,
            provider=connection.provider,
            endpoint=connection.endpoint or "",
            billing=connection.billing,
            uses_key=connection.auth.secret is not None,
            key="", key_header=connection.key_header or "",
            enabled=connection.enabled, models=list(connection.models),
            models_for=(connection.endpoint.strip()
                        if connection.endpoint else None),
            secret=connection.auth.secret)

    def set_models(self, address: str, models: list) -> bool:
        """Record models found by a test at ``address``, the field
        value when the test started. An answer for an address no
        longer in the form is refused without changing its models.
        The caller must not substitute the current field value for
        the tested address."""
        if address.strip() != self.endpoint.strip():
            return False
        self.models = list(models)
        self.models_for = self.endpoint.strip()
        return True

    def models_fit(self, address: str) -> bool:
        """Known provenance must match the saved address, including
        after repeated edits. A new custom draft can carry manually
        declared models without prior provenance."""
        return self.models_for is None or self.models_for == address

    def locality(self):
        """Where this would send data, as far as the address can
        tell. ``None`` while it is not a usable address, so a form
        can show the badge live."""
        try:
            parsed = Endpoint.parse(self.endpoint.strip())
        except (ValueError, AiError):
            return None
        return Locality.LOCAL if parsed.is_loopback() else Locality.REMOTE

    def prepare(self, existing: list) -> Prepared:
        """Check everything and build the connection. ``existing`` is
        every saved connection, so new ids stay unique and an edit
        finds its original."""
        label = self.label.strip()
        if not label:
            raise DraftError(Field.LABEL, "Give the connection a name.")
        if (len(label) > MAX_LABEL_CHARS
                or any(ord(c) < 0x20 or ord(c) == 0x7F for c in label)):
            raise DraftError(
                Field.LABEL,
                f"The name must be one line of at most {MAX_LABEL_CHARS} "
                "characters.")
        provider = self.provider.strip()
        try:
            validate_id(provider)
        except ValueError as exc:
            raise DraftError(
                Field.PROVIDER,
                f"The service type is not valid ({exc}).") from None
        address = self.endpoint.strip()
        if not address:
            raise DraftError(Field.ADDRESS,
                             "Enter the address of the server.")
        if len(address.encode("utf-8")) > MAX_ENDPOINT_BYTES:
            raise DraftError(Field.ADDRESS, "That address is too long.")
        try:
            parsed = Endpoint.parse(address)
        except (ValueError, AiError) as exc:
            raise DraftError(
                Field.ADDRESS,
                f"That address cannot be used ({exc}). Use https://, or "
                "http:// only for this computer.") from None
        locality = (Locality.LOCAL if parsed.is_loopback()
                    else Locality.REMOTE)
        typed = self._typed_key()
        key_header = self._checked_key_header()
        if self.editing is not None:
            editing_id = (self.editing.as_str()
                          if hasattr(self.editing, "as_str")
                          else str(self.editing))
            if not any(c.id.as_str() == editing_id for c in existing):
                raise DraftError(
                    Field.LABEL,
                    "This connection was removed elsewhere. Close the form "
                    "and start again.")
            connection_id = self.editing
        else:
            connection_id = new_id(label, existing)
        auth, key_change = self._auth_and_key(typed, connection_id,
                                              existing)
        connection = Connection(
            connection_id, label, provider, endpoint=address,
            locality=locality, auth=auth, billing=self.billing,
            models=(list(self.models) if self.models_fit(address) else []),
            enabled=self.enabled, key_header=key_header)
        try:
            connection.validate()
        except ValueError as exc:
            raise DraftError(Field.ADDRESS, f"{exc}.") from None
        return Prepared(connection, key_change)

    def _typed_key(self):
        text = self.key.strip()
        if not text:
            return None
        check_key(text)
        return text

    def _checked_key_header(self):
        header = self.key_header.strip()
        if not header:
            return None
        try:
            check_key_header(header)
        except ValueError as exc:
            raise DraftError(Field.KEY_HEADER, str(exc)) from None
        return header

    def _auth_and_key(self, typed, connection_id, existing):
        if not self.uses_key:
            change = (KeyChange.clear() if self._secret is not None
                      else KeyChange.keep())
            return Auth.none(), change
        secret = self._secret or _new_secret_ref(connection_id,
                                                 existing).as_str()
        if typed is not None:
            return Auth.api_key(secret), KeyChange.set(typed)
        if self._secret is not None:
            return Auth.api_key(secret), KeyChange.keep()
        raise DraftError(
            Field.KEY,
            "Paste the key. If this server needs none, turn off "
            "\"Uses a key\".")
