"""Secret stores (Craft-AI port of ai-native ``secrets/``).

A place that can hold secrets by reference. The key itself is never
in the store file or the plan — only its locator. The environment
store reads ``CRAFT_AI_SECRET_<REF>`` (the reference flattened to
upper case); it is read-only, so a write is refused, never silently
dropped. The memory store is the in-process one. The layered store
takes the first hit on read and the first store that accepts the
write on set. The OS-keychain and passphrase-encrypted file stores
are NOT ported: there is no stdlib keychain binding, and the file
store's encryption needs a crypto lib Devon does not vendor. Devon's
key store stays :mod:`nomorals.accounts.vault` for accounts plus
these stores for AI connections.
"""

from __future__ import annotations

import os

from .error import AiError, ErrorKind

__all__ = [
    "EnvStore",
    "LayeredStore",
    "MemoryStore",
    "SecretStore",
    "secret_for",
]

PREFIX = "CRAFT_AI_SECRET_"


class SecretStore:
    """A place that can hold secrets by reference."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    def get(self, key: str):
        """The secret for ``key``, or ``None`` if this store has
        none."""
        raise NotImplementedError

    def set(self, key: str, value: str) -> None:
        """Save ``value`` under ``key``, replacing any earlier
        value. Read-only stores raise."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove ``key``. Removing a key that is not there is not
        an error."""
        raise NotImplementedError


class EnvStore(SecretStore):
    """Reads the process environment. Read-only."""

    def __init__(self, lookup=None) -> None:
        self._lookup = lookup or (lambda name: os.environ.get(name))

    @staticmethod
    def var_name(key: str) -> str:
        """The variable name for a reference."""
        tail = "".join(
            c.upper() if c.isascii() and c.isalnum() else "_"
            for c in str(key))
        return f"{PREFIX}{tail}"

    @property
    def name(self) -> str:
        return "environment"

    def get(self, key: str):
        value = self._lookup(self.var_name(key))
        if value is None or value == "":
            return None
        return value

    def set(self, key: str, value: str) -> None:
        raise AiError.unsupported(
            "the environment secret store is read-only")

    def delete(self, key: str) -> None:
        raise AiError.unsupported(
            "the environment secret store is read-only")


class MemoryStore(SecretStore):
    """An in-process map of secrets. Gone when the process ends."""

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    @property
    def name(self) -> str:
        return "memory"

    def get(self, key: str):
        return self._values.get(str(key))

    def set(self, key: str, value: str) -> None:
        self._values[str(key)] = str(value)

    def delete(self, key: str) -> None:
        self._values.pop(str(key), None)


class LayeredStore(SecretStore):
    """Several stores, in order. Reads take the first hit; writes go
    to the first store that accepts them."""

    def __init__(self, layers: list) -> None:
        self._layers = list(layers)

    @property
    def name(self) -> str:
        return "layered"

    def layer_names(self) -> list:
        return [layer.name for layer in self._layers]

    def get(self, key: str):
        for layer in self._layers:
            value = layer.get(key)
            if value is not None:
                return value
        return None

    def set(self, key: str, value: str) -> None:
        first_error = None
        for layer in self._layers:
            try:
                layer.set(key, value)
                return
            except AiError as exc:
                if exc.kind is ErrorKind.UNSUPPORTED:
                    continue
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error
        raise AiError.unsupported(
            "no secret store in the layer accepts writes")

    def delete(self, key: str) -> None:
        first_error = None
        for layer in self._layers:
            try:
                layer.delete(key)
            except AiError as exc:
                if exc.kind is ErrorKind.UNSUPPORTED:
                    continue
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


def secret_for(store: SecretStore, connection):
    """The secret a connection needs, or ``None`` for a connection
    with no credentials.

    A connection that needs a secret the store does not hold is an
    auth error that says which connection to fix, so an app can send
    the user to the launcher."""
    ref = connection.auth.secret
    if ref is None:
        return None
    value = store.get(ref)
    if value:
        return value
    route = "sign-in" if connection.auth.kind == "subscription" else "key"
    raise AiError.new(
        ErrorKind.AUTH,
        f"no {route} is saved for connection \"{connection.label}\"; "
        "add it in the launcher")
