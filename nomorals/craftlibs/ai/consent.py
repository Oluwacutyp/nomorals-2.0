"""Consent (Craft-Libs port of ai-core ``consent.rs``).

Before data goes to a remote connection, the app shows a
:class:`Disclosure` — what kinds of data, how big, to where, under
which capability, and the names of the parameters, never their
contents — and the :class:`ConsentLedger` records the agreement.
Consent is per app, connection, destination and capability, and a
new kind of data asks again. Local connections never ask: nothing
leaves the machine.
"""

from __future__ import annotations

from .capability import CapabilityId
from .connection import Connection, Locality
from .ids import AppId
from .payload import PayloadKind
from .request import Request

__all__ = ["ConsentLedger", "ConsentRecord", "Disclosure"]

# Payload kinds in the order the upstream enum declares them;
# a disclosure lists its kinds in this order.
_KIND_ORDER = [PayloadKind.TEXT, PayloadKind.IMAGE, PayloadKind.VIDEO,
               PayloadKind.AUDIO, PayloadKind.DOCUMENT, PayloadKind.OTHER]


class Disclosure:
    """What one request would send, and where. Kinds and sizes
    only — never contents, never parameter values."""

    def __init__(self, app: str, connection: str, destination: str,
                 capability: CapabilityId, kinds, items: list,
                 param_names: list, local: bool = False) -> None:
        self.app = app
        self.connection = connection
        self.destination = destination
        self.capability = capability
        self._kinds = list(kinds)
        self.items = items
        self.param_names = list(param_names)
        self.local = local

    @classmethod
    def for_request(cls, app, connection: Connection,
                    request: Request) -> Disclosure:
        """Build the disclosure for a request on a connection."""
        if connection.endpoint is not None:
            from .endpoint import Endpoint
            destination = Endpoint.parse(connection.endpoint).fingerprint()
        else:
            destination = "local"
        # Locality is a property of the connection, not of the
        # endpoint: a local connection on loopback still never
        # asks, even though it has an endpoint to fingerprint.
        local = connection.locality is Locality.LOCAL
        items: list[dict] = []
        kinds: set[PayloadKind] = set()
        for _role, payload in request.inputs:
            kind = payload.kind()
            kinds.add(kind)
            items.append({"kind": kind.value,
                          "bytes": payload.size_hint()})
        if request.params:
            kinds.add(PayloadKind.TEXT)
        ordered = [k for k in _KIND_ORDER if k in kinds]
        return cls(str(app), connection.id.as_str(), destination,
                   request.capability, ordered, items,
                   list(request.params.keys()), local)

    def kinds(self) -> list[PayloadKind]:
        return list(self._kinds)

    def to_dict(self) -> dict:
        return {"app": self.app, "connection": self.connection,
                "destination": self.destination,
                "capability": self.capability.as_str(),
                "kinds": [k.value for k in self._kinds],
                "items": self.items,
                "param_names": self.param_names,
                "local": self.local}


class ConsentRecord:
    """One agreement: an app may send these kinds of data to this
    destination under this capability, until ``expires_at``
    (``None`` = no expiry). A record from before capability
    scoping (``capability`` ``None``) covers nothing."""

    def __init__(self, app: str, connection: str, destination: str,
                 capability: CapabilityId | None, kinds,
                 expires_at: int | None = None) -> None:
        self.app = app
        self.connection = connection
        self.destination = destination
        self.capability = capability
        self.kinds = set(kinds)
        self.expires_at = expires_at

    def covers(self, disclosure: Disclosure, now: int) -> bool:
        if self.capability is None:
            return False
        if self.capability != disclosure.capability:
            return False
        if self.expires_at is not None and now >= self.expires_at:
            return False
        return set(disclosure.kinds()).issubset(self.kinds)


class ConsentLedger:
    """Agreements, keyed by app / connection / destination /
    capability. One record per key; a later grant merges its
    kinds into the record."""

    def __init__(self) -> None:
        self.records: list[ConsentRecord] = []

    def _find(self, disclosure: Disclosure) -> ConsentRecord | None:
        for record in self.records:
            if (record.app == disclosure.app
                    and record.connection == disclosure.connection
                    and record.destination == disclosure.destination
                    and record.capability == disclosure.capability):
                return record
        return None

    def needs_consent(self, disclosure: Disclosure, now: int) -> bool:
        """True when this request may not go out yet."""
        if disclosure.local:
            return False
        record = self._find(disclosure)
        if record is None:
            return True
        return not record.covers(disclosure, now)

    def grant(self, disclosure: Disclosure) -> None:
        self._grant(disclosure, None)

    def grant_until(self, disclosure: Disclosure,
                    expires_at: int) -> None:
        self._grant(disclosure, expires_at)

    def _grant(self, disclosure: Disclosure,
               expires_at: int | None) -> None:
        record = self._find(disclosure)
        if record is None:
            record = ConsentRecord(disclosure.app, disclosure.connection,
                                   disclosure.destination,
                                   disclosure.capability,
                                   set(disclosure.kinds()), expires_at)
            self.records.append(record)
            return
        record.kinds.update(disclosure.kinds())
        record.expires_at = expires_at

    def revoke(self, app, connection) -> None:
        """Withdraw every agreement for one app on one
        connection."""
        app_id = app if isinstance(app, str) else AppId.new(app).as_str()
        conn_id = connection if isinstance(connection, str) \
            else connection.as_str()
        self.records = [r for r in self.records
                        if not (r.app == app_id
                                and r.connection == conn_id)]
