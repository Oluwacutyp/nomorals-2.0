"""Trust tiers and the policy that decides which are accepted.

A port of the rules in ``craft-extensions`` (``crates/ext-core/src/
tier.rs``, dual MIT/Apache): the tier is assigned by the host from
where a package came, never read from the package. The third-party
tier is refused until the review gates pass, and no release build
turns it on. Only the first-party tier may use first-party id
prefixes, so a file or an outside publisher cannot impersonate or
replace one of Devon's own extensions.
"""

from __future__ import annotations

from enum import Enum

from .ids import ExtensionId

__all__ = ["Policy", "Tier", "TierError"]


class Tier(str):
    """Where a package came from, as assigned by the host."""

    FIRST_PARTY = "first-party"
    FROM_FILE = "from-file"
    REVIEWED_THIRD_PARTY = "reviewed-third-party"

    @classmethod
    def parse(cls, value: object) -> Tier:
        if value in ("first-party", "from-file",
                     "reviewed-third-party"):
            return cls(str(value))
        raise ValueError(f"not a trust tier: {value!r}")


class TierError(Enum):
    """Why a tier was refused."""

    THIRD_PARTY_CLOSED = ("extensions from outside publishers cannot "
                          "be installed yet")
    RESERVED_NAMESPACE = ("this package uses a name that only "
                          "Devon's own extensions may use")


class Policy:
    """What this build accepts. Built only through
    ``Policy.standard()``, so a caller cannot opt in."""

    def __init__(self) -> None:
        self._private = ()

    @classmethod
    def standard(cls) -> Policy:
        """The one policy this build runs."""
        return cls()

    def admit(self, tier: Tier | str) -> TierError | None:
        """Whether packages of this tier may be installed."""
        tier = Tier.parse(tier) if isinstance(tier, str) else tier
        if tier in (Tier.FIRST_PARTY, Tier.FROM_FILE):
            return None
        return TierError.THIRD_PARTY_CLOSED

    def check_id(self, tier: Tier | str,
                 ext_id: ExtensionId | str) -> TierError | None:
        """Whether this tier may use this id."""
        tier = Tier.parse(tier) if isinstance(tier, str) else tier
        ident = (ext_id if isinstance(ext_id, ExtensionId)
                 else ExtensionId.new(str(ext_id)))
        if tier != Tier.FIRST_PARTY and ident.in_first_party_namespace():
            return TierError.RESERVED_NAMESPACE
        return None
