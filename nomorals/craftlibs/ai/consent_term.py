"""How long a person's yes lasts (Craft-AI port of ai-core
``consent_term.rs``).

The prompt is each app's own, but the choices and what they mean are
shared, so "for an hour" means the same in every app and a person is
never asked about something they already agreed to in another one's
wording. :meth:`ConsentTerm.id` is the stable name to store or
translate against.
"""

from __future__ import annotations

from .consent import ConsentLedger, Disclosure

__all__ = ["ConsentTerm"]

HOUR = 60 * 60
DAY = 24 * HOUR


class ConsentTerm:
    """How long a yes lasts."""

    _INSTANCES: dict[str, ConsentTerm] = {}

    def __init__(self, kind: str) -> None:
        self._kind = kind

    @classmethod
    def this_time(cls) -> ConsentTerm:
        return cls._of("this_time")

    @classmethod
    def one_hour(cls) -> ConsentTerm:
        return cls._of("one_hour")

    @classmethod
    def one_day(cls) -> ConsentTerm:
        return cls._of("one_day")

    @classmethod
    def until_revoked(cls) -> ConsentTerm:
        return cls._of("until_revoked")

    @classmethod
    def _of(cls, kind: str) -> ConsentTerm:
        if kind not in cls._INSTANCES:
            cls._INSTANCES[kind] = cls(kind)
        return cls._INSTANCES[kind]

    @classmethod
    def from_id(cls, term_id: str) -> ConsentTerm:
        """The term for a stored id. Unknown ids are refused."""
        if term_id not in ("this_time", "one_hour", "one_day",
                           "until_revoked"):
            raise ValueError(f"unknown consent term {term_id!r}")
        return cls._of(term_id)

    @classmethod
    def all(cls) -> list[ConsentTerm]:
        """The choices in the order a prompt lists them, shortest first."""
        return [cls.this_time(), cls.one_hour(), cls.one_day(),
                cls.until_revoked()]

    def id(self) -> str:
        """A stable name, for storing a default choice or looking up
        a translation."""
        return self._kind

    def remembers(self) -> bool:
        """Whether the yes is written to the consent file."""
        return self._kind != "this_time"

    def expires_at(self, now: int) -> int | None:
        """When the yes lapses (unix seconds) if given at ``now``;
        ``None`` if it does not."""
        if self._kind == "this_time":
            return now
        if self._kind == "one_hour":
            return now + HOUR
        if self._kind == "one_day":
            return now + DAY
        return None

    def __eq__(self, other) -> bool:
        return isinstance(other, ConsentTerm) and self._kind == other._kind

    def __hash__(self) -> int:
        return hash(self._kind)

    def __repr__(self) -> str:
        return f"ConsentTerm.{self._kind}"


# The class-level list upstream spells ``ConsentTerm::ALL``.
ConsentTerm.ALL = [ConsentTerm.this_time(), ConsentTerm.one_hour(),
                   ConsentTerm.one_day(), ConsentTerm.until_revoked()]


def _copy_ledger(ledger: ConsentLedger) -> ConsentLedger:
    from .consent import ConsentRecord

    copy = ConsentLedger()
    for record in ledger.records:
        copy.records.append(ConsentRecord(
            record.app, record.connection, record.destination,
            record.capability, set(record.kinds), record.expires_at))
    return copy


def _agree(self: ConsentLedger, disclosure: Disclosure,
           term: ConsentTerm, now: int) -> ConsentLedger:
    """Apply a person's choice and return the ledger that covers the
    job they said yes to. Save ``self`` only if ``term.remembers()``;
    start the job with the returned ledger either way."""
    if not term.remembers():
        for_job = _copy_ledger(self)
        for_job.grant(disclosure)
        return for_job
    end = term.expires_at(now)
    if end is None:
        self.grant(disclosure)
    else:
        self.grant_until(disclosure, end)
    return _copy_ledger(self)


def _forget_lapsed(self: ConsentLedger, now: int) -> None:
    """Drop agreements that have lapsed, so the file does not grow
    with old yeses."""
    self.records = [r for r in self.records
                    if r.expires_at is None or now < r.expires_at]


ConsentLedger.agree = _agree  # type: ignore[method-assign]
ConsentLedger.forget_lapsed = _forget_lapsed  # type: ignore[method-assign]
