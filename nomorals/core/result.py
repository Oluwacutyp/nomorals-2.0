"""The ``Outcome`` type: explicit, unavoidable error handling.

Every fallible boundary in the framework returns ``Outcome[T]`` rather than raising.
The type is a small discriminated union::

    Ok(value)   -> .ok is True,  .value is the payload
    Err(error)  -> .ok is False, .error is a NoMoralsError

The API is deliberately close to Rust's ``Result`` and Haskell's ``Either``, because
those are the two designs that have stood up to a decade of production use.

    >>> r = Ok(3).map(lambda x: x * 2)
    >>> r.value
    6
    >>> Err(ValidationError("nope")).map(lambda x: x * 2).ok
    False
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterator, TypeVar, Union, overload

from .errors import NoMoralsError, classify

T = TypeVar("T")
U = TypeVar("U")

__all__ = ["Err", "Ok", "Outcome", "outcome_from", "unwrap_all"]


class Ok(Generic[T]):
    """A successful outcome carrying a value."""

    __slots__ = ("_value",)
    ok: bool = True

    def __init__(self, value: T) -> None:
        self._value = value

    @property
    def value(self) -> T:
        return self._value

    @property
    def error(self) -> None:
        return None

    # -- monadic interface ---------------------------------------------------
    def map(self, fn: Callable[[T], U]) -> "Outcome[U]":
        try:
            return Ok(fn(self._value))
        except Exception as exc:  # noqa: BLE001 - boundary of user code
            return Err(classify(exc))

    def and_then(self, fn: Callable[[T], "Outcome[U]"]) -> "Outcome[U]":
        try:
            return fn(self._value)
        except Exception as exc:  # noqa: BLE001
            return Err(classify(exc))

    def or_else(self, fn: Callable[[NoMoralsError], "Outcome[T]"]) -> "Outcome[T]":
        return self

    def unwrap(self) -> T:
        return self._value

    def unwrap_or(self, default: T) -> T:
        return self._value

    def unwrap_or_else(self, fn: Callable[[NoMoralsError], T]) -> T:
        return self._value

    def expect(self, message: str) -> T:
        return self._value

    def ok_value(self) -> "Outcome[T]":
        return self

    def err_value(self) -> None:
        return None

    # -- ergonomics ----------------------------------------------------------
    def __bool__(self) -> bool:
        return True

    def __iter__(self) -> Iterator[T]:
        yield self._value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Ok) and other._value == self._value

    def __hash__(self) -> int:
        try:
            return hash((True, self._value))
        except TypeError:  # unhashable payload
            return hash(True)

    def __repr__(self) -> str:
        return f"Ok({self._value!r})"


class Err(Generic[T]):
    """A failed outcome carrying a :class:`NoMoralsError`."""

    __slots__ = ("_error",)
    ok: bool = False

    def __init__(self, error: NoMoralsError | str | BaseException) -> None:
        if isinstance(error, NoMoralsError):
            self._error = error
        elif isinstance(error, str):
            self._error = NoMoralsError(error)
        else:
            self._error = classify(error)

    @property
    def value(self) -> None:
        return None

    @property
    def error(self) -> NoMoralsError:
        return self._error

    # -- monadic interface ---------------------------------------------------
    def map(self, fn: Callable[[Any], Any]) -> "Outcome[Any]":
        return self

    def and_then(self, fn: Callable[[Any], "Outcome[Any]"]) -> "Outcome[Any]":
        return self

    def or_else(self, fn: Callable[[NoMoralsError], "Outcome[T]"]) -> "Outcome[T]":
        try:
            return fn(self._error)
        except Exception as exc:  # noqa: BLE001
            return Err(classify(exc))

    def unwrap(self) -> T:
        raise self._error

    def unwrap_or(self, default: T) -> T:
        return default

    def unwrap_or_else(self, fn: Callable[[NoMoralsError], T]) -> T:
        return fn(self._error)

    def expect(self, message: str) -> T:
        raise NoMoralsError(f"{message}: {self._error.message}", details=self._error.to_dict())

    def ok_value(self) -> None:
        return None

    def err_value(self) -> NoMoralsError:
        return self._error

    # -- ergonomics ----------------------------------------------------------
    def __bool__(self) -> bool:
        return False

    def __iter__(self) -> Iterator[Any]:
        return iter(())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Err):
            return NotImplemented
        return other._error.code == self._error.code and other._error.message == self._error.message

    def __hash__(self) -> int:
        return hash((False, self._error.code, self._error.message))

    def __repr__(self) -> str:
        return f"Err({self._error.code}: {self._error.message})"


Outcome = Union[Ok[T], Err[T]]


def outcome_from(
    fn: Callable[..., T], *args: Any, **kwargs: Any
) -> Outcome[T]:
    """Call ``fn`` and capture either its return value or a classified error.

    If ``fn`` already returns an ``Outcome`` it is passed through untouched.
    """
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - this is the boundary
        return Err(exc)
    if isinstance(result, (Ok, Err)):
        return result
    return Ok(result)


def unwrap_all(outcomes: list[Outcome[T]]) -> "Outcome[list[T]]":
    """Collect a list of outcomes into a single outcome of a list.

    Fails fast on the first ``Err``, aggregating the remaining codes in ``details``.
    """
    values: list[T] = []
    for item in outcomes:
        if item.ok:
            values.append(item.value)  # type: ignore[union-attr]
        else:
            remaining = [
                o.error.code for o in outcomes if not o.ok  # type: ignore[union-attr]
            ]
            err = item.error  # type: ignore[union-attr]
            err.details["all_codes"] = remaining
            return Err(err)
    return Ok(values)


def partition(outcomes: list[Outcome[T]]) -> tuple[list[T], list[NoMoralsError]]:
    """Split outcomes into (successes, failures) without failing fast."""
    oks: list[T] = []
    errs: list[NoMoralsError] = []
    for item in outcomes:
        if item.ok:
            oks.append(item.value)  # type: ignore[union-attr]
        else:
            errs.append(item.error)  # type: ignore[union-attr]
    return oks, errs
