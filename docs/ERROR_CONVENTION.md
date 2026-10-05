# Error Handling Convention

## The Rule

**`Outcome[T]` for expected failures. Exceptions for programmer errors.**

This is the single convention for all new Devon code. The two patterns
(`errors.py` exceptions vs `result.py` Outcome) are not competing —
they serve different purposes:

### Use `Outcome[T]` (Ok/Err) when:

- The failure is **expected** and the caller should handle it
- Examples: network down, file missing, model overloaded, validation failed,
  user provided bad input, external API returned an error
- The function signature makes the fallibility **explicit** and **unavoidable**
- Callers **cannot** accidentally ignore the error (must check `.ok`)

```python
from nomorals.core.result import Ok, Err, Outcome
from nomorals.core.errors import NotFound

def find_user(user_id: str) -> Outcome[User]:
    user = db.get(user_id)
    if user is None:
        return Err(NotFound(f"user {user_id} not found"))
    return Ok(user)

# Caller MUST handle both cases:
result = find_user("123")
if result.ok:
    print(result.value.name)
else:
    print(f"failed: {result.error.message}")
```

### Use exceptions (raise) when:

- The failure is a **programmer error** (bug in the code)
- The state is **unrecoverable** (invariant violated, corrupted data)
- Examples: `ValueError` for invalid arguments, `AssertionError` for
  broken invariants, `RuntimeError` for impossible states

```python
def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError("divisor cannot be zero")  # programmer error
    return a / b
```

### Why both?

- **Outcome** forces handling at the type level. A function returning
  `Outcome[User]` tells you "this might fail, deal with it." You can't
  forget — the compiler (and code review) sees it.
- **Exceptions** are for "this should never happen." They crash loudly
  so bugs get fixed, not silently swallowed.

### At the boundaries

Tool calls (`ToolRegistry.call`) always return `Outcome` — a tool
failing is an expected outcome, not a crash. The registry converts
unexpected exceptions into `Err(classify(exc))` so even buggy tools
don't take down the agent.

### Migration note

Older code mixes both patterns. Don't refactor it all at once.
For **new code**, follow this convention. When touching old code,
migrate the function you're changing to `Outcome` if it has expected
failure modes.
