"""Solana chain primitives for the pump desk — stdlib only.

Everything the desk needs from the chain, with zero third-party
dependencies (the house rule: the core runs anywhere, Termux ARM
included — no solders wheels required).  When ``solders`` happens
to be installed nothing here changes: this module IS the decode
path, audited in-tree, not a vendored binary (the design's
supply-chain rule: the fake pumpfun-bot exfiltration case is why
this desk vendors NO external bot code).

Contents:

* base58 encode/decode (the Solana alphabet);
* program-derived addresses (pure-python ed25519 off-curve check);
* Anchor event discriminators (``sha256("event:<Name>")[:8]`` —
  derived, never copied from a blog post; §7.12 of the design);
* Anchor event decoders for the pump program's Create / Trade /
  Complete / CompletePumpAmmMigration events (prefix layouts —
  newer trailing fields are tolerated, never required);
* the bonding-curve account parser — virtual/real reserves,
  token_total_supply and creator READ PER TOKEN (§7.2: supply is
  never hard-coded; §7.4: the quote mint defaults to wrapped SOL
  and is only overridden by account evidence);
* SPL mint parsing (authority flags, supply, decimals);
* curve math in the QUOTE asset (SOL for standard pairs): price,
  market cap, curve-% — never USD (§7.3);
* a small JSON-RPC client over urllib (proxy-honouring, like the
  rest of the house) with an outcome hook the desk feeds into the
  ONE source registry.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .wsclient import TRANSPORT_DENIED_CLASS, TransportDeniedError
from .failures import record_failure

__all__ = [
    "PUMP_PROGRAM", "PUMPSWAP_PROGRAM", "SOL_MINT",
    "b58encode", "b58decode", "find_program_address",
    "bonding_curve_pda", "pumpswap_pool_pda",
    "event_discriminator", "decode_program_data",
    "BondingCurveState", "parse_bonding_curve",
    "MintInfo", "parse_mint_account",
    "price_quote_per_token", "market_cap_quote", "curve_pct",
    "SolanaRPC", "RoutedRPC", "RPCError",
    "TransportDeniedError", "TRANSPORT_DENIED_CLASS",
    "SOURCE_BUDGET_CLASS",
]

#: Registry error class recorded when a source's per-cycle hard
#: time budget is spent: the source degrades for the rest of the
#: cycle (its route fails over / its reads name the gap) instead
#: of stalling the desk.
SOURCE_BUDGET_CLASS = "source-budget-exhausted"

#: pump.fun program (design §1.4/§7.12, verified).
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
#: PumpSwap (Pump AMM) — the graduation venue (design §1.3).
PUMPSWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
#: Wrapped SOL mint — the default quote asset.
SOL_MINT = "So11111111111111111111111111111111111111112"

#: Sellable fraction of supply sold along the curve — the verified
#: SDK split (793,100,000 of 1,000,000,000; the rest seeds the
#: graduation pool).  Used ONLY as the curve-% basis when the desk
#: did not observe the curve's pre-buy state; the basis is always
#: named on the dossier (design §7.2).
SELLABLE_FRACTION = 7931 / 10000

#: Real SOL (lamports) in the curve at sell-out — the virtual 115
#: SOL completion point minus the 30 SOL virtual advance (§7.3).
#: A cross-check estimate for curve %, never the primary basis.
GRADUATION_REAL_SOL_LAMPORTS = 85_000_000_000


# ── base58 ────────────────────────────────────────────────────────────

_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(_B58_ALPHABET)}


def b58encode(data: bytes) -> str:
    if not data:
        return ""
    zeros = 0
    for b in data:
        if b == 0:
            zeros += 1
        else:
            break
    n = int.from_bytes(data, "big")
    out = ""
    while n > 0:
        n, rem = divmod(n, 58)
        out = _B58_ALPHABET[rem] + out
    return "1" * zeros + out


def b58decode(text: str) -> bytes:
    if not text:
        return b""
    n = 0
    for ch in text.strip():
        if ch not in _B58_INDEX:
            raise ValueError(f"invalid base58 character {ch!r}")
        n = n * 58 + _B58_INDEX[ch]
    zeros = 0
    for ch in text:
        if ch == "1":
            zeros += 1
        else:
            break
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * zeros + body


# ── program-derived addresses ─────────────────────────────────────────

_ED_P = 2 ** 255 - 19
_ED_D = (-121665 * pow(121666, _ED_P - 2, _ED_P)) % _ED_P


def _is_on_curve(point: bytes) -> bool:
    """True when the 32 bytes are a valid ed25519 point (a PDA must
    NOT be one).  Pure-python decompression check: y is read
    little-endian (top bit = x sign); the point exists iff
    (y^2-1)/(d*y^2+1) is a quadratic residue mod p."""
    if len(point) != 32:
        return True
    y = int.from_bytes(point, "little") & ((1 << 255) - 1)
    if y >= _ED_P:
        return False
    y2 = (y * y) % _ED_P
    u = (y2 - 1) % _ED_P
    v = (_ED_D * y2 + 1) % _ED_P
    if v == 0:
        return False
    x2 = u * pow(v, _ED_P - 2, _ED_P) % _ED_P
    if x2 == 0:
        return True
    return pow(x2, (_ED_P - 1) // 2, _ED_P) == 1


def _create_program_address(seeds: list[bytes], program_id: bytes) -> bytes:
    h = hashlib.sha256()
    for s in seeds:
        h.update(s)
    h.update(program_id)
    h.update(b"ProgramDerivedAddress")
    return h.digest()


def find_program_address(seeds: list[bytes], program_id_b58: str) -> tuple[str, int]:
    """(address, bump) — bump searched 255 down, Solana-canonical."""
    program = b58decode(program_id_b58)
    for bump in range(255, -1, -1):
        addr = _create_program_address(seeds + [bytes([bump])], program)
        if not _is_on_curve(addr):
            return b58encode(addr), bump
    raise ValueError("no viable program address bump")


def bonding_curve_pda(mint_b58: str) -> str:
    """The per-token bonding curve account (seeds §7.12)."""
    addr, _ = find_program_address([b"bonding-curve", b58decode(mint_b58)],
                                   PUMP_PROGRAM)
    return addr


def pumpswap_pool_pda(creator_b58: str, base_mint_b58: str,
                      quote_mint_b58: str = SOL_MINT, index: int = 0) -> str:
    """The canonical PumpSwap pool address for a graduated token
    (seeds §7.12: ["pool", index u16 LE, creator, base, quote])."""
    addr, _ = find_program_address(
        [b"pool", struct.pack("<H", index), b58decode(creator_b58),
         b58decode(base_mint_b58), b58decode(quote_mint_b58)],
        PUMPSWAP_PROGRAM)
    return addr


# ── Anchor event decode ───────────────────────────────────────────────

def event_discriminator(name: str) -> bytes:
    """Anchor event discriminator: sha256("event:<Name>")[:8]."""
    return hashlib.sha256(f"event:{name}".encode()).digest()[:8]


EVENT_NAMES = ("CreateEvent", "TradeEvent", "CompleteEvent",
               "CompletePumpAmmMigrationEvent")
_DISCRIMINATORS = {event_discriminator(n): n for n in EVENT_NAMES}


class _Reader:
    def __init__(self, data: bytes):
        self.d = data
        self.off = 0

    def take(self, n: int) -> bytes:
        chunk = self.d[self.off:self.off + n]
        if len(chunk) < n:
            raise ValueError("event payload truncated")
        self.off += n
        return chunk

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def bool(self) -> bool:
        return self.take(1)[0] != 0

    def pubkey(self) -> str:
        return b58encode(self.take(32))

    def string(self) -> str:
        n = struct.unpack("<I", self.take(4))[0]
        if n > 4096:
            raise ValueError("implausible string length in event")
        return self.take(n).decode("utf-8", errors="replace")


def decode_program_data(payload: bytes) -> Optional[dict[str, Any]]:
    """Decode one Anchor event payload (the bytes behind a
    ``Program data:`` log line).  Returns None for anything that is
    not one of the four pump events the desk tracks.  Prefix layouts:
    fields added by later program versions trail the ones parsed
    here and are ignored, never required."""
    if len(payload) < 8:
        return None
    name = _DISCRIMINATORS.get(payload[:8])
    if name is None:
        return None
    r = _Reader(payload[8:])
    try:
        if name == "CreateEvent":
            ev: dict[str, Any] = {
                "event": name,
                "name": r.string(), "symbol": r.string(),
                "uri": r.string(),
                "mint": r.pubkey(), "bonding_curve": r.pubkey(),
                "user": r.pubkey(),
            }
            if r.off + 32 <= len(r.d):  # newer IDL appends creator
                ev["creator"] = r.pubkey()
            else:
                ev["creator"] = ev["user"]
            return ev
        if name == "TradeEvent":
            ev = {
                "event": name,
                "mint": r.pubkey(),
                "sol_amount": r.u64(), "token_amount": r.u64(),
                "is_buy": r.bool(), "user": r.pubkey(),
                "timestamp": r.i64(),
                "virtual_sol_reserves": r.u64(),
                "virtual_token_reserves": r.u64(),
            }
            if r.off + 16 <= len(r.d):  # real reserves trail in newer IDL
                ev["real_sol_reserves"] = r.u64()
                ev["real_token_reserves"] = r.u64()
            return ev
        if name == "CompleteEvent":
            return {"event": name, "mint": r.pubkey(),
                    "user": r.pubkey(), "timestamp": r.i64(),
                    "bonding_curve": r.pubkey()}
        # CompletePumpAmmMigrationEvent — layout is not relied on:
        # the desk confirms graduation by the canonical pool's
        # existence (§7.5); here it only needs the mint, which the
        # payload is scanned for by the caller when known.
        return {"event": name, "raw_len": len(payload)}
    except (ValueError, struct.error):
        return None


def decode_log_lines(logs: list[str]) -> list[dict[str, Any]]:
    """All decodable pump events in a transaction's log lines."""
    import base64
    out: list[dict[str, Any]] = []
    for line in logs or []:
        if "Program data:" not in line:
            continue
        blob = line.split("Program data:", 1)[1].strip()
        try:
            payload = base64.b64decode(blob)
        except Exception:  # noqa: BLE001 - a bad line is not an event
            continue
        ev = decode_program_data(payload)
        if ev is not None:
            out.append(ev)
    return out


# ── account parsing ───────────────────────────────────────────────────

@dataclass
class BondingCurveState:
    """The bonding curve account, read live (never from constants)."""
    virtual_token_reserves: int = 0
    virtual_sol_reserves: int = 0
    real_token_reserves: int = 0
    real_sol_reserves: int = 0
    token_total_supply: int = 0
    complete: bool = False
    creator: str = ""
    quote_mint: str = SOL_MINT
    account_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "virtual_token_reserves": self.virtual_token_reserves,
            "virtual_sol_reserves": self.virtual_sol_reserves,
            "real_token_reserves": self.real_token_reserves,
            "real_sol_reserves": self.real_sol_reserves,
            "token_total_supply": self.token_total_supply,
            "complete": self.complete,
            "creator": self.creator,
            "quote_mint": self.quote_mint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BondingCurveState":
        return cls(**{k: v for k, v in (d or {}).items()
                      if k in cls.__dataclass_fields__})


#: BondingCurve account layout after the 8-byte discriminator
#: (pump-public-docs IDL): five u64s, a bool, the creator pubkey.
_CURVE_MIN_LEN = 8 + 40 + 1 + 32


def parse_bonding_curve(data: bytes) -> Optional[BondingCurveState]:
    if data is None or len(data) < _CURVE_MIN_LEN:
        return None
    vt, vs, rt, rs, supply = struct.unpack_from("<QQQQQ", data, 8)
    return BondingCurveState(
        virtual_token_reserves=vt, virtual_sol_reserves=vs,
        real_token_reserves=rt, real_sol_reserves=rs,
        token_total_supply=supply,
        complete=data[48] != 0,
        creator=b58encode(data[49:81]),
        account_len=len(data))


@dataclass
class MintInfo:
    mint_authority: Optional[str] = None    # None = authority revoked
    freeze_authority: Optional[str] = None
    supply: int = 0
    decimals: int = 0
    initialized: bool = False


def parse_mint_account(data: bytes) -> Optional[MintInfo]:
    """SPL Token mint layout (82 bytes): COption<Pubkey> mint
    authority (u32 tag + 32 bytes), u64 supply, u8 decimals,
    u8 is_initialized, COption<Pubkey> freeze authority."""
    if data is None or len(data) < 82:
        return None
    def _copt(off: int) -> Optional[str]:
        tag = struct.unpack_from("<I", data, off)[0]
        if tag == 0:
            return None
        return b58encode(data[off + 4:off + 36])
    return MintInfo(
        mint_authority=_copt(0),
        supply=struct.unpack_from("<Q", data, 36)[0],
        decimals=data[44],
        initialized=data[45] != 0,
        freeze_authority=_copt(46))


# ── curve math (quote-asset denominated, never USD) ───────────────────

def price_quote_per_token(state: BondingCurveState,
                          decimals: int = 6) -> float:
    """Marginal price of one WHOLE token, in quote lamports
    (divide by 1e9 for SOL when the quote mint is wrapped SOL)."""
    if state.virtual_token_reserves <= 0:
        return 0.0
    return (state.virtual_sol_reserves / state.virtual_token_reserves) \
        * (10 ** decimals)


def market_cap_quote(state: BondingCurveState) -> float:
    """Market cap in quote lamports: marginal price x total supply
    (the raw-unit ratio makes the decimals cancel exactly)."""
    if state.virtual_token_reserves <= 0 or state.token_total_supply <= 0:
        return 0.0
    return (state.virtual_sol_reserves * state.token_total_supply
            / state.virtual_token_reserves)


def curve_pct(state: BondingCurveState,
              initial_real_token_reserves: int = 0) -> tuple[float, str]:
    """(fraction of the curve sold, basis).

    Primary basis ``observed-init``: the desk saw the curve before
    any buy (real_sol == 0 at first read) and recorded its
    real_token_reserves — sell-through is measured against that
    exact per-token figure.  Fallback basis ``supply-split``: the
    verified protocol split of THIS token's own total supply
    (§7.2).  Both are per-token; neither hard-codes 1B."""
    if initial_real_token_reserves > 0:
        sold = initial_real_token_reserves - state.real_token_reserves
        return (max(0.0, min(1.0, sold / initial_real_token_reserves)),
                "observed-init")
    sellable = state.token_total_supply * SELLABLE_FRACTION
    if sellable <= 0:
        return 0.0, "unknown"
    sold = sellable - state.real_token_reserves
    return (max(0.0, min(1.0, sold / sellable)), "supply-split")


# ── JSON-RPC client ───────────────────────────────────────────────────

class RPCError(Exception):
    pass


def _provider_error(body: bytes) -> Optional[dict[str, Any]]:
    """The JSON-RPC error object inside an error body, if any:
    {code, message}.  Providers that reject at the HTTP layer
    still speak JSON-RPC in the body (dRPC's plan refusal is an
    HTTP 400 carrying ``code 35``) — that object is the actual
    diagnosis and must not be thrown away."""
    try:
        data = json.loads((body or b"").decode("utf-8", "replace"))
    except ValueError:
        return None
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict) and (err.get("message") or err.get("code")):
        return {"code": err.get("code"),
                "message": str(err.get("message") or "")}
    if isinstance(err, str) and err:
        return {"code": None, "message": err}
    return None


def _is_transport_denied(exc: BaseException) -> bool:
    """True when *exc* is the OS refusing the connection itself:
    a builtin PermissionError, the desk's wrapped form, or a
    URLError whose reason is one (urllib wraps the socket error
    on some paths/versions and lets it escape bare on others —
    both are the same failure and get the same name)."""
    if isinstance(exc, (PermissionError, TransportDeniedError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, (PermissionError,
                                       TransportDeniedError))
    return False


class SolanaRPC:
    """Minimal Solana JSON-RPC client over urllib (stdlib — the
    house HTTP idiom; proxy env honoured by urllib's opener).

    ``on_outcome(ok, latency_s, error_class)`` is the desk's hook
    into the ONE source registry: every call grades the endpoint,
    no bespoke health code.

    ``budget_s`` is the source's hard time budget for one desk
    cycle (0 = unbounded, the standalone default): elapsed call
    time accrues against it across calls, each call's socket
    timeout is clamped to what remains, and a call that arrives
    with the budget spent fails fast with
    :data:`SOURCE_BUDGET_CLASS` — the route degrades for the rest
    of the cycle instead of stalling the desk (the 2026-10-23
    failure: ~16.5s stalls repeated across routes and dossiers
    until the whole watch job timed out).  The desk resets the
    budget at the top of every cycle (``reset_budget``)."""

    def __init__(self, endpoint: str = "https://api.mainnet-beta.solana.com",
                 timeout: float = 20.0,
                 on_outcome: Optional[Callable[..., None]] = None,
                 user_agent: str = "",
                 budget_s: float = 0.0):
        self.endpoint = endpoint
        self.timeout = timeout
        self.on_outcome = on_outcome
        if not user_agent:
            from .sources import desk_user_agent
            user_agent = desk_user_agent()
        self.user_agent = user_agent
        self.budget_s = max(0.0, float(budget_s or 0.0))
        self._budget_spent = 0.0
        self._id = 0

    # ── per-cycle source budget ───────────────────────────────────

    def reset_budget(self) -> None:
        """Start a fresh cycle for this source: its spent time
        zeroes.  Called by the desk at the top of every cycle."""
        self._budget_spent = 0.0

    def budget_left(self) -> float:
        """Seconds of this cycle's budget remaining (inf when the
        client is unbounded)."""
        if self.budget_s <= 0:
            return float("inf")
        return max(0.0, self.budget_s - self._budget_spent)

    def _report(self, ok: bool, t0: float, error_class: str = "") -> None:
        if self.on_outcome is not None:
            try:
                self.on_outcome(ok, time.time() - t0, error_class)
            except Exception as exc:  # noqa: BLE001 - reporting never sinks a call
                record_failure("chain.SolanaRPC._report", exc)

    #: bounded retries on HTTP 429 — the free public RPC throttles
    #: bursts (observed live 2026-10-22: getTokenLargestAccounts 429
    #: mid-dossier); a short backoff keeps the free floor useful
    #: without hammering the endpoint.
    RATE_LIMIT_RETRIES = 3

    def call(self, method: str, params: list) -> Any:
        t0 = time.time()
        attempt = 0
        try:
            while True:
                attempt += 1
                left = self.budget_left()
                if left <= 0:
                    self._report(False, t0, SOURCE_BUDGET_CLASS)
                    raise RPCError(
                        f"{method}: source budget exhausted "
                        f"({self.budget_s:.0f}s this cycle) for "
                        f"{self.endpoint} — the source degrades, "
                        f"the desk fails over")
                sock_timeout = min(self.timeout, left) \
                    if left != float("inf") else self.timeout
                self._id += 1
                body = json.dumps({"jsonrpc": "2.0", "id": self._id,
                                   "method": method,
                                   "params": params}).encode()
                req = urllib.request.Request(
                    self.endpoint, data=body,
                    headers={"Content-Type": "application/json",
                             "User-Agent": self.user_agent})
                try:
                    with urllib.request.urlopen(
                            req, timeout=sock_timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code == 429 \
                            and attempt <= self.RATE_LIMIT_RETRIES:
                        time.sleep(0.5 * (2 ** (attempt - 1)))
                        continue
                    detail = None
                    try:
                        detail = _provider_error(exc.read())
                    except Exception:  # noqa: BLE001 - body is a bonus
                        detail = None
                    if detail is not None:
                        cls = (f"rpc-error-{detail['code']}"
                               if detail["code"] is not None
                               else "rpc-error")
                        self._report(False, t0, cls)
                        raise RPCError(
                            f"{method}: HTTP {exc.code} — provider "
                            f"error {detail['code']}: "
                            f"{detail['message']}") from exc
                    self._report(False, t0, f"http-{exc.code}")
                    raise RPCError(f"{method}: HTTP {exc.code}") from exc
                except Exception as exc:  # noqa: BLE001
                    if _is_transport_denied(exc):
                        self._report(False, t0, TRANSPORT_DENIED_CLASS)
                        raise RPCError(
                            f"{method}: transport denied (the OS "
                            f"refused the connection): {exc}") from exc
                    self._report(False, t0, type(exc).__name__)
                    raise RPCError(f"{method}: {exc}") from exc
        finally:
            self._budget_spent += time.time() - t0
        if isinstance(data, dict) and data.get("error"):
            detail = _provider_error(json.dumps(data).encode())
            if detail is not None:
                cls = (f"rpc-error-{detail['code']}"
                       if detail["code"] is not None else "rpc-error")
                self._report(False, t0, cls)
                raise RPCError(
                    f"{method}: provider error {detail['code']}: "
                    f"{detail['message']}")
            self._report(False, t0, "rpc-error")
            raise RPCError(f"{method}: {data['error']}")
        self._report(True, t0)
        return data.get("result") if isinstance(data, dict) else None

    # ── typed helpers ─────────────────────────────────────────────

    def get_account(self, pubkey: str) -> Optional[bytes]:
        """Raw account bytes, or None when the account doesn't exist."""
        import base64
        res = self.call("getAccountInfo",
                        [pubkey, {"encoding": "base64",
                                  "commitment": "confirmed"}])
        value = (res or {}).get("value")
        if not value:
            return None
        return base64.b64decode(value["data"][0])

    def get_curve(self, mint: str) -> Optional[BondingCurveState]:
        raw = self.get_account(bonding_curve_pda(mint))
        return parse_bonding_curve(raw) if raw else None

    def get_mint_info(self, mint: str) -> Optional[MintInfo]:
        raw = self.get_account(mint)
        return parse_mint_account(raw) if raw else None

    def get_token_largest(self, mint: str) -> list[dict[str, Any]]:
        res = self.call("getTokenLargestAccounts",
                        [mint, {"commitment": "confirmed"}])
        return list((res or {}).get("value") or [])

    def get_token_accounts_by_owner(self, owner: str,
                                    mint: str = "") -> list[dict[str, Any]]:
        """Every token account of ``owner`` (optionally filtered to
        one mint), normalized to {address, owner, mint, amount}
        with amount in raw units.  jsonParsed is answered by every
        Solana RPC for both SPL Token and Token-2022 accounts —
        this is the read that keeps working on the free floor when
        getTokenLargestAccounts is throttled (observed live:
        largest 429s while by-owner answers 200)."""
        filt: dict[str, Any] = {"mint": mint} if mint else \
            {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}
        res = self.call("getTokenAccountsByOwner",
                        [owner, filt, {"encoding": "jsonParsed",
                                       "commitment": "confirmed"}])
        out: list[dict[str, Any]] = []
        for row in (res or {}).get("value") or []:
            info = (((row.get("account") or {}).get("data") or {})
                    .get("parsed") or {}).get("info") or {}
            amount = (info.get("tokenAmount") or {}).get("amount")
            if amount is None:
                continue
            out.append({
                "address": row.get("pubkey", ""),
                "owner": info.get("owner", owner),
                "mint": info.get("mint", mint),
                "amount": int(amount),
            })
        return out

    def get_token_supply(self, mint: str) -> Optional[int]:
        """Total supply in raw units, or None when unreadable."""
        res = self.call("getTokenSupply",
                        [mint, {"commitment": "confirmed"}])
        value = (res or {}).get("value") or {}
        amount = value.get("amount")
        return int(amount) if amount is not None else None

    def get_signatures(self, address: str, limit: int = 50,
                       before: str = "") -> list[dict[str, Any]]:
        opts: dict[str, Any] = {"limit": int(limit),
                                "commitment": "confirmed"}
        if before:
            opts["before"] = before
        res = self.call("getSignaturesForAddress", [address, opts])
        return list(res or [])

    def get_transaction(self, signature: str) -> Optional[dict[str, Any]]:
        res = self.call("getTransaction",
                        [signature, {"encoding": "jsonParsed",
                                     "commitment": "confirmed",
                                     "maxSupportedTransactionVersion": 0}])
        return res

    def get_slot(self) -> int:
        return int(self.call("getSlot", [{"commitment": "confirmed"}]) or 0)


class RoutedRPC:
    """The desk's chain reads, routed over several endpoints.

    The free public RPC throttles per method (getTokenLargestAccounts
    429s persistently while other reads sail through), and no single
    public endpoint is a foundation.  RoutedRPC holds an ordered
    list of SolanaRPC clients — provisioned Helius first when a key
    exists, then the registry's public RPC sources in health order
    (see sources.rpc_routes) — and serves each typed read from the
    first route that answers.  A route RAISING (RPCError: HTTP
    error, throttle-after-retries, transport failure) fails over to
    the next; a route answering — including "account does not
    exist" (None) or an empty list — IS the answer and stops the
    walk.  Every underlying client reports its own outcomes to the
    registry, so the health loop grades each route from real calls
    and the order self-corrects.  When every route fails the LAST
    error is re-raised: failure stays a named gap upstream, never a
    silent zero.

    ``served_by`` records which endpoint answered each method last
    — the provenance the dossier prints on its holder picture."""

    def __init__(self, clients: list[SolanaRPC]):
        if not clients:
            raise ValueError("RoutedRPC needs at least one route")
        self.clients = list(clients)
        self.served_by: dict[str, str] = {}
        #: endpoint of the first route (the primary) — duck-type
        #: parity with SolanaRPC for callers that log it.
        self.endpoint = self.clients[0].endpoint

    @property
    def endpoints(self) -> list[str]:
        return [c.endpoint for c in self.clients]

    def reset_budgets(self) -> None:
        """Fresh per-cycle budget for every route (the desk calls
        this at the top of each cycle).  Duck-typed: route doubles
        without budgets are simply skipped."""
        for client in self.clients:
            reset = getattr(client, "reset_budget", None)
            if callable(reset):
                reset()

    def _route(self, method: str, *args: Any, **kwargs: Any) -> Any:
        last_exc: Optional[Exception] = None
        for client in self.clients:
            try:
                out = getattr(client, method)(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - failover is the job
                last_exc = exc
                continue
            self.served_by[method] = client.endpoint
            return out
        assert last_exc is not None
        raise last_exc

    # ── the SolanaRPC surface the desk consumes ──────────────────

    def get_account(self, pubkey: str) -> Optional[bytes]:
        return self._route("get_account", pubkey)

    def get_curve(self, mint: str) -> Optional[BondingCurveState]:
        return self._route("get_curve", mint)

    def get_mint_info(self, mint: str) -> Optional[MintInfo]:
        return self._route("get_mint_info", mint)

    def get_token_largest(self, mint: str) -> list[dict[str, Any]]:
        return self._route("get_token_largest", mint)

    def get_token_accounts_by_owner(self, owner: str,
                                    mint: str = "") -> list[dict[str, Any]]:
        return self._route("get_token_accounts_by_owner", owner, mint)

    def get_token_supply(self, mint: str) -> Optional[int]:
        return self._route("get_token_supply", mint)

    def get_signatures(self, address: str, limit: int = 50,
                       before: str = "") -> list[dict[str, Any]]:
        return self._route("get_signatures", address, limit, before)

    def get_transaction(self, signature: str) -> Optional[dict[str, Any]]:
        return self._route("get_transaction", signature)

    def get_slot(self) -> int:
        return self._route("get_slot")
