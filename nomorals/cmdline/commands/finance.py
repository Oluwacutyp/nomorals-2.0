"""``nm finance`` — bank account linking and transactions (Mono + Plaid)."""

from __future__ import annotations

import argparse
from typing import Any
from ..emit import _emit


def _get_vault(context: Any):
    try:
        from ...accounts.vault import open_vault
        return open_vault(getattr(context, "db", None), context=context,
                          actor="finance-cli")
    except Exception:
        return None


def _cmd_finance(args: argparse.Namespace, context: Any) -> int:
    """Bank account linking and transactions via Mono and Plaid connectors."""
    from ...connectors.mono import MonoConnector
    from ...connectors.plaid import PlaidConnector

    action = getattr(args, "action", "status")
    provider = getattr(args, "provider", "auto").lower()
    account_id = getattr(args, "account_id", "")

    db = getattr(context, "db", None)
    vault = _get_vault(context)

    connectors: dict[str, Any] = {}
    if provider in ("auto", "mono"):
        try:
            connectors["mono"] = MonoConnector(vault=vault)  # type: ignore[arg-type]
        except Exception as exc:
            _emit(args, {"error": f"mono init failed: {exc}"},
                  f"mono init failed: {exc}")
            return 1
    if provider in ("auto", "plaid"):
        try:
            connectors["plaid"] = PlaidConnector(vault=vault)  # type: ignore[arg-type]
        except Exception as exc:
            _emit(args, {"error": f"plaid init failed: {exc}"},
                  f"plaid init failed: {exc}")
            return 1

    if action == "status":
        result = {}
        lines = []
        for name, conn in connectors.items():
            try:
                st = conn.status()
                d = st.to_dict() if hasattr(st, "to_dict") else {"status": str(st)}
            except Exception as exc:
                d = {"connected": False, "error": str(exc)}
            result[name] = d
            lines.append(f"{name.capitalize()}: {'connected' if d.get('connected') else 'disconnected'}")
        _emit(args, result, "\n".join(lines))
        return 0

    # For non-status actions, pick the single connector
    if len(connectors) == 1:
        conn = next(iter(connectors.values()))
        conn_name = next(iter(connectors.keys()))
    elif provider in connectors:
        conn = connectors[provider]
        conn_name = provider
    else:
        _emit(args, {"error": "specify --provider mono|plaid for this action"},
              "Use --provider mono or --provider plaid")
        return 1

    if action == "link":
        # Both connectors drive the human linking flow through
        # begin_link(): it prints the widget guide and, with a db, parks
        # a human checkpoint the owner resolves with the widget code.
        # Starting that flow IS the action — "linked": False plus a
        # checkpoint/next step is the designed non-interactive outcome.
        try:
            result = conn.begin_link(db=db, context=context)
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        result = result if isinstance(result, dict) else {"result": result}
        if result.get("linked"):
            text = f"account linked via {conn_name}"
        elif result.get("checkpoint_id"):
            text = (f"link started via {conn_name} — checkpoint "
                    f"{result['checkpoint_id']}: "
                    f"{result.get('next', 'complete the widget, then resolve it')}")
        else:
            text = (f"link started via {conn_name}: "
                    f"{result.get('next', 'complete the widget to finish linking')}")
        _emit(args, {"provider": conn_name, **result}, text)
        return 0

    if action == "accounts":
        # The real surfaces differ per connector: Mono lists its linked
        # accounts directly; Plaid returns per-institution entries whose
        # "accounts" lists need flattening. (The old code called
        # list_accounts()/accounts(), which NEITHER connector has —
        # this action failed with AttributeError for both providers.)
        try:
            if hasattr(conn, "linked_accounts"):
                accounts = conn.linked_accounts()
            elif hasattr(conn, "get_accounts"):
                accounts = []
                for entry in conn.get_accounts():
                    if isinstance(entry, dict) and isinstance(
                            entry.get("accounts"), list):
                        for acct in entry["accounts"]:
                            if isinstance(acct, dict):
                                acct = {**acct, "institution":
                                        entry.get("institution_name", "")}
                            accounts.append(acct)
                    else:
                        accounts.append(entry)
            elif hasattr(conn, "list_accounts"):
                accounts = conn.list_accounts()
            else:
                raise AttributeError(
                    f"{conn_name} connector exposes no account listing")
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        _emit(args, {"provider": conn_name, "count": len(accounts), "accounts": accounts},
              f"Found {len(accounts)} account(s) via {conn_name}")
        return 0

    if action == "balance":
        # Plaid derives balances from /accounts/get (get_balances);
        # Mono carries the balance on the account record itself — the
        # same read finance/overview.py performs for the money view.
        try:
            rows: list[dict[str, Any]] = []
            if hasattr(conn, "get_balances"):
                for entry in conn.get_balances():
                    if isinstance(entry, dict) and isinstance(
                            entry.get("accounts"), list):
                        for acct in entry["accounts"]:
                            rows.append({
                                "institution": entry.get("institution_name", ""),
                                "name": acct.get("name", ""),
                                "available": acct.get("available"),
                                "current": acct.get("current"),
                                "currency": acct.get("currency", ""),
                            })
                    elif isinstance(entry, dict):
                        rows.append(entry)
            elif hasattr(conn, "linked_accounts"):
                for linked in conn.linked_accounts():
                    aid = str(linked.get("account_id")
                              or linked.get("id") or "")
                    info = conn.get_account(aid) if aid else {}
                    acc = info.get("account") or info
                    rows.append({
                        "account_id": aid,
                        "name": linked.get("name") or acc.get("name", ""),
                        "balance": acc.get("balance"),
                        "currency": acc.get("currency", "NGN"),
                    })
            else:
                raise AttributeError(
                    f"{conn_name} connector exposes no balance read")
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        lines = []
        for r in rows:
            if "balance" in r:
                lines.append(f"{r.get('name') or r.get('account_id')}: "
                             f"{r.get('balance')} {r.get('currency', '')}".rstrip())
            else:
                lines.append(f"{r.get('name')}: available "
                             f"{r.get('available')}, current "
                             f"{r.get('current')} {r.get('currency', '')}".rstrip())
        _emit(args, {"provider": conn_name, "balances": rows},
              "\n".join(lines) if lines
              else f"no balances reported via {conn_name}")
        return 0

    if action == "transactions":
        if not account_id:
            _emit(args, {"error": "account_id required"},
                  "Usage: nm finance transactions --provider mono --account-id <id>")
            return 1
        # --days is the window for BOTH providers: Mono takes ISO
        # start/end kwargs; Plaid takes a (start_date, end_date) range
        # positionally. The old code passed the account id positionally
        # to both — for Plaid it landed in start_date and end_date was
        # never supplied — and then tried to slice Mono's dict result.
        from datetime import date, timedelta

        days = int(getattr(args, "days", 30) or 30)
        end = date.today()
        start = end - timedelta(days=days)
        try:
            if conn_name == "plaid":
                entries = conn.get_transactions(
                    start.isoformat(), end.isoformat(),
                    account_ids=[account_id])
                txs = []
                for entry in entries:
                    if isinstance(entry, dict) and isinstance(
                            entry.get("transactions"), list):
                        txs.extend(entry["transactions"])
                    elif isinstance(entry, dict):
                        txs.append(entry)
            else:
                res = conn.get_transactions(
                    account_id, start=start.isoformat(),
                    end=end.isoformat())
                txs = (res.get("transactions", [])
                       if isinstance(res, dict) else list(res))
        except Exception as exc:
            _emit(args, {"error": str(exc)}, f"Failed: {exc}")
            return 1
        _emit(args, {"provider": conn_name, "count": len(txs),
                     "transactions": txs[:10],
                     "window": {"start": start.isoformat(),
                                "end": end.isoformat(), "days": days}},
              f"Found {len(txs)} transaction(s) via {conn_name} "
              f"({start.isoformat()} to {end.isoformat()})")
        return 0

    _emit(args, {"error": f"unknown action: {action}"},
          f"Unknown action: {action}. Use: status|link|accounts|transactions|balance")
    return 1
