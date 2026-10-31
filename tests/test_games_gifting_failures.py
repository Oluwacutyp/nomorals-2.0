"""Failure pins for nomorals/games/gifting.py (tail-72).

One pin per census site. All six sites are adjudicated EXCLUSION except
148, which was FLAGGED and then CONVERTED by parent amendment
(see SURVEY.md in the tail-72 verify dir):

- 102 pending_for query failure -> None (degrade-to-absence, fail-closed)
- 148 _set_status UPDATE failure -> traced + False; callers report
  honestly (CONVERTED — was a bare traceless pass)
- 172 history query failure -> [] (degrade-to-absence, display only)
- 194 _canonical_key import failure -> key unchanged (fallback is contract)
- 231 find_by_username failure -> scan fallback still resolves
- 238 store.all failure -> "no player found" (absence-conflation, stated)
"""

import logging
import sqlite3
import sys
import types
from contextlib import contextmanager

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.gifting import GiftStore, _canonical_key, resolve_recipient


class Q:
    """DB shim over sqlite3 with fault injection + a call log (modeled on
    tests/test_gifting.py's _Q)."""

    def __init__(self, conn, fail=()):
        self.conn = conn
        self.fail = set(fail)       # method names that raise
        self.fail_sql = ()          # execute() raises if SQL contains these
        self.calls = []

    def _gate(self, name, sql=""):
        self.calls.append((name, sql))
        if name in self.fail or any(s in sql for s in self.fail_sql):
            raise RuntimeError(f"injected {name} failure")

    def execute(self, sql, params=()):
        self._gate("execute", sql)
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        self._gate("query", sql)
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def transaction(self):
        @contextmanager
        def tx():
            yield
            self.conn.commit()
        return tx()


def _fresh_q(**kw):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return Q(conn, **kw)


def _store_with_gift():
    q = _fresh_q()
    store = GiftStore(q)
    gift = store.create("tg:1", "Ada", "tg:2", "Bob", "coins", "100",
                        amount=100)
    q.calls.clear()
    return q, store, gift


class _LogCap:
    """Manual handler on the module logger (probe-proven capture path)."""

    def __init__(self):
        self.messages = []
        cap = self

        class H(logging.Handler):
            def emit(self, record):
                cap.messages.append(record.getMessage())

        self.handler = H()
        self.logger = logging.getLogger("nomorals.games.gifting")

    def __enter__(self):
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *exc):
        self.logger.removeHandler(self.handler)
        return False


# --------------------------------------------------------------------- 102
def test_pending_for_query_failure_returns_none():
    q, store, gift = _store_with_gift()
    assert store.pending_for("tg:1") is not None  # healthy control
    q.fail = {"query"}
    q.calls.clear()
    # A sick DB degrades to the no-gift / no-db value: None (fail-closed —
    # confirm/cancel then honestly report "nothing pending").
    assert store.pending_for("tg:1") is None
    assert any(c[0] == "query" for c in q.calls)  # attempted (control)
    assert GiftStore(_fresh_q()).pending_for("tg:9") is None  # empty store


# --------------------------------------------------------------------- 148
class _Wallet:
    """Player-store stub for the runtime handler: real balance moves."""

    def __init__(self):
        self.balances = {}
        self.calls = []

    def spend_coins(self, player, amount, ref):
        self.calls.append(("spend", player.key, amount, ref))
        self.balances[player.key] = \
            self.balances.get(player.key, 0) - amount
        return self.balances[player.key]

    def add_coins(self, player, amount, ref):
        self.calls.append(("add", player.key, amount, ref))
        self.balances[player.key] = \
            self.balances.get(player.key, 0) + amount


class _GiftHarness(RuntimeGamesMixin):
    """RuntimeGamesMixin with a stub engine (pattern of
    tests/test_game_identity_username.py's _DeleteHarness)."""

    def __init__(self, engine):
        self._engine = engine
        self.context = types.SimpleNamespace(db=None)
        self.notified = []

    def _game_engine(self):
        return self._engine

    def _notify_gift(self, recipient, gift, giver):
        self.notified.append((recipient.key, gift.id))


def _harness(q, wallet=None):
    from nomorals.games.players import Player
    engine = types.SimpleNamespace(
        db=q, store=wallet or _Wallet(),
        economy=types.SimpleNamespace(_items={}))
    return _GiftHarness(engine), Player(
        key="tg:1", platform="tg", name="Ada")


def _status_of(q, gift_id):
    row = q.query_one("SELECT status FROM game_gifts WHERE id = ?",
                      (gift_id,))
    return row["status"]


def test_set_status_failure_is_traced_reported_and_warns():
    """CONVERTED (tail-72 parent amendment): the UPDATE inside
    _set_status IS the operation. Its failure is now (a) traced on the
    module logger in the file's own idiom, (b) returned as False from
    mark_done/_set_status, (c) surfaced on the confirm reply with a
    do-not-confirm-again warning (the transfer really landed; a blind
    re-confirm would send twice), and (d) cancel() no longer stamps
    'cancelled' on a gift whose write failed — the cancel handler
    replies honestly instead of "nothing moved"."""
    # --- store level: traced + False, healthy control returns True
    q, store, gift = _store_with_gift()
    assert store.mark_done(gift.id) is True  # healthy control
    assert _status_of(q, gift.id) == "done"
    assert store.pending_for("tg:1") is None

    q2, store2, gift2 = _store_with_gift()
    with _LogCap() as cap:
        q2.fail_sql = ("UPDATE game_gifts SET status",)
        assert store2.mark_done(gift2.id) is False
        assert any("UPDATE game_gifts" in sql for _n, sql in q2.calls
                   if _n == "execute")  # attempted (control)
    assert any("set_status failed" in m for m in cap.messages)
    assert _status_of(q2, gift2.id) == "pending"  # DB truth unchanged
    # Degraded mode (db=None): no durable record, nothing to chain —
    # the write is vacuously landed.
    assert GiftStore(None).mark_done("gift_x") is True

    # --- runtime confirm (coins): transfer lands, warning rides reply
    q3 = _fresh_q()
    GiftStore(q3).create("tg:1", "Ada", "tg:2", "Bob", "coins", "100",
                         amount=100)
    harness, player = _harness(q3)
    harness._engine.store.balances["tg:1"] = 500
    q3.fail_sql = ("UPDATE game_gifts SET status",)
    with _LogCap() as cap3:
        reply = harness._control_gift("confirm", player=player)
    assert "🎁 sent 100 coins to Bob!" in reply
    assert "do NOT /gift confirm again" in reply
    assert "would send twice" in reply
    assert harness._engine.store.balances["tg:2"] == 100  # value moved
    assert harness.notified  # recipient still notified (value moved)
    assert any("set_status failed" in m for m in cap3.messages)

    # Healthy confirm control: no warning, row done.
    q4 = _fresh_q()
    g4 = GiftStore(q4).create("tg:1", "Ada", "tg:2", "Bob", "coins",
                              "100", amount=100)
    harness4, player4 = _harness(q4)
    harness4._engine.store.balances["tg:1"] = 500
    reply4 = harness4._control_gift("confirm", player=player4)
    assert "🎁 sent 100 coins to Bob!" in reply4
    assert "do NOT" not in reply4
    assert _status_of(q4, g4.id) == "done"

    # --- cancel: true status at store level + honest handler reply
    q5, store5, gift5 = _store_with_gift()
    q5.fail_sql = ("UPDATE game_gifts SET status",)
    dropped = store5.cancel("tg:1")
    assert dropped is not None and dropped.status == "pending"
    harness5, player5 = _harness(q5)
    reply5 = harness5._control_gift("cancel", player=player5)
    assert "still pending" in reply5
    assert "nothing moved" not in reply5
    # Healthy cancel control: old reply, gift stamped cancelled.
    q6, store6, gift6 = _store_with_gift()
    ok_drop = store6.cancel("tg:1")
    assert ok_drop is not None and ok_drop.status == "cancelled"
    harness6, player6 = _harness(q6)
    # gift6 was already cancelled above; stage a fresh one to cancel
    # through the handler.
    GiftStore(q6).create("tg:1", "Ada", "tg:2", "Bob", "coins", "5",
                         amount=5)
    reply6 = harness6._control_gift("cancel", player=player6)
    assert reply6.startswith("cancelled:") and "nothing moved." in reply6


def test_execute_gift_item_and_gear_branches_warn_on_failed_record(
        monkeypatch):
    """The item and gear branches of _execute_gift carry the same
    warning: mark_done False after the value moved -> the reply says
    the gift went through but must not be confirmed again."""
    import nomorals.games.gear as gear_mod

    # --- item branch (direct handler call, stub wallet store)
    class _ItemStore:
        def __init__(self):
            self.moves = []

        def consume_item(self, player, ref):
            self.moves.append(("consume", player.key, ref))
            return True

        def grant_item(self, player, ref):
            self.moves.append(("grant", player.key, ref))

    q = _fresh_q()
    store = GiftStore(q)
    gift = store.create("tg:1", "Ada", "tg:2", "Bob", "item", "potion")
    engine = types.SimpleNamespace(
        db=q, store=_ItemStore(),
        economy=types.SimpleNamespace(
            _items={"potion": types.SimpleNamespace(name="Potion")}))
    harness = _GiftHarness(engine)
    from nomorals.games.players import Player
    player = Player(key="tg:1", platform="tg", name="Ada")
    q.fail_sql = ("UPDATE game_gifts SET status",)
    reply = harness._execute_gift(engine, store, gift, player)
    assert "🎁 sent Potion to Bob!" in reply
    assert "do NOT /gift confirm again" in reply
    assert ("grant", "tg:2", "potion") in engine.store.moves

    # --- gear branch (GearStore stubbed at its import site)
    class _GearStore:
        def __init__(self, db):
            self.db = db

        def find(self, key, ref):
            return types.SimpleNamespace(id="inst1", slug=ref)

        def transfer(self, inst_id, to_key):
            return True, "Katana"

    monkeypatch.setattr(gear_mod, "GearStore", _GearStore)
    q2 = _fresh_q()
    store2 = GiftStore(q2)
    gift2 = store2.create("tg:1", "Ada", "tg:2", "Bob", "gear",
                          "katana_rare")
    engine2 = types.SimpleNamespace(
        db=q2, store=_ItemStore(),
        economy=types.SimpleNamespace(_items={}))
    harness2 = _GiftHarness(engine2)
    q2.fail_sql = ("UPDATE game_gifts SET status",)
    reply2 = harness2._execute_gift(engine2, store2, gift2, player)
    assert "🎁 sent Katana to Bob!" in reply2
    assert "do NOT /gift confirm again" in reply2


# --------------------------------------------------------------------- 172
def test_history_query_failure_returns_empty():
    q, store, gift = _store_with_gift()
    store.mark_done(gift.id)
    assert len(store.history("tg:1")) == 1  # healthy control
    q.fail = {"query"}
    q.calls.clear()
    assert store.history("tg:1") == []  # degrade-to-absence (display only)
    assert any(c[0] == "query" for c in q.calls)  # attempted (control)


# --------------------------------------------------------------------- 194
def test_canonical_key_import_failure_returns_key_unchanged(monkeypatch):
    assert _canonical_key("telegram-bot:42") == "telegram:42"  # control
    fake = types.ModuleType("nomorals.games.players")  # no aliases attr
    monkeypatch.setitem(sys.modules, "nomorals.games.players", fake)
    # Import failure degrades to the identity fold — the only honest key.
    assert _canonical_key("telegram-bot:42") == "telegram-bot:42"


# ------------------------------------------------------------- 231 and 238
class _Prof:
    platform = "telegram"

    def __init__(self, key, name, username):
        self.key, self.name, self.username = key, name, username


class _Store:
    def __init__(self, profs, fail_all=False, fail_user=False):
        self._p = profs
        self.fail_all, self.fail_user = fail_all, fail_user
        self.calls = []

    def find_by_username(self, label):
        self.calls.append(("find_by_username", label))
        if self.fail_user:
            raise RuntimeError("user lookup down")
        return None

    def all(self, limit=500):
        self.calls.append(("all", limit))
        if self.fail_all:
            raise RuntimeError("store down")
        return list(self._p)


def test_resolve_recipient_username_lookup_failure_falls_back_to_scan():
    ada = _Prof("telegram:111", "Ada", "ada_lovelace")
    store = _Store([ada], fail_user=True)
    prof, err = resolve_recipient(store, "@ada_lovelace")
    # The priority-1 rung failed (control: it was attempted); the scan
    # rungs still resolve by username — the ladder is the contract.
    assert store.calls[0] == ("find_by_username", "ada_lovelace")
    assert ("all", 500) in store.calls
    assert prof is ada and err == ""


def test_resolve_recipient_store_all_failure_reads_as_no_player():
    ada = _Prof("telegram:111", "Ada", "ada_lovelace")
    store = _Store([ada], fail_all=True)
    prof, err = resolve_recipient(store, "@ada")
    assert ("all", 500) in store.calls  # attempted (control)
    assert prof is None
    # Byte-identical to a genuinely empty store (absence-conflation,
    # stated in SURVEY; the flow fails closed — no gift is staged).
    empty_prof, empty_err = resolve_recipient(_Store([]), "@ada")
    assert empty_prof is None and err == empty_err
    assert "no player found matching 'ada'" in err
