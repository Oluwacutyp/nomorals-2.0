"""RuntimeMetaMixin: PartnerRuntime command group (meta)."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from ...storage.kv import KVStore
from .approvals import _key_set
from .failures import record_failure
_log = get_logger(__name__)


class RuntimeMetaMixin:
    """RuntimeMetaMixin for :class:`PartnerRuntime`."""


    def _model_status_line(self) -> str:
        """Which model is ACTUALLY answering, and whether it's failing over.

        This is the line that should have caught the silent mock fallback:
        a failing provider must be visible in /status, not buried in a warn
        log line that scrolls away.

        Honest by construction: a provider is only called "answering" if it
        has actually succeeded (``last_success``).  The old line showed the
        first fallback *by position* — "hf_serverless is answering for now"
        while every message went unanswered because that fallback was
        failing too (no token).  That lie is what made a dead chain look
        alive in /status.
        """
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return "n/a"
        try:
            snap = snapshot()
        except Exception as exc:  # noqa: BLE001 - status must never crash a chat
            record_failure("runtime_meta._model_status_line", exc)
            return "n/a"
        active = str(snap.get("active") or "?")
        chain = [str(p) for p in (snap.get("chain") or [])]
        health = (snap.get("health") or {})
        active_health = health.get(active) or {}
        if not int(active_health.get("failures") or 0):
            return f"{active} (chain: {', '.join(chain)})" if chain else active
        last_error = str(active_health.get("last_error") or "unknown error")
        reason = self._plain_model_reason(last_error)
        # Who has ACTUALLY answered recently?  By last_success, not by
        # position in the chain.
        succeeded = [
            (float((health.get(p) or {}).get("last_success") or 0.0), p)
            for p in chain
            if p != active
        ]
        succeeded = [(ts, p) for ts, p in succeeded if ts > 0]
        if succeeded:
            top = max(succeeded)[1]
            return (f"{top} is answering for now — {active} is down ({reason}); "
                    f"{active} gets another try on the next message")
        # Nothing in the chain has ever answered.  Say so, with the backup's
        # own failure reason — that is the difference between "down and
        # recovering" and "silently dead".
        backups = [p for p in chain if p != active]
        if not backups:
            return f"no model is answering — {active} is down ({reason}) and " \
                   "there is no backup configured"
        caps = {str(n): set(v) for n, v in (snap.get("chain_caps") or {}).items()}
        backup_reasons = []
        for p in backups:
            known_caps = caps.get(p)
            if known_caps is not None \
                    and "chat" not in known_caps and "complete" not in known_caps:
                # this backup can never answer a chat message — saying
                # "has not been tried yet" for it is a soft lie
                # (unknown capability lists are treated as chat-capable)
                continue
            b_health = health.get(p) or {}
            if int(b_health.get("failures") or 0):
                backup_reasons.append(f"{p} failed too ({self._plain_model_reason(str(b_health.get('last_error') or 'unknown error'))})")
            else:
                backup_reasons.append(f"{p} in chain (not yet verified this session)")
        if backup_reasons:
            return (f"no model is answering — {active} is down ({reason}); "
                    + "; ".join(backup_reasons)
                    + ". Messages get no reply until one of them works")
        # every backup is a non-chat provider (e.g. ocr) — only the local
        # model can answer
        return (f"no model is answering — {active} is down ({reason}) and the "
                "other providers in the chain cannot chat. Only fixing "
                f"{active} restores replies")

    @staticmethod
    def _plain_model_reason(error: str) -> str:
        """A raw exception string in chat reads like the system is broken.
        Translate the common ones into plain words."""
        low = error.lower()
        if "still loading" in low or "loading" in low:
            return "still loading the model — normal on a phone, takes a few minutes"
        if "400" in low or "not served by this hf" in low or "not in the inference" in low:
            return "the model name is not hosted there — it tries to swap to a hosted model"
        if "timed out" in low or "timeout" in low:
            return "took too long — if it's your phone's model, close other apps"
        if "connection refused" in low or "errno 111" in low:
            return "not running — start it with `nm models --start-local`"
        if "not responding" in low:
            return "frozen — kill and restart it with `nm models --start-local`"
        if "rate limit" in low or "429" in low:
            return "rate limited — it recovers on its own"
        if "401" in low or "unauthorized" in low or "invalid" in low:
            return "bad credentials"
        return error[:60]

    def _control_status(self) -> str:
        """The /status reply as a designed Card (presentation
        Phase 3): the same facts the old eight-line dump carried,
        composed as titled State + Traffic stat blocks, with the
        degradation and failure-ledger lines surfacing ONLY when
        real — as verbatim prose, so their copy ("degradation:",
        "partner failures (N total") reads identically.  Rendered
        in the canonical house dialect; the gateway converts per
        platform at send, exactly like any producer string."""
        mood = self.brain.mood.current()
        rel = self.brain.relationship
        aut = self._autonomy.status() if self._autonomy else {"mode": "off"}
        from ..power import power_mode_for

        power_state = "active" if power_mode_for(self.context).active else "locked"
        platforms = ", ".join(
            name for name, info in self.gateway.status().items()
            if not name.startswith("_") and isinstance(info, dict)
            and info.get("running_in_session")
        ) or "none"
        stats = self._stats_snapshot()
        from ...social.chat.typography import Card, card_text

        card = Card("settings", title="Devon — status")
        card.stats([
            ("mood", f"{mood.label} (affection "
                     f"{mood.values.get('affection', 0):.0f}, "
                     f"energy {mood.values.get('energy', 0):.0f}, "
                     f"frustration {mood.values.get('frustration', 0):.0f})"),
            ("relationship", f"{rel.stage}, trust {rel.trust:.0f}, "
                             f"{len(rel.milestones)} milestones, "
                             f"{len(rel.fights)} fights logged"),
            ("autonomy", f"{aut.get('mode', 'off')}, "
                         f"{aut.get('pending_proposals', 0)} pending"),
            ("model", self._model_status_line()),
            ("platforms running", platforms),
            ("power mode", power_state),
        ], title="State")
        card.stats([
            ("messages", stats["messages"]),
            ("replies", stats["replies"]),
            ("errors", stats["errors"]),
            ("commands", stats.get("controls", 0)),
        ], title="Traffic")
        # Degradation ladders: surfaced ONLY when a subsystem is
        # actually serving from a fallback rung — silence when healthy.
        degradation = self._degradation_status()
        if degradation.get("degraded"):
            rungs = ", ".join(
                f"{name} on rung {degradation['ladders'][name]['rung']} "
                f"({degradation['ladders'][name]['rung_name']})"
                for name in degradation["degraded"])
            card.block_lines([f"degradation: {rungs}"])
        # Partner failure ledger: surfaced ONLY when best-effort
        # fallbacks have actually fired — silence when healthy.
        from .failures import failure_count, format_failures
        if failure_count():
            card.block_lines(format_failures(limit=5).splitlines())
        return card_text(card)

    def _control_museum(self, tail: str, message: Any = None) -> str:
        """The /museum reply: the Museum census over every system, on
        demand. A thin dispatch to the same ``museum_census`` tool the
        brain uses (audit R3-73) — no Museum construction duplicated
        here, so the command's output is byte-identical to the tool's.
        Read-only (FS_READ); a failing shelf reports a named error
        inside the census, never here. A tool failure is an honest
        refusal in words, never silence and never a fake census.

        ``/museum <shelf>`` shows only that shelf's cards — the same
        card lines as the full census, filtered through the tool's
        ``shelf`` parameter, so the handler still never builds a
        Museum itself. An unknown shelf name is refused in words,
        naming the shelves that hold cards; the check is against the
        live census's ``by_shelf``, never a hardcoded list.

        ``/museum send <id>`` (the R3-73 remainder) delivers one
        sendable card through the same ``file_send`` tool ``/file``
        and ``/artifact send`` use: the ``museum_find`` tool resolves
        the id (exact or unique prefix), a non-sendable card is
        refused in words — never the store file behind it — and the
        file goes back to the chat the command came from. The
        census-time ``sendable`` is a fact about then; the send
        re-checks the file exists now, in case it raced away."""
        words = (tail or "").split()
        if not words:
            outcome = self.context.tools.call("museum_census")
            if not outcome.ok:
                return ("museum census failed: "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            value = outcome.value or {}
            return str(value.get("card_text") or "museum is empty")
        if words[0].lower() != "send":
            if len(words) > 1:
                return ("usage: /museum <shelf> — one shelf name, or "
                        "/museum send <id> to send one card")
            outcome = self.context.tools.call("museum_census",
                                              shelf=words[0])
            if not outcome.ok:
                return ("museum census failed: "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            value = outcome.value or {}
            return str(value.get("card_text") or "museum is empty")
        if len(words) < 2:
            return ("usage: /museum send <id> — /museum lists the "
                    "cards and their ids")
        if message is None:
            return ("send needs a live chat — run /museum send from "
                    "a chat, not the console")
        ref = words[1]
        outcome = self.context.tools.call("museum_find", ref=ref)
        if not outcome.ok:
            return ("museum find failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        card = (outcome.value or {}).get("card")
        if not card:
            return (f"no single card matches {ref!r} — /museum lists "
                    "them, use the full id")
        if not card.get("sendable"):
            return (f"that's exhibit-only — {card.get('title')} "
                    f"[{card.get('id')}] has no file to send "
                    f"({card.get('kind')} cards point at their owning "
                    "store, they never ship it)")
        path = str(card.get("path") or "")
        if not os.path.isfile(path):
            return (f"the file for {card.get('title')} "
                    f"[{card.get('id')}] is gone from the museum — "
                    "nothing left to send")
        chat = message.chat
        outcome = self.context.tools.call(
            "file_send",
            platform=str(getattr(chat, "platform", "") or ""),
            chat_id=str(getattr(chat, "chat_id", "") or ""),
            path=path, caption=f"🏛️ {card.get('title')}")
        if not outcome.ok:
            return ("file send failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        value = outcome.value or {}
        size_kb = int(value.get("bytes") or 0) // 1024
        name = os.path.basename(str(value.get("path") or path))
        return (f"📎 sent {card.get('title')} [{card.get('id')}] — "
                f"{name} ({size_kb} KB)")

    def _control_analytics(self, tail: str, message: Any = None) -> str:
        """The /analytics reply: analytical queries + exports over the
        live store, on demand. A thin dispatch to the same engine the
        ``nm analytics`` CLI uses (storage/analytics.py: DuckDB when
        installed, the SQLite passthrough otherwise) — no engine
        construction duplicated here, so chat and console answer from
        the same data.

        ``/analytics`` shows the backend + stats; ``tables`` lists
        every live table with row counts; ``query "<select sql>"`` runs
        a read-only query (SELECT/WITH only, 100 rows shown);
        ``export <table> [csv|parquet]`` / ``export query "<select
        sql>" [csv|parquet]`` writes a file under <home>/exports/ and
        sends it back to this chat through the same ``file_send`` tool
        ``/museum send`` uses. Parquet needs the DuckDB backend and
        refuses honestly on the SQLite passthrough, naming the backend,
        never silently writing a CSV with a .parquet name. A query
        export is capped at 1000 rows and says so when the cap bites.
        Owner-only; read-only throughout — nothing here writes the
        store. A tool failure is an honest refusal in words, never
        silence and never a fake export."""
        from ...storage.analytics import (
            analytics_source_rows,
            default_export_path,
            open_analytics,
            referenced_tables,
        )

        words = (tail or "").split()
        verb = words[0].lower() if words else "status"
        if verb not in ("status", "tables", "query", "export"):
            return ("usage: /analytics [status | tables | query "
                    "\"<select sql>\" | export <table> [csv|parquet] | "
                    "export query \"<select sql>\" [csv|parquet]]")
        db = getattr(self.context, "db", None)
        if db is None:
            return ("analytics needs a database — none is attached "
                    "to this session.")
        try:
            engine = open_analytics(db)
        except Exception as exc:  # noqa: BLE001 - honest refusal
            return f"analytics unavailable: {exc}"
        if verb == "status":
            try:
                return str(engine.format_stats())
            except Exception as exc:  # noqa: BLE001 - status never crashes chat
                return f"analytics status failed: {exc}"
        if verb == "tables":
            rows: list[tuple[str, int]] = []
            for name in db.tables():
                try:
                    seam = analytics_source_rows(db, name)
                    count = len(seam) if seam is not None else db.row_count(name)
                except Exception:  # noqa: BLE001 - one bad table must not kill the listing
                    count = -1
                rows.append((name, count))
            rows.sort(key=lambda r: -r[1])
            lines = [f"tables ({len(rows)}):"]
            for name, count in rows[:100]:
                lines.append(
                    f"  {name} — {'?' if count < 0 else count} rows")
            return "\n".join(lines)
        if verb == "query":
            _, _, sql = (tail or "").partition(" ")
            sql = sql.strip()
            if not sql:
                return ("usage: /analytics query \"<select sql>\" — "
                        "read-only (SELECT/WITH)")
            try:
                if getattr(engine, "name", "") == "duckdb":
                    for table in referenced_tables(sql):
                        engine.load_table(db, table)
                result = engine.query(sql)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"analytics query failed: {exc}"
            shown = result[:100]
            lines = [f"{len(result)} rows"
                     + (f" (showing {len(shown)})"
                        if len(shown) < len(result) else "") + ":"]
            for row in shown:
                lines.append("  " + ", ".join(
                    f"{k}={v}" for k, v in row.items()))
            return "\n".join(lines)
        # verb == "export"
        rest = words[1:]
        if not rest:
            return ("usage: /analytics export <table> [csv|parquet] | "
                    "/analytics export query \"<select sql>\" [csv|parquet]")
        fmt = "csv"
        if rest[-1].lower() in ("csv", "parquet"):
            fmt = rest[-1].lower()
            rest = rest[:-1]
            if not rest:
                return ("usage: /analytics export <table> [csv|parquet] | "
                        "/analytics export query \"<select sql>\" [csv|parquet]")
        if fmt == "parquet" and getattr(engine, "name", "") != "duckdb":
            return ("parquet export needs the duckdb backend — this "
                    "install runs the sqlite passthrough; use csv")
        if message is None:
            return ("export needs a live chat — run /analytics export "
                    "from a chat, not the console")
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home", None) if settings is not None else None
        home = str(home or "~/.nomorals")
        try:
            if rest[0].lower() == "query":
                sql = " ".join(rest[1:])
                if not sql:
                    return ("usage: /analytics export query "
                            "\"<select sql>\" [csv|parquet]")
                if getattr(engine, "name", "") == "duckdb":
                    for table in referenced_tables(sql):
                        engine.load_table(db, table)
                target = default_export_path(home, "query", fmt)
                rows = engine.export_query(sql, target, format=fmt,
                                           limit=1000)
                capped = rows >= 1000
                label = "query"
            else:
                table = rest[0]
                if getattr(engine, "name", "") == "duckdb":
                    engine.load_table(db, table)
                target = default_export_path(home, table, fmt)
                if fmt == "parquet":
                    engine.to_parquet(table, target)
                else:
                    engine.to_csv(table, target)
                seam = analytics_source_rows(db, table)
                if seam is not None:
                    rows = len(seam)
                else:
                    count = engine.query(
                        f'SELECT COUNT(*) AS n FROM "{table}"')
                    rows = int(count[0]["n"]) if count else 0
                capped = False
                label = table
        except Exception as exc:  # noqa: BLE001 - honest refusal
            return f"analytics export failed: {exc}"
        size = target.stat().st_size if target.is_file() else 0
        summary = (f"📊 exported {rows} rows from {label} to "
                   f"{target.name} ({size} bytes)"
                   + (" — capped at 1000 rows; narrow the query "
                      "for the rest" if capped else ""))
        tools = getattr(self.context, "tools", None)
        if tools is None:
            return (f"{summary} The file is built at {target}, but "
                    "this session has no file-send tool.")
        chat = message.chat
        outcome = tools.call(
            "file_send",
            platform=str(getattr(chat, "platform", "") or ""),
            chat_id=str(getattr(chat, "chat_id", "") or ""),
            path=str(target), caption=summary)
        if not outcome.ok:
            return (f"{summary} Sending the file failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)} "
                    f"(it is kept at {target}.)")
        return summary

    def _control_airports(self, tail: str, message: Any = None) -> str:
        """The /airports reply: the airport→country table, in chat.

        The chat twin of ``nm airports`` (airport refresh ship
        0bd2bd1f): ``/airports`` shows the loaded table (IATA codes,
        countries), the last refresh, and the next monthly run;
        ``/airports refresh`` runs one refresh tick now. The same
        ``refresh_airports`` / ``read_refresh_state`` pair the console
        command calls, so chat and console answer from the same state
        and the same words. A failed refresh reports the reason and
        leaves the old table in place; the handler never raises.
        Owner-only. Nothing here needs a live chat — no file is sent
        back — so ``message`` is accepted and ignored."""
        from ...travel.airports import (
            airport_countries_count,
            airport_country_count,
        )
        from ...travel.refresh import (
            AIRPORTS_REFRESH_SPEC,
            AIRPORTS_SOURCE_URL,
            next_refresh_run,
            read_refresh_state,
            run_airport_refresh_tick_once,
        )

        words = (tail or "").split()
        verb = words[0].lower() if words else "status"
        if verb not in ("status", "refresh"):
            return "usage: /airports [status | refresh]"
        if verb == "status":
            try:
                rows = airport_country_count()
                countries = airport_countries_count()
            except Exception as exc:  # noqa: BLE001 - status never crashes chat
                return f"airport table unavailable: {exc}"
            lines = [f"airport table: {rows} IATA codes, "
                     f"{countries} countries"]
            state = read_refresh_state(self.context)
            if state is None:
                lines.append("last refresh: never")
            elif state.get("ok"):
                lines.append(
                    f"last refresh: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(state.get('ran_at') or 0.0)))}"
                    f" — {state.get('rows', 0)} rows, "
                    f"{state.get('countries', 0)} countries")
            else:
                lines.append(
                    f"last refresh: {time.strftime('%Y-%m-%d %H:%M', time.localtime(float(state.get('ran_at') or 0.0)))}"
                    f" — failed: {state.get('error') or 'unknown error'}")
            lines.append(f"source: {AIRPORTS_SOURCE_URL}")
            lines.append(f"schedule: {AIRPORTS_REFRESH_SPEC}")
            lines.append(f"next run: {next_refresh_run()}")
            return "\n".join(lines)
        try:
            out = run_airport_refresh_tick_once(context=self.context)
        except Exception as exc:  # noqa: BLE001 - refresh never crashes chat
            return f"refresh failed: {exc}"
        if out.get("ok"):
            return (f"refreshed: {out.get('rows', 0)} IATA codes, "
                    f"{out.get('countries', 0)} countries, "
                    f"{out.get('dropped', 0)} rows dropped")
        return f"refresh failed: {out.get('error') or 'unknown error'}"

    def _control_flight(self, tail: str, message: Any = None) -> str:
        """The /flight reply: live flight tracking, in chat.

        ``/flight BA249`` follows one flight — live position,
        altitude, speed, heading, ETA, data age and source — and
        sends the track map she draws herself. ``/flight area LOS``
        lists live aircraft over an airport (or a lat/lon box).
        ``/flight watch BA249 LHR [expect-epoch]`` arms a watch
        that pings on landing, diversion, delay or loss;
        ``/flight unwatch BA249`` drops it, ``/flight watches``
        lists the armed watches. Owner-only. The feed is the open
        ADS-B feeds (OpenSky) — never the schedule; a down or empty
        feed is reported in words with the reason, never a faked
        position. A render or send failure reports in words — never
        silence, never a fake image.
        """
        from ...travel.tracking import (
            FlightWatcher,
            OpenSkyFeed,
            TrackStore,
            aircraft_in_area,
            coords_for_iata,
            render_track_map,
            track_flight,
        )

        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home", None) if settings is not None else None
        home = str(home or "~/.nomorals")

        def _map_for(out: dict[str, Any],
                     destination: str | None) -> str | None:
            """Draw the track map for a successful track. Returns the
            PNG path, or None when there is no home to write to."""
            try:
                store = TrackStore(home)
                points = store.points(out["callsign"])
                current = {"latitude": out.get("latitude"),
                           "longitude": out.get("longitude")}
                png, _digest = render_track_map(
                    points, destination=destination or None,
                    current=current)
                path = (Path(home) / "travel"
                        / f"track_{out['callsign']}.png")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png)
                return str(path)
            except Exception:  # noqa: BLE001 - a failed map never kills the track
                return None

        words = (tail or "").split()
        verb = words[0].lower() if words else ""
        if verb == "watch":
            if len(words) < 2:
                return ("usage: /flight watch <flight> "
                        "[destination] [expect-epoch]")
            destination = ""
            expect_at = 0.0
            if len(words) >= 3 and len(words[2]) == 3:
                destination = words[2].upper()
            for token in words[2:]:
                try:
                    expect_at = float(token)
                except (TypeError, ValueError):
                    continue
            try:
                watch = FlightWatcher(home=home).watch(
                    words[1], destination=destination,
                    expect_at=expect_at)
            except Exception as exc:  # noqa: BLE001 - watch never crashes chat
                return f"watch failed: {exc}"
            line = f"watching {watch.flight}"
            if watch.destination:
                line += f" → {watch.destination}"
            if watch.expect_at:
                line += (f" — expected "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(watch.expect_at))}")
            return line + "."
        if verb == "unwatch":
            if len(words) < 2:
                return "usage: /flight unwatch <flight>"
            watcher = FlightWatcher(home=home)
            query = words[1].upper()
            removed = 0
            for watch in watcher.list_watches(active_only=False):
                if watch.flight == query:
                    removed += int(watcher.unwatch(watch.id))
            if removed:
                return f"no longer watching {query}."
            return f"no watch on {query}."
        if verb == "watches":
            watches = FlightWatcher(home=home).list_watches()
            if not watches:
                return "no flights watched."
            lines = ["watched flights:"]
            for watch in watches:
                line = f"• {watch.flight}"
                if watch.destination:
                    line += f" → {watch.destination}"
                if watch.expect_at:
                    line += (f" — expected "
                             f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(watch.expect_at))}")
                if watch.alerted:
                    line += f" (alerted: {', '.join(watch.alerted)})"
                lines.append(line)
            return "\n".join(lines)
        if verb == "area":
            if len(words) < 2:
                return ("usage: /flight area <IATA> | /flight area "
                        "<lat_min> <lon_min> <lat_max> <lon_max>")
            box: tuple[float, float, float, float] | None = None
            label = ""
            if len(words) == 2 and len(words[1]) == 3:
                coords = coords_for_iata(words[1])
                if coords is None:
                    return f"no airport {words[1].upper()!r}"
                lat, lon = coords
                box = (lat - 1.0, lon - 1.0, lat + 1.0, lon + 1.0)
                label = f"over {words[1].upper()}"
            elif len(words) == 5:
                try:
                    box = (float(words[1]), float(words[2]),
                           float(words[3]), float(words[4]))
                except (TypeError, ValueError):
                    return ("usage: /flight area <IATA> | /flight area "
                            "<lat_min> <lon_min> <lat_max> <lon_max>")
                label = "in that box"
            if box is None:
                return ("usage: /flight area <IATA> | /flight area "
                        "<lat_min> <lon_min> <lat_max> <lon_max>")
            try:
                states, meta = OpenSkyFeed().fetch(box)
            except Exception as exc:  # noqa: BLE001 - area never crashes chat
                return f"area lookup failed: {exc}"
            if meta.get("error"):
                return (f"live feed unavailable: {meta['error']} "
                        "(the schedule is not a substitute)")
            found = aircraft_in_area(states, box)
            if not found:
                return (f"no live aircraft {label} right now "
                        f"({meta.get('source') or 'OpenSky'}, "
                        f"data {meta.get('fetched_at', 0)} fetched)")
            lines = [f"live aircraft {label} "
                     f"({len(found)}, {meta.get('source') or 'OpenSky'}):"]
            for state in found[:15]:
                lines.append(
                    f"• {state.callsign or state.icao24} — "
                    f"{state.altitude_ft if state.altitude_ft is not None else '?'} ft, "
                    f"{state.speed_kts if state.speed_kts is not None else '?'} kt, "
                    f"{state.true_track if state.true_track is not None else '?'}°")
            if len(found) > 15:
                lines.append(f"… and {len(found) - 15} more")
            return "\n".join(lines)
        if not verb:
            return ("usage: /flight <flight> | /flight watch <flight> "
                    "[destination] [expect-epoch] | /flight unwatch "
                    "<flight> | /flight watches | /flight area <IATA>")
        # Default: track one flight, and send the map she draws.
        try:
            out = track_flight(OpenSkyFeed(), words[0],
                               store=TrackStore(home))
        except Exception as exc:  # noqa: BLE001 - track never crashes chat
            return f"flight lookup failed: {exc}"
        if not out.get("ok"):
            return (f"no live position for {words[0].upper()} right now"
                    f" — {out.get('reason') or 'unknown reason'}")
        lines = [f"✈️ {out['callsign'] or out['icao24']} "
                 f"({out.get('flight') or words[0].upper()})"]
        if out.get("latitude") is not None:
            lines.append(f"position: {out['latitude']:.4f}, "
                         f"{out['longitude']:.4f}")
        lines.append(f"altitude: {out.get('altitude_ft') or '?'} ft · "
                     f"speed: {out.get('speed_kts') or '?'} kt · "
                     f"heading: {out.get('heading') or '?'}°")
        if out.get("on_ground"):
            lines.append("on the ground")
        if out.get("destination"):
            lines.append(f"→ {out['destination']} "
                         f"{out.get('distance_km', '?')} km away, "
                         f"bearing {out.get('bearing', '?')}°"
                         + (f", ETA ~{out['eta_minutes']} min"
                            if out.get("eta_minutes") is not None
                            else ""))
        lines.append(f"data {out.get('age_s', 0)}s old "
                     f"({out.get('freshness') or 'unknown'}) · "
                     f"source: {out.get('source') or 'OpenSky'}")
        text = "\n".join(lines)
        map_path = _map_for(out, out.get("destination"))
        if map_path and message is not None:
            failed = self._scene_send_file(
                str(map_path), message, "track map")
            if failed is not None:
                return text + f"\n{failed}"
            return text + "\nmap sent."
        if map_path:
            return text + f"\nmap: {map_path}"
        return text

    def _control_burn(self, tail: str, message: Any = None) -> str:
        """The /burn reply: seal and read burn-after-reading notes.

        ``/burn seal <text> [--expire 1h|1d|7d|30d|never]
        [--password WORD]`` seals a note under a fresh random key
        and hands back the link — the key travels inside the link
        and never lives in storage. ``/burn read <code>
        [--password WORD]`` reads it once and burns it; a second
        read finds nothing. ``/burn list`` lists live notes as
        metadata only, never contents; ``/burn purge <id>`` deletes
        an unread note early. Owner-only. A wrong code or a wrong
        password refuses and leaves the note sealed; only a
        successful read burns it. The honest limit is stated in
        the seal reply: anyone holding the link can read it once,
        so the link is the key — keep it off public chats.
        """
        from ...memory.ephemeral_notes import (
            BurnNoteStore,
            list_notes,
            parse_expiry,
            purge_note,
            read_note,
            seal_note,
        )

        context = getattr(self, "context", None)
        settings = getattr(context, "settings", None) if context else None
        home = getattr(settings, "home", None) if settings else None
        home = str(home or "~/.nomorals")

        def _flag(rest: str, flag: str) -> tuple[str, str | None]:
            match = re.search(rf"\s{re.escape(flag)}\s+(\S+)\s*$", rest)
            if not match:
                return rest, None
            return rest[:match.start()].strip(), match.group(1)

        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        try:
            store = BurnNoteStore(home)
        except Exception as exc:  # noqa: BLE001 - burn never crashes chat
            return f"burn notes unavailable: {exc}"
        if verb == "seal":
            if not rest:
                return ("usage: /burn seal <text> "
                        "[--expire 1h|1d|7d|30d|never] [--password WORD]")
            rest, expire = _flag(rest, "--expire")
            rest, password = _flag(rest, "--password")
            if not rest:
                return ("usage: /burn seal <text> "
                        "[--expire 1h|1d|7d|30d|never] [--password WORD]")
            try:
                seconds = parse_expiry(expire if expire else None)
                sealed = seal_note(store, rest, expires_in_s=seconds,
                                   password=password or None)
            except ValueError as exc:
                return str(exc)
            lines = ["sealed — read once with:",
                     f"/burn read {sealed['link']}"]
            expires_at = sealed["expires_at"]
            when = ("never" if expires_at is None else
                    time.strftime("%Y-%m-%d %H:%M",
                                  time.localtime(expires_at)))
            lines.append(f"expires: {when}"
                         + (" · password-gated"
                            if sealed["has_password"] else ""))
            lines.append("anyone holding the link can read it once — "
                         "the link is the key, so keep it off public "
                         "chats.")
            return "\n".join(lines)
        if verb == "read":
            if not rest:
                return "usage: /burn read <code> [--password WORD]"
            code, password = _flag(rest, "--password")
            if not code:
                return "usage: /burn read <code> [--password WORD]"
            try:
                out = read_note(store, code.split()[0],
                                password=password or None)
            except Exception as exc:  # noqa: BLE001 - read never crashes chat
                return f"cannot read: {exc}"
            if not out.get("ok"):
                return f"cannot read: {out.get('error')}"
            return f"{out['text']}\n— note burned: it is gone now."
        if verb == "list":
            notes = list_notes(store)
            if not notes:
                return "no live burn notes."
            lines = ["live burn notes:"]
            for meta in notes:
                expires_at = meta["expires_at"]
                when = ("never" if expires_at is None else
                        time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(expires_at)))
                lines.append(f"• {meta['note_id']} · expires {when}"
                             + (" · password-gated"
                                if meta["has_password"] else ""))
            return "\n".join(lines)
        if verb == "purge":
            if not rest:
                return "usage: /burn purge <note-id>"
            if purge_note(store, rest.split()[0]):
                return f"purged {rest.split()[0]}."
            return f"no live note {rest.split()[0]}."
        return ("usage: /burn seal <text> [--expire 1h|1d|7d|30d|never] "
                "[--password WORD] | /burn read <code> [--password WORD] "
                "| /burn list | /burn purge <note-id>")

    def _control_share(self, tail: str, message: Any = None) -> str:
        """The /share reply: expiring file shares.

        ``/share <file> [downloads N] [--expire 1h|1d|7d|30d|never|
        --expire friday|--expire 2026-12-25]`` mints a share over a
        file Devon already holds and hands back the link — after the
        download budget is spent or the expiry passes, the link 404s
        and the bytes are gone from storage. ``/share list`` lists
        live shares as metadata with links, never contents;
        ``/share revoke <id>`` kills a share early; ``/share
        status`` reports the live count, the journal length and the
        sweep job. Owner-only. Anyone holding the link can download
        until the budget dies or the expiry passes, and the link is
        host-network-only unless the API is fronted with a tunnel —
        both are stated in the mint reply.
        """
        from ...storage.share import (
            SHARE_JOB,
            SHARE_SPEC,
            ShareStore,
            list_shares,
            mint_share,
            parse_share_expiry,
            revoke_share,
        )

        context = getattr(self, "context", None)
        settings = getattr(context, "settings", None) if context else None
        home = getattr(settings, "home", None) if settings else None
        home = str(home or "~/.nomorals")
        api = getattr(settings, "api", None) if settings else None
        host = str(getattr(api, "host", "") or "").strip()
        port = getattr(api, "port", None)
        base_url = ""
        if host and port:
            if host in ("0.0.0.0", "::"):
                host = "127.0.0.1"
            base_url = f"http://{host}:{int(port)}"

        def _flag(rest: str, flag: str) -> tuple[str, str | None]:
            match = re.search(rf"(?:^|\s){re.escape(flag)}\s+(\S+)\s*$", rest)
            if not match:
                return rest, None
            return rest[:match.start()].strip(), match.group(1)

        parts = (tail or "").split(None, 1)
        first = parts[0] if parts else ""
        verb = first.lower()
        rest = parts[1].strip() if len(parts) > 1 else ""
        try:
            store = ShareStore(home)
        except Exception as exc:  # noqa: BLE001 - share never crashes chat
            return f"shares unavailable: {exc}"
        if verb == "list":
            shares = list_shares(store, base_url=base_url)
            if not shares:
                return "no live shares."
            lines = ["live shares:"]
            for meta in shares:
                expires_at = meta["expires_at"]
                when = ("never" if expires_at is None else
                        time.strftime("%Y-%m-%d %H:%M",
                                      time.localtime(expires_at)))
                lines.append(f"• {meta['share_id']} · {meta['filename']} · "
                             f"{meta['downloads_left']} left · expires {when}\n"
                             f"  {meta['link']}")
            return "\n".join(lines)
        if verb == "revoke":
            if not rest:
                return "usage: /share revoke <share-id>"
            if revoke_share(store, rest.split()[0]):
                return f"revoked {rest.split()[0]} — the link is dead now."
            return f"no live share {rest.split()[0]}."
        if verb == "status":
            shares = list_shares(store)
            return (f"live shares: {len(shares)} · "
                    f"journal: {len(store.journal_entries())} · "
                    f"sweep job: {SHARE_JOB} ({SHARE_SPEC})")
        if not first:
            return ("usage: /share <file> [downloads N] "
                    "[--expire 1h|1d|7d|30d|never|friday|2026-12-25] "
                    "| /share list | /share revoke <share-id> "
                    "| /share status")
        # Anything else is a mint: the first token is the file.
        file_arg = first
        downloads = 1
        match = re.search(r"\bdownloads\s+(\d+)\b", rest)
        if match:
            downloads = int(match.group(1))
            rest = (rest[:match.start()] + rest[match.end():]).strip()
        rest, expire = _flag(rest, "--expire")
        if rest:
            return ("usage: /share <file> [downloads N] "
                    "[--expire 1h|1d|7d|30d|never|friday|2026-12-25]")
        try:
            parsed = parse_share_expiry(expire if expire else None)
        except ValueError as exc:
            return str(exc)
        try:
            if parsed is None:
                out = mint_share(store, file_arg, downloads=downloads,
                                 never=True)
            else:
                out = mint_share(store, file_arg, downloads=downloads,
                                 expires_in_s=max(0.0, parsed - time.time()))
        except Exception as exc:  # noqa: BLE001 - a failed mint never crashes chat
            return f"could not share: {exc}"
        if not out.get("ok"):
            return f"could not share: {out.get('error')}"
        link = str(out["link"])
        if base_url:
            link = base_url + link
        expires_at = out["expires_at"]
        when = ("never" if expires_at is None else
                time.strftime("%Y-%m-%d %H:%M",
                              time.localtime(expires_at)))
        lines = [f"shared — {out['filename']} ({out['size']} bytes)",
                 link,
                 f"{out['downloads_left']} download(s) left · expires {when}",
                 "anyone holding this link can download until the "
                 "budget dies or the expiry passes; the link is "
                 "host-network-only unless the API is fronted with "
                 "a tunnel."]
        return "\n".join(lines)

    def _impossible_diagram(self, claim_id: str,
                            message: Any = None) -> str:
        """``/impossible diagram <id>``: the dossier's prior-art map
        as a diagram image in this chat.

        The scene organ renders the map; the file goes back through
        the same ``file_send`` tool ``/analytics export`` uses. A
        missing dossier or a failed render reports in words — never
        silence, never a fake image.
        """
        from ...impossible.core import ImpossibleMadePossible
        from ...visuals.impossible import render_dossier_png

        claim_id = (claim_id or "").strip()
        if not claim_id:
            return "usage: /impossible diagram <id>"
        if message is None:
            return ("diagram needs a live chat — run "
                    "/impossible diagram from a chat, not the console")
        try:
            dossier = ImpossibleMadePossible().dossier(claim_id)
        except Exception as exc:  # noqa: BLE001 - never crashes chat
            return f"no dossier for '{claim_id}': {exc}"
        if not dossier:
            return f"No claim with id '{claim_id}'."
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home", None) if settings is not None else None
        home = str(home or "~/.nomorals")
        try:
            target = render_dossier_png(
                dossier, kind="prior_art",
                out_dir=str(Path(home) / "visuals"))
        except Exception as exc:  # noqa: BLE001 - honest refusal
            return f"diagram failed: {exc}"
        tools = getattr(self.context, "tools", None)
        if tools is None:
            return (f"prior-art map rendered to {target}, but this "
                    "session has no file-send tool.")
        chat = message.chat
        outcome = tools.call(
            "file_send",
            platform=str(getattr(chat, "platform", "") or ""),
            chat_id=str(getattr(chat, "chat_id", "") or ""),
            path=str(target),
            caption="prior-art map — what exists, what failed")
        if not outcome.ok:
            return (f"prior-art map rendered to {target}, but sending "
                    f"it failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        return "prior-art map sent."

    def _maybe_send_impossible_diagram(self, reply: str,
                                       message: Any = None) -> None:
        """After a ``/impossible solve``, send the prior-art map.

        Fail-open by design: the report is the answer and is already
        returned; anything missing here (no claim id, no chat, a
        failed render) simply means no diagram — never an exception
        into the reply path.
        """
        try:
            claim_id = ""
            for line in (reply or "").splitlines():
                if line.startswith("Claim id: "):
                    claim_id = line[len("Claim id: "):].strip()
                    break
            if not claim_id or message is None:
                return
            self._impossible_diagram(claim_id, message)
        except Exception:  # noqa: BLE001 - diagram is additive only
            return

    def _scene_send_file(self, path: str, message: Any,
                         caption: str) -> str | None:
        """Send a rendered scene file back in this chat.

        Returns ``None`` when the file went out, else the honest
        reason it did not (the path is still named, so the owner can
        open it from the box).
        """
        tools = getattr(self.context, "tools", None)
        if tools is None:
            return (f"rendered to {path}, but this session has no "
                    "file-send tool.")
        chat = message.chat
        outcome = tools.call(
            "file_send",
            platform=str(getattr(chat, "platform", "") or ""),
            chat_id=str(getattr(chat, "chat_id", "") or ""),
            path=str(path),
            caption=caption)
        if not outcome.ok:
            return (f"rendered to {path}, but sending it failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        return None

    def _control_scene(self, tail: str, message: Any = None) -> str:
        """The /scene reply: the interaction half of the visual
        surface, in chat.

        ``/scene point <x> <y> [label]`` flies the cursor to a
        normalized 0..1 coordinate, leaves the marker, and sends the
        scene PNG back. ``/scene tag <text with [POINT:...] tags>``
        parses the tags out of the text (coordinates are 0..1 in
        chat), places them, and sends the PNG. ``/scene page`` writes
        the interactive live page — the flight visible on it — and
        sends the HTML file. ``/scene clear`` empties the board.
        ``/scene status`` reports the board, the cursor, the idle
        trigger and the bridge. ``/scene idle on|off|status`` arms
        the breakpoint detector. Owner-only. A render or send
        failure reports in words — never silence, never a fake
        image.
        """
        from ..visuals.annotations import get_board
        from ..visuals.flight import flight_frames
        from ..visuals.idle import get_idle_trigger
        from ..visuals.interactive import page_for_board
        from ..visuals.render import render_scene as _render_scene
        from ..visuals.tags import parse_tags, tags_to_annotations

        words = (tail or "").split()
        verb = words[0].lower() if words else "status"
        if verb not in ("point", "tag", "clear", "status", "page",
                        "idle"):
            return ("usage: /scene point <x> <y> [label] | tag <text "
                    "with [POINT:...] tags> | clear | status | page | "
                    "idle [on|off|status]")
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home", None) if settings is not None else None
        home = str(home or Path("~/.nomorals").expanduser())
        board = get_board(home)
        out_dir = str(Path(home) / "visuals")

        def _send_png(spec: dict[str, Any], caption: str) -> str:
            if message is None:
                record = _render_scene(spec, target="chat",
                                       out_dir=out_dir)
                return (f"rendered to {record['path']} — run /scene "
                        "from a chat to have the image sent there.")
            try:
                record = _render_scene(spec, target="chat",
                                       out_dir=out_dir)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"scene failed: {exc}"
            failed = self._scene_send_file(record["path"], message,
                                           caption)
            if failed is not None:
                return failed
            return "scene sent."

        if verb == "point":
            if len(words) < 3:
                return "usage: /scene point <x> <y> [label]"
            try:
                x, y = float(words[1]), float(words[2])
            except ValueError:
                return "/scene point needs numeric x and y (0..1)"
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return "/scene point coordinates are 0..1"
            label = " ".join(words[3:])
            start = (board.cursor[0], board.cursor[1])
            frames = flight_frames(start, (x, y))
            board.point(x, y, label=label)
            board.set_last_flight(frames)
            spec = board.to_overlay_spec("scene point")
            if spec is None:
                return "point placed, but the board is empty."
            note = _send_png(spec, f"scene point — {label or 'here'}")
            return (f"pointed at ({x:.2f}, {y:.2f})"
                    + (f" — {label}" if label else "")
                    + f" — flight {frames[-1]['t']:.2f}s. {note}")
        if verb == "tag":
            text = (tail or "")[len(words[0]):].strip()
            if not text:
                return ("usage: /scene tag <text with [POINT:...] tags> "
                        "— coordinates are 0..1 in chat")
            parsed = parse_tags(text)
            if not parsed.tags and not parsed.none:
                return ("no tags parsed — kept the text as-is: "
                        f"{parsed.text}")
            for ann in tags_to_annotations(parsed.tags):
                board.create(
                    kind=ann["kind"], x=ann["x"], y=ann["y"],
                    w=ann.get("w", 0.0), h=ann.get("h", 0.0),
                    points=ann.get("points", ()),
                    shape=ann.get("shape", ""),
                    text=ann.get("text", ""),
                    screen=ann.get("screen", 1))
            points_tags = [t for t in parsed.tags if t.kind == "point"]
            if points_tags:
                first = points_tags[0]
                start = (board.cursor[0], board.cursor[1])
                frames = flight_frames(start, (first.x, first.y))
                board.move_cursor(first.x, first.y, first.screen)
                board.set_last_flight(frames)
            spec = board.to_overlay_spec("scene tags")
            if spec is None:
                return f"tags placed: {parsed.text}"
            note = _send_png(spec, "scene tags")
            return (f"{len(parsed.tags)} tag(s) placed"
                    + (" — [POINT:none]" if parsed.none else "")
                    + f". {note}")
        if verb == "clear":
            cleared = board.clear()
            board.set_last_flight([])
            return f"board cleared — {cleared} annotation(s) removed."
        if verb == "page":
            path = page_for_board(board, title="scene",
                                  out_dir=out_dir)
            if path is None:
                return ("nothing live on the board — place an "
                        "annotation first (/scene point or /scene tag).")
            if message is None:
                return f"live page written to {path}"
            failed = self._scene_send_file(
                path, message,
                "live scene page — open it in a browser to see the "
                "flight")
            if failed is not None:
                return failed
            return "live page sent — open it to see the flight."
        if verb == "idle":
            trigger = get_idle_trigger(home)
            sub = words[1].lower() if len(words) > 1 else "status"
            if sub == "on":
                trigger.arm()
                return "idle trigger on — I speak up at the next pause."
            if sub == "off":
                trigger.disarm()
                return "idle trigger off."
            if sub != "status":
                return "usage: /scene idle [on|off|status]"
            idle = trigger.idle_seconds()
            idle_text = ("never acted" if idle is None
                         else f"{idle:.0f}s since your last message")
            return ("idle trigger: "
                    f"{'on' if trigger.armed else 'off'}, {idle_text}")
        # verb == "status"
        from ..visuals.bridge import bridge_fingerprint

        counts: dict[str, int] = {}
        for ann in board.list():
            counts[ann.state] = counts.get(ann.state, 0) + 1
        trigger = get_idle_trigger(home)
        fingerprint = bridge_fingerprint(home)
        lines = [
            "scene board:",
            f"  annotations: {len(board.list())} "
            f"({counts.get('armed', 0)} armed, "
            f"{counts.get('completed', 0)} completed, "
            f"{counts.get('missed', 0)} missed)",
            f"  cursor: ({board.cursor[0]:.2f}, {board.cursor[1]:.2f}) "
            f"screen {board.cursor[2]}",
            f"  idle trigger: {'on' if trigger.armed else 'off'}",
            ("  bridge: token "
             + (f"set ({fingerprint})" if fingerprint else "not set")),
        ]
        return "\n".join(lines)

    def _control_board(self, tail: str, message: Any = None) -> str:
        """The /board reply: the whiteboard organ, in chat.

        ``/board new <title> [--theme sketch]`` makes a board.
        ``/board add <board> <type> <label> [x y w h]`` adds a shape
        (coordinates optional — shapes auto-place on a grid).
        ``/board move <board> <shape> <x> <y>`` moves one.
        ``/board remove <board> <shape>`` deletes one (its bindings
        drop with it). ``/board bind <board> <from> <to> [label]``
        connects two shapes with an arrow. ``/board undo|redo
        <board>`` steps the op history. ``/board show <board>`` and
        ``/board list`` read boards. ``/board render <board>
        [--theme sketch]`` sends the PNG; ``/board page <board>``
        sends the editable live page. Boards and shapes take an id or
        a unique id prefix. Owner-only. A failure reports in words —
        never silence, never a fake image.
        """
        import shlex

        from ...media.edit_engine.board import (
            SHAPE_TYPES,
            BoardStore,
            render_board,
            write_board_page,
        )

        usage = ("usage: /board new <title> [--theme sketch|clean] | "
                 "add <board> <type> <label> [x y w h] | move <board> "
                 "<shape> <x> <y> | remove <board> <shape> | bind "
                 "<board> <from> <to> [label] | undo|redo <board> | "
                 "show <board> | list | render <board> [--theme sketch] "
                 "| page <board>")
        try:
            words = shlex.split(tail or "")
        except ValueError:
            return usage
        verb = words[0].lower() if words else "list"
        if verb not in ("new", "add", "move", "remove", "bind",
                        "unbind", "undo", "redo", "show", "list",
                        "render", "page"):
            return usage
        settings = getattr(self.context, "settings", None)
        home = getattr(settings, "home", None) if settings is not None else None
        home = str(home or Path("~/.nomorals").expanduser())
        store = BoardStore(home=home)
        out_dir = str(Path(home) / "boards")

        def _find_board(ref: str):
            found = store.matches(ref)
            if not found:
                return None, f"no board {ref!r} — try /board list."
            if len(found) > 1:
                return None, (f"board {ref!r} matches {len(found)} "
                              "boards — use the full id.")
            doc = store.load(found[0])
            if doc is None:
                return None, f"board {ref!r} could not be read."
            return doc, None

        def _find_shape(doc, ref: str):
            if doc.shape(ref) is not None:
                return doc.shape(ref), None
            hits = [s for s in doc.shapes if s.id.startswith(ref)]
            if len(hits) == 1:
                return hits[0], None
            if not hits:
                return None, f"no shape {ref!r} on this board."
            return None, (f"shape {ref!r} matches {len(hits)} shapes "
                          "— use the full id.")

        def _send(path: str, caption: str) -> str:
            if message is None:
                return f"written to {path} — run /board from a chat " \
                    "to have the file sent there."
            failed = self._scene_send_file(path, message, caption)
            if failed is not None:
                return failed
            return "sent."

        if verb == "list":
            rows = store.list_boards()
            if not rows:
                return ("no boards yet — /board new <title> makes "
                        "one.")
            lines = ["boards:"]
            for row in rows:
                lines.append(
                    f"  {row['id'][:8]}  {row['title'] or '(untitled)'}"
                    f" — {row['shapes']} shape(s), "
                    f"{row['bindings']} binding(s), {row['theme']}")
            return "\n".join(lines)
        if verb == "new":
            theme = "clean"
            title_parts: list[str] = []
            i = 1
            while i < len(words):
                if words[i] == "--theme" and i + 1 < len(words):
                    theme = words[i + 1].lower()
                    i += 2
                    continue
                title_parts.append(words[i])
                i += 1
            if theme not in ("clean", "sketch"):
                return "/board new theme must be clean or sketch."
            from ...media.edit_engine.board import BoardDocument

            doc = BoardDocument.create(" ".join(title_parts),
                                       theme=theme)
            store.save(doc)
            return (f"board {doc.id[:8]} — "
                    f"{doc.title or '(untitled)'} ({theme}). "
                    f"Add a shape: /board add {doc.id[:8]} rect "
                    f"\"First box\"")
        # Every other verb names a board first.
        if len(words) < 2:
            return usage
        doc, error = _find_board(words[1])
        if doc is None:
            return error or usage
        if verb == "show":
            lines = [f"board {doc.id[:8]} — "
                     f"{doc.title or '(untitled)'} ({doc.theme}, "
                     f"v{doc.version}):"]
            for shape in doc.shapes:
                label = (shape.props.get("text", "")
                         or shape.props.get("label", ""))
                lines.append(
                    f"  {shape.id[:8]}  {shape.type} "
                    f"({shape.x:.0f}, {shape.y:.0f}) "
                    f"{shape.w:.0f}x{shape.h:.0f}"
                    + (f" — {label}" if label else ""))
            for binding in doc.bindings:
                lines.append(
                    f"  ↳ {binding.arrow_shape[:8]} binds "
                    f"{binding.from_shape[:8]} → "
                    f"{binding.to_shape[:8]}"
                    + (f" — {binding.label}" if binding.label else ""))
            if not doc.shapes:
                lines.append("  (empty — add a shape to draw on it)")
            return "\n".join(lines)
        if verb == "add":
            if len(words) < 4:
                return ("usage: /board add <board> <type> <label> "
                        "[x y w h]")
            shape_type = words[2].lower()
            if shape_type not in SHAPE_TYPES:
                return (f"shape type must be one of "
                        f"{list(SHAPE_TYPES)}, got {shape_type!r}")
            label = words[3]
            numbers: list[float] = []
            for raw in words[4:8]:
                try:
                    numbers.append(float(raw))
                except ValueError:
                    return f"/board add needs numeric x y w h, got {raw!r}"
            count = len(doc.shapes)
            x = numbers[0] if len(numbers) > 0 else 40.0 + (count % 4) * 260.0
            y = numbers[1] if len(numbers) > 1 else 40.0 + (count // 4) * 200.0
            shape: dict[str, Any] = {"type": shape_type, "x": x, "y": y}
            if len(numbers) > 2:
                shape["w"] = numbers[2]
            if len(numbers) > 3:
                shape["h"] = numbers[3]
            if shape_type in ("rect", "ellipse", "text", "sticky"):
                shape["props"] = {"text": label}
            elif shape_type == "arrow":
                shape["props"] = {"label": label}
            try:
                created = doc.add_shape(shape)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board add failed: {exc}"
            store.save(doc)
            return (f"added {created.type} {created.id[:8]} "
                    f"({created.x:.0f}, {created.y:.0f}) to board "
                    f"{doc.id[:8]}.")
        if verb == "move":
            if len(words) < 5:
                return "usage: /board move <board> <shape> <x> <y>"
            shape, error = _find_shape(doc, words[2])
            if shape is None:
                return error or usage
            try:
                x, y = float(words[3]), float(words[4])
            except ValueError:
                return "/board move needs numeric x and y"
            try:
                doc.update_shape(shape.id, x=x, y=y)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board move failed: {exc}"
            store.save(doc)
            return (f"moved {shape.id[:8]} to ({x:.0f}, {y:.0f}).")
        if verb == "remove":
            if len(words) < 3:
                return "usage: /board remove <board> <shape>"
            shape, error = _find_shape(doc, words[2])
            if shape is None:
                return error or usage
            try:
                result = doc.remove_shape(shape.id)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board remove failed: {exc}"
            store.save(doc)
            dropped = result.get("removed_bindings") or []
            return (f"removed {shape.id[:8]}"
                    + (f" — {len(dropped)} binding(s) dropped with it"
                       if dropped else "."))
        if verb == "bind":
            if len(words) < 4:
                return ("usage: /board bind <board> <from> <to> "
                        "[label]")
            src, error = _find_shape(doc, words[2])
            if src is None:
                return error or usage
            dst, error = _find_shape(doc, words[3])
            if dst is None:
                return error or usage
            label = " ".join(words[4:])
            try:
                binding = doc.bind(src.id, dst.id, label=label)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board bind failed: {exc}"
            store.save(doc)
            return (f"bound {src.id[:8]} → {dst.id[:8]} "
                    f"(arrow {binding.arrow_shape[:8]}).")
        if verb == "unbind":
            if len(words) < 3:
                return "usage: /board unbind <board> <binding>"
            binding = doc.binding(words[2])
            if binding is None:
                hits = [b for b in doc.bindings
                        if b.id.startswith(words[2])]
                if len(hits) == 1:
                    binding = hits[0]
                else:
                    return f"no binding {words[2]!r} on this board."
            try:
                doc.unbind(binding.id)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board unbind failed: {exc}"
            store.save(doc)
            return f"unbound {binding.id[:8]} (the arrow stays)."
        if verb == "undo":
            try:
                doc.undo()
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board undo failed: {exc}"
            store.save(doc)
            return f"undid — {len(doc.shapes)} shape(s) now."
        if verb == "redo":
            try:
                doc.redo()
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board redo failed: {exc}"
            store.save(doc)
            return f"redid — {len(doc.shapes)} shape(s) now."
        if verb == "render":
            theme = doc.theme
            for i, word in enumerate(words):
                if word == "--theme" and i + 1 < len(words):
                    theme = words[i + 1].lower()
            if theme not in ("clean", "sketch"):
                return "/board render theme must be clean or sketch."
            try:
                record = render_board(doc, theme=theme,
                                      out_dir=out_dir)
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board render failed: {exc}"
            note = _send(record["path"],
                         f"board — {doc.title or doc.id[:8]}")
            return (f"board rendered ({record['shapes']} shape(s), "
                    f"{record['theme']} theme). {note}")
        if verb == "page":
            try:
                path = write_board_page(
                    doc, Path(out_dir) / f"board-{doc.id[:8]}.html")
            except Exception as exc:  # noqa: BLE001 - honest refusal
                return f"board page failed: {exc}"
            note = _send(path, f"board — {doc.title or doc.id[:8]} "
                               "(open it to keep drawing)")
            return f"live board page written. {note}"
        return usage

    def _control_photo(self, tail: str, message: Any = None) -> str:
        """The /photo reply: the Studio photo editor, in chat.

        A thin dispatch to the same ``photo_editor`` tool the brain
        uses — no organ logic duplicated here, so the command's
        output is the tool's output.

        ``/photo open <path.psd>`` opens a PSD into the Studio
        store. ``/photo edit <id> crop <l> <t> <r> <b>`` crops the
        canvas; ``/photo edit <id> enhance [brightness <n>]``
        brightens the active layer; ``/photo edit <id> develop
        [brightness <n>] [contrast <n>]`` runs the develop stage.
        ``/photo undo|redo <id>`` steps the op history.
        ``/photo show <id>`` and ``/photo list`` read documents.
        ``/photo save <id> [path]`` writes the PSD back.
        ``/photo render <id> <png path>`` writes the composite PNG;
        ``/photo page <id> <html path>`` writes the Studio page.
        Owner-only. A failure reports in words — never silence,
        never a fake image.
        """
        import shlex

        usage = ("usage: /photo open <path.psd> | edit <id> crop "
                 "<l> <t> <r> <b> | edit <id> enhance [brightness "
                 "<n>] [contrast <n>] | edit <id> develop "
                 "[brightness <n>] [contrast <n>] | undo|redo <id> "
                 "| show <id> | list | save <id> [path] | render "
                 "<id> <png path> | page <id> <html path>")
        try:
            words = shlex.split(tail or "")
        except ValueError:
            return usage
        verb = words[0].lower() if words else "list"
        if verb not in ("open", "edit", "undo", "redo", "show",
                        "list", "save", "render", "page"):
            return usage

        def _call(action: str, **kwargs: Any) -> tuple[bool, dict]:
            outcome = self.context.tools.call(
                "photo_editor", action=action, **kwargs)
            if not outcome.ok:
                error = getattr(outcome, "error", None)
                text = getattr(error, "message", None) or str(error)
                return False, {"error": text}
            value = getattr(outcome, "value", None) or {}
            if not isinstance(value, dict):
                value = {"value": value}
            return True, value

        def _describe(value: dict) -> str:
            size = value.get("size") or []
            head = f"photo {value.get('id', '')[:8]}"
            if len(size) == 2:
                head += (f" — {value.get('title') or 'untitled'} "
                         f"{size[0]}x{size[1]}")
            lines = [head,
                     f"  {value.get('layers', 0)} layer(s)"]
            for layer in value.get("layer_list") or []:
                lines.append(
                    f"  {layer.get('name')} — {layer.get('kind')} "
                    f"{layer.get('width')}x{layer.get('height')}")
            for op in value.get("history") or []:
                params = ", ".join(
                    f"{k}={v}" for k, v in
                    (op.get("params") or {}).items())
                lines.append(f"  edit: {op.get('op')}({params})")
            for note in value.get("notes") or []:
                lines.append(f"  note: {note}")
            return "\n".join(lines)

        def _send(path: str, caption: str) -> str:
            if message is None:
                return (f"written to {path} — run /photo from a "
                        "chat to have the file sent there.")
            failed = self._scene_send_file(path, message, caption)
            if failed is not None:
                return failed
            return "sent."

        if verb == "list":
            ok, value = _call("list")
            if not ok:
                return f"photo list failed: {value['error']}"
            docs = value.get("documents") or []
            if not docs:
                return ("no photos open yet — /photo open "
                        "<path.psd> opens one.")
            lines = ["photos:"]
            for row in docs:
                lines.append(
                    f"  {row.get('id', '')[:8]} — "
                    f"{row.get('title') or 'untitled'} "
                    f"{row.get('width')}x{row.get('height')}, "
                    f"{len(row.get('layers') or [])} layer(s)")
            return "\n".join(lines)
        if verb == "open":
            if len(words) < 2:
                return usage
            path = str(Path(words[1]).expanduser())
            ok, value = _call("open", path=path)
            if not ok:
                return f"photo open failed: {value['error']}"
            return _describe(value)
        ref = words[1] if len(words) > 1 else ""
        if verb in ("show", "undo", "redo"):
            if not ref:
                return usage
            ok, value = _call(verb, doc_id=ref)
            if not ok:
                return f"photo {verb} failed: {value['error']}"
            if verb == "show":
                return _describe(value)
            done = value.get("undone") or value.get("redone")
            if done is None:
                return value.get("note") or "nothing to step."
            return (f"{verb} — {done.get('op')} stepped. "
                    f"{len(value.get('history') or [])} edit(s) now.")
        if verb == "edit":
            if len(words) < 4:
                return usage
            name = words[2].lower()
            if name not in ("crop", "enhance", "develop"):
                return usage
            rest = words[3:]
            params: dict[str, Any] = {}
            if name == "crop":
                if len(rest) != 4:
                    return usage
                try:
                    box = tuple(int(v) for v in rest)
                except ValueError:
                    return "/photo edit crop needs four integers."
                params["box"] = list(box)
            else:
                i = 0
                while i < len(rest):
                    key = rest[i].lower()
                    if key == "layer" and i + 1 < len(rest):
                        try:
                            params["layer"] = int(rest[i + 1])
                        except ValueError:
                            return ("/photo edit layer needs an "
                                    "integer.")
                        i += 2
                        continue
                    if i + 1 >= len(rest):
                        return usage
                    try:
                        params[key] = float(rest[i + 1])
                    except ValueError:
                        return (f"/photo edit {key} needs a "
                                f"number, got {rest[i + 1]!r}")
                    i += 2
            ok, value = _call("edit", doc_id=ref, op=name,
                              params=params)
            if not ok:
                return f"photo edit failed: {value['error']}"
            return _describe(value)
        if verb == "save":
            target = str(Path(words[2]).expanduser()) \
                if len(words) > 2 else ""
            ok, value = _call("save", doc_id=ref, path=target)
            if not ok:
                return f"photo save failed: {value['error']}"
            return (f"saved as PSD: {value.get('saved')}\n"
                    + _describe(value))
        if verb in ("render", "page"):
            if len(words) < 3:
                return usage
            target = str(Path(words[2]).expanduser())
            ok, value = _call(verb, doc_id=ref, out=target)
            if not ok:
                return f"photo {verb} failed: {value['error']}"
            key = "rendered" if verb == "render" else "page"
            note = _send(value.get(key, target),
                         f"photo — {value.get('title') or ref[:8]}")
            return (f"photo {verb} written. {note}")
        return usage

    def _control_extensions(self, tail: str, message: Any = None) -> str:
        """The /extensions reply: the Studio's extensions, in chat.

        A thin dispatch to the same ``studio_extensions`` tool the
        brain uses — no organ logic duplicated here, so the
        command's output is the tool's output.

        ``/extensions`` lists what is installed, what is
        switched on, and which theme the Studio wears.
        ``/extensions install <path-or-url>`` installs a package (a URL
        needs its pinned digest: ``sha256=… size=…``).
        ``/extensions enable|disable <id>`` switches one on or off.
        ``/extensions uninstall <id>`` removes one everywhere.
        ``/extensions theme <id>|none`` chooses the Studio's look.
        ``/extensions themes`` lists the installed themes.
        ``/extensions panels [photo]`` lists the panels enabled for a
        page. Owner-only. A failure reports in words — never
        silence, never a fake install.
        """
        import shlex

        usage = ("/extensions | install <path-or-url> "
                 "[sha256=… size=…] | enable|disable <id> | "
                 "uninstall <id> | theme <id>|none | themes | "
                 "panels [page]")
        try:
            words = shlex.split(tail or "")
        except ValueError:
            return "usage: " + usage
        verb = words[0].lower() if words else "extensions"
        if verb not in ("extensions", "install", "enable", "disable",
                        "uninstall", "theme", "themes", "panels"):
            return "usage: " + usage

        def _call(action: str, **kwargs: Any) -> tuple[bool, dict]:
            outcome = self.context.tools.call(
                "studio_extensions", action=action, **kwargs)
            if not outcome.ok:
                error = getattr(outcome, "error", None)
                text = getattr(error, "message", None) or str(error)
                return False, {"error": text}
            value = getattr(outcome, "value", None) or {}
            if not isinstance(value, dict):
                value = {"value": value}
            return True, value

        if verb == "extensions":
            ok, value = _call("list")
            if not ok:
                return f"studio extensions failed: {value['error']}"
            installed = value.get("installed") or []
            lines = ["studio extensions:"]
            if not installed:
                lines.append("  none installed yet — /extensions install "
                             "<path-or-url> installs one.")
            for row in installed:
                lines.append(
                    f"  {row.get('id')} {row.get('version')} — "
                    f"{row.get('kind')} — {row.get('name')} "
                    f"({row.get('tier')})"
                    + (" — on" if row.get("enabled_for_studio")
                       else " — off"))
            active = value.get("active_theme")
            lines.append("theme: " + (active or "the Studio's own look"))
            return "\n".join(lines)
        if verb == "install":
            if len(words) < 2:
                return "usage: " + usage
            target = words[1]
            kwargs: dict[str, Any] = {}
            for extra in words[2:]:
                if "=" in extra:
                    key, val = extra.split("=", 1)
                    kwargs[key.strip().lower()] = val.strip()
            if target.startswith(("http://", "https://")):
                sha = kwargs.get("sha256", "")
                size = kwargs.get("size", "")
                if not sha or not size:
                    return ("installing from a URL needs its pinned "
                            "digest: /extensions install <url> "
                            "sha256=<hex> size=<bytes>")
                try:
                    size_int = int(size)
                except ValueError:
                    return "/extensions install size= needs an integer."
                ok, value = _call("install", url=target, sha256=sha,
                                  size=size_int)
            else:
                ok, value = _call(
                    "install", path=str(Path(target).expanduser()))
            if not ok:
                return f"studio install failed: {value['error']}"
            return (f"installed {value.get('id')} "
                    f"{value.get('version')} — {value.get('kind')} — "
                    f"{value.get('name')} ({value.get('tier')})")
        if verb in ("enable", "disable"):
            if len(words) < 2:
                return "usage: " + usage
            ok, value = _call(verb, ext_id=words[1])
            if not ok:
                return f"studio {verb} failed: {value['error']}"
            state = "on" if verb == "enable" else "off"
            return f"{value.get('id')} is {state} for the Studio."
        if verb == "uninstall":
            if len(words) < 2:
                return "usage: " + usage
            ok, value = _call("uninstall", ext_id=words[1])
            if not ok:
                return f"studio uninstall failed: {value['error']}"
            return f"{value.get('id')} is gone from the Studio."
        if verb == "theme":
            if len(words) < 2:
                return "usage: " + usage
            ok, value = _call("theme", ext_id=words[1])
            if not ok:
                return f"studio theme failed: {value['error']}"
            active = value.get("active_theme")
            if active is None:
                return "the Studio is back to its own look."
            return f"the Studio now wears {active}."
        if verb == "themes":
            ok, value = _call("themes")
            if not ok:
                return f"studio themes failed: {value['error']}"
            themes = value.get("themes") or []
            if not themes:
                return ("no themes installed yet — /extensions install "
                        "<theme zip> installs one.")
            lines = ["themes:"]
            for row in themes:
                lines.append(f"  {row.get('id')} {row.get('version')} — "
                             f"{row.get('name')}")
            return "\n".join(lines)
        if verb == "panels":
            page = words[1] if len(words) > 1 else "photo"
            ok, value = _call("panels", page=page)
            if not ok:
                return f"studio panels failed: {value['error']}"
            panels = value.get("panels") or []
            if not panels:
                return f"no panels enabled for the {page} page."
            lines = [f"panels on the {page} page:"]
            for row in panels:
                lines.append(f"  {row.get('id')} — {row.get('title')} "
                             f"({row.get('slot')})")
            return "\n".join(lines)
        return "usage: " + usage

    def _control_arena_research(self, tail: str,
                                 message: Any = None) -> str:
        """The /arena-research reply: Devon researching herself, on
        demand. A thin dispatch to the same ``arena_researcher`` tool
        the brain uses (owner phase 2026-10-21 item 2) — no survey,
        ranking, or queue construction duplicated here, so the
        command's output is the tool's output.

        ``/arena-research`` (no args) shows the status: job, gates,
        last run, pending proposals. ``tick`` runs one cycle now and
        reports what it found and filed. ``survey`` lists the current
        ranked findings without filing anything. ``proposals`` lists
        what this organ filed. ``ensure`` / ``enable`` / ``disable``
        manage the durable job. A tool failure is an honest refusal
        in words, never silence and never a fake status. Nothing is
        ever auto-applied from here — proposals wait for the owner's
        word in the upgrade queue."""
        words = (tail or "").split()
        verb = words[0].lower() if words else "status"
        if verb not in ("status", "tick", "survey", "proposals", "ensure",
                        "enable", "disable"):
            return ("usage: /arena-research [status | tick | survey | "
                    "proposals | ensure | enable | disable]")
        outcome = self.context.tools.call("arena_researcher",
                                          action=verb)
        if not outcome.ok:
            return ("arena researcher failed: "
                    f"{getattr(outcome.error, 'message', outcome.error)}")
        value = outcome.value or {}
        if verb == "status":
            job = value.get("job") or {}
            gates = value.get("gates") or {}
            last = value.get("last_run") or {}
            lines = ["arena researcher:",
                     f"  job: {'scheduled' if job.get('scheduled') else 'not scheduled'}"
                     + (f", every {job.get('interval_hours')}h"
                        if job.get("scheduled") else "")
                     + ("" if job.get("enabled", True) else " (disabled)"),
                     f"  gates: feature={'on' if gates.get('feature_research') else 'off'}, "
                     f"proactive={'on' if gates.get('proactive_master') else 'off'}, "
                     f"quiet_hours={'yes' if gates.get('quiet_hours') else 'no'}",
                     "  last run: "
                     + ("never" if not last else
                        ("ok" if last.get("ok")
                         else "deferred: " + str(
                             last.get("skipped_reason")
                             or last.get("error") or ""))),
                     f"  pending proposals: {value.get('pending_proposals', 0)}"]
            if last:
                lines.append(
                    f"  last cycle: {last.get('findings_count', 0)} "
                    f"finding(s), {last.get('proposals_created', 0)} "
                    "filed")
            return "\n".join(lines)
        if verb == "tick":
            return (f"tick {'ok' if value.get('ok') else 'deferred'}"
                    + (f" — {value.get('skipped_reason')}"
                       if value.get("skipped_reason") else "")
                    + f" — findings {value.get('findings_count', 0)}, "
                      f"proposals {value.get('proposals_created', 0)}"
                      f" ({value.get('proposals_deduped', 0)} deduped)")
        if verb == "survey":
            findings = value.get("findings") or []
            if not findings:
                return "arena survey: no findings — the tree is clean"
            lines = [f"arena survey: {len(findings)} finding(s)"]
            for f in findings[:10]:
                lines.append(f"  · [{f.get('kind')}] {f.get('path')}:"
                             f"{f.get('line')} — {f.get('evidence')}")
            if len(findings) > 10:
                lines.append(f"  … and {len(findings) - 10} more")
            return "\n".join(lines)
        if verb == "proposals":
            rows = value.get("proposals") or []
            if not rows:
                return "arena researcher: no proposals filed yet"
            lines = [f"arena proposals: {len(rows)}"]
            for r in rows[:10]:
                lines.append(f"  · [{r.get('status')}] {r.get('title')}")
            if len(rows) > 10:
                lines.append(f"  … and {len(rows) - 10} more")
            return "\n".join(lines)
        if verb == "ensure":
            return (f"arena researcher job: "
                    f"{'already scheduled' if value.get('already_scheduled') else 'scheduled'} "
                    f"(every {value.get('interval_hours')}h)")
        return (f"arena researcher job "
                f"{'enabled' if verb == 'enable' else 'disabled'}")

    def _control_ping(self, message: Any = None) -> str:
        """The life of the /ping message itself, stage by stage.

        Owner spec (2026-11-01): arrival, receipt, processing,
        inbound and sent stages plus system condition. Every number
        comes from a real stamp — the platform timestamp on the
        message (arrival), ``meta['received_at']`` stamped in
        :meth:`PartnerRuntime.on_message` (receipt), and
        ``meta['process_at']`` stamped in ``_process_inner``
        (processing start; the gap between receipt and processing is
        the inbound queue wait). A stage with no stamp says so —
        nothing is estimated or invented. The sent stage is this
        reply: it is composed now and handed to the platform the
        moment this handler returns.
        """
        now = time.time()
        meta = getattr(message, "meta", None) or {}
        arrived = float(getattr(message, "ts", 0.0) or 0.0)
        received = float(meta.get("received_at") or 0.0)
        processing = float(meta.get("process_at") or 0.0)

        def clock(ts: float) -> str:
            return time.strftime("%H:%M:%S", time.localtime(ts))

        def delta(a: float, b: float) -> str:
            secs = b - a
            if secs < 1:
                return f"{secs * 1000:.0f}ms"
            if secs < 90:
                return f"{secs:.1f}s"
            return f"{secs / 60:.1f}m"

        lines = ["🏓 pong — this message's journey:"]
        lines.append(
            f"· arrival: {clock(arrived)} (platform stamp)"
            if arrived else "· arrival: not stamped by the platform")
        if received:
            gap = (f" (+{delta(arrived, received)} in transit)"
                   if arrived and received >= arrived else "")
            lines.append(f"· receipt: {clock(received)}{gap}")
        else:
            lines.append("· receipt: not stamped")
        if received and processing:
            lines.append(
                f"· inbound: {delta(received, processing)} in the queue")
        if processing:
            lines.append(
                f"· processing: from {clock(processing)}, "
                f"reply composed in {delta(processing, now)}")
        else:
            lines.append("· processing: not stamped")
        total = f", {delta(arrived, now)} end to end" if arrived else ""
        lines.append(f"· sent: this reply{total}")
        # system condition — the same sources /status reads, compressed.
        try:
            running = [
                name for name, info in self.gateway.status().items()
                if not name.startswith("_") and isinstance(info, dict)
                and info.get("running_in_session")
            ]
            platforms = ", ".join(running) if running else "none running"
        except Exception as exc:  # noqa: BLE001 - ping must never crash
            record_failure("runtime_meta._control_ping:platforms", exc)
            platforms = "unknown"
        try:
            self.context.db.query_one("SELECT 1 AS ok")
            db_line = "db ok"
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_meta._control_ping:db", exc)
            db_line = f"db failing ({type(exc).__name__})"
        try:
            stats = self._stats_snapshot()
            traffic = (f"{stats['messages']} msgs in, {stats['replies']} "
                       f"replies, {stats['errors']} errors")
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_meta._control_ping:stats", exc)
            traffic = "traffic unknown"
        lines.append(f"condition: platforms {platforms} · {db_line} · {traffic}")
        return "\n".join(lines)

    def _control_mood(self, arg: str) -> str:
        engine = self.brain.mood
        try:
            if not arg:
                state = engine.current()
                dims = ", ".join(f"{k} {int(v)}" for k, v in sorted(state.values.items()))
                return f"mood: {state.label} — {dims}"
            if arg.strip().lower() == "reset":
                engine.reset()
                return "mood reset to baselines"
            if "=" in arg:
                pairs = dict(
                    (p.split("=", 1)[0].strip().lower(), float(p.split("=", 1)[1]))
                    for p in arg.split() if "=" in p
                )
                engine.set_dimensions(pairs)
                state = engine.current()
                return f"mood set: {state.label} ({', '.join(f'{k} {int(v)}' for k, v in pairs.items())})"
            engine.set_label(arg)
            return f"mood forced: {engine.current().label}"
        except ValueError as exc:
            return str(exc)

    def _control_mode(self, arg: str) -> str:
        mode = arg.strip().lower()
        if mode not in {"off", "suggest", "auto"}:
            return "usage: /mode off|suggest|auto"
        self.settings.partner.autonomy_mode = mode
        try:
            KVStore(self.context.db).set("partner.autonomy_mode", {"mode": mode})
        except Exception as exc:  # noqa: BLE001 - a write failure must not block the switch
            record_failure("runtime_meta._control_mode", exc)
        try:
            from ..autonomy import AutonomyAgent

            if self._autonomy is None and mode != "off":
                dm_cap, group_cap = self._tuned_autonomy_caps()
                self._autonomy = AutonomyAgent(
                    self.context, self.brain, self.gateway,
                    mode=mode,
                    owner_chats=self._owner_chats,
                    group_chats=_key_set(self.settings.partner.group_chats),
                    quiet_start=self.settings.partner.quiet_start,
                    quiet_end=self.settings.partner.quiet_end,
                    max_dm_per_day=dm_cap,
                    max_group_per_day=group_cap,
                )
                self._autonomy.start()
            elif self._autonomy is not None:
                self._autonomy.set_mode(mode)
        except Exception as exc:  # noqa: BLE001
            _log.warning("autonomy mode switch failed: %s", exc)
            return f"mode set to {mode} (agent switch failed: {exc})"
        return f"autonomy: {mode}"

    def _control_agent(self, arg: str, chat_key: str = "") -> str:
        """``/agent ...`` — the selectable agent modes for this chat.

        Adaptive learns from outcomes, autonomous self-initiates,
        planning plans first and replans on failure.  Modes combine
        (``/agent adaptive+planning``) and persist per chat, so the
        CLI (``nm agent ...``) and chat see the same set.

        Uses the runtime's ONE persistent ``agent_modes`` — a fresh
        engine per command would have no sources and no pending
        initiatives, so ``/agent tick`` would silently do nothing.
        """
        try:
            return self.agent_modes.apply_command(arg or "", chat_key or "")
        except Exception as exc:  # noqa: BLE001 - a mode switch never kills chat
            _log.warning("agent mode switch failed: %s", exc)
            return f"agent command failed: {exc}"

    def _control_chatmode(self, command: Any, chat_key: str,
                          message=None) -> str:
        """Per-chat reply mode + per-chat feature toggles — the single
        entry point for group controls. Owner-only (enforced by the
        grant check in handle_control before dispatch).

        ::

            /chatmode                          status for this chat
            /chatmode status [target]
            /chatmode list                     every chat with custom settings
            /chatmode games-only [target]     silent for normal chat, games work
            /chatmode games-off [target]      normal chat works, games refused
            /chatmode games-on [target]       games back on (mode → normal)
            /chatmode quiet [target]          never messages first here
            /chatmode normal [target]         mode → normal (toggles kept)
            /chatmode feature <games|music|voice|images|proactive> <on|off> [target]
            /chatmode feature games.<game> <on|off> [target]

        No target sets the chat the command was sent in. A target
        names a remote chat the way a human would (register #30 —
        see nomorals/partner/chat_targets.py): a full key
        (``platform:chat_id``), a phone number (that person's
        WhatsApp DM), or the chat's NAME, resolved against the
        title index and adapter chat listings. A name matching
        several chats lists the candidates instead of guessing;
        a name matching none reports what was tried. ``feature``
        changes one toggle in one chat only — disabling games in
        Group A never affects Group B or DMs.
        Modes gate reply-vs-silence; toggles gate features; no toggle
        re-enables what a mode silences. Turning games off while a game
        is live ends it honestly — scores kept, no further moves.
        Voice off is text-replies-only: voice notes still transcribe.
        """
        from ...partner.chat_modes import (
            FEATURES, MODE_GAMES_OFF, MODE_GAMES_ONLY, MODE_NORMAL,
            MODE_QUIET, get_chat_settings, list_chat_settings,
            set_chat_mode, set_feature, set_game)

        usage = ("usage: /chatmode [games-only|games-off|games-on|quiet|"
                 "normal|status|list] [target]  ·  /chatmode feature "
                 "<games|music|voice|images|proactive|games.<game>> <on|off> "
                 "[target]  ·  a target is a chat name, a phone number, "
                 "or a full key (platform:chat_id)")
        words = (command.tail or "").split()
        sub = (words[0].lower() if words else "status")

        def _resolve(raw: str) -> tuple[str, str, str | None]:
            """Resolve a target reference → (key, title, error).

            Exactly one of key / error is set. Resolution follows
            chat_targets: full key, phone number, chat name — with
            ambiguity and exhaustion reported, never guessed.
            """
            from ...partner.chat_targets import resolve_chat_target

            chat = getattr(message, "chat", None)
            adapters: list = []
            gateway = getattr(self, "gateway", None)
            if gateway is not None:
                try:
                    adapters = list(gateway.adapters.values())
                except Exception:  # noqa: BLE001 - listing is a hint
                    adapters = []
            res = resolve_chat_target(
                raw,
                current_key=chat_key,
                current_title=str(getattr(chat, "title", "") or ""),
                current_platform=str(getattr(chat, "platform", "") or ""),
                db=getattr(self.context, "db", None),
                adapters=adapters)
            if res.resolved:
                return res.key, res.title, None
            return "", "", res.describe(raw)

        def _where(target: str, title: str = "") -> str:
            if target == chat_key:
                return "this chat"
            return f"{title} ({target})" if title else target

        def _end_live_game(target: str, *, game: str = "") -> str:
            """D2: games going off ends the live game honestly — scores
            kept (the engine records results on finish), no further
            moves accepted. Returns the extra reply line ("" when no
            game was live). Never raises."""
            try:
                engine = self._game_engine()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - no engine, no live game
                record_failure("runtime_meta._end_live_game:280", exc)
                return ""
            try:
                live = engine.live(target)
            except Exception as exc:  # noqa: BLE001 - a broken lookup is not a game
                record_failure("runtime_meta._end_live_game:284", exc)
                return ""
            if live is None:
                return ""
            live_game = str(getattr(live, "game", "") or "")
            if game and live_game.lower() != game.lower():
                return ""
            try:
                engine.quit(target)
            except Exception as exc:  # noqa: BLE001 - toggle still applied
                return (f" (the live {live_game or 'game'} couldn't be "
                        f"closed cleanly: {exc} — /game quit to close it)")
            return (f" Games are now off in this chat — ending the live "
                    f"{live_game or 'game'}, scores kept.")

        def _status_line(target: str, title: str = "") -> str:
            settings = get_chat_settings(target)
            mode = settings["mode"]
            disabled = settings["disabled"]
            games_off = settings.get("games_disabled", [])
            where = _where(target, title)
            toggles = ", ".join(
                f"{f} {'off' if f in disabled else 'on'}" for f in FEATURES)
            extra = ""
            if games_off:
                extra = (" games off: " + ", ".join(games_off) + ".")
            if mode == MODE_GAMES_ONLY:
                head = (f"{where} is games-only — normal chat stays silent "
                        "here; game commands still work.")
                tail = (" In games-only, ALL non-game messages are "
                        "silenced — including voice notes (even after "
                        "transcription). Game commands + in-game moves "
                        "still work. /chatmode normal restores replies.")
            elif mode == MODE_GAMES_OFF:
                head = (f"{where} is games-off — I answer normal chat "
                        "here, but game commands are refused.")
                tail = " /chatmode games-on turns games back on."
            elif mode == MODE_QUIET:
                head = (f"{where} is quiet — I never message first here, "
                        "but I reply when addressed.")
                tail = (" /chatmode normal restores proactive messages; "
                        "/chatmode feature proactive on does the same.")
            elif disabled:
                head = (f"{where} is normal with features off: "
                        f"{', '.join(disabled)}.")
                tail = " /chatmode feature <name> on turns one back on."
            else:
                head = f"{where} is normal — I answer everything here."
                tail = ""
            if "voice" in disabled:
                tail += (" Voice off = text replies only — voice notes "
                         "still transcribe.")
            return f"{head} features: {toggles}.{extra}{tail}"

        if sub == "list":
            if len(words) > 1:
                return f"/chatmode list takes no target — {usage}"
            all_settings = list_chat_settings()
            if not all_settings:
                return ("no per-chat settings — every chat is normal with "
                        "all features on.")
            lines = ["per-chat settings:"]
            for key in sorted(all_settings):
                settings = all_settings[key]
                off = settings["disabled"]
                games_off = settings.get("games_disabled", [])
                suffix = ""
                if off:
                    suffix += f" (off: {', '.join(off)})"
                if games_off:
                    suffix += f" (games off: {', '.join(games_off)})"
                lines.append(
                    f"  {key}: {settings['mode'].replace('_', '-')}{suffix}")
            return "\n".join(lines)

        if sub == "feature":
            if len(words) < 3:
                return f"/chatmode feature needs a feature and on/off — {usage}"
            feature = words[1].lower()
            state = words[2].lower()
            if state not in ("on", "off"):
                return f"want on or off, got {state!r} — {usage}"
            # The target is the rest of the line — a chat NAME may
            # be several words ("think creativity think peace").
            target_raw = " ".join(words[3:])
            target, target_title, err = _resolve(target_raw)
            if err:
                return err
            where = _where(target, target_title)
            verb = "back on" if state == "on" else "off"
            if feature.startswith("games."):
                # Per-game toggle (D5): /chatmode feature games.chess off
                game = feature.split(".", 1)[1].strip()
                if not game:
                    return f"want a game name after games. — {usage}"
                set_game(target, game, state == "on")
                extra = ""
                if state == "off":
                    extra = _end_live_game(target, game=game)
                    if not extra:
                        extra = (f" {game} doesn't run here now, but your "
                                 f"other games still do.")
                return f"games.{game} is {verb} in {where}.{extra}"
            if feature not in FEATURES:
                return (f"unknown feature {feature!r} — want one of: "
                        f"{', '.join(FEATURES)} or games.<game> — {usage}")
            set_feature(target, feature, state == "on")
            extra = ""
            if feature == "games" and state == "off":
                extra = (" Game commands here now get an honest refusal "
                         "instead of running.")
                extra += _end_live_game(target)
            elif feature == "games" and state == "on":
                extra = (" Games work here again (a games-off mode is "
                         "cleared too).")
            elif feature == "voice" and state == "off":
                extra = (" Replies here stay text; voice notes still "
                         "transcribe.")
            elif feature == "proactive" and state == "off":
                extra = (" I won't message first here anymore (same as "
                         "/chatmode quiet).")
            elif feature == "proactive" and state == "on":
                extra = (" Proactive messages resume here (a quiet mode "
                         "is cleared too).")
            return f"{feature} is {verb} in {where}.{extra}"

        if sub == "status":
            target_raw = " ".join(words[1:])
            target, target_title, err = _resolve(target_raw)
            if err:
                return err
            return _status_line(target, target_title)

        if sub in ("games-only", "games_only", "gamesonly", "games"):
            mode: str | None = MODE_GAMES_ONLY
            mode_word = "games-only"
        elif sub in ("games-off", "games_off", "nogames", "no-games"):
            mode = MODE_GAMES_OFF
            mode_word = "games-off"
        elif sub in ("games-on", "games_on"):
            # games-on = games back on: clear the games toggle AND any
            # games-off mode. set_feature("games", True) does both.
            target_raw = " ".join(words[1:])
            target, target_title, err = _resolve(target_raw)
            if err:
                return err
            set_feature(target, "games", True)
            where = _where(target, target_title)
            return (f"games are back on in {where} — game commands work "
                    f"here again.")
        elif sub in ("quiet", "silent"):
            mode = MODE_QUIET
            mode_word = "quiet"
        elif sub in ("normal", "off", "clear", "default"):
            mode = MODE_NORMAL
            mode_word = "normal"
        else:
            return f"unknown mode {sub!r} — {usage}"
        target_raw = " ".join(words[1:])
        target, target_title, err = _resolve(target_raw)
        if err:
            return err
        where = _where(target, target_title)
        set_chat_mode(target, mode)
        if mode == MODE_GAMES_ONLY:
            return (f"{where} is now games-only — I'll stay silent for "
                    "normal chat here, but game commands still work. "
                    "/chatmode normal to restore. In games-only, ALL "
                    "non-game messages are silenced — including voice "
                    "notes (even after transcription). Game commands + "
                    "in-game moves still work.")
        if mode == MODE_GAMES_OFF:
            return (f"{where} is now games-off — I'll answer normal chat "
                    "here, but game commands are refused honestly "
                    "(never silently dropped, never run). "
                    "/chatmode games-on turns games back on."
                    + _end_live_game(target))
        if mode == MODE_QUIET:
            return (f"{where} is now quiet — I'll never message first "
                    "here, but I reply when addressed. /chatmode normal "
                    "or /chatmode feature proactive on restores proactive "
                    "messages.")
        return (f"{where} is back to normal — I'll answer everything here "
                f"again. (Feature toggles are kept: "
                f"/chatmode status shows them.)")

    def _control_doctor(self) -> str:
        """System health — the Telegram twin of ``nm doctor`` (spec §8.6).

        Owner-only (enforced by the grant check in handle_control
        before dispatch). Six checks, each pass/fail with the REAL
        cause — never a bare "unhealthy":

        - registry: every registered command has details + a list
          slot + a dispatch branch (the orphan audit, live).
        - config: every template env key mapped (ignored must be 0).
        - stores: chat-modes + style-overrides + database readable.
        - disk: free space on the install volume vs. what one
          update needs, plus the snapshot store's footprint.
        - adapters: every adapter's running state + health error.
        - providers: the LLM chain's keys present (no values shown).
        """
        import os
        import re

        from ...social.chat.control import (
            COMMAND_DETAILS, CONTROL_COMMANDS, LIST_GROUPS, LIST_ONELINERS)

        results: list[tuple[bool, str]] = []

        # ── registry ──
        try:
            runtime_src = open(
                os.path.join(os.path.dirname(__file__), "runtime.py"),
                encoding="utf-8").read()
            start = runtime_src.index("    def _handle_control_inner(")
            rest = runtime_src[start:]
            nxt = re.search(r"\n    def ", rest[1:])
            body = rest[: nxt.start() + 1] if nxt else rest
            handled: set[str] = set(re.findall(r'if kind == "([^"]+)"', body))
            for m in re.finditer(r"if kind in \{([^}]+)\}", body):
                handled.update(re.findall(r'"([^"]+)"', m.group(1)))
            for m in re.finditer(r"if kind in \(([^)]+)\)", body):
                handled.update(re.findall(r'"([^"]+)"', m.group(1)))
            # Dispatch also routes through NAMED membership sets
            # (the games via GAME_START_COMMANDS / GAME_COMMANDS,
            # the platform kind sets): a kind covered by a constant
            # the dispatch tests is handled, even though no literal
            # branch names it. Reading only literals reported every
            # game undispatchable — a false FAIL (nav audit, Devon
            # half). Unresolvable names contribute nothing, so a
            # genuinely uncovered kind still flags.
            from ...social.chat import control as _control_mod
            for m in re.finditer(r"kind in ([A-Z][A-Z0-9_]*)", body):
                members = getattr(_control_mod, m.group(1), None)
                if members:
                    handled.update(str(x) for x in members)
            missing = sorted(
                k for k in CONTROL_COMMANDS
                if k not in handled
                or k not in COMMAND_DETAILS
                or k not in LIST_ONELINERS
                or sum(k in kinds for _, kinds in LIST_GROUPS) != 1)
            if missing:
                results.append((False, "registry: undispatchable or "
                                f"undocumented commands: {', '.join(missing)}"))
            else:
                results.append((True, "registry: "
                                f"{len(CONTROL_COMMANDS)} commands, all with "
                                "details + list slot + dispatch"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"registry: check failed: {exc_str(exc)}"))

        # ── config ──
        try:
            from ...core.config import config_env_status
            status = config_env_status()
            ignored = status["ignored"]
            if ignored:
                results.append((False, "config: "
                                f"{len(ignored)} ignored key(s): "
                                + ", ".join(sorted(ignored))))
            else:
                results.append((True, "config: "
                                f"{len(status['set_mapped'])} set / "
                                f"{len(status['unset_mapped'])} unset, "
                                "0 ignored"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"config: check failed: {exc_str(exc)}"))

        # ── stores ──
        try:
            from ..startup_progress import verify_stores
            problems = verify_stores(getattr(self, "context", None))
            if problems:
                results.append((False, "stores: " + "; ".join(problems)))
            else:
                results.append((True, "stores: chat-modes + "
                                "style-overrides + database readable"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"stores: check failed: {exc_str(exc)}"))

        # ── disk ──
        try:
            from ...core import diskspace as _diskspace
            _settings = getattr(getattr(self, "context", None),
                                "settings", None)
            if _settings is None:
                results.append((False, "disk: no settings on this "
                                "runtime — cannot measure"))
            else:
                rep = _diskspace.disk_report(
                    _settings.home_path, _settings.db_path,
                    getattr(_settings, "blob_dir", None))
                free = _diskspace.human_bytes(rep["free_bytes"])
                need = _diskspace.human_bytes(rep["need_bytes"])
                store = (f"snapshots {rep['store_count']} using "
                         f"{_diskspace.human_bytes(rep['store_bytes'])}")
                # The whole-tree accounting (media-cache
                # discipline): when the biggest area outweighs
                # the snapshot, it is named — the pile is a
                # measured fact on the line, not a discovery.
                area = _diskspace.largest_area_note(rep)
                if area:
                    store = f"{store}; {area}"
                if rep["low"]:
                    results.append((False,
                                    f"disk: LOW — {free} free on the "
                                    f"install volume, an update needs "
                                    f"~{need}; {store}. Free space "
                                    "before updating."))
                else:
                    results.append((True,
                                    f"disk: {free} free (an update "
                                    f"needs ~{need}); {store}"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"disk: check failed: {exc_str(exc)}"))

        # ── adapters ──
        try:
            gateway = getattr(self, "gateway", None)
            if gateway is None:
                results.append((False, "adapters: no gateway attached"))
            else:
                status = gateway.status() or {}
                parts = []
                any_running = False
                for name in sorted(status):
                    if name.startswith("_"):
                        continue
                    info = status[name] or {}
                    running = bool(info.get("running_in_session"))
                    any_running = any_running or running
                    err = str(info.get("error") or "")
                    parts.append(f"{name} {'up' if running else 'down'}"
                                 + (f" ({err[:80]})" if err and not running else ""))
                if not parts:
                    results.append((False, "adapters: no adapters registered"))
                elif not any_running:
                    results.append((False, "adapters: none running — "
                                    + ", ".join(parts)))
                else:
                    results.append((True, "adapters: " + ", ".join(parts)))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"adapters: check failed: {exc_str(exc)}"))

        # ── providers ──
        try:
            # Audit the EFFECTIVE settings the runtime was built from —
            # the same source build_router consumed. Reading os.environ
            # here audited a different configuration: on a file-based
            # install (.env read by load_settings, never exported) the
            # chain looked empty and this check passed vacuously while
            # a real provider ran, or failed while it worked.
            llm = getattr(getattr(getattr(self, "context", None),
                                  "settings", None), "llm", None)
            if llm is None:
                results.append((False, "providers: no settings on this "
                                "runtime — cannot audit the chain"))
            else:
                chain: list[str] = []
                for raw in [getattr(llm, "provider", "")] + list(
                        getattr(llm, "fallback_chain", None) or []):
                    name = str(raw or "").strip().lower()
                    if name and name not in chain:
                        chain.append(name)

                def _key_present(provider: str) -> bool | None:
                    """True/False for key-backed providers; None = keyless."""
                    if provider == "groq":
                        return bool(getattr(llm, "groq_api_key", ""))
                    if provider == "openrouter":
                        return bool(getattr(llm, "openrouter_api_key", ""))
                    if provider in ("hf", "huggingface", "hf_serverless",
                                    "hf_endpoint"):
                        return bool(getattr(llm, "hf_token", ""))
                    if provider in ("openai", "openai_compat"):
                        return bool(getattr(llm, "openai_api_key", ""))
                    # Env-only providers (no Settings field): mirror
                    # build_router's own env checks.
                    if provider == "anthropic":
                        return bool(os.environ.get("ANTHROPIC_API_KEY")
                                    or os.environ.get("NM_ANTHROPIC_API_KEY"))
                    if provider == "gemini":
                        return bool(os.environ.get("GEMINI_API_KEY")
                                    or os.environ.get("NM_GEMINI_API_KEY"))
                    if provider == "deepseek":
                        return bool(os.environ.get("DEEPSEEK_API_KEY")
                                    or os.environ.get("NM_DEEPSEEK_API_KEY"))
                    return None

                missing = [p for p in chain if _key_present(p) is False]
                if not chain:
                    results.append((True, "providers: no provider chain "
                                    "configured (local/default brain)"))
                elif missing:
                    results.append((False, "providers: no API key in "
                                    "settings for: " + ", ".join(missing)))
                else:
                    results.append((True, "providers: chain "
                                    f"{' → '.join(chain)}, keys present"))
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            from ...core.errors import exc_str
            results.append((False, f"providers: check failed: {exc_str(exc)}"))

        passed = sum(1 for ok, _ in results if ok)
        lines = [f"doctor — {passed}/{len(results)} checks pass"]
        for ok, text in results:
            lines.append(f"{'OK  ' if ok else 'FAIL'} {text}")
        return "\n".join(lines)

    def _control_style(self, command: Any, chat_key: str,
                       message: Any = None) -> str:
        """Inspect + tune the mined style profile (spec §3.4). Owner-only
        (enforced by the grant check in handle_control before dispatch).

        The profile drives OUTPUT styling (the presentation layer:
        command / tool / game / scheduled outputs) — conversational replies keep
        the persona's own voice and are NOT reformatted by the style
        system (owner correction 2026-10-17: style was meant for
        outputs, not normal chats).

        ::

            /style                          show this chat's live profile
            /style <field> <value> [target]  set an override delta
            /style reset [target]           clear this chat's overrides
            /style list                     every chat with overrides

        Fields: emoji_cap (0..3), target_chars, max_chars,
        sections|lists|markup (on|off). Overrides are deltas on top of
        the mined profile — they persist, carry provenance, and reset
        restores the mined numbers. No target sets the chat the command
        was sent in. A target names a remote chat the way a human
        would (register #35 — see nomorals/partner/chat_targets.py):
        a full key (``platform:chat_id``), a phone number (that
        person's WhatsApp DM), or the chat's NAME, resolved against
        the title index and adapter chat listings. A name matching
        several chats lists the candidates instead of guessing;
        a name matching none reports what was tried. The override
        always lands under the resolved chat's canonical key.
        """
        from ...partner.style import style_for_chat
        from ...partner.style_overrides import (
            FIELDS, clear_overrides, get_overrides, get_overrides_detailed,
            list_scopes, set_override)

        usage = ("usage: /style  ·  /style <emoji_cap|target_chars|max_chars|"
                 "sections|lists|markup> <value> [target]  ·  "
                 "/style reset [target]  ·  /style list  ·  a target is a "
                 "chat name, a phone number, or a full key "
                 "(platform:chat_id)")
        words = (command.tail or "").split()
        sub = (words[0].lower() if words else "")

        def _resolve(raw: str) -> tuple[str, str, str | None]:
            """Resolve a target reference → (key, title, error).

            Exactly one of key / error is set. Resolution follows
            chat_targets (register #35, the /chatmode pattern):
            full key, phone number, chat name — with ambiguity
            and exhaustion reported, never guessed. Raw keys keep
            working verbatim; human forms resolve to the same
            canonical key the override is stored under.
            """
            from ...partner.chat_targets import resolve_chat_target

            chat = getattr(message, "chat", None)
            adapters: list = []
            gateway = getattr(self, "gateway", None)
            if gateway is not None:
                try:
                    adapters = list(gateway.adapters.values())
                except Exception:  # noqa: BLE001 - listing is a hint
                    adapters = []
            res = resolve_chat_target(
                raw,
                current_key=chat_key,
                current_title=str(getattr(chat, "title", "") or ""),
                current_platform=str(getattr(chat, "platform", "") or ""),
                db=getattr(self.context, "db", None),
                adapters=adapters)
            if res.resolved:
                return res.key, res.title, None
            return "", "", res.describe(raw)

        def _where(target: str, title: str = "") -> str:
            if target == chat_key:
                return "this chat"
            return f"{title} ({target})" if title else target

        def _audience_for(target: str) -> str:
            if target == chat_key and message is not None:
                kind = str(getattr(getattr(message, "chat", None), "kind", "") or "")
                if "group" in kind.lower() or "channel" in kind.lower():
                    return "group"
            return "owner"

        def _profile_lines(target: str) -> str:
            overrides = get_overrides(target)
            profile = style_for_chat(
                audience=_audience_for(target),
                platform=(target.partition(":")[0] or "telegram"),
                mood_values=self.mood.current().values
                if hasattr(self, "mood") else None,
                overrides=overrides)
            detailed = get_overrides_detailed(target)
            lines = [
                f"style for {target} (audience: {profile.audience}):",
                f"  emoji budget: {profile.emoji_budget}  ·  target: "
                f"{profile.target_chars} chars  ·  hard cap: {profile.max_chars}",
                f"  sections: {'on' if profile.allow_sections else 'off'}  ·  "
                f"lists: {'on' if profile.allow_lists else 'off'}  ·  "
                f"markup: {'on' if profile.allow_markup else 'off'}",
            ]
            if profile.measured.n >= 5:
                lines.append(
                    f"  measured from your own messages: median "
                    f"{int(profile.measured.median_chars)} chars over "
                    f"{profile.measured.n} messages")
            if detailed:
                lines.append("  overrides (deltas on the mined numbers):")
                for field in sorted(detailed):
                    rec = detailed[field]
                    when = ""
                    try:
                        import datetime as _dt
                        when = _dt.datetime.fromtimestamp(
                            float(rec.get("set_at") or 0)).strftime(" %Y-%m-%d")
                    except (TypeError, ValueError, OSError, OverflowError):
                        when = ""
                    lines.append(
                        f"    {field} = {rec['value']} (set by {rec.get('set_by', '?')}{when})")
                lines.append("  /style reset clears these.")
            else:
                lines.append("  no overrides — mined numbers in effect.")
            lines.append(
                "  mood + measured history bend these at reply time; "
                "style is mined from YOUR messages only, never other people's.")
            return "\n".join(lines)

        if not words:
            return _profile_lines(chat_key)
        if sub == "list":
            if len(words) > 1:
                return f"/style list takes no target — {usage}"
            scopes = list_scopes()
            if not scopes:
                return "no style overrides anywhere — mined numbers in effect."
            lines = ["chats with style overrides:"]
            for scope in scopes:
                ov = get_overrides(scope)
                lines.append(f"  {scope}: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(ov.items())))
            return "\n".join(lines)
        if sub == "reset":
            # The target is the rest of the line — a chat NAME may
            # be several words (the /chatmode rule, register #35).
            target_raw = " ".join(words[1:])
            target, target_title = chat_key, ""
            if target_raw:
                target, target_title, err = _resolve(target_raw)
                if err:
                    return err
            cleared = clear_overrides(target)
            where = _where(target, target_title)
            return (f"overrides cleared for {where} — mined numbers back in "
                    f"effect." if cleared else
                    f"no overrides to clear for {where}.")
        field_aliases = {"sections": "allow_sections", "lists": "allow_lists",
                         "markup": "allow_markup"}
        field = field_aliases.get(sub, sub)
        if field not in FIELDS:
            return (f"unknown style field {sub!r} — want one of: "
                    f"{', '.join(sorted(list(FIELDS) + list(field_aliases)))} — {usage}")
        if len(words) < 2:
            return f"/style {sub} needs a value — {usage}"
        # The target is the rest of the line — a chat NAME may
        # be several words (the /chatmode rule, register #35).
        target_raw = " ".join(words[2:])
        target, target_title = chat_key, ""
        if target_raw:
            target, target_title, err = _resolve(target_raw)
            if err:
                return err
        try:
            set_override(target, field, words[1])
        except ValueError as exc:
            return f"{exc} — {usage}"
        where = _where(target, target_title)
        return (f"style override for {where}: {field} = {words[1]} "
                f"(delta on the mined profile; /style reset clears it).")

    def _control_model(self, arg: str) -> str:
        """Live model switch — no restart.

        ``/model`` reports what is answering; ``/model <provider> [fallback …]``
        rebuilds the router in place. The choice is persisted to the kv_store
        (the same store ``nm models --set-provider`` writes), so it also
        survives a restart.
        """
        parts = [p for p in (arg or "").split() if p]
        if not parts:
            return f"model: {self._model_status_line()}"
        provider = parts[0].lower()
        chain = [p.lower() for p in parts[1:]]

        from ...llm.providers import build_provider

        try:
            build_provider(provider, model="probe")  # validates the name
        except Exception as exc:  # noqa: BLE001
            return (f"can't use {provider!r}: {exc} — /model alone shows "
                    f"what's answering and the current fallbacks")

        from ..context import _build_router, persist_provider_override

        db = self.context.db
        try:
            persist_provider_override(db, provider, chain or None)
        except Exception as exc:  # noqa: BLE001
            return f"could not persist the switch: {exc}"
        try:
            new_router = _build_router(self.settings, self.context.bus, db=db, context=self.context)
        except Exception as exc:  # noqa: BLE001
            return f"could not build the {provider} router: {exc}"
        self.brain.responder.router = new_router
        self.context.router = new_router
        try:
            from ...llm.power import invalidate as _invalidate_power
            _invalidate_power(self.context)
        except Exception as exc:  # noqa: BLE001 - power cache is best-effort
            record_failure("runtime_meta._control_model", exc)
        _log.info("model switched live: provider=%s chain=%s", provider, chain)
        return f"model switched — now: {self._model_status_line()}"

    def _control_providers(self) -> str:
        """Probe every registered provider and report live/dead with reasons.

        ``/providers`` — the diagnostic for a silent brain.  Runs each
        provider's liveness probe in parallel (bounded), then joins it with
        the router's recorded health (calls, failures, last error, cooldown)
        and translates the last error into an actionable hint: bad key,
        dead model id, rate limit, or network.
        """
        from concurrent.futures import ThreadPoolExecutor
        from ...core.errors import (
            is_auth_error,
            is_not_found_error,
            is_rate_limited_error,
        )

        router = getattr(self.context, "router", None)
        if router is None:
            return "providers: no router on this runtime."
        names = list(router.providers())
        if not names:
            return "providers: chain is empty — nothing can answer. /model to configure."

        def _probe(name: str) -> tuple[str, bool, float, str]:
            provider = router.get(name)
            if provider is None:
                return name, False, 0.0, "provider not registered"
            started = time.perf_counter()
            try:
                detail_fn = getattr(provider, "health_detail", None)
                if callable(detail_fn):
                    ok, detail = detail_fn()
                    ok = bool(ok)
                    detail = str(detail or "")
                else:
                    ok = bool(provider.health())
                    detail = ""
            except Exception as exc:  # noqa: BLE001 - a probe must never raise
                ok = False
                detail = str(exc) or exc.__class__.__name__
            return name, ok, (time.perf_counter() - started), detail

        # Parallel: sequential probes would cost sum(health timeouts).
        live: dict[str, tuple[bool, float, str]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(names)),
                                thread_name_prefix="providers-probe") as pool:
            for name, ok, elapsed, detail in pool.map(_probe, names):
                live[name] = (ok, elapsed, detail)

        try:
            snap = router.stats_snapshot()
        except Exception as exc:  # noqa: BLE001 - status must never crash a chat
            record_failure("runtime_meta._control_providers:893", exc)
            snap = {}
        health = snap.get("health") or {}
        active = str(snap.get("active") or "")

        def _hint(last_error: str) -> str:
            if not last_error:
                return ""
            probe_exc: Exception = Exception(last_error)
            # Rebuild enough signal for the classifiers: the recorded error
            # already embeds the HTTP status ("401 unauthorized for …").
            if is_auth_error(probe_exc):
                return "→ auth failed: check the key/token for this provider"
            if is_not_found_error(probe_exc):
                return "→ model id not hosted: fix the model name in config"
            if is_rate_limited_error(probe_exc):
                return "→ rate limited: wait for the quota window"
            lowered = last_error.lower()
            if "timed out" in lowered or "timeout" in lowered:
                return "→ network timeout: check connectivity"
            if "cool" in lowered:
                return ""
            return ""

        lines = [f"providers (active: {active or 'none'}):"]
        for name in names:
            provider = router.get(name)
            model_id = ""
            if provider is not None:
                try:
                    model_id = str(provider.model_id or "")
                except Exception as exc:  # noqa: BLE001
                    record_failure("runtime_meta._control_providers:924", exc)
                    model_id = ""
            h = health.get(name) or {}
            ok, elapsed, probe_detail = live.get(name, (False, 0.0, ""))
            calls = int(h.get("calls") or 0)
            failures = int(h.get("failures") or 0)
            cooling = bool(h.get("cooling_down"))
            last_error = str(h.get("last_error") or "")
            marker = "★" if name == active else " "
            if cooling:
                state = "🧊 cooling down"
            elif ok:
                state = f"✅ live ({elapsed:.1f}s)"
            elif probe_detail:
                # The live probe ran and failed — show WHY. The old
                # "(no calls yet)" label implied the probe never ran,
                # which hid the real failure (bad key, dead model, …).
                state = f"⚠️ probe failed ({probe_detail[:120]})"
            elif calls and failures >= calls:
                state = "❌ failing every call"
            elif calls:
                state = f"⚠️ probe failed ({failures}/{calls} failed)"
            else:
                # Probe failed with no reason recorded anywhere: say
                # so plainly instead of implying it was never tested.
                state = "⚠️ probe failed (no detail)"
            detail = f" — {model_id}" if model_id else ""
            lines.append(f" {marker} {name}{detail}: {state}")
            if last_error:
                short = last_error[:160] + ("…" if len(last_error) > 160 else "")
                lines.append(f"    last error: {short}")
                hint = _hint(last_error)
                if hint:
                    lines.append(f"    {hint}")
            elif probe_detail:
                # Zero recorded calls but the live probe just failed:
                # the probe detail IS the error — hint from it so the
                # owner still gets the actionable line (bad key, …).
                hint = _hint(probe_detail)
                if hint:
                    lines.append(f"    {hint}")
            elif calls and not failures:
                lines.append(f"    {calls} calls, 0 failures")
        # The llm ladder's all-down circuit breaker, when open: callers
        # are fast-failing instead of paying full provider walks, and
        # the owner should see that state (and its expiry) here rather
        # than wonder why the brain went quiet.
        try:
            es = getattr(self.context, "error_system", None)
            ladder = es.ladders.get("llm") if es is not None else None
            if ladder is not None:
                ladder_status = ladder.status()
                if ladder_status.get("circuit_open"):
                    lines.append(
                        f" ⚡ llm circuit open — brain fast-failing; "
                        f"retry in ~{float(ladder_status.get('retry_in_s') or 0):.0f}s "
                        f"({int(ladder_status.get('fast_fails') or 0)} calls held)")
        except Exception as exc:  # noqa: BLE001 - status must never crash a chat
            record_failure("runtime_meta._control_providers:ladder", exc)
        return "\n".join(lines)

    # ── devon: the autonomous dev & investigation agent ──────────────────────
    def _control_devon(self, tail: str, chat_key: str) -> str:
        """Owner types ``/devon <free text>``: Devon plans the tools that fit
        the question, runs them within a budget, journals every step into his
        own memory box, and digests the findings into a plain-English reply.

        With no task, it's a status line: what he's on and his last digests.
        """
        from ..devon import DevonAgent

        task = (tail or "").strip()
        agent = DevonAgent(self.context, brain=self.brain, gateway=self.gateway)
        if not task:
            digests = agent.recent_digests(3)
            if not digests:
                return (
                    "devon: give me a task — e.g. /devon check if the brain replied to the last messages\n"
                    "tools: read/grep code, git status+diff, live db, message flow, turn traces, "
                    "run tests, log tail, mood, chat+game stats, model chain health, config snapshot, "
                    "his own history, web research, scheduled watches, background missions.\n"
                    "every run is journaled in his memory box, so he remembers what he already checked."
                )
            lines = ["devon — recent digests:"]
            for d in digests:
                lines.append(f"  · {d['task'][:60]} → {d['digest'][:160]}")
            return "\n".join(lines)

        result = agent.run(task, chat_key=chat_key)
        tools = ", ".join(dict.fromkeys(s.tool for s in result.steps)) or "none"
        header = f"🔧 devon (planned by {result.planned_by}, {result.seconds:.0f}s · tools: {tools})"
        # wave 68: when the plan degraded, say WHY — an honest
        # "planned by heuristic (model down)" beats a silent fallback.
        if result.planned_by == "heuristic" and result.plan_error:
            header += f" — {result.plan_error[:90]}"
        if result.plan_error:
            # Persist EVERY plan degradation (heuristic AND reasoning-engine
            # fallback) so `nm mind` shows the last plan_error with a
            # timestamp (wave E router telemetry). Goes through the CoreMind
            # helper built for exactly this — never raises, never breaks
            # the reply.
            try:
                self.mind.record_plan_error(result.plan_error, route="devon")
            except Exception:  # noqa: BLE001 - telemetry never breaks a reply
                _log.debug("devon plan-error telemetry failed", exc_info=True)
        text = f"{header}\n{result.digest}"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── reasoning: /think — explicit, auditable multi-step thought ──────────
    def _control_think(self, tail: str, chat_key: str) -> str:
        """Owner types ``/think <question> [strategy]``: the reasoning
        engine works it out with an explicit trace — plan, subgoals,
        actions, observations, critiques, verdict — and answers with the
        full work shown, not just the conclusion."""
        from ..reasoning import ReasoningEngine, trace_text

        parts = (tail or "").strip().split(None, 1)
        if not parts or not parts[0].strip():
            return ("usage: /think <question> [strategy]\n"
                    "strategies: auto (default) · cot · decompose · "
                    "hypothesize · critique · tree\n"
                    "it shows the work — every step, then the answer.")
        question = parts[0].strip()
        strategy = parts[1].strip() if len(parts) > 1 and parts[1].strip() \
            else "auto"
        if strategy not in {"auto", "cot", "decompose", "hypothesize",
                            "critique", "tree"}:
            return (f"unknown strategy {strategy!r} — use auto, cot, "
                    "decompose, hypothesize, critique, or tree")
        engine = ReasoningEngine(self.context, max_llm_calls=14,
                                 max_seconds=120.0)
        try:
            result = engine.reason(question, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 — never let a bad run kill chat
            return f"reasoning failed: {type(exc).__name__}: {exc}"
        text = trace_text(result)
        if len(text) > 3500:
            text = text[:3480] + "\n…(trace trimmed)"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── /benchmark — how sharp is the system right now ──────────────────────
    def _control_benchmark(self, tail: str) -> str:
        """Owner types ``/benchmark [dimension]``: runs the agent benchmark
        (1 task per dimension for speed in-chat) and reports the score."""
        from ..benchmark import run_benchmark

        dim = (tail or "").strip()
        dims = [dim] if dim in {"reasoning", "planning", "tool_use",
                                "self_correction"} else None
        if tail and not dims:
            return "dimensions: reasoning, planning, tool_use, self_correction"
        report = run_benchmark(self.context, dimensions=dims, limit=1)
        if not report.measurable:
            return (f"can't benchmark with the {report.provider!r} provider "
                    f"(mock/offline) — switch to a real model and ask again")
        overall = report.overall if report.overall is not None else 0.0
        lines = [f"benchmark: {overall:.2f} ({report.provider}, "
                 f"{report.seconds:.0f}s)"]
        for name, d in report.scores.items():
            score = "n/a" if d.score is None else f"{d.score:.0%}"
            lines.append(f"  {name}: {score} ({d.passed}/{d.total})")
        return "\n".join(lines)

    def _control_identity(self, tail: str) -> str:
        """Identity bank for signups: /identity show | set <field> <value> | clear.

        Stores the owner's real identity once (name, email, phone) so account
        creation flows can use it without asking every time. Persisted to DB.
        """
        from ...accounts.creator import AccountCreator
        from ...accounts.vault import VaultLockedError, open_vault

        parts = (tail or "").strip().split(None, 2)
        verb = parts[0].lower() if parts else "show"

        # Get or create the AccountCreator — the vault opens through
        # the one unlock path (any owner credential), never a raw
        # passphrase construction.
        try:
            vault = open_vault(
                self.context.db if hasattr(self.context, "db") else None,
                context=self.context, actor="identity")
        except VaultLockedError as exc:
            return f"identity bank unavailable — {exc}"
        creator = AccountCreator(vault, db=getattr(self.context, "db", None))

        if verb == "show":
            identity = creator.get_owner_identity()
            if not identity:
                return ("no identity bank set yet.\n"
                        "usage:\n"
                        "  /identity set name <your name>\n"
                        "  /identity set email <your email>\n"
                        "  /identity set phone <your phone>  (optional)")
            lines = ["identity bank:"]
            lines.append(f"  name: {identity.get('name', '')}")
            lines.append(f"  email: {identity.get('email', '')}")
            if identity.get("phone"):
                lines.append(f"  phone: {identity.get('phone', '')}")
            return "\n".join(lines)

        if verb == "set":
            if len(parts) < 3:
                return "usage: /identity set <name|email|phone> <value>"
            field, value = parts[1].lower(), parts[2].strip()
            if field not in ("name", "email", "phone"):
                return f"unknown field {field!r} — use name, email, or phone"
            if not value:
                return f"value for {field} cannot be empty"
            # Load existing (may be partial), update the field, persist.
            # Partial saves are allowed — signup flows validate completeness.
            current = creator.get_owner_identity() or {}
            current[field] = value
            try:
                if creator.db is not None:
                    KVStore(creator.db).set(creator.IDENTITY_KV_KEY, current)
                creator._owner_identity = dict(current)
            except Exception as exc:  # noqa: BLE001
                return f"save failed: {exc}"
            missing = [f for f in ("name", "email") if not current.get(f)]
            if missing:
                return (f"set {field}. still need: {', '.join(missing)}\n"
                        f"  /identity set {missing[0]} <value>")
            return "identity bank complete."

        if verb == "clear":
            # Clear by setting empty (will fail validation, so do direct kv delete)
            try:
                if creator.db is not None:
                    KVStore(creator.db).delete(creator.IDENTITY_KV_KEY)
                creator._owner_identity = None
                return "identity bank cleared."
            except Exception as exc:  # noqa: BLE001
                return f"clear failed: {exc}"

        return "usage: /identity [show|set <field> <value>|clear]"

    #: Commands the tour never probes — destructive, state-changing, slow,
    #: or money-moving. The tour is read-only; these stay untouched.
    _TOUR_DENYLIST = frozenset({
        # code execution
        "exec",
        # outbound messaging
        "send", "dm", "email", "gmail", "mail", "inbox",
        # destructive / state-changing
        "quit", "upgrade", "train", "repair", "trial",
        # starts background work
        "mission", "schedule", "routine", "monitor", "watch", "track",
        # installs software / downloads multi-GB weights (minutes-long)
        "models",
        # state toggles / autonomous loops
        "proactive", "model", "features", "arena",
        # money
        "money", "shop", "bet", "desk", "apk", "spend",
        # slow (attack suites, benchmarks, media generation)
        "redteam", "benchmark", "vnote", "audio", "video", "music",
        # no recursion
        "tour", "heal",
    })

    _TOUR_PROBE_TIMEOUT = 6.0  # seconds per command; slower → skipped

    def _control_tour(self, tail: str, chat_key: str = "") -> str:
        """Smoke-test every registered command with safe read-only probes.

        /tour — probe all commands, report OK / broken / skipped.
        /tour <command> — deep-probe one command.
        """
        target = (tail or "").strip().lstrip("/").lower().split(None, 1)[0] \
            if (tail or "").strip() else ""
        if target:
            return self._tour_single(target, chat_key)
        return self._tour_all(chat_key)

    def _tour_probe(self, kind: str, chat_key: str) -> dict:
        """One safe probe of a command. Never raises. Returns a result dict."""
        from ...social.chat.control import (
            CONTROL_COMMANDS, GAME_START_COMMANDS, detailed_help)

        spec = CONTROL_COMMANDS.get(kind)
        if spec is None:
            return {"kind": kind, "verdict": "skip", "reason": "not registered"}
        min_args = spec[0]
        if min_args > 0:
            return {"kind": kind, "verdict": "skip",
                    "reason": f"needs {min_args} arg(s)"}
        if kind in self._TOUR_DENYLIST:
            return {"kind": kind, "verdict": "skip", "reason": "denylisted"}
        if kind in GAME_START_COMMANDS:
            # Starting a game is interactive state (a live session in
            # this chat), never a read-only probe — and the probe must
            # not mint one. The dispatch path itself is covered by the
            # registry test (GAME_START_COMMANDS → _control_game branch);
            # the tour verifies the help page and reports the skip
            # honestly instead of faking an OK.
            try:
                h = detailed_help(kind)
                help_ok = bool(h) and "no help page" not in h[:60].lower()
            except Exception:  # noqa: BLE001
                help_ok = False
            return {"kind": kind, "verdict": "skip",
                    "reason": "interactive game — start it to play",
                    "help_ok": help_ok}

        # Help coverage check — pure read-only.
        help_ok = True
        try:
            h = detailed_help(kind)
            if not h or "unknown" in h[:60].lower():
                help_ok = False
        except Exception:  # noqa: BLE001
            help_ok = False

        # Dispatch probe — no-arg call in a worker thread with a timeout.
        # Never raises out of here; timeouts and exceptions are captured.
        import concurrent.futures as _cf
        import traceback as _tb

        def _run():
            return self.handle_control(f"/{kind}", chat_key, message=None)

        reply, error, location = "", "", ""
        timed_out = False
        with _cf.ThreadPoolExecutor(max_workers=1,
                                     thread_name_prefix="tour") as _ex:
            fut = _ex.submit(_run)
            try:
                reply = fut.result(timeout=self._TOUR_PROBE_TIMEOUT)
            except _cf.TimeoutError:
                timed_out = True
                fut.cancel()
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                frames = _tb.extract_tb(exc.__traceback__)
                if frames:
                    location = (f"{frames[-1].filename.split('/')[-1]}:"
                                f"{frames[-1].lineno}")

        if not isinstance(reply, str):
            # ActionableMessage replies (menus, hub screens, game
            # boards) are a healthy dispatch shape — the probe reads
            # their text like the send channel does, instead of
            # dying on .strip() and reporting the command broken.
            reply = str(getattr(reply, "text", "") or "")
        if timed_out:
            return {"kind": kind, "verdict": "skip", "reason": "slow (>6s)",
                    "help_ok": help_ok}
        if error:
            return {"kind": kind, "verdict": "broken", "error": error,
                    "location": location, "help_ok": help_ok}
        # A reply is only OK if it is a real answer. Two lies the tour
        # used to bless as OK: handle_control's "unknown command"
        # fallback for a REGISTERED kind (a dispatch-table gap), and
        # an empty reply (nothing to send, nothing delivered — a
        # command that returns nothing is a broken command).
        _stripped = (reply or "").strip()
        if _stripped.startswith("unknown command"):
            return {"kind": kind, "verdict": "broken",
                    "error": f"dispatch gap: {reply[:100]}",
                    "location": "runtime.py:handle_control", "help_ok": help_ok}
        if not _stripped:
            return {"kind": kind, "verdict": "broken",
                    "error": "empty reply — the command returned nothing",
                    "location": "", "help_ok": help_ok}
        return {"kind": kind, "verdict": "ok",
                "reply": (reply or "")[:120], "help_ok": help_ok}

    def _tour_all(self, chat_key: str) -> str:
        """Probe every registered command, return a compact report."""
        from ...social.chat.control import CONTROL_COMMANDS

        kinds = sorted(k for k in CONTROL_COMMANDS if k != "error")
        ok, broken, skipped = [], [], []
        skip_reasons: dict[str, int] = {}
        for kind in kinds:
            try:
                res = self._tour_probe(kind, chat_key)
            except Exception as exc:  # noqa: BLE001 — the tour never dies
                res = {"kind": kind, "verdict": "broken",
                       "error": f"{type(exc).__name__}: {exc}", "location": ""}
            if res["verdict"] == "ok":
                ok.append(kind)
            elif res["verdict"] == "broken":
                broken.append(res)
            else:
                skipped.append(res)
                r = res.get("reason", "?")
                skip_reasons[r] = skip_reasons.get(r, 0) + 1

        lines = [f"🧪 tour — {len(kinds)} commands probed",
                 f"✅ {len(ok)} OK · ❌ {len(broken)} broken · "
                 f"⏭️ {len(skipped)} skipped"]
        if broken:
            lines.append("")
            lines.append("❌ broken:")
            for res in broken[:15]:
                loc = f" [{res['location']}]" if res.get("location") else ""
                lines.append(f"  /{res['kind']} — {res['error'][:100]}{loc}")
            if len(broken) > 15:
                lines.append(f"  …and {len(broken) - 15} more")
        if skipped:
            lines.append("")
            detail = ", ".join(f"{n} {r}" for r, n in
                               sorted(skip_reasons.items(),
                                      key=lambda kv: -kv[1]))
            lines.append(f"⏭️ skipped: {detail}")
        lines.append("")
        lines.append("deep-probe one: /tour <command>")
        return "\n".join(lines)

    def _tour_single(self, kind: str, chat_key: str) -> str:
        """Deep-probe one command with safe probes."""
        from ...social.chat.control import CONTROL_COMMANDS

        if kind not in CONTROL_COMMANDS:
            return (f"no such command: /{kind}\n"
                    f"/tour (no args) lists everything probed.")
        try:
            res = self._tour_probe(kind, chat_key)
        except Exception as exc:  # noqa: BLE001 — the tour never dies
            res = {"kind": kind, "verdict": "broken",
                   "error": f"{type(exc).__name__}: {exc}", "location": ""}

        lines = [f"🧪 /{kind}"]
        lines.append(f"  📖 help: {'yes' if res.get('help_ok') else 'missing/broken'}")
        verdict = res["verdict"]
        if verdict == "ok":
            reply = res.get("reply", "")
            lines.append(f"  ⚙️ probe: OK")
            if reply:
                lines.append(f"  💬 reply: {reply[:200]}")
        elif verdict == "broken":
            loc = f" [{res['location']}]" if res.get("location") else ""
            lines.append(f"  ⚙️ probe: BROKEN — {res.get('error', '?')[:200]}{loc}")
        else:
            lines.append(f"  ⚙️ probe: SKIPPED — {res.get('reason', '?')}")
        if kind in self._TOUR_DENYLIST:
            lines.append(f"  🛡️ denylisted: never auto-probed (destructive/slow)")
        return "\n".join(lines)

    # ── /heal: the self-healing loop ──────────────────────────────────────
    # tour (find broken) → error doctor (diagnose) → conservative auto-fix
    # → re-probe (verify) → report. Only two fix patterns are ever
    # auto-applied: unbound-variable init and dead-method rescue. Both are
    # semantics-preserving by construction; everything else is reported.

    def _control_heal(self, tail: str, chat_key: str = "") -> str:
        """Self-healing loop over broken commands.

        /heal — find broken commands, auto-fix the safe ones, verify, report.
        /heal --dry — show what WOULD be fixed without changing anything.
        /heal <command> — heal just one command.
        """
        from ...core.self_heal import heal_all, heal_one

        args = (tail or "").strip().split()
        dry = "--dry" in args or "-n" in args
        target = next((a.lstrip("/").lower() for a in args
                       if not a.startswith("-")), "")

        # /heal never probes itself or the tour (recursion), and respects
        # the same denylist as the tour for everything else. Game-start
        # commands are skipped too: probing one would START a live game
        # in this chat — interactive state, never a safe probe.
        from ...social.chat.control import GAME_START_COMMANDS
        skip = (frozenset(self._TOUR_DENYLIST) | {"heal"}
                | set(GAME_START_COMMANDS))

        if target:
            if target == "heal":
                return "🩹 /heal doesn't heal itself."
            if target in GAME_START_COMMANDS:
                return (f"🩹 /{target} is an interactive game — probing it "
                        "would start a live game in this chat. Start it to "
                        "play; nothing to heal.")
            try:
                res = heal_one(self.handle_control, target, chat_key,
                               dry_run=dry)
            except Exception as exc:  # noqa: BLE001 — heal never raises
                return f"🩹 heal of /{target} failed safely: {exc}"
            return self._heal_report_single(res)

        from ...social.chat.control import CONTROL_COMMANDS
        kinds = sorted(k for k in CONTROL_COMMANDS
                       if k != "error" and k not in skip)
        try:
            summary = heal_all(self.handle_control, kinds, chat_key,
                               dry_run=dry, skip=skip)
        except Exception as exc:  # noqa: BLE001 — heal never raises
            return f"🩹 heal loop failed safely: {exc}"
        return self._heal_report(summary)

    def _heal_report_single(self, res) -> str:
        """One-command heal report."""
        if not res.broken:
            return f"🩹 /{res.kind} — not broken, nothing to heal."
        lines = [f"🩹 /{res.kind} — {res.error[:120]}"]
        if res.dry_run and res.fixable:
            lines.append(f"  would fix: {res.fix_detail}")
            if res.diff_preview:
                lines.append(f"  {res.diff_preview}")
        elif res.fixed:
            lines.append(f"  ✅ healed: {res.fix_detail}")
            if res.committed:
                lines.append("  📝 fix committed (reversible via git)")
        else:
            lines.append(f"  ❌ still broken: {res.skip_reason or res.error[:100]}")
            if res.suggestion:
                lines.append(f"  💡 suggestion: {res.suggestion[:160]}")
        return "\n".join(lines)

    def _heal_report(self, summary: dict) -> str:
        """Full-loop heal report."""
        dry = summary.get("dry_run", False)
        healed = summary.get("healed", [])
        still = summary.get("still_broken", [])
        would = summary.get("would_fix", [])
        total = summary.get("total", 0)

        if dry:
            lines = [f"🩹 heal --dry — {total} commands probed"]
            if would:
                lines.append(f"  would fix {len(would)}:")
                for r in would:
                    lines.append(f"    /{r['kind']} — {r['fix_detail'][:100]}")
            else:
                lines.append("  nothing auto-fixable found.")
            if still:
                lines.append(f"  still broken (no safe fix): "
                             f"{', '.join('/' + r['kind'] for r in still[:10])}")
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "\n".join(lines),
                [[("🩹 Apply the fixes", "heal")]])

        lines = [f"🩹 heal — {total} commands probed"]
        if healed:
            lines.append(f"  ✅ healed {len(healed)}:")
            for r in healed:
                c = " 📝" if r.get("committed") else ""
                lines.append(f"    /{r['kind']} — {r['fix_detail'][:100]}{c}")
        if still:
            lines.append(f"  ❌ still broken {len(still)}:")
            for r in still[:12]:
                why = r.get("skip_reason") or r.get("error", "")[:80]
                lines.append(f"    /{r['kind']} — {why[:100]}")
            if len(still) > 12:
                lines.append(f"    …and {len(still) - 12} more")
        if not healed and not still:
            lines.append("  ✨ nothing broken — nothing to heal.")
        return "\n".join(lines)
