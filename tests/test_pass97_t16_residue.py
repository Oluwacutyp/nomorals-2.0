"""Pass #97 — T1.6 residue burn-down: two defects proven on pristine by #96.

1. Vision ``room:`` references could never resolve. Pass #96 moved the
   room-resolver body verbatim into ``workspace.rooms.resolve_room_ref``
   and wired it via ``tools.vision.set_room_resolver`` from
   ``agents.context.build_context`` — but the body built a db-less
   ``RoomManager``, and ``RoomManager.get()`` returns None without a db,
   so every real ``room:`` reference failed with "no such room". These
   tests drive the vision tool's public path (``resolve_image``) through
   a context assembled by the REAL composition root (``build_context``),
   against a real room created in a real database.

2. ``NaijaShoppingEngine.match_cross_site`` read ``s.price`` while the
   snapshot dataclass's field is ``price_ngn`` — AttributeError on any
   real input, zero callers, nothing caught it. These tests exercise the
   method with realistic multi-site snapshots and pin the grouping
   contract (word-order-invariant title match via naija_deals).

No network anywhere in this file.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.agents.context import build_context
from nomorals.core.config import load_settings
from nomorals.core.errors import ToolError
from nomorals.integrations.naija_shopping import (
    NaijaShoppingEngine,
    PriceSnapshot,
)
from nomorals.storage.db import Database
from nomorals.tools import vision as V
from nomorals.tools.browser import BrowserSession
from nomorals.workspace.rooms import RoomManager


class VisionRoomRefTests(unittest.TestCase):
    """room: references resolve through the build_context wiring."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-p97-")
        self.addCleanup(self._tmp.cleanup)
        # the vision resolver is a module global — always leave it as
        # found-at-import (None) so no wiring leaks into other suites
        self.addCleanup(V.set_room_resolver, None)
        settings = load_settings(overrides={
            "home": self._tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.enabled": "false",
        })
        # the real composition root: installing the room resolver is part
        # of build_context, exactly as at production boot
        self.context = build_context(
            settings, with_executor=False, with_router=False,
            with_memory=False, with_tools=False)
        # a real room in the context's real database, built the way the
        # room tool builds its manager (root + context.db)
        mgr = RoomManager(self.context.settings.workspace_dir,
                          db=self.context.db)
        self.room = mgr.create("Vision Lab")
        self.payload = b"\x89PNG\r\n\x1a\npass97-room-bytes"
        target = (Path(self.context.settings.workspace_dir)
                  / "rooms" / self.room.slug / "files" / "chart.png")
        target.write_bytes(self.payload)

    def test_room_reference_resolves_real_room_bytes(self) -> None:
        data, label = V.resolve_image(
            self.context,
            reference=f"room:{self.room.slug}:files/chart.png")
        self.assertEqual(data, self.payload)
        self.assertEqual(label, f"room:{self.room.slug}:files/chart.png")

    def test_missing_room_still_errors_by_name(self) -> None:
        with self.assertRaises(ToolError) as cm:
            V.resolve_image(self.context,
                            reference="room:no-such-room:files/x.png")
        self.assertIn("no such room", str(cm.exception))

    def test_missing_file_in_real_room_errors_by_name(self) -> None:
        with self.assertRaises(ToolError) as cm:
            V.resolve_image(
                self.context,
                reference=f"room:{self.room.slug}:files/absent.png")
        self.assertIn("not a file", str(cm.exception))

    def test_unwired_resolver_fails_honestly_by_name(self) -> None:
        V.set_room_resolver(None)
        with self.assertRaises(ToolError) as cm:
            V.resolve_image(
                self.context,
                reference=f"room:{self.room.slug}:files/chart.png")
        self.assertIn("no room resolver wired", str(cm.exception))


class MatchCrossSiteTests(unittest.TestCase):
    """match_cross_site groups the same product across sites."""

    def setUp(self) -> None:
        self.engine = NaijaShoppingEngine(Database(":memory:"),
                                          BrowserSession())

    @staticmethod
    def _snap(site: str, title: str, price: float, url: str) -> PriceSnapshot:
        return PriceSnapshot(product_url=url, site=site, price_ngn=price,
                             title=title)

    def test_groups_same_product_across_sites(self) -> None:
        snapshots = [
            self._snap("jumia", "Samsung Galaxy A55 5G 128GB", 150_000.0,
                       "https://www.jumia.com.ng/a55-a"),
            # same product, word order shuffled — the deal hunter's
            # word-order-invariant key must still group it
            self._snap("konga", "128GB Samsung Galaxy A55 5G", 142_500.0,
                       "https://www.konga.com/a55-b"),
            self._snap("slot", "Samsung Galaxy A55 5G 128GB", 155_000.0,
                       "https://www.slot.ng/a55-c"),
            self._snap("jumia", "Infinix Hot 40i 128GB", 98_000.0,
                       "https://www.jumia.com.ng/hot40i"),
        ]
        groups = self.engine.match_cross_site(snapshots)
        self.assertEqual(len(groups), 2)

        samsung = next(g for g in groups if "Galaxy A55" in g["title"])
        # ranked cheapest-first by the shared matcher
        self.assertEqual(
            samsung["sites"],
            [("konga", 142_500.0), ("jumia", 150_000.0),
             ("slot", 155_000.0)])
        self.assertEqual(samsung["best_site"], "konga")
        self.assertEqual(samsung["best_price"], 142_500.0)

        infinix = next(g for g in groups if "Infinix" in g["title"])
        self.assertEqual(infinix["sites"], [("jumia", 98_000.0)])
        self.assertEqual(infinix["best_site"], "jumia")
        self.assertEqual(infinix["best_price"], 98_000.0)

    def test_empty_input_gives_no_groups(self) -> None:
        self.assertEqual(self.engine.match_cross_site([]), [])


if __name__ == "__main__":
    unittest.main()
