"""Offline tests for build-map #107 — live audio rooms. All offline, no I/O
beyond temp dirs, mock platform/tip/transcriber seams."""

from __future__ import annotations

import tempfile
import unittest

from nomorals.community import audio_rooms as ar
from nomorals.community.groups import GroupStore


def _fresh():
    """A community group + a room store in temp dirs."""
    gs = GroupStore(data_dir=tempfile.mkdtemp())
    g = gs.create_group("Devs", "coding", created_by="host1")
    st = ar.RoomStore(data_dir=tempfile.mkdtemp())
    return gs, g.id, st


def _room(st, gid, title="Friday AMA", gs=None):
    res = ar.create_room(gid, title, "host1", "Ada", store=st, group_store=gs)
    assert res["ok"], res
    return res["room"]


class TestRoomLifecycle(unittest.TestCase):
    def test_create_requires_real_community(self):
        st = ar.RoomStore(data_dir=tempfile.mkdtemp())
        res = ar.create_room("nope", "X", "h", "H", store=st)
        self.assertFalse(res["ok"])
        self.assertIn("no such community", res["reason"])

    def test_create_needs_fields(self):
        st = ar.RoomStore(data_dir=tempfile.mkdtemp())
        self.assertFalse(ar.create_room("", "X", "h", "H", store=st)["ok"])
        self.assertFalse(ar.create_room("g", "", "h", "H", store=st)["ok"])
        self.assertFalse(ar.create_room("g", "X", "", "H", store=st)["ok"])

    def test_create_and_get(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        self.assertEqual(room.host_id, "host1")
        self.assertEqual(room.state, "live")
        self.assertTrue(room.captions)  # on by default
        self.assertFalse(room.recording)
        got = st.get(room.room_id)
        self.assertEqual(got.title, "Friday AMA")

    def test_list_scoped_to_community(self):
        gs, gid, st = _fresh()
        r1 = _room(st, gid, "One", gs=gs)
        _room(st, gid, "Two", gs=gs)
        rooms = st.list(community_id=gid)
        self.assertEqual(len(rooms), 2)
        self.assertEqual({r.community_id for r in rooms}, {gid})
        self.assertNotIn(r1.room_id, [r.room_id for r in st.list(community_id="other")])

    def test_join_leave(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        res = ar.join_room(room.room_id, "u2", "Bola", store=st)
        self.assertTrue(res["ok"])
        self.assertEqual(res["role"], "listener")
        # re-join is idempotent
        res2 = ar.join_room(room.room_id, "u2", "Bola", store=st)
        self.assertTrue(res2["ok"])
        self.assertEqual(res2["role"], "listener")
        left = ar.leave_room(room.room_id, "u2", store=st)
        self.assertTrue(left["ok"])
        self.assertTrue(left["left"])
        # host can't leave
        nope = ar.leave_room(room.room_id, "host1", store=st)
        self.assertFalse(nope["ok"])

    def test_join_ended_room_fails(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.end_room(room.room_id, "host1", store=st)
        res = ar.join_room(room.room_id, "u2", "Bola", store=st)
        self.assertFalse(res["ok"])

    def test_never_raises_on_garbage(self):
        st = ar.RoomStore(data_dir=tempfile.mkdtemp())
        self.assertIsNone(ar.get_room("", store=st))
        self.assertFalse(ar.join_room("", "", "", store=st)["ok"])
        self.assertFalse(ar.raise_hand("zzz", "u", "N", store=st)["ok"])
        self.assertFalse(ar.end_room("zzz", "h", store=st)["ok"])


class TestHandRaise(unittest.TestCase):
    def test_raise_promote_fifo(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.join_room(room.room_id, "u2", "Bola", store=st)
        ar.join_room(room.room_id, "u3", "Emeka", store=st)
        p2 = ar.raise_hand(room.room_id, "u3", "Emeka", store=st)["position"]
        p1 = ar.raise_hand(room.room_id, "u2", "Bola", store=st)["position"]
        self.assertEqual((p2, p1), (1, 2))
        # host promotes in FIFO order
        sp = ar.promote_speaker(room.room_id, "u2", "host1", store=st)
        self.assertTrue(sp["ok"])
        self.assertEqual(sp["speaker"], "Bola")
        room = st.get(room.room_id)
        self.assertEqual(len(room.hand_raise_queue), 1)
        self.assertEqual(len(room.speakers), 1)

    def test_only_host_promotes(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        res = ar.promote_speaker(room.room_id, "u2", "u3", store=st)
        self.assertFalse(res["ok"])
        self.assertIn("host", res["reason"])

    def test_speaker_cannot_raise(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        ar.promote_speaker(room.room_id, "u2", "host1", store=st)
        res = ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        self.assertFalse(res["ok"])

    def test_lower_hand(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        res = ar.lower_hand(room.room_id, "u2", store=st)
        self.assertTrue(res["was_queued"])
        self.assertEqual(len(st.get(room.room_id).hand_raise_queue), 0)

    def test_demote_back_to_audience(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        ar.promote_speaker(room.room_id, "u2", "host1", store=st)
        res = ar.demote_speaker(room.room_id, "u2", "host1", store=st)
        self.assertTrue(res["ok"])
        room = st.get(room.room_id)
        self.assertEqual(len(room.speakers), 0)
        self.assertEqual(len(room.listeners), 1)


class TestModeration(unittest.TestCase):
    def test_mute_unmute(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        ar.promote_speaker(room.room_id, "u2", "host1", store=st)
        m = ar.mute_user(room.room_id, "u2", "host1", store=st)
        self.assertTrue(m["ok"])
        self.assertEqual(m["action"], "muted")
        room = st.get(room.room_id)
        self.assertTrue(room.speakers[0].muted)
        u = ar.mute_user(room.room_id, "u2", "host1", muted=False, store=st)
        self.assertEqual(u["action"], "unmuted")
        # non-host cannot mute
        bad = ar.mute_user(room.room_id, "u2", "u3", store=st)
        self.assertFalse(bad["ok"])
        # host cannot mute self
        self.assertFalse(ar.mute_user(room.room_id, "host1", "host1", store=st)["ok"])

    def test_block_removes_everywhere(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.join_room(room.room_id, "u2", "Bola", store=st)
        ar.raise_hand(room.room_id, "u2", "Bola", store=st)
        res = ar.block_user(room.room_id, "u2", "host1", store=st)
        self.assertTrue(res["ok"])
        room = st.get(room.room_id)
        self.assertIn("u2", room.blocked)
        self.assertEqual(len(room.listeners), 0)
        # blocked user can't rejoin
        self.assertFalse(ar.join_room(room.room_id, "u2", "Bola", store=st)["ok"])

    def test_report_anyone(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.join_room(room.room_id, "u2", "Bola", store=st)
        res = ar.report_user(room.room_id, "u2", "u3", "spam", store=st)
        self.assertTrue(res["ok"])
        room = st.get(room.room_id)
        self.assertEqual(len(room.reports), 1)
        self.assertEqual(room.reports[0]["reason"], "spam")


class TestRecordingCaptions(unittest.TestCase):
    def test_recording_opt_in_visible(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        self.assertFalse(room.recording_visible)
        res = ar.set_recording(room.room_id, True, "host1", store=st)
        self.assertTrue(res["recording"])
        room = st.get(room.room_id)
        self.assertTrue(room.recording_visible)  # always visible when on
        rendered = ar.render_room(room)
        self.assertIn("REC", rendered)

    def test_only_host_records(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        self.assertFalse(ar.set_recording(room.room_id, True, "u2", store=st)["ok"])

    def test_captions_default_on(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        res = ar.add_caption(room.room_id, "hello from the stage", store=st)
        self.assertTrue(res["ok"])
        # turn off → captions refused
        ar.set_captions(room.room_id, False, "host1", store=st)
        self.assertFalse(ar.add_caption(room.room_id, "x", store=st)["ok"])

    def test_end_generates_transcript(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        ar.add_caption(room.room_id, "welcome everyone", store=st)
        res = ar.end_room(room.room_id, "host1", store=st)
        self.assertTrue(res["ok"])
        room = st.get(room.room_id)
        self.assertEqual(room.state, "ended")
        self.assertIn("welcome everyone", room.transcript)
        # with a real transcriber seam
        gs2, gid2, st2 = _fresh()
        room2 = _room(st2, gid2, gs=gs2)
        res2 = ar.end_room(room2.room_id, "host1",
                           transcriber=lambda ref: "full STT transcript here",
                           store=st2)
        self.assertTrue(res2["ok"])
        self.assertIn("STT transcript", st2.get(room2.room_id).transcript)


class TestTips(unittest.TestCase):
    def test_tip_without_processor_records_honestly(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        res = ar.tip_user(room.room_id, "u3", "Emeka", "host1", 50000, store=st)
        self.assertTrue(res["ok"])
        self.assertEqual(res["tip"].status, "recorded")
        self.assertIn("wired up", res["note"])

    def test_tip_with_processor(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        calls = []
        def fake_tip(from_id, to_id, amount_kobo, room_id):
            calls.append((from_id, to_id, amount_kobo))
            return {"ok": True}
        res = ar.tip_user(room.room_id, "u3", "Emeka", "host1", 150000,
                          tip_fn=fake_tip, store=st)
        self.assertTrue(res["ok"])
        self.assertEqual(res["tip"].status, "sent")
        self.assertEqual(calls, [("u3", "host1", 150000)])
        # failed processor
        res2 = ar.tip_user(room.room_id, "u3", "Emeka", "host1", 1000,
                           tip_fn=lambda **k: {"ok": False, "reason": "declined"},
                           store=st)
        self.assertFalse(res2["ok"])
        self.assertEqual(res2["tip"].status, "failed")

    def test_tip_validation(self):
        gs, gid, st = _fresh()
        room = _room(st, gid, gs=gs)
        self.assertFalse(ar.tip_user(room.room_id, "u3", "E", "host1", 0, store=st)["ok"])
        self.assertFalse(ar.tip_user(room.room_id, "", "E", "host1", 100, store=st)["ok"])
        ar.end_room(room.room_id, "host1", store=st)
        self.assertFalse(ar.tip_user(room.room_id, "u3", "E", "host1", 100, store=st)["ok"])


class TestChat(unittest.TestCase):
    def _chat(self, tail, sender="Ada", sender_id="host1", **kw):
        st = kw.pop("store", None)
        return ar.control_room(tail, sender=sender, sender_id=sender_id,
                               store=st, **kw)

    def test_usage(self):
        out = self._chat("")
        self.assertIn("/room create", out)

    def test_full_chat_cycle(self):
        gs = GroupStore(data_dir=tempfile.mkdtemp())
        g = gs.create_group("Devs", "coding", created_by="host1")
        st = ar.RoomStore(data_dir=tempfile.mkdtemp())
        out = self._chat(f"create {g.id} | Friday AMA", store=st, group_store=gs)
        self.assertIn("room is live", out)
        room_id = st.list()[0].room_id
        # another user joins + raises
        j = self._chat(f"join {room_id}", sender="Bola", sender_id="u2", store=st)
        self.assertIn("listener", j)
        r = self._chat(f"raise {room_id}", sender="Bola", sender_id="u2", store=st)
        self.assertIn("#1", r)
        # host promotes
        s = self._chat(f"speak {room_id} u2", store=st)
        self.assertIn("now speaking", s)
        # recording
        rec = self._chat(f"record {room_id} on", store=st)
        self.assertIn("recording", rec)
        # show renders REC
        show = self._chat(f"show {room_id}", store=st)
        self.assertIn("REC", show)
        # tip
        tip = self._chat(f"tip {room_id} u2 500", sender="Emeka", sender_id="u3", store=st)
        self.assertIn("₦500", tip)
        # end
        end = self._chat(f"end {room_id}", store=st)
        self.assertIn("room ended", end)

    def test_chat_garbage_never_raises(self):
        st = ar.RoomStore(data_dir=tempfile.mkdtemp())
        for tail in ["", "frobnicate", "create", "speak", "tip x y zzz",
                     "record x maybe", "mute", None]:
            out = ar.control_room(tail, sender_id="u", store=st)
            self.assertIsInstance(out, str)
            self.assertTrue(out)


class TestIsolation(unittest.TestCase):
    def test_no_owner_data_in_module(self):
        import ast
        from pathlib import Path
        src = Path(ar.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        for m in mods:
            self.assertFalse(m.startswith("nomorals.memory"), m)
            self.assertFalse(m.startswith("nomorals.accounts"), m)
            self.assertFalse(m.startswith("nomorals.connectors"), m)
            self.assertNotIn("vault", m)


if __name__ == "__main__":
    unittest.main()
