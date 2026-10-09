"""Tests for the live DJ performer (nomorals/media/dj_live.py).

The live loop itself needs audio + composer; these tests cover the
parts that must be exactly right: room sensing, request parsing, the
persona, and set lifecycle with mocked delivery.
"""

import time

from nomorals.media import dj_live as live


def test_sensor_hype_detection():
    s = live.RoomSensor()
    s.note_message("🔥🔥🔥", sender="ada")
    s.note_message("this is FIRE", sender="bob")
    assert s.energy() > 0.1
    assert s.trend() in ("rising", "flat")


def test_sensor_dead_detection():
    s = live.RoomSensor()
    s.note_message("skip this boring track", sender="ada")
    assert s.dead_signals() >= 1


def test_sensor_request_parsing():
    s = live.RoomSensor()
    s.note_message("play some afrobeats please", sender="ada")
    assert s.pending_requests() == 1
    req = s.pop_request()
    assert req is not None
    assert "afrobeats" in req.query.lower()
    assert req.requester == "ada"
    assert s.pending_requests() == 0


def test_sensor_request_dedupe():
    s = live.RoomSensor()
    s.note_message("play amapiano", sender="ada")
    s.note_message("play amapiano", sender="ada")
    assert s.pending_requests() == 1


def test_sensor_silence_is_dead():
    s = live.RoomSensor()
    assert s.energy() == 0.0
    assert s.trend() == "dead"


def test_sensor_energy_decays():
    s = live.RoomSensor()
    s.note_message("🔥", sender="a", ts=time.time() - 290)
    e_old = s.energy()
    s2 = live.RoomSensor()
    s2.note_message("🔥", sender="a")
    assert s2.energy() >= e_old


def test_persona_talk_varies():
    p = live.DJPersona()
    lines = {p.talk(situation="hype", energy=0.9) for _ in range(6)}
    assert len(lines) >= 3  # not a fixed script


def test_persona_talk_references_context():
    p = live.DJPersona()
    line = p.talk(situation="request_yes", track_title="Midnight Oil",
                  track_style="afrobeats", requester="Ada", energy=0.8)
    assert "Ada" in line or "Midnight Oil" in line


def test_persona_never_repeats_back_to_back():
    p = live.DJPersona()
    # force the RNG path to collide by talking twice with same args
    a = p.talk(situation="outro", energy=0.5)
    b = p.talk(situation="outro", energy=0.5)
    # de-repeat guard appends a suffix on collision
    assert a != b or "for real this time" in b


def test_set_lifecycle():
    delivered = []
    said = []
    s = live.LiveSet(
        chat_key="test:1",
        deliver_audio=lambda p, c: delivered.append((p, c)),
        deliver_text=lambda t: said.append(t),
        context=None, workdir="/tmp/dj_live_test")
    assert not s.live
    msg = s.start(styles=["afrobeats"], n_tracks=3)
    assert "digging" in msg.lower() or "live" in msg.lower()
    # second start while live refuses
    assert "already live" in s.start()
    assert s.live
    # notify feeds the sensor
    s.notify("🔥🔥🔥", sender="ada")
    assert s.sensor.energy() > 0
    stop_msg = s.stop()
    assert "signing off" in stop_msg
    assert not s.live


def test_registry():
    live._SETS.clear()
    assert live.get_live_set("nope") is None
    assert live.stop_live_set("nope") == "no live DJ set in this chat"


def test_style_for_query():
    assert live.LiveSet._style_for_query("play some amapiano") == "amapiano"
    assert live.LiveSet._style_for_query("uk drill please") == "uk-drill"
    assert live.LiveSet._style_for_query("something chill") == "afrobeats"


def test_dj_character_identity():
    p = live.DJPersona()
    # One Vrede: the character bank's Vrede, not a separate "DJ Vrede".
    # DJ_NAME remains the stage name for announcements.
    assert p.character is None or p.character.name == "Vrede"
    assert live.DJ_NAME == "DJ Vrede"
