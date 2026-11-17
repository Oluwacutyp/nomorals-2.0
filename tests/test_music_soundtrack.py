"""Soundtrack organ tests: placements, trust model, fingerprint round-trip, wiring.

The organ is nomorals/music_sources/placements.py — the TuneFind core:
a placement relation (work -> episode/film -> scene -> song) with a
verified-vs-community trust model, seeded from public soundtrack
listings whose pairings were verified against live sources at build
time (Stranger Things S4, Sopranos finale, Pulp Fiction, Guardians,
Wayne's World, Reservoir Dogs, Titanic, Rocky III, Ghost, Saturday
Night Fever, The Bodyguard). Every seed carries provenance "curated"
and confidence 0.8 — labelled, never presented as supervisor-verified.
"""

import pytest

from nomorals.music_sources.placements import (
    SEED_PLACEMENTS,
    Placement,
    PlacementStore,
    format_placement_reply,
    lookup,
    parse_soundtrack_query,
    placements_for_clip,
    play_handoff,
)


@pytest.fixture()
def store():
    """A fresh in-memory store, seeded."""
    s = PlacementStore(":memory:")
    s.seed()
    return s


class TestSeeds:
    def test_seed_count_and_provenance(self, store):
        assert store.count() == len(SEED_PLACEMENTS)
        assert len(SEED_PLACEMENTS) >= 12
        for p in SEED_PLACEMENTS:
            assert p.provenance == "curated"
            assert 0.0 < p.confidence <= 0.8
            assert not p.is_verified

    def test_seed_is_idempotent(self, store):
        store.seed()
        assert store.count() == len(SEED_PLACEMENTS)


class TestEpisodeLookup:
    def test_stranger_things_s4e4(self, store):
        ans = lookup(store, "Stranger Things S4E4")
        assert ans["ok"] is True
        assert ans["work"] == "Stranger Things"
        assert ans["season"] == 4 and ans["episode"] == 4
        titles = [p["track_title"] for p in ans["placements"]]
        assert "Running Up That Hill (A Deal with God)" in titles
        bush = [p for p in ans["placements"]
                if p["track_title"].startswith("Running Up That Hill")][0]
        assert bush["track_artist"] == "Kate Bush"
        assert "Vecna" in bush["scene_note"]

    def test_stranger_things_s4e9(self, store):
        ans = lookup(store, "Stranger Things season 4 episode 9")
        assert ans["ok"] is True
        titles = [p["track_title"] for p in ans["placements"]]
        assert "Master of Puppets" in titles
        assert "Running Up That Hill (A Deal with God)" in titles

    def test_sopranos_finale(self, store):
        ans = lookup(store, "The Sopranos S6E21")
        assert ans["ok"] is True
        assert ans["placements"][0]["track_title"] == "Don't Stop Believin'"
        assert ans["placements"][0]["track_artist"] == "Journey"

    def test_film_lookup(self, store):
        ans = lookup(store, "Pulp Fiction")
        assert ans["ok"] is True
        assert ans["kind"] == "film"
        titles = [p["track_title"] for p in ans["placements"]]
        assert "Misirlou" in titles
        assert "Son of a Preacher Man" in titles

    def test_unknown_work_refuses_honestly(self, store):
        ans = lookup(store, "Some Show That Does Not Exist S1E1")
        assert ans["ok"] is False
        assert "Some Show That Does Not Exist" in ans["error"]
        assert "add" in ans["error"].lower()

    def test_known_work_unknown_episode_names_what_exists(self, store):
        ans = lookup(store, "Stranger Things S4E2")
        assert ans["ok"] is False
        assert "S4E2" in ans["error"] or "episode" in ans["error"].lower()
        assert ans.get("suggestions")


class TestQueryParser:
    def test_sxe_form(self):
        q = parse_soundtrack_query("Severance S2E3")
        assert q["work"] == "Severance"
        assert q["season"] == 2 and q["episode"] == 3

    def test_season_episode_words(self):
        q = parse_soundtrack_query("Stranger Things season 4 episode 4")
        assert q["work"] == "Stranger Things"
        assert q["season"] == 4 and q["episode"] == 4

    def test_plain_film_title(self):
        q = parse_soundtrack_query("Dune: Part Two")
        assert q["work"] == "Dune: Part Two"
        assert q["season"] is None and q["episode"] is None

    def test_prefix_stripped(self):
        q = parse_soundtrack_query("what song plays in Stranger Things S4E4")
        assert q["work"] == "Stranger Things"
        assert q["season"] == 4 and q["episode"] == 4


class TestReverseIndex:
    def test_song_to_works(self, store):
        works = store.works_for_track(
            "Running Up That Hill (A Deal with God)")
        assert works
        assert all(w["work_title"] == "Stranger Things" for w in works)
        episodes = {(w["season"], w["episode"]) for w in works}
        assert (4, 1) in episodes and (4, 4) in episodes and (4, 9) in episodes

    def test_song_to_works_with_artist(self, store):
        works = store.works_for_track("Come and Get Your Love", "Redbone")
        assert len(works) == 1
        assert works[0]["work_title"] == "Guardians of the Galaxy"
        assert works[0]["work_kind"] == "film"

    def test_unknown_song_empty(self, store):
        assert store.works_for_track("No Such Song Ever") == []

    def test_search_works(self, store):
        found = store.search_works("stranger")
        assert "Stranger Things" in found
        assert store.search_works("zzzz") == []


class TestTrustModel:
    def test_curated_label_never_claims_verified(self, store):
        ans = lookup(store, "Pulp Fiction")
        reply = format_placement_reply(ans)
        assert "not supervisor-verified" in reply
        assert "verified by the music supervisor" not in reply

    def test_community_entry_labelled_unconfirmed(self, store):
        store.add(Placement(
            work_id="the-wire", work_kind="tv", work_title="The Wire",
            season=1, episode=1, episode_title="The Target",
            scene_note="opening montage",
            track_title="Way Down in the Hole", track_artist="The Blind Boys of Alabama",
            provenance="community", confidence=0.5))
        ans = lookup(store, "The Wire S1E1")
        assert ans["ok"] is True
        assert "not yet confirmed" in format_placement_reply(ans)

    def test_community_confirms_flip_to_verified(self, store):
        store.add(Placement(
            work_id="the-wire", work_kind="tv", work_title="The Wire",
            season=1, episode=1, episode_title="The Target",
            scene_note="opening montage",
            track_title="Way Down in the Hole", track_artist="The Blind Boys of Alabama",
            provenance="community", confidence=0.5))
        for _ in range(3):
            updated = store.vote("the-wire", "Way Down in the Hole", confirm=True)
        assert updated is not None
        assert updated.is_verified
        assert updated.confidence > 0.5
        ans = lookup(store, "The Wire S1E1")
        assert "confirmed by 3 votes" in format_placement_reply(ans)

    def test_denies_pull_confidence_down(self, store):
        store.add(Placement(
            work_id="the-wire", work_kind="tv", work_title="The Wire",
            season=1, episode=1, episode_title="The Target",
            scene_note="opening montage",
            track_title="Way Down in the Hole", track_artist="The Blind Boys of Alabama",
            provenance="community", confidence=0.5))
        updated = store.vote("the-wire", "Way Down in the Hole", confirm=False)
        assert updated.confidence < 0.5
        assert not updated.is_verified

    def test_owner_add_is_from_own_watching(self, store):
        store.add(Placement(
            work_id="severance", work_kind="tv", work_title="Severance",
            season=2, episode=3, episode_title="Who Is Alive?",
            scene_note="closing credits",
            track_title="Unknown Song", track_artist="Unknown Artist",
            provenance="owner", confidence=0.95))
        ans = lookup(store, "Severance S2E3")
        assert ans["ok"] is True
        assert "your own watching" in format_placement_reply(ans)


class TestPlayHandoff:
    def test_handoff_names_real_chain(self):
        assert play_handoff("Running Up That Hill (A Deal with God)", "Kate Bush") == (
            "/play Running Up That Hill (A Deal with God) — Kate Bush")

    def test_reply_carries_play_line_per_song(self, store):
        reply = format_placement_reply(lookup(store, "Pulp Fiction"))
        assert "/play Misirlou — Dick Dale" in reply
        assert "/play Son of a Preacher Man — Dusty Springfield" in reply


class TestFingerprintRoundTrip:
    def test_clip_to_placements(self, store):
        def fake_identify(path):
            assert path == "/tmp/clip.wav"
            return {"ok": True, "title": "Running Up That Hill (A Deal with God)",
                    "artist": "Kate Bush", "score": 0.9}

        ans = placements_for_clip(store, "/tmp/clip.wav", identify_fn=fake_identify)
        assert ans["ok"] is True
        assert ans["track"] == "Running Up That Hill (A Deal with God)"
        assert ans["artist"] == "Kate Bush"
        assert ans["placements"]
        assert ans["placements"][0]["work_title"] == "Stranger Things"

    def test_clip_no_match_refuses_honestly(self, store):
        ans = placements_for_clip(
            store, "/tmp/clip.wav",
            identify_fn=lambda p: {"ok": False, "reason": "no match in library"})
        assert ans["ok"] is False
        assert "no match" in ans["error"]

    def test_clip_identified_but_no_placements(self, store):
        ans = placements_for_clip(
            store, "/tmp/clip.wav",
            identify_fn=lambda p: {"ok": True, "title": "Never Gonna Give You Up",
                                   "artist": "Rick Astley", "score": 0.9})
        assert ans["ok"] is False
        assert "Never Gonna Give You Up" in ans["error"]
        assert "add" in ans["error"].lower()

    def test_no_identifier_refuses_honestly(self, store):
        ans = placements_for_clip(store, "/tmp/eddie.wav")
        assert ans["ok"] is False
        assert "identify" in ans["error"]

    def test_chat_clip_calls_audio_fingerprint(self, tmp_path, monkeypatch):
        import importlib
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        seen = {}

        def fake_identify(path, **kwargs):
            seen["path"] = path
            return {"ok": True, "title": "Master of Puppets",
                    "artist": "Metallica", "score": 0.8}

        # NOTE: nomorals.audio.fingerprint is shadowed on the package
        # by the fingerprint() function itself — patch the real module
        # object from sys.modules.
        fp_module = importlib.import_module("nomorals.audio.fingerprint")
        monkeypatch.setattr(fp_module, "identify", fake_identify)
        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        reply = runtime._control_soundtrack("clip /tmp/eddie.wav")
        assert seen["path"] == "/tmp/eddie.wav"
        assert "Stranger Things" in reply
        assert "Master of Puppets" in reply


class TestChatRegistration:
    def test_soundtrack_command_registered(self):
        from nomorals.social.chat.control import (
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
            LIST_GROUPS,
            LIST_ONELINERS,
            PUBLIC_COMMANDS,
            caller_may_run,
            list_catalog,
        )

        assert "soundtrack" in CONTROL_COMMANDS
        assert CONTROL_COMMANDS["soundtrack"] == (1, None)
        assert "soundtrack" in PUBLIC_COMMANDS
        assert len(PUBLIC_COMMANDS) == 274
        assert "soundtrack" in COMMAND_DETAILS
        assert "/soundtrack" in COMMAND_DETAILS["soundtrack"]["usage"]
        assert "soundtrack" in LIST_ONELINERS
        assert "soundtrack" in [c for _, kinds in LIST_GROUPS for c in kinds]
        assert [c for _, kinds in LIST_GROUPS for c in kinds].count("soundtrack") == 1
        assert "soundtrack" in list_catalog()
        assert caller_may_run("soundtrack", False) is True

    def test_soundtrack_handler_answers_and_never_raises(self, tmp_path):
        from types import SimpleNamespace

        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

        runtime = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
        runtime.settings = SimpleNamespace(home_path=str(tmp_path))
        handler = runtime._control_soundtrack
        assert "Kate Bush" in handler("Stranger Things S4E4")
        assert "usage" in handler("").lower()
        assert "not in the catalogue" in handler("Some Show That Does Not Exist S1E1")


class TestCli:
    @staticmethod
    def _context(tmp_path):
        from types import SimpleNamespace

        return SimpleNamespace(
            settings=SimpleNamespace(home_path=str(tmp_path)))

    def test_nm_soundtrack_answers(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.soundtrack import _cmd_soundtrack

        args = argparse.Namespace(soundtrack_query=["Stranger Things", "S4E4"],
                                  json=False)
        assert _cmd_soundtrack(args, self._context(tmp_path)) == 0
        assert "Kate Bush" in capsys.readouterr().out

    def test_nm_soundtrack_unknown_exits_1(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.soundtrack import _cmd_soundtrack

        args = argparse.Namespace(
            soundtrack_query=["Some Show That Does Not Exist", "S1E1"],
            json=False)
        assert _cmd_soundtrack(args, self._context(tmp_path)) == 1
        assert "not in the catalogue" in capsys.readouterr().out

    def test_nm_soundtrack_song_reverse(self, capsys, tmp_path):
        import argparse

        from nomorals.cmdline.commands.soundtrack import _cmd_soundtrack

        args = argparse.Namespace(
            soundtrack_query=["song", "Come and Get Your Love"],
            json=False)
        assert _cmd_soundtrack(args, self._context(tmp_path)) == 0
        assert "Guardians of the Galaxy" in capsys.readouterr().out
