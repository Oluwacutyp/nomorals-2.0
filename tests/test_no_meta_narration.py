"""No meta-narration (owner rule, 2026-11-01, group build P0 sweep):
Devon never sends a second message narrating her own action — the
artifact IS the message. Status text only when the user asked for a
report or something failed, plainly, once.

The sweep silenced the success receipts at every send site found:
_control_tts ("voice note sent"), _deliver_voice_note ("said it"),
_control_decode ("(file sent)"), the /play send line ("▶ sent … to
this chat"), _control_caption ("captioned video sent"), /ekiti say
("reference audio sent"), /proxy's "sent the clean file" lead-in and
/artifact send's "📎 sent" line — with their pins updated in place.
Failure and degradation texts at all of those sites are unchanged:
they are information, not narration.

This file pins the background-download delta contract the sweep
created: when the job body's whole text is what the ack already
showed, the follow-up delta is EMPTY — no follow-up posts, and the
job records delivered=True (see _play_download_bg) so the harvester
never invents a "report never reached this chat" failure for a
delivery that landed silently on purpose.
"""
from __future__ import annotations

from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

_delta = RuntimeMediaMixin._play_job_delta_text


def test_delta_empty_when_body_text_is_the_ack_text():
    assert _delta("queued 1:\n  - Song", "queued 1:\n  - Song") == ""


def test_delta_carries_a_real_outcome_past_the_prefix():
    out = _delta("queued 1:\n  - Song",
                 "queued 1:\n  - Song\ncouldn't download the audio: x")
    assert out == "couldn't download the audio: x"


def test_delta_without_prefix_returns_the_whole_text():
    assert _delta("queued 1", "some other outcome") == "some other outcome"


def test_delta_empty_prefix_returns_the_whole_text():
    assert _delta("", "outcome") == "outcome"
