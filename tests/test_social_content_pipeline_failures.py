"""Tail-50 pins for nomorals/social/content_pipeline.py —
1 convert (518) / 6 exclusions / 1 flagged (467).

8 tests for 8 sites (counts equal). Each test pins one site
with a control half proving the guarded path was actually
exercised, asserting via the double's own call log that the
failing operation itself was attempted. The file contains
no exception handlers, so it adds zero census sites.

content_pipeline.py calls no failures-ledger record
function anywhere (grep-proven), so NOT-recording is pinned
against the social package ledger (nomorals/social/
failures.py, pass #150) with the tail-8 snapshot method:
recent_failures() before/after each trigger is unchanged.

Verdict notes (full grounds in the tail-50 SURVEY.md):
- 97: _engagement_of malformed JSON metrics -> 0.0, the
  genuine no-engagement value; consumers only average/rank.
- 137: optimal_windows unparseable posted_at row -> skip;
  no honest bucket exists for it, other rows still bucket.
- 296: _generate_drafts apply_voice failure -> pass; the
  genuine unshaped template draft stands, nothing claimed.
- 451: VoiceProfile.load failure -> VoiceProfile(), the
  documented default/no-profile value; load handles the
  routine corrupt-file case itself, so this is exotic-only.
- 467: FLAGGED (parent adjudication) — virality_score
  failure -> ViralityScore(50.0, ["scoring unavailable"]);
  the test pins OBSERVED behavior only: the reasons are
  discarded, drafts render/sort as a measured 50/"weak"
  and escape the weak flag (50 >= 40 threshold).
- 518: CONVERT — schedule_post failure -> blind q.schedule
  re-schedule raised on the scheduled->scheduled transition
  (schedule_post queues first), so scheduled_for stayed 0
  and the batch message falsely said "unscheduled" for
  queue-scheduled drafts. Fix: check the queue state and
  re-schedule only when not already scheduled.
- 546: _past_winners db failure -> None, the genuine
  no-winners value; the novelty input is simply off.
- 553: _past_winners unusable row -> skip; an unreadable
  row cannot honestly be ranked, other rows still rank.
"""

import json
import time

import nomorals.social.content_pipeline as CP
import nomorals.social.voice as voice
from nomorals.social import failures as soc_fail
from nomorals.social.drafts import DraftQueue

SAMPLE = [
    {"title": "AI agents ship real code now",
     "snippet": "New benchmarks show agents completing full PRs.",
     "url": "https://example.com/1"},
    {"title": "Local LLMs beat cloud on cost",
     "snippet": "A 4B model on your laptop vs $200/mo API bills.",
     "url": "https://example.com/2"},
]


def _snap():
    return list(soc_fail.recent_failures())


def _queue(tmp_path):
    return DraftQueue(db_path=str(tmp_path / "drafts.db"))


def _sub(tmp_path):
    d = tmp_path / "b"
    d.mkdir(exist_ok=True)
    return d


def test_97_malformed_metrics_json_is_zero_engagement(monkeypatch):
    before = _snap()
    loads_log = []
    orig_loads = json.loads

    def spy_loads(s, *a, **k):
        loads_log.append(s)
        return orig_loads(s, *a, **k)

    monkeypatch.setattr(json, "loads", spy_loads)
    assert CP._engagement_of("{not json") == 0.0
    assert loads_log[0] == "{not json"  # the parse was attempted
    assert CP._engagement_of('{"likes": 3, "replies": 2}') == 5.0
    assert CP._engagement_of({"likes": 3}) == 3.0
    assert _snap() == before


def test_137_bad_posted_at_row_skipped(monkeypatch):
    before = _snap()
    float_log = []

    class BadTS:
        def __float__(self):
            float_log.append("float")
            raise ValueError("bogus ts")

    class DB:
        def __init__(self, rows):
            self.rows = rows
            self.queries = []

        def query(self, sql, params=()):
            self.queries.append(sql)
            return [dict(r) for r in self.rows]

    good = {"posted_at": time.time() - 86400,
            "metrics": '{"likes": 10}'}
    bad = {"posted_at": BadTS(), "metrics": '{"likes": 999}'}
    db = DB([bad, good])
    windows, from_data = CP.optimal_windows(db, n=1)
    assert from_data is True and len(windows) == 1
    assert float_log == ["float"]  # float(posted_at) attempted
    assert db.queries  # the query itself ran
    windows2, from_data2 = CP.optimal_windows(DB([bad]), n=2)
    assert from_data2 is False and len(windows2) == 2
    _, from_data3 = CP.optimal_windows(DB([good]), n=1)
    assert from_data3 is True
    assert _snap() == before


def test_296_apply_voice_failure_keeps_template_draft(monkeypatch):
    before = _snap()
    av_log = []

    def bad_av(text, profile):
        av_log.append(text)
        raise RuntimeError("voice boom")

    monkeypatch.setattr(voice, "apply_voice", bad_av)
    drafts = CP._generate_drafts(SAMPLE, voice.VoiceProfile())
    assert av_log  # shaping was attempted
    assert len(drafts) >= 5
    assert drafts[0] == CP._template_draft(SAMPLE[0], 0)
    monkeypatch.setattr(voice, "apply_voice",
                        lambda text, profile: text + " [v]")
    marked = CP._generate_drafts(SAMPLE, voice.VoiceProfile())
    assert any("[v]" in d for d in marked)
    assert _snap() == before


def test_451_profile_load_failure_uses_default_profile(
        monkeypatch, tmp_path):
    before = _snap()
    load_log = []
    seen = []
    real_av = voice.apply_voice

    def bad_load(cls, path=None):
        load_log.append("load")
        raise RuntimeError("profile path boom")

    def capture_av(text, profile):
        seen.append(profile)
        return real_av(text, profile)

    monkeypatch.setattr(voice.VoiceProfile, "load",
                        classmethod(bad_load))
    monkeypatch.setattr(voice, "apply_voice", capture_av)
    res = CP.weekly_content("AI agents", findings=SAMPLE,
                            queue=_queue(tmp_path),
                            voice_profile=None)
    assert load_log == ["load"]  # load was attempted
    assert len(res.drafts) >= 5
    assert seen and all(
        p.n_posts == 0 and p.avg_sentence_len == 15.0 for p in seen)
    sentinel = voice.VoiceProfile(avg_sentence_len=99.0)
    seen.clear()
    monkeypatch.setattr(
        voice.VoiceProfile, "load",
        classmethod(lambda cls, path=None: sentinel))
    res2 = CP.weekly_content("AI agents", findings=SAMPLE,
                             queue=_queue(_sub(tmp_path)),
                             voice_profile=None)
    assert len(res2.drafts) >= 5
    assert seen and all(p is sentinel for p in seen)
    assert _snap() == before


def test_467_scoring_failure_observed_neutral_50(
        monkeypatch, tmp_path):
    """FLAGGED site: pins OBSERVED behavior only (see SURVEY)."""
    before = _snap()
    vs_log = []

    def bad_vs(text, platform="x", past_winners=None):
        vs_log.append(text)
        raise RuntimeError("scorer boom")

    monkeypatch.setattr(voice, "virality_score", bad_vs)
    res = CP.weekly_content("AI agents", findings=SAMPLE,
                            queue=_queue(tmp_path),
                            voice_profile=voice.VoiceProfile())
    assert vs_log  # scoring was attempted
    assert all(d.virality == 50.0 for d in res.drafts)
    assert all(d.virality_grade == "weak" for d in res.drafts)
    assert all(d.weak is False for d in res.drafts)
    assert "virality 50" in res.message
    monkeypatch.setattr(
        voice, "virality_score",
        lambda text, platform="x", past_winners=None:
            voice.ViralityScore(90.0, ["ok"]))
    res2 = CP.weekly_content("AI agents", findings=SAMPLE,
                             queue=_queue(_sub(tmp_path)),
                             voice_profile=voice.VoiceProfile())
    assert all(d.virality == 90.0 and d.virality_grade == "strong"
               for d in res2.drafts)
    assert _snap() == before


def test_518_schedule_post_failure_keeps_queue_schedule(
        monkeypatch, tmp_path):
    """CONVERT: the fallback must not re-schedule a draft that
    schedule_post already queued (its first action), which made
    scheduled_for stay 0 and the message claim "unscheduled"."""
    before = _snap()
    q = _queue(tmp_path)
    calls = []
    real_schedule = q.schedule

    def counting(draft_id, when_ts):
        calls.append(draft_id)
        return real_schedule(draft_id, when_ts)

    monkeypatch.setattr(q, "schedule", counting)

    class NoOnce:
        pass

    res = CP.weekly_content("AI agents", findings=SAMPLE, queue=q,
                            voice_profile=voice.VoiceProfile(),
                            scheduler=NoOnce())
    n = len(res.drafts)
    assert n >= 5
    # One queue schedule per draft (inside schedule_post); the
    # fallback saw the draft already scheduled and did not raise.
    assert len(calls) == n
    assert all(d.scheduled_for > 0 for d in res.drafts)
    assert "unscheduled" not in res.message
    d0 = q.get(res.drafts[0].draft_id)
    assert d0 is not None and d0.status == "scheduled"
    assert abs(d0.scheduled_time - res.drafts[0].scheduled_for) < 1.0

    q2 = _queue(_sub(tmp_path))
    calls2 = []
    real2 = q2.schedule

    def counting2(draft_id, when_ts):
        calls2.append(draft_id)
        return real2(draft_id, when_ts)

    monkeypatch.setattr(q2, "schedule", counting2)

    class GoodSched:
        def __init__(self):
            self.once = []

        def register_action(self, task_id, fn):
            pass

        def schedule_once(self, task_id, when_ts, action, params):
            self.once.append(task_id)
            return None

    gs = GoodSched()
    res2 = CP.weekly_content("AI agents", findings=SAMPLE, queue=q2,
                             voice_profile=voice.VoiceProfile(),
                             scheduler=gs)
    assert len(calls2) == len(res2.drafts)
    assert len(gs.once) == len(res2.drafts)
    assert all(d.scheduled_for > 0 for d in res2.drafts)
    assert _snap() == before


def test_546_past_winners_db_failure_is_none():
    before = _snap()
    q_log = []

    class DBFail:
        def query(self, sql, params=()):
            q_log.append(sql)
            raise RuntimeError("no social_posts")

    assert CP._past_winners(DBFail()) is None
    assert q_log and "social_posts" in q_log[0]

    class DBOk:
        def query(self, sql, params=()):
            return [
                {"content": "winner post", "metrics": '{"likes": 10}'},
                {"content": "quiet post", "metrics": '{"likes": 1}'},
            ]

    assert CP._past_winners(DBOk()) == ["winner post", "quiet post"]
    assert _snap() == before


def test_553_past_winners_bad_row_skipped():
    before = _snap()
    keys_log = []

    class BadRow:
        def keys(self):
            keys_log.append("keys")
            raise RuntimeError("row boom")

    class DB:
        def __init__(self, rows):
            self.rows = rows

        def query(self, sql, params=()):
            return list(self.rows)

    good = {"content": "good post", "metrics": '{"likes": 7}'}
    assert CP._past_winners(DB([BadRow(), good])) == ["good post"]
    assert keys_log == ["keys"]  # row access was attempted
    assert CP._past_winners(DB([good])) == ["good post"]
    assert _snap() == before
