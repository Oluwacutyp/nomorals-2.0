"""Mock screens Phase 3 (FakeDetail port, census §27):
social posts, profiles, stories, comments and video
pages — Facebook / Instagram / Threads / X / TikTok
posts, Facebook / Instagram / X / TikTok profiles and
the YouTube channel, the Instagram Story, the TikTok
comment sheet and the YouTube video page.

Network-free: renders are local Pillow work, media
fixtures are tiny generated files, and the delivery seam
is a recording stub — the phase-2b suite's idiom.
"""
from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.social.chat.mock_screens import (
    SOCIAL_DESIGNS,
    CommentScreen,
    DeviceChrome,
    MockScreensError,
    MockScreenService,
    PostScreen,
    ProfileCount,
    ProfileItem,
    ProfileScreen,
    SocialProfile,
    StoryScreen,
    StorySticker,
    VideoScreen,
    shorten_count,
)
from nomorals.social.chat.mock_screens.authoring import (
    demo_comment_screen,
    demo_post_screen,
    demo_profile_screen,
    demo_story_screen,
    demo_video_screen,
    parse_mock_tail,
)
from nomorals.social.chat.mock_screens.render_social import (
    render_comment_screen,
    render_post_screen,
    render_profile_screen,
    render_story_screen,
    render_video_screen,
)

# ── fixtures ─────────────────────────────────────────────


def _png_bytes(color=(30, 120, 200), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _solid_png(path, color=(200, 30, 30), size=(300, 200)):
    Image.new("RGB", size, color).save(path)
    return path


class _Send:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return SimpleNamespace(ok=self.ok, error="")


def _author():
    return SocialProfile("adaeze", "Adaeze Nwosu",
                         handle="adaeze.creates", verified=True)


def _el(drawn, **want):
    for entry in drawn["elements"]:
        if all(entry.get(k) == v for k, v in want.items()):
            return entry
    raise AssertionError("no element " + repr(want)
                         + " in " + repr(drawn["elements"]))


def _els(drawn, **want):
    return [e for e in drawn["elements"]
            if all(e.get(k) == v for k, v in want.items())]


# ── census accounting (census §27 build list) ────────────

#: every Phase 3 tool the census names, mapped to the
#: registry that renders it
CENSUS_RENDERED = {
    "Facebook Post": "social:facebook",
    "Facebook Profile": "social:facebook",
    "Instagram Post": "social:instagram",
    "Instagram Profile": "social:instagram",
    "Instagram Threads Post": "social:threads",
    "Instagram Story": "social:instagram",
    "X Tweet": "social:x",
    "X Profile": "social:x",
    "TikTok Post": "social:tiktok",
    "TikTok Profile": "social:tiktok",
    "TikTok Comment": "social:tiktok",
    "YouTube Video Post": "social:youtube",
    "YouTube Channel": "social:youtube",
}

#: census-named scope held back, with its census reason —
#: never filled by guessing
CENSUS_REMAINDERS = {
    "Instagram Story's five text styles":
        "census §5.4 names five text styles without "
        "enumerating them — no style vocabulary is "
        "invented; story text wears the design's default "
        "style",
    "YouTube live-chat / Ask / community / Hype cards":
        "census §5.9 names the cards without detailing "
        "them — the video page draws the verified "
        "elements only",
    "TikTok Post's detailed layout controls":
        "census §5.7 records the elements from earlier "
        "research without re-inspecting the layout — "
        "the post rides the shared post anatomy",
    "The other eleven of the homepage's 24 social tools":
        "census §5.1: the category claims twenty-four, "
        "the extraction names thirteen — the remainder "
        "must not be invented",
    "Story / list video export":
        "census §5.4: recording is Phase 4 of the same "
        "organ, not this build",
    "Receipts of any kind":
        "owner 2026-11-11: the receipts family is "
        "dropped entirely — never built in any form",
}


def test_census_phase3_catalogue_is_accounted():
    for target in CENSUS_RENDERED.values():
        kind, key = target.split(":")
        assert kind == "social"
        assert key in SOCIAL_DESIGNS, target
    assert set(SOCIAL_DESIGNS) == {
        "facebook", "instagram", "threads", "x", "tiktok",
        "youtube"}
    for reason in CENSUS_REMAINDERS.values():
        assert reason.startswith(("census §", "owner "))


def test_phase3_palettes_are_complete_per_design():
    for name, design in SOCIAL_DESIGNS.items():
        assert set(design["light"]) \
            == set(design["dark"]), name
    keysets = {frozenset(d["light"])
               for d in SOCIAL_DESIGNS.values()}
    assert len(keysets) == 1


# ── counts ───────────────────────────────────────────────


@pytest.mark.parametrize("value,style,want", [
    (0, "intl", "0"),
    (999, "intl", "999"),
    (1000, "intl", "1K"),
    (1240, "intl", "1.2K"),
    (25400, "intl", "25.4K"),
    (1200000, "intl", "1.2M"),
    (4500000, "intl", "4.5M"),
    (1200000000, "intl", "1.2B"),
    (999, "indian", "999"),
    (99999, "indian", "100K"),
    (100000, "indian", "1 Lakh"),
    (150000, "indian", "1.5 Lakh"),
    (2540000, "indian", "25.4 Lakh"),
    (10000000, "indian", "1 Crore"),
    (12000000, "indian", "1.2 Crore"),
])
def test_shorten_count_boundaries(value, style, want):
    assert shorten_count(value, style) == want


def test_shorten_count_rejects_unknown_style():
    with pytest.raises(ValueError):
        shorten_count(100, "roman")


# ── schema ───────────────────────────────────────────────


def test_phase3_schema_round_trips_are_lossless():
    post = demo_post_screen("facebook-post", "android")
    assert PostScreen.from_dict(post.to_dict()).to_dict() \
        == post.to_dict()
    profile = demo_profile_screen("tiktok-profile", "android")
    assert ProfileScreen.from_dict(
        profile.to_dict()).to_dict() == profile.to_dict()
    story = demo_story_screen("instagram-story", "android")
    assert StoryScreen.from_dict(story.to_dict()).to_dict() \
        == story.to_dict()
    comments = demo_comment_screen("android")
    assert CommentScreen.from_dict(
        comments.to_dict()).to_dict() == comments.to_dict()
    video = demo_video_screen("youtube-video", "android")
    assert VideoScreen.from_dict(video.to_dict()).to_dict() \
        == video.to_dict()


def test_phase3_schema_rejects_states_the_apps_lack():
    with pytest.raises(ValueError):
        PostScreen("facebook", _author(), visibility="secret")
    with pytest.raises(ValueError):
        ProfileCount("followers", 1, style="roman")
    with pytest.raises(ValueError):
        ProfileItem("x", kind="hologram")
    with pytest.raises(ValueError):
        StoryScreen(kind="video")
    with pytest.raises(ValueError):
        StoryScreen(bg="purple")
    with pytest.raises(ValueError):
        StorySticker("confetti")
    with pytest.raises(ValueError):
        VideoScreen(layout="live")
    with pytest.raises(ValueError):
        VideoScreen(count_style="roman")


def test_story_progress_model_clamps_to_the_census():
    screen = StoryScreen(
        segments=3, index=9, played=4.5)
    assert screen.index == 2
    assert screen.played == 1.0


def test_story_sticker_positions_clamp_to_fractions():
    sticker = StorySticker("mention", "adaeze", x=2.0,
                           y=-1.0)
    assert sticker.x == 1.0
    assert sticker.y == 0.0


# ── post renders ─────────────────────────────────────────


@pytest.mark.parametrize("name,style,width", [
    ("facebook-post", "android", 800),
    ("facebook-post", "iphone", 780),
    ("instagram-post", "android", 800),
    ("threads-post", "iphone", 780),
    ("x-tweet", "android", 800),
    ("tiktok-post", "android", 800),
])
def test_post_demos_are_real_images(name, style, width):
    screen = demo_post_screen(name, style)
    data, drawn = render_post_screen(screen)
    img = Image.open(io.BytesIO(data))
    assert img.size[0] == width
    assert img.size[1] > 200
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    author = _el(drawn, kind="post_author")
    assert author["verified"] is True
    assert author["avatar_source"] == "placeholder"
    engagement = _el(drawn, kind="post_engagement")
    assert engagement["likes"] == screen.likes
    assert engagement["comments"] == screen.comments
    assert engagement["shares"] == screen.shares
    again, _d = render_post_screen(screen)
    assert again == data


def test_post_media_and_video_markers_are_recorded():
    fb = demo_post_screen("facebook-post", "android")
    _data, drawn = render_post_screen(fb)
    assert _els(drawn, kind="post_media") == []
    tk = demo_post_screen("tiktok-post", "android")
    _data, drawn = render_post_screen(tk)
    media = _el(drawn, kind="post_media")
    assert media["video"] is True
    assert media["media_source"] == "placeholder"


def test_post_visibility_rides_the_record():
    fb = demo_post_screen("facebook-post", "android")
    assert fb.visibility == "public"
    ig = demo_post_screen("instagram-post", "android")
    assert ig.visibility == ""


# ── profile renders ──────────────────────────────────────


@pytest.mark.parametrize("name,style,width", [
    ("facebook-profile", "android", 800),
    ("instagram-profile", "android", 800),
    ("instagram-profile", "iphone", 780),
    ("x-profile", "android", 800),
    ("tiktok-profile", "android", 800),
    ("youtube-channel", "android", 800),
    ("youtube-channel", "iphone", 780),
])
def test_profile_demos_are_real_images(name, style, width):
    screen = demo_profile_screen(name, style)
    data, drawn = render_profile_screen(screen)
    img = Image.open(io.BytesIO(data))
    assert img.size[0] == width
    assert img.size[1] > 300
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    avatar = _el(drawn, kind="profile_avatar")
    assert avatar["avatar_source"] == "placeholder"
    again, _d = render_profile_screen(screen)
    assert again == data


def test_profile_grids_preserve_data_order_and_pins():
    # census §27 exit proof: user-defined order is the
    # drawn order, and TikTok's pinned videos come first
    tiktok = demo_profile_screen("tiktok-profile", "android")
    _data, drawn = render_profile_screen(tiktok)
    items = _els(drawn, kind="profile_item")
    assert [i["id"] for i in items] == \
        [i.id for i in tiktok.items]
    pinned = [i["pinned"] for i in items]
    assert pinned == [True, True, False, False, False, False]
    ig = demo_profile_screen("instagram-profile", "android")
    _data, drawn = render_profile_screen(ig)
    items = _els(drawn, kind="profile_item")
    assert len(items) == 6
    assert all(i["item_kind"] == "photo" for i in items)


def test_youtube_profile_has_video_rows_and_shorts():
    yt = demo_profile_screen("youtube-channel", "android")
    _data, drawn = render_profile_screen(yt)
    items = _els(drawn, kind="profile_item")
    kinds = [i["item_kind"] for i in items]
    assert kinds == ["video", "video", "video",
                     "short", "short"]
    assert "cover" in drawn["regions"]


def test_facebook_profile_has_cover_and_post_rows():
    fb = demo_profile_screen("facebook-profile", "android")
    _data, drawn = render_profile_screen(fb)
    assert "cover" in drawn["regions"]
    items = _els(drawn, kind="profile_item")
    assert [i["item_kind"] for i in items] == ["post", "post"]


def test_profile_counts_follow_the_design_style():
    # YouTube's design shortens Indian-style; the demo
    # leaves count_style blank, so the design governs
    yt = demo_profile_screen("youtube-channel", "android")
    assert shorten_count(1200000, "indian") == "12 Lakh"
    assert yt.count_style == ""
    from nomorals.social.chat.mock_screens.designs import (
        SOCIAL_DESIGNS,
    )
    assert SOCIAL_DESIGNS["youtube"]["count_style"] \
        == "indian"


# ── story renders ────────────────────────────────────────


@pytest.mark.parametrize("name,style,width,height", [
    ("instagram-story", "android", 800, 1600),
    ("instagram-story", "iphone", 780, 1688),
    ("instagram-story-text", "android", 800, 1600),
])
def test_story_demos_are_full_screen_images(
        name, style, width, height):
    screen = demo_story_screen(name, style)
    data, drawn = render_story_screen(screen)
    img = Image.open(io.BytesIO(data))
    assert img.size == (width, height)
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert drawn["progress"]["segments"] == screen.segments
    assert drawn["progress"]["index"] == screen.index
    assert drawn["progress"]["played"] == screen.played
    again, _d = render_story_screen(screen)
    assert again == data


def test_story_sticker_positions_are_in_the_record():
    story = demo_story_screen("instagram-story", "android")
    _data, drawn = render_story_screen(story)
    stickers = _els(drawn, kind="story_sticker")
    assert [s["sticker_kind"] for s in stickers] == \
        ["mention", "poll", "music"]
    by_kind = {s["sticker_kind"]: s for s in stickers}
    assert by_kind["mention"]["x"] == 0.5
    assert by_kind["mention"]["y"] == 0.32
    assert by_kind["poll"]["options"] == ["Yes", "No"]
    assert by_kind["music"]["text"] == "Midnight City"
    assert "reply_bar" in drawn["regions"]


def test_story_text_slide_has_no_media():
    story = demo_story_screen("instagram-story-text",
                              "android")
    _data, drawn = render_story_screen(story)
    assert drawn["media_source"] == "none"
    assert _el(drawn, kind="story_text")["text"] \
        == story.text


# ── comment renders ──────────────────────────────────────


def test_comment_demo_is_a_real_image():
    screen = demo_comment_screen("android")
    data, drawn = render_comment_screen(screen)
    img = Image.open(io.BytesIO(data))
    assert img.size[0] == 800
    assert img.size[1] > 400
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    header = _el(drawn, kind="comment_header")
    assert header["total"] == 128
    assert header["shown"] == 3
    again, _d = render_comment_screen(screen)
    assert again == data


def test_comment_creator_marks_are_data_not_paint():
    # census §27 exit proof: replies and creator marks
    # are represented as data
    screen = demo_comment_screen("android")
    _data, drawn = render_comment_screen(screen)
    creator = _el(drawn, id="c1")
    assert creator["creator"] is True
    assert creator["liked_by_creator"] is False
    liked = _el(drawn, id="c2")
    assert liked["liked_by_creator"] is True
    reply = _el(drawn, id="c1r1")
    assert reply["reply_to"] == "c1"
    more = _el(drawn, kind="comment_more")
    assert more["view_more"] == 2  # 12 replies, 10 shown


# ── video renders ────────────────────────────────────────


@pytest.mark.parametrize("name,style,width", [
    ("youtube-video", "android", 800),
    ("youtube-video", "iphone", 780),
    ("youtube-video-comments", "android", 800),
    ("youtube-video-description", "android", 800),
])
def test_video_demos_are_real_images(name, style, width):
    screen = demo_video_screen(name, style)
    data, drawn = render_video_screen(screen)
    img = Image.open(io.BytesIO(data))
    assert img.size[0] == width
    assert img.size[1] > 500
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert _el(drawn, kind="video_player")["media_source"] \
        == "placeholder"
    channel = _el(drawn, kind="video_channel")
    assert channel["subscribed"] is False
    again, _d = render_video_screen(screen)
    assert again == data


def test_video_layouts_draw_their_own_tail():
    watch = demo_video_screen("youtube-video", "android")
    _data, drawn = render_video_screen(watch)
    assert len(_els(drawn, kind="video_up_next")) == 3
    assert _els(drawn, kind="video_description")[0] \
        ["shortened"] is True
    comments = demo_video_screen("youtube-video-comments",
                                 "android")
    _data, drawn = render_video_screen(comments)
    assert _els(drawn, kind="video_up_next") == []
    assert _el(drawn, id="vc1")["text"].startswith(
        "The zone held")
    desc = demo_video_screen("youtube-video-description",
                             "android")
    _data, drawn = render_video_screen(desc)
    assert _els(drawn, kind="video_description")[0] \
        ["shortened"] is False


def test_video_subscribed_state_wears_the_plain_button():
    screen = demo_video_screen("youtube-video", "android")
    screen.subscribed = True
    _data, drawn = render_video_screen(screen)
    assert _el(drawn, kind="video_channel")["subscribed"] \
        is True


# ── media honesty (the hand-off contract) ────────────────


def test_broken_post_photo_fails_with_the_reason(tmp_path):
    junk = tmp_path / "post.png"
    junk.write_text("not an image")
    screen = PostScreen(
        "facebook", _author(), text="hi", media=str(junk),
        chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError) as caught:
        render_post_screen(screen)
    assert "could not be used" in str(caught.value)
    service = MockScreenService(media_dir=tmp_path)
    out = service.render_post(screen)
    assert out["sent"] is False
    assert "could not be used" in out["reason"]


def test_broken_cover_and_thumbnail_fail_the_render(
        tmp_path):
    junk = tmp_path / "cover.png"
    junk.write_text("not an image")
    profile = ProfileScreen(
        "youtube",
        SocialProfile("gs", "GoldenSwing Signals"),
        cover_media=str(junk),
        chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError):
        render_profile_screen(profile)
    video = VideoScreen(
        thumbnail=str(junk), title="t",
        chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError):
        render_video_screen(video)
    story = StoryScreen(
        media=str(junk), chrome=DeviceChrome(style="android"))
    with pytest.raises(MockScreensError):
        render_story_screen(story)


def test_real_avatar_and_post_photo_are_composited(tmp_path):
    face = _solid_png(tmp_path / "face.png", (10, 180, 90),
                      size=(80, 80))
    photo = _solid_png(tmp_path / "photo.png", (200, 30, 30),
                       size=(300, 200))
    screen = PostScreen(
        "facebook",
        SocialProfile("adaeze", "Adaeze Nwosu", avatar=str(face)),
        text="hi", media=str(photo),
        chrome=DeviceChrome(style="android"))
    data, drawn = render_post_screen(screen)
    author = _el(drawn, kind="post_author")
    assert author["avatar_source"] == "real"
    media = _el(drawn, kind="post_media")
    assert media["media_source"] == "real"
    pixels = Image.open(io.BytesIO(data)).convert("RGB")
    # the photo frame's centre IS the source's colour
    _x, y, _w, h = media["box"]
    assert pixels.getpixel((400, y + h // 2)) == (200, 30, 30)


def test_supplied_avatar_via_handoff_is_strict(tmp_path):
    screen = demo_post_screen("facebook-post", "android")
    out = MockScreenService(media_dir=tmp_path).render_post(
        screen, avatars={"adaeze": b"not an image"})
    assert out["sent"] is False
    assert "could not be used" in out["reason"]
    good = MockScreenService(media_dir=tmp_path).render_post(
        screen, avatars={"adaeze": _png_bytes((10, 180, 90))})
    author = _el(good["drawn"], kind="post_author")
    assert author["avatar_source"] == "real"


# ── service + command reachability ───────────────────────


def test_service_delivers_each_phase3_class(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "m")
    cases = [
        (service.render_post,
         demo_post_screen("facebook-post", "android")),
        (service.render_profile,
         demo_profile_screen("instagram-profile", "android")),
        (service.render_story,
         demo_story_screen("instagram-story", "android")),
        (service.render_comment, demo_comment_screen("android")),
        (service.render_video,
         demo_video_screen("youtube-video", "android")),
    ]
    for door, screen in cases:
        send = _Send()
        out = door(screen, audience="personal", send=send)
        assert out["sent"] is True
        ref, caption = send.calls[0]
        assert ref.kind == "image" and ref.mime == "image/png"
        assert Path(ref.path).exists()
        assert "mock screen" in caption


def test_parse_dispatch_modes():
    cases = {
        "demo facebook-post": "demo-post",
        "demo instagram-post": "demo-post",
        "demo threads-post": "demo-post",
        "demo x-tweet": "demo-post",
        "demo tiktok-post": "demo-post",
        "demo facebook-profile": "demo-profile",
        "demo instagram-profile": "demo-profile",
        "demo x-profile": "demo-profile",
        "demo tiktok-profile": "demo-profile",
        "demo youtube-channel": "demo-profile",
        "demo instagram-story": "demo-story",
        "demo instagram-story-text": "demo-story",
        "demo tiktok-comment": "demo-comment",
        "demo youtube-video": "demo-video",
        "demo youtube-video-comments": "demo-video",
        "demo youtube-video-description": "demo-video",
    }
    for tail, mode in cases.items():
        _obj, got = parse_mock_tail(tail)
        assert got == mode, tail
    with pytest.raises(MockScreensError):
        parse_mock_tail("demo myspace-post")


class _Runtime(RuntimeMockScreensMixin):
    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _Send()

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


@pytest.mark.parametrize("tail", [
    "demo facebook-post", "demo instagram-post",
    "demo threads-post", "demo x-tweet", "demo tiktok-post",
    "demo facebook-profile", "demo instagram-profile",
    "demo x-profile", "demo tiktok-profile",
    "demo youtube-channel", "demo instagram-story",
    "demo instagram-story-text", "demo tiktok-comment",
    "demo youtube-video", "demo youtube-video-comments",
    "demo youtube-video-description",
])
def test_command_delivers_each_phase3_demo(tmp_path, tail):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock("mock", tail, _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "image"
    assert Path(ref.path).exists()
    assert "mock screen" in caption


def test_command_phase3_stays_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "demo facebook-post", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_chat_renders_untouched_by_the_new_classes():
    # the phase-1 regression contract lives in the older
    # suite; here the seam fact: adding the classes did
    # not change the chat registry's contents
    from nomorals.social.chat.mock_screens import (
        CHAT_DESIGNS,
    )
    from nomorals.social.chat.mock_screens.authoring import (
        demo_screen,
    )
    from nomorals.social.chat.mock_screens.render import (
        render_screen,
    )

    assert set(CHAT_DESIGNS) == {
        "whatsapp", "instagram", "telegram", "messenger",
        "snapchat", "imessage", "sms"}
    _data, drawn = render_screen(
        demo_screen("whatsapp", "android"))
    assert drawn["platform"] == "whatsapp"
