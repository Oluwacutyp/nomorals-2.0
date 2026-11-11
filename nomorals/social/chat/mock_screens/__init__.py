"""Mock screens: FakeDetail-style platform screen generators.

A transport-free L4 organ beside ``makers``.  A screen is a
typed model first (``schema``), drawn second (``render``)
through platform designs held as data (``designs``) — the
makers doctrine: designs are data, one layout engine per
screen class, typography through the typography service,
avatars through the makers mask/placeholder machinery, and a
drawn-region record returned with the pixels so tests prove
structure, not vibes.

Phase 0+1: the schema, the design registry, the Pillow
renderer foundation, and the flagship chats — WhatsApp 1:1,
WhatsApp group and Instagram DM, each in iPhone and Android
chrome.  Phase 2 (this build): the remaining chat families
enter the same registry — Telegram, Messenger, Snapchat,
iMessage (incl. the green-SMS Apple-to-Android variant)
and Android Messages, 1:1 and group — each family's
choreography carried as design data (where times and ticks
live, bubble tails, run avatars, read-state lines, input
bars).  Chat tools the census names but does not inspect
to flagship depth stay named remainders (see the census
§3.10 note in the tests) rather than guessed designs.
The media hand-off: image and voice messages carry REAL
media — a generated image or any image bytes compose
into the photo frame, a real voice note is drawn with
its measured waveform and true duration (``media``
probes in :mod:`.media`) — and the placeholder scene /
synthetic waveform remain only for messages carrying no
media.  Phase 2b: the lists / status / calls classes —
WhatsApp, Instagram, Telegram and Snapchat chat lists,
the WhatsApp call list and status (Updates) list, the
WhatsApp Status screen and its Views sheet, the generic
+ named-platform call screens, and the iPhone Call
History — each a typed model in ``schema``, designs as
data in ``designs`` (``LIST_DESIGNS`` /
``STATUS_DESIGNS`` / ``CALL_DESIGNS``), and its own
layout engine in ``render_screens`` under the same
media hand-off contract.  Phase 3 (this build): the
social classes — Facebook / Instagram / Threads / X /
TikTok posts, Facebook / Instagram / X / TikTok
profiles and the YouTube channel, the Instagram Story,
the TikTok comment sheet and the YouTube video page —
each a typed model in ``schema``, designs as data in
``designs`` (``SOCIAL_DESIGNS``), and its own layout
engine in ``render_social`` under the same media
hand-off contract.  Phase 4 (this build): the
animation organ (``animation``) — a data timeline per
screen class (chat reveal + typing + true-length voice
playback, list row arrival, status/story segment fill,
call timer, page scroll) rendered through the same
engines and piped to ffmpeg by Motion Studio's frame
pipe; the video ends on exactly the previewed screen.
Phase 5 (this build): bulk test data — a seeded
generator of obviously synthetic personal / identity
test data (``demo_data``) that populates whole screens
from one seed + count (``bulk_list_screen`` /
``bulk_call_history`` / ``bulk_chat_screen`` /
``bulk_status_views``) and exports records as text /
JSON / CSV / XML; the ``/mock demo bulk-*`` and
``/mock data`` forms ride it.  NL authoring is a later
phase of the same organ.

The organ never claims a render is a real captured
screenshot: it is a mock screen, generated from data the
caller supplied.
"""

from __future__ import annotations

from .animation import (
    Clip,
    Recording,
    RecordingError,
    TimelineSpec,
    build_call_recording,
    build_chat_recording,
    build_comment_recording,
    build_hold_recording,
    build_list_recording,
    build_profile_recording,
    build_status_recording,
    build_story_recording,
    build_video_recording,
    record,
)
from .demo_data import (
    COUNTRIES,
    CREATIVE_FAMILIES,
    MAIL_PROVIDERS,
    MAX_BULK_DATA,
    MAX_BULK_SCREEN_ROWS,
    REFERENCE_DATE,
    BulkGenerator,
    DemoDataError,
    bulk_call_history,
    bulk_chat_screen,
    bulk_list_screen,
    bulk_status_views,
    export_records,
)
from .designs import (
    CALL_DESIGNS,
    CHAT_DESIGNS,
    LIST_DESIGNS,
    SOCIAL_DESIGNS,
    STATUS_BG_COLORS,
    STATUS_DESIGNS,
    STORY_BG_COLORS,
)
from .nl_authoring import (
    MAKE_USAGE,
    PROJECT_USAGE,
    NLAuthoringError,
    NLParseResult,
    model_summary,
    parse_nl_description,
    revise_screen,
)
from .projects import (
    MockProject,
    ProjectError,
    ProjectStore,
    missing_media,
    valid_project_name,
)
from .schema import (
    CALL_CONTROLS,
    CALL_STATES,
    COUNT_STYLES,
    DEVICE_STYLES,
    LIST_FLAVORS,
    LIST_PREVIEW_KINDS,
    LIST_ROW_STATES,
    LIST_ROW_TYPES,
    MESSAGE_KINDS,
    POST_VISIBILITIES,
    PROFILE_ITEM_KINDS,
    RING_STATES,
    STATUS_KINDS,
    STORY_KINDS,
    STORY_STICKER_KINDS,
    TICK_STATES,
    VIDEO_LAYOUTS,
    CallHistoryEntry,
    CallHistoryScreen,
    CallScreen,
    CommentScreen,
    DeviceChrome,
    FolderTab,
    ListRow,
    ListScreen,
    Message,
    Participant,
    PostScreen,
    ProfileCount,
    ProfileItem,
    ProfileScreen,
    Screen,
    SocialComment,
    SocialProfile,
    StatusScreen,
    StatusViewer,
    StatusViewsScreen,
    StoryItem,
    StoryScreen,
    StorySticker,
    UpNextItem,
    VideoScreen,
    shorten_count,
)
from .service import MockScreensError, MockScreenService

#: The /mock command kinds this organ answers (the runtime
#: dispatch set and this tuple are pinned equal by tests,
#: the makers idiom).
MOCK_COMMANDS: tuple[str, ...] = ("mock",)

__all__ = [
    "BulkGenerator",
    "CALL_CONTROLS",
    "CALL_DESIGNS",
    "CALL_STATES",
    "CHAT_DESIGNS",
    "COUNTRIES",
    "COUNT_STYLES",
    "CREATIVE_FAMILIES",
    "Clip",
    "DEVICE_STYLES",
    "DemoDataError",
    "LIST_DESIGNS",
    "LIST_FLAVORS",
    "LIST_PREVIEW_KINDS",
    "LIST_ROW_STATES",
    "LIST_ROW_TYPES",
    "MAIL_PROVIDERS",
    "MAX_BULK_DATA",
    "MAX_BULK_SCREEN_ROWS",
    "MESSAGE_KINDS",
    "MOCK_COMMANDS",
    "POST_VISIBILITIES",
    "PROFILE_ITEM_KINDS",
    "REFERENCE_DATE",
    "RING_STATES",
    "Recording",
    "RecordingError",
    "SOCIAL_DESIGNS",
    "STATUS_BG_COLORS",
    "STATUS_DESIGNS",
    "STATUS_KINDS",
    "STORY_BG_COLORS",
    "STORY_KINDS",
    "STORY_STICKER_KINDS",
    "TICK_STATES",
    "TimelineSpec",
    "VIDEO_LAYOUTS",
    "build_call_recording",
    "build_chat_recording",
    "build_comment_recording",
    "build_hold_recording",
    "build_list_recording",
    "build_profile_recording",
    "build_status_recording",
    "build_story_recording",
    "build_video_recording",
    "record",
    "CallHistoryEntry",
    "CallHistoryScreen",
    "CallScreen",
    "CommentScreen",
    "DeviceChrome",
    "FolderTab",
    "ListRow",
    "ListScreen",
    "Message",
    "MockScreenService",
    "MockScreensError",
    "Participant",
    "PostScreen",
    "ProfileCount",
    "ProfileItem",
    "ProfileScreen",
    "Screen",
    "SocialComment",
    "SocialProfile",
    "StatusScreen",
    "StatusViewer",
    "StatusViewsScreen",
    "StoryItem",
    "StoryScreen",
    "StorySticker",
    "UpNextItem",
    "VideoScreen",
    "bulk_call_history",
    "bulk_chat_screen",
    "bulk_list_screen",
    "bulk_status_views",
    "export_records",
    "shorten_count",
    "MAKE_USAGE",
    "MockProject",
    "NLAuthoringError",
    "NLParseResult",
    "PROJECT_USAGE",
    "ProjectError",
    "ProjectStore",
    "missing_media",
    "model_summary",
    "parse_nl_description",
    "revise_screen",
    "valid_project_name",
]
