"""``nm`` command implementations, grouped by domain."""

from __future__ import annotations

from .agent import _cmd_ask, _cmd_run
from .autonomy import _cmd_autonomy, _set_daily_call_cap
from .backup import _cmd_backup
from .benchmark import _BuildersBuildBackend, _cmd_benchmark
from .bet import _cmd_bet
from .book import _cmd_book
from .briefing import _cmd_briefing
from .captcha import _cmd_captcha
from .cards import _cmd_cards
from .code import _cmd_code, _cmd_code_review, _cmd_code_run, _cmd_code_test, _critic_verdict
from .data import _cmd_data
from .doctor import _cmd_doctor, _cmd_models, _cmd_models_doctor, _cmd_setup, _env_update_home
from .exec import _cmd_apps, _cmd_exec
from .finance import _cmd_finance
from .games import _cmd_arena, _cmd_simulate, _cmd_skill, _cmd_trial
from .goal import _cmd_goal
from .golden import _cmd_golden
from .hub import _cmd_hub
from .improve import _cmd_improve, _inbox_obj
from .inbox import _cmd_inbox
from .media import (
    _VIDEO_EXTS,
    _cmd_media,
    _cmd_media_convert,
    _cmd_media_edit,
    _cmd_media_jobs,
    _cmd_media_probe,
    _cmd_media_wait,
    _cmd_studio,
    _is_video_file,
    _media_call,
    _media_tools,
)
from .memory import _cmd_memory, _cmd_memory_action
from .meta import (
    _cli_command_help,
    _cli_overview,
    _cli_subparsers,
    _cmd_commands,
    _cmd_connectors,
    _cmd_deliver,
    _cmd_help,
    _cmd_zip,
    _reply_path_report,
)
from .mind import _cmd_mind
from .mission import _cmd_mission
from .missions import _cmd_missions, _render_result
from .models import _cmd_model_broker
from .money import _cmd_money
from .music import _cmd_music
from .native import _cmd_native
from .owner import _cmd_owner
from .partner import _cmd_partner_ask, _cmd_reason, _cmd_workspace
from .power import _cmd_power
from .project import _cmd_project
from .queue import _cmd_queue
from .recover import _cmd_recover
from .research import _cmd_cookies, _cmd_kg, _cmd_research_loop, _cmd_structure
from .room import _cmd_room, _room_obj
from .security import _cmd_cipher, _cmd_crack, _cmd_decode, _cmd_monitor, _cmd_osint, _cmd_watch
from .serve import _cmd_serve
from .skills import _cmd_skill_pkg
from .snapshot import _cmd_snapshot
from .status import _cmd_status, _status_section
from .swarm import _cmd_swarm, _critique_from_json, _work_from_json
from .timeline import _cmd_timeline, _render_timeline_row
from .tools import _cmd_tools
from .trade import _cmd_trade
from .train import _cmd_train
from .tui import _cmd_tui, _dispatch_tui_command
from .update import _cmd_update
from .vision import (
    _cmd_vision,
    _confirm,
    _print_vision_result,
    _vision_source,
    _vision_tool_call,
    _vision_tools,
)
from .voice import (
    _cmd_voice,
    _cmd_voice_call,
    _cmd_voice_catalogue_list,
    _cmd_voice_clone,
    _cmd_voice_consent,
    _cmd_voice_current,
    _cmd_voice_decrypt,
    _cmd_voice_describe,
    _cmd_voice_fetch,
    _cmd_voice_listen,
    _cmd_voice_purge,
    _cmd_voice_rm,
    _cmd_voice_say,
    _cmd_voice_stats,
    _cmd_voice_transcribe,
    _cmd_voice_transcript,
    _cmd_voice_use,
    _voice_data_dir,
    _voice_read_key_file,
    _voice_stt_from_tools,
    _voice_think_runtime,
)
from .weather import _cmd_weather

__all__ = [
    "_cmd_run",
    "_cmd_ask",
    "_cmd_autonomy",
    "_set_daily_call_cap",
    "_cmd_backup",
    "_BuildersBuildBackend",
    "_cmd_benchmark",
    "_cmd_bet",
    "_cmd_book",
    "_cmd_briefing",
    "_cmd_captcha",
    "_cmd_cards",
    "_cmd_code",
    "_cmd_code_run",
    "_critic_verdict",
    "_cmd_code_review",
    "_cmd_code_test",
    "_cmd_data",
    "_cmd_doctor",
    "_env_update_home",
    "_cmd_models",
    "_cmd_models_doctor",
    "_cmd_setup",
    "_cmd_exec",
    "_cmd_apps",
    "_cmd_finance",
    "_cmd_arena",
    "_cmd_trial",
    "_cmd_skill",
    "_cmd_simulate",
    "_cmd_goal",
    "_cmd_hub",
    "_inbox_obj",
    "_cmd_improve",
    "_cmd_inbox",
    "_VIDEO_EXTS",
    "_media_tools",
    "_media_call",
    "_cmd_media",
    "_cmd_studio",
    "_is_video_file",
    "_cmd_media_edit",
    "_cmd_media_wait",
    "_cmd_media_probe",
    "_cmd_media_jobs",
    "_cmd_media_convert",
    "_cmd_memory",
    "_cmd_memory_action",
    "_reply_path_report",
    "_cmd_commands",
    "_cmd_deliver",
    "_cmd_zip",
    "_cmd_connectors",
    "_cli_subparsers",
    "_cli_command_help",
    "_cli_overview",
    "_cmd_help",
    "_cmd_mind",
    "_cmd_mission",
    "_cmd_missions",
    "_render_result",
    "_cmd_model_broker",
    "_cmd_money",
    "_cmd_music",
    "_cmd_native",
    "_cmd_owner",
    "_cmd_reason",
    "_cmd_workspace",
    "_cmd_partner_ask",
    "_cmd_power",
    "_cmd_project",
    "_cmd_queue",
    "_cmd_research_loop",
    "_cmd_kg",
    "_cmd_cookies",
    "_cmd_structure",
    "_room_obj",
    "_cmd_room",
    "_cmd_crack",
    "_cmd_decode",
    "_cmd_osint",
    "_cmd_cipher",
    "_cmd_monitor",
    "_cmd_watch",
    "_cmd_serve",
    "_cmd_skill_pkg",
    "_status_section",
    "_cmd_status",
    "_cmd_snapshot",
    "_cmd_recover",
    "_cmd_update",
    "_cmd_golden",
    "_cmd_swarm",
    "_work_from_json",
    "_critique_from_json",
    "_render_timeline_row",
    "_cmd_timeline",
    "_cmd_tools",
    "_cmd_trade",
    "_cmd_train",
    "_cmd_tui",
    "_dispatch_tui_command",
    "_vision_tools",
    "_vision_tool_call",
    "_vision_source",
    "_confirm",
    "_print_vision_result",
    "_cmd_vision",
    "_voice_data_dir",
    "_voice_read_key_file",
    "_voice_stt_from_tools",
    "_voice_think_runtime",
    "_cmd_voice",
    "_cmd_voice_call",
    "_cmd_voice_say",
    "_cmd_voice_fetch",
    "_cmd_voice_clone",
    "_cmd_voice_catalogue_list",
    "_cmd_voice_use",
    "_cmd_voice_current",
    "_cmd_voice_rm",
    "_cmd_voice_describe",
    "_cmd_voice_transcript",
    "_cmd_voice_listen",
    "_cmd_voice_transcribe",
    "_cmd_voice_stats",
    "_cmd_voice_consent",
    "_cmd_voice_purge",
    "_cmd_voice_decrypt",
    "_cmd_weather",
]
