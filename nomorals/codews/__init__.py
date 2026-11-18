"""L5 code workspace object: repos, patches, tests, builds for agents."""

from .compute import Computation, compute_answer, format_compute_reply
from .explain import (
    Fragment,
    explain_command,
    format_explanation_reply,
    open_explanation_store,
    parse_bash,
)
from .man_corpus import (
    BUILTIN_DOCS,
    ManCorpus,
    ManPage,
    fetch_tldr_page,
    open_man_corpus,
    parse_help_text,
    parse_man_page_file,
    parse_man_page_text,
    parse_tldr_markdown,
)
from .patch import (
    apply_edit_blocks,
    apply_patch,
    check_patch,
    preview_patch,
    record_patch,
    review_patch,
    split_patch,
)
from .run import detect_stack, parse_junit_xml, run_build, run_tests
from .textfmt import (
    Diagnostic,
    TransformResult,
    beautify,
    convert,
    detect_format,
    fix_json,
    format_transform_reply,
    minify,
    transform,
    validate,
)
from .wolfram import WolframClient
from .workspace import CodeWorkspace, WorkspaceError

__all__ = [
    "CodeWorkspace",
    "WorkspaceError",
    "Computation",
    "WolframClient",
    "compute_answer",
    "format_compute_reply",
    "Fragment",
    "explain_command",
    "format_explanation_reply",
    "open_explanation_store",
    "parse_bash",
    "BUILTIN_DOCS",
    "ManCorpus",
    "ManPage",
    "fetch_tldr_page",
    "open_man_corpus",
    "parse_help_text",
    "parse_man_page_file",
    "parse_man_page_text",
    "parse_tldr_markdown",
    "review_patch",
    "apply_patch",
    "preview_patch",
    "record_patch",
    "split_patch",
    "check_patch",
    "apply_edit_blocks",
    "Diagnostic",
    "TransformResult",
    "beautify",
    "minify",
    "validate",
    "convert",
    "fix_json",
    "transform",
    "detect_format",
    "format_transform_reply",
    "run_tests",
    "run_build",
    "detect_stack",
    "parse_junit_xml",
]
