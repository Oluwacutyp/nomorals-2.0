"""L5 code workspace object: repos, patches, tests, builds for agents."""

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
from .workspace import CodeWorkspace, WorkspaceError

__all__ = [
    "CodeWorkspace",
    "WorkspaceError",
    "review_patch",
    "apply_patch",
    "preview_patch",
    "record_patch",
    "split_patch",
    "check_patch",
    "apply_edit_blocks",
    "run_tests",
    "run_build",
    "detect_stack",
    "parse_junit_xml",
]
