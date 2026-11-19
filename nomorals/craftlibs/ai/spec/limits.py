"""Spec limits (Craft-Libs port of ai-core ``spec.rs`` caps).

The constants live here, in a module with no imports, so the
spec package and its checker can both use them without a
circular import.
"""

SPEC_VERSION = 1
MAX_SPEC_BYTES = 32 * 1024
MAX_MODELS = 16
MAX_TEMPLATE_BYTES = 4 * 1024
MAX_RENDERED_BYTES = 4 * 1024 * 1024
MAX_POINTER_DEPTH = 8
MAX_STATES = 16
MAX_PATH_BYTES = 256
