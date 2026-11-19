"""Devon's shared libraries (Craft-Libs port).

The one home for the code the craft organs share: the
deterministic ZIP package codec (cad + layout), the two noise
engines (textures' f32-exact point engine + material's grid
engine), the ai-core data model, the shared PDF surface and the
i18n formatting rung. Each module is importable on its own; this
package only names them.
"""

from . import ai, attribution, codec, i18n, noise, parity, pdf

__all__ = ["ai", "attribution", "codec", "i18n", "noise", "parity",
           "pdf"]
