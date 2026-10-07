"""Vision ("eyes") for Devon.

One brain, vision as a tool. The text model is the only agent — it runs the
agentic loop. The vision model is a sensor: called as a function, returns
text descriptions, has no agency, no tool access, no loop.

Primary path: Groq's vision API (Llama 3.2 Vision) via the LLM router.
Fallback: local GGUF vision model via the lifecycle/provisioner.
"""

from .screenshot import capture_screenshot, screenshot_from_file
from .seer import Seer, see

__all__ = ["Seer", "see", "capture_screenshot", "screenshot_from_file",
           "register"]


def register(registry) -> None:
    """Register the ``see`` sensor tool with the tool registry."""
    from ..core.policy import Capability

    @registry.register(
        "see",
        description=(
            "Look at an image and answer a question about it. "
            "image_path= path to a PNG/JPG screenshot. question= what to "
            "look for (e.g. 'where is the attack button?'). Returns a text "
            "description. The vision model only describes — it never decides."
        ),
        capability=Capability.MODEL_CALL,
    )
    def see_tool(image_path: str, question: str = "") -> str:
        """Look at an image and answer a question about it."""
        return see(image_path, question)
