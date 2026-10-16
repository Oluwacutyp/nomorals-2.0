"""Tier 2 moat — Nigerian / African reference library as style packs.

Each pack is prompt fragments + palette + critique notes, so the
critique engine can check a render against the pack it claims to be
in. Packs cover: Nollywood lighting, Lagos street, owambe, harmattan
haze. Prompts and critique notes code-switch in Yoruba / Pidgin —
the language the scenes live in — with English glosses so the
pipeline stays checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

NIGERIAN_PACKS: dict[str, dict[str, Any]] = {
    "nollywood_lighting": {
        "prompt": "Nollywood drama lighting, warm tungsten key, deep shadows, rich skin tones, Lagos living room",
        "palette": ["#f5c396", "#8b4513", "#1a0f08", "#e8b44a", "#5c1a1a"],
        "critique_notes": [
            "Skin tones must stay warm and rich, never ashy or grey.",
            "Owambe shine: gele and aso-oke should read glossy, not flat.",
        ],
        "code_switch": "Imole gbona bi ti Nollywood — warm light, make the skin glow (jẹ ki awọ rẹ tàn).",
    },
    "lagos_street": {
        "prompt": "Lagos street scene, yellow danfo buses, market stalls, okada, humid midday light, bustling crowd",
        "palette": ["#f7d117", "#2e7d32", "#d84315", "#37474f", "#ffcc80"],
        "critique_notes": [
            "Danfo yellow must read yellow, not mustard-brown.",
            "Street must feel crowded and alive — no empty boulevards.",
        ],
        "code_switch": "Ojú ọ̀nà Èkó — Lagos street, e don scatter with life, no dull moment.",
    },
    "owambe": {
        "prompt": "Owambe party, aso-oke, gele, Ankara, live band, dancing crowd, colourful canopy, celebration",
        "palette": ["#e91e63", "#ffd54f", "#4a148c", "#00bcd4", "#f48fb1"],
        "critique_notes": [
            "Aso-oke patterns must be coherent, not melted geometry.",
            "Celebration energy: dancing, colour, movement — no stiff poses.",
        ],
        "code_switch": "Owambe gidi — real party, everybody dey dance, colour full everywhere.",
    },
    "harmattan_haze": {
        "prompt": "Harmattan haze, dusty golden morning light, soft diffused sun, dry-season atmosphere, muted distance",
        "palette": ["#d7ccc8", "#a1887f", "#ffcc80", "#8d6e63", "#ece0d0"],
        "critique_notes": [
            "Haze must soften distance, not fog out the subject.",
            "Golden dust light, not grey overcast.",
        ],
        "code_switch": "Ìgbà ẹ̀rùn harmattan — dusty golden air, sun dey soft, no harsh shadow.",
    },
}


@dataclass
class StylePack:
    name: str
    prompt: str = ""
    palette: list[str] = field(default_factory=list)
    critique_notes: list[str] = field(default_factory=list)
    code_switch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "prompt": self.prompt,
                "palette": list(self.palette),
                "critique_notes": list(self.critique_notes),
                "code_switch": self.code_switch}


def nigerian_pack(name: str) -> StylePack:
    if name not in NIGERIAN_PACKS:
        raise ValueError(f"unknown pack {name!r}; use {sorted(NIGERIAN_PACKS)}")
    data = NIGERIAN_PACKS[name]
    return StylePack(name=name, prompt=data["prompt"], palette=list(data["palette"]),
                     critique_notes=list(data["critique_notes"]),
                     code_switch=data["code_switch"])


def nigerian_prompt(name: str, scene: str = "") -> str:
    """Pack prompt + scene, with the code-switch line for flavour."""
    pack = nigerian_pack(name)
    base = f"{pack.prompt}. {scene}".strip()
    return f"{base} [{pack.code_switch}]"
