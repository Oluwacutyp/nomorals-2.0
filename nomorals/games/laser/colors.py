"""Named colours in the laser's 16-bit colour space.

Ported from ``colors.py`` in echelon/laser-asteroids (MIT, Copyright
(c) 2012 Brandon Thomas).
"""
from __future__ import annotations

from .constants import CMAX

COLOR_RED = (CMAX, 0, CMAX // 4)
COLOR_GREEN = (0, CMAX, int(CMAX / 3.2))
COLOR_BLUE = (CMAX // 4, CMAX, CMAX)

COLOR_YELLOW = (CMAX, CMAX, 0)
COLOR_ORANGE = (CMAX, CMAX, CMAX // 3)

COLOR_COOL_PINK = (CMAX // 2, 0, CMAX)
COLOR_WARM_PINK = (CMAX, 0, CMAX // 3)
COLOR_PINK = COLOR_WARM_PINK
COLOR_PURPLE = (CMAX // 3, 0, CMAX)

COLOR_WHITE = (CMAX // 2, CMAX, int(CMAX / 2.2))

#: Bullet colours cycled one per shot, as in the original.
BULLET_COLORS = (COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_YELLOW)
