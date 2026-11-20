"""Occupancy grid — the map a robot builds from its scans.

The echelon/robot TODO wanted "multiple sensors to help
localization"; the clone never built it. Devon builds the map
itself: a log-odds grid, updated from range scans by carving
free space along each beam and marking the hit cell occupied.
Log-odds is the honest representation — a cell that was seen
empty ten times and full once is probably empty, and the grid
says so in numbers instead of flipping a bit.

    p = 1 - 1 / (1 + exp(l))     l = log(p / (1 - p))

Updates: L_OCC = +0.85 per hit, L_FREE = -0.12 per pass, clamped
to ±4.0 (p in 0.018..0.982). Unknown cells sit at l = 0 (p = 0.5).
"""

from __future__ import annotations

import base64
import json
import math
from array import array
from pathlib import Path

from .errors import RobotError
from .sensors import Scan
from .state import Pose

L_OCC = 0.85
L_FREE = -0.12
L_MIN = -4.0
L_MAX = 4.0


class OccupancyGrid:
    """A 2D log-odds occupancy grid over a fixed world rectangle."""

    def __init__(self, width_m: float = 20.0, height_m: float = 20.0,
                 cell_m: float = 0.05,
                 origin_x: float = -10.0, origin_y: float = -10.0) -> None:
        if width_m <= 0 or height_m <= 0 or cell_m <= 0:
            raise RobotError(
                "bad_scan",
                f"grid dims must be positive, got {width_m}x{height_m}@{cell_m}")
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.cell_m = float(cell_m)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.width = max(1, int(self.width_m / self.cell_m))
        self.height = max(1, int(self.height_m / self.cell_m))
        # Log-odds per cell, row-major (iy * width + ix), 0 = unknown.
        self._cells = array("f", bytes(4 * self.width * self.height))
        self.observations = 0

    # ── coordinates ────────────────────────────────────────────────
    def to_cell(self, x: float, y: float) -> tuple[int, int]:
        """World (x, y) -> cell (ix, iy)."""
        return (int(math.floor((x - self.origin_x) / self.cell_m)),
                int(math.floor((y - self.origin_y) / self.cell_m)))

    def to_world(self, ix: int, iy: int) -> tuple[float, float]:
        """Cell (ix, iy) -> world coords of the cell centre."""
        return (self.origin_x + (ix + 0.5) * self.cell_m,
                self.origin_y + (iy + 0.5) * self.cell_m)

    def in_bounds(self, ix: int, iy: int) -> bool:
        return 0 <= ix < self.width and 0 <= iy < self.height

    def _index(self, ix: int, iy: int) -> int:
        return iy * self.width + ix

    def log_odds(self, ix: int, iy: int) -> float:
        """Log-odds at a cell; out of bounds reads as unknown (0)."""
        if not self.in_bounds(ix, iy):
            return 0.0
        return float(self._cells[self._index(ix, iy)])

    def probability(self, ix: int, iy: int) -> float:
        """Occupancy probability at a cell (0.5 = unknown)."""
        log_odds = self.log_odds(ix, iy)
        return 1.0 - 1.0 / (1.0 + math.exp(log_odds))

    def probability_at(self, x: float, y: float) -> float:
        """Occupancy probability at world coords."""
        ix, iy = self.to_cell(x, y)
        return self.probability(ix, iy)

    # ── updates ────────────────────────────────────────────────────
    def _add(self, ix: int, iy: int, delta: float) -> None:
        if not self.in_bounds(ix, iy):
            return
        index = self._index(ix, iy)
        value = self._cells[index] + delta
        self._cells[index] = max(L_MIN, min(L_MAX, value))

    @staticmethod
    def _ray_cells(x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        """Bresenham cells from (x0, y0) to (x1, y1), inclusive of
        both ends, in order."""
        cells = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                return cells
            err2 = 2 * err
            if err2 >= dy:
                err += dy
                x += sx
            if err2 <= dx:
                err += dx
                y += sy

    def update(self, pose: Pose, scan: Scan) -> dict[str, int]:
        """Fold one scan into the grid from the given pose. Beams
        with a return carve free space to the hit and mark the hit
        occupied; no-return beams carve free to max range. Returns
        counts of cells touched."""
        origin_ix, origin_iy = self.to_cell(pose.x, pose.y)
        free = 0
        hit = 0
        for angle, value in zip(scan.angles, scan.ranges, strict=True):
            world = pose.theta + angle
            dist = scan.max_range if value is None else float(value)
            end_x = pose.x + dist * math.cos(world)
            end_y = pose.y + dist * math.sin(world)
            end_ix, end_iy = self.to_cell(end_x, end_y)
            ray = self._ray_cells(origin_ix, origin_iy, end_ix, end_iy)
            for cell in ray[:-1]:
                self._add(cell[0], cell[1], L_FREE)
                free += 1
            if value is not None and self.in_bounds(end_ix, end_iy):
                self._add(end_ix, end_iy, L_OCC)
                hit += 1
        self.observations += 1
        return {"free": free, "hit": hit, "scans": self.observations}

    # ── localization ───────────────────────────────────────────────
    def _score(self, pose: Pose, scan: Scan) -> float:
        """How well a scan fits the grid from a pose: the sum of
        log-odds at the hit endpoints. A pose whose hits land on
        occupied cells scores high; one whose hits land in known
        free space scores low."""
        total = 0.0
        for angle, value in zip(scan.angles, scan.ranges, strict=True):
            if value is None:
                continue
            world = pose.theta + angle
            x = pose.x + float(value) * math.cos(world)
            y = pose.y + float(value) * math.sin(world)
            ix, iy = self.to_cell(x, y)
            total += self.log_odds(ix, iy)
        return total

    def localize(self, pose: Pose, scan: Scan) -> dict:
        """Re-find the robot in its own map: a coarse-to-fine search
        over (dx, dy, dθ) around the given pose, maximizing the
        scan's log-odds score. Refuses when the grid has too little
        in it to match against (a match on an empty map is a fake)."""
        known = sum(1 for value in self._cells if abs(value) > 0.05)
        if known < 20:
            raise RobotError(
                "pose_unknown",
                f"only {known} observed cells in the map — drive and "
                "scan first so there is something to match against")
        if scan.hits < 3:
            raise RobotError(
                "pose_unknown",
                f"only {scan.hits} scan hits — need at least 3 to localize")
        best = (self._score(pose, scan), pose.x, pose.y, pose.theta)
        # Coarse pass, then a fine pass around the coarse winner.
        for step_m, step_deg in ((0.10, 5.0), (0.02, 1.0)):
            _, bx, by, bt = best
            for dx in (-step_m, 0.0, step_m):
                for dy in (-step_m, 0.0, step_m):
                    for dt in (-math.radians(step_deg), 0.0,
                               math.radians(step_deg)):
                        trial = Pose(bx + dx, by + dy, bt + dt)
                        score = self._score(trial, scan)
                        if score > best[0]:
                            best = (score, trial.x, trial.y, trial.theta)
        _, x, y, theta = best
        return {"pose": Pose(x, y, theta).as_dict(),
                "score": best[0], "hits": scan.hits}

    # ── views ──────────────────────────────────────────────────────
    def counts(self) -> dict[str, int]:
        """Cell tallies: observed, occupied (p > 0.65), free (p < 0.35)."""
        observed = occupied = free = 0
        for value in self._cells:
            if abs(value) <= 0.05:
                continue
            observed += 1
            if value > 0.65:
                occupied += 1
            elif value < -0.35:
                free += 1
        return {"cells": self.width * self.height,
                "observed": observed, "occupied": occupied, "free": free,
                "scans": self.observations}

    def occupied_cells(self, threshold: float = 0.65) -> list[tuple[float, float]]:
        """World coords of cells with p >= threshold."""
        out = []
        for iy in range(self.height):
            for ix in range(self.width):
                if self.probability(ix, iy) >= threshold:
                    out.append(self.to_world(ix, iy))
        return out

    def render_ascii(self, width: int = 80) -> str:
        """The map as text: '#' occupied, '.' free, ' ' unknown.
        Downsampled to ``width`` columns; the robot's pose is not
        drawn here (the caller knows it)."""
        step_x = max(1, self.width // width)
        # Text cells are ~twice as tall as wide, so step twice as
        # far vertically to keep the map's aspect honest.
        step_y = max(1, step_x * 2)
        lines = []
        for iy in range(self.height - 1, -1, -step_y):
            row = []
            for ix in range(0, self.width, step_x):
                # Aggregate the whole sampled block: a one-cell-wide
                # ray would vanish under point sampling, so the block
                # shows occupied if ANY cell in it is occupied, else
                # free if ANY cell in it has free evidence.
                block_max = -99.0
                block_min = 99.0
                for by in range(iy - step_y + 1, iy + 1):
                    for bx in range(ix, min(ix + step_x, self.width)):
                        value = self.log_odds(bx, by)
                        block_max = max(block_max, value)
                        block_min = min(block_min, value)
                if block_max >= 0.65:
                    row.append("#")
                elif block_min <= -0.05:
                    row.append(".")
                else:
                    row.append(" ")
            lines.append("".join(row))
        return "\n".join(lines)

    def to_pgm(self) -> bytes:
        """The map as a P5 PGM image (stdlib only, no PIL): 0 =
        occupied, 127 = unknown, 255 = free. Top row is +y."""
        header = f"P5\n{self.width} {self.height}\n255\n".encode("ascii")
        body = bytearray()
        for iy in range(self.height - 1, -1, -1):
            for ix in range(self.width):
                p = self.probability(ix, iy)
                if p >= 0.65:
                    body.append(0)
                elif p <= 0.35:
                    body.append(255)
                else:
                    body.append(127)
        return header + bytes(body)

    # ── persistence ────────────────────────────────────────────────
    def save(self, path: str | Path) -> dict:
        """Save the grid as JSON (log-odds packed as base64 floats)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "width_m": self.width_m, "height_m": self.height_m,
            "cell_m": self.cell_m,
            "origin_x": self.origin_x, "origin_y": self.origin_y,
            "observations": self.observations,
            "cells_b64": base64.b64encode(self._cells.tobytes()).decode("ascii"),
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return {"path": str(path), "cells": self.width * self.height,
                "observations": self.observations}

    @classmethod
    def load(cls, path: str | Path) -> OccupancyGrid:
        """Load a grid saved by :meth:`save`."""
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        grid = cls(width_m=raw["width_m"], height_m=raw["height_m"],
                   cell_m=raw["cell_m"],
                   origin_x=raw["origin_x"], origin_y=raw["origin_y"])
        blob = base64.b64decode(raw["cells_b64"])
        cells = array("f")
        cells.frombytes(blob)
        if len(cells) != grid.width * grid.height:
            raise RobotError(
                "bad_scan",
                f"saved grid has {len(cells)} cells, expected "
                f"{grid.width * grid.height}")
        grid._cells = cells
        grid.observations = int(raw.get("observations", 0))
        return grid
