"""What an eye (or a single camera) can measure of each ball: where it is in the visual field and
how big it looks. No distance, no velocity.

The eye sits at the point the operator is steering to. Frames arrive at a fixed rate, late, and
noisy. Which blob is which ball is assumed known.
"""

from dataclasses import dataclass

import numpy as np

NOISE = {  # name -> (bearing noise in radians, relative noise on apparent size)
    "none": (0.0, 0.0),
    "moderate": (np.radians(0.5), 0.03),
    "heavy": (np.radians(1.5), 0.10),
}


@dataclass
class Frame:
    t: float          # when the frame was taken, seconds
    azimuth: float    # radians, positive to the left, 0 straight ahead (+x)
    elevation: float  # radians above level
    size: float       # apparent diameter, radians


def see(t: float, position: np.ndarray, radius: float, bearing_noise: float, size_noise: float,
        rng: np.random.Generator) -> Frame:
    distance = float(np.linalg.norm(position))
    azimuth = np.arctan2(position[1], position[0]) + bearing_noise * rng.standard_normal()
    elevation = np.arctan2(position[2], np.hypot(position[0], position[1])) + bearing_noise * rng.standard_normal()
    size = 2 * np.arctan(radius / max(distance, radius)) * (1 + size_noise * rng.standard_normal())
    return Frame(t, float(azimuth), float(elevation), float(max(size, 1e-4)))


def direction(azimuth: float, elevation: float) -> np.ndarray:
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


class Eye:
    def __init__(self, n_balls: int, noise: str = "none", rate: float = 60.0, delay_frames: int = 1,
                 memory: int = 12, seed: int = 0):
        self.bearing_noise, self.size_noise = NOISE[noise]
        self.interval = 1.0 / rate
        self.delay = delay_frames * self.interval
        self.memory = memory
        self.rng = np.random.default_rng(seed)
        self.frames: list[list[Frame]] = [[] for _ in range(n_balls)]
        self._last_pos: list[np.ndarray | None] = [None] * n_balls
        self._next = 0.0

    def look(self, t: float, positions: list[np.ndarray], radius: float) -> list[list[Frame]]:
        """Call every simulation step. Returns, per ball, the frames that have arrived by now."""
        if t >= self._next:
            self._next = t + self.interval
            for i, position in enumerate(positions):
                last = self._last_pos[i]
                if last is not None and np.linalg.norm(position - last) > 0.5:
                    self.frames[i].clear()   # the ball was thrown afresh or taken away: a new track
                self._last_pos[i] = position.copy()
                self.frames[i].append(see(t, position, radius, self.bearing_noise, self.size_noise, self.rng))
                del self.frames[i][:-self.memory - 2]
        return [[f for f in track if f.t <= t - self.delay][-self.memory:] for track in self.frames]
