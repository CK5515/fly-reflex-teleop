"""Two ways to turn what the eye sees into avoidance.

`fit_flight` is the standard geometric route: recover each ball's distance from its apparent size,
fit a ballistic flight through the recent frames, and hand position and velocity to the usual
geometric avoidance. `looming_cue` is the fly's route: no distance at all, only how long until
contact (apparent size divided by its growth rate) and where in the visual field.
"""

from dataclasses import replace

import numpy as np

from reflex.continuous import ContinuousReflex
from reflex.interface import ReflexCommand, SceneEvents
from sim.eye import Frame, direction

GRAVITY = np.array([0.0, 0.0, -9.81])


def fit_flight(frames: list[Frame], radius: float, now: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Position and velocity now, by least squares over the frames, knowing gravity. Needs 3 frames."""
    if len(frames) < 3:
        return None
    when = np.array([f.t - now for f in frames])
    seen = np.array([direction(f.azimuth, f.elevation) * radius / np.tan(f.size / 2) for f in frames])
    design = np.stack([np.ones_like(when), when], axis=1)
    solution = np.linalg.lstsq(design, seen - 0.5 * when[:, None] ** 2 * GRAVITY, rcond=None)[0]
    return solution[0], solution[1]


def looming_cue(frames: list[Frame], now: float, lag: int = 4) -> tuple[float, Frame, Frame] | None:
    """(seconds to contact, latest frame, a frame from a little earlier), or None if not approaching.

    Time to contact is apparent size over its growth rate, the growth rate being the slope of size
    over the recent frames.
    """
    if len(frames) < 3:
        return None
    when = np.array([f.t - now for f in frames])
    size = np.array([f.size for f in frames])
    growth, size_now = np.polyfit(when, size, 1)
    if growth <= 1e-4 or size_now <= 0:
        return None
    return float(size_now / growth), frames[-1], frames[max(len(frames) - 1 - lag, 0)]


class EyeGeometricReflex(ContinuousReflex):
    """Geometric avoidance fed by the eye instead of by exact positions and velocities."""

    relative_to_target = True   # the eye already measures from the operator's target point

    def step(self, events: SceneEvents) -> ReflexCommand:
        fitted = [fit_flight(frames, events.object_radius, events.t) for frames in events.sights or []]
        return super().step(replace(events, objects=[f for f in fitted if f is not None]))
