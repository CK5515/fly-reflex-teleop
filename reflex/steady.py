"""Keep a loose load on a carried tray: the steadying reflex.

Three parts, all continuous:
- hold the tray level whatever the operator's wrist does;
- lean into the motion the operator asks for, the way a waiter does, so the load is not left behind;
- lean a little more to roll the load back to the middle and stop it.
"""

import numpy as np

from reflex.interface import SceneEvents

G = 9.81


class Steady:
    def __init__(self, stiffness: float = 16.0, damping: float = 6.5, lean_into: float = 1.0,
                 max_tilt: float = 0.3, max_tilt_planned: float = 0.6, max_rate: float = 5.0, smoothing: float = 0.03):
        self.stiffness = stiffness   # pull toward the tray's centre, 1/s^2 (a 4 rad/s spring)
        self.damping = damping       # braking of the load's roll, 1/s
        self.lean_into = lean_into   # share of the asked-for acceleration cancelled by leaning
        self.max_tilt = max_tilt     # radians, in ordinary carrying
        self.max_tilt_planned = max_tilt_planned  # radians, during a movement the reflex planned itself
        self.max_rate = max_rate     # radians per second the lean may change
        self.smoothing = smoothing   # seconds
        self._tilt = np.zeros(2)
        self._time = None

    def tilt(self, events: SceneEvents, planned_acc: np.ndarray | None = None) -> np.ndarray:
        """Lean about the world x and y axes. Leaning by +a about x pushes the load toward -y at g*a;
        leaning by +a about y pushes it toward +x."""
        dt = 0.0 if self._time is None else max(events.t - self._time, 0.0)
        self._time = events.t
        if not events.load_on:
            self._tilt = np.zeros(2)
            return self._tilt
        # The acceleration we want the load to have: follow the tray, and return to its centre. The
        # tray's acceleration is what the operator asks for plus whatever the reflex itself has planned.
        tray_acc = events.target_acc[:2] + (0.0 if planned_acc is None else planned_acc[:2])
        limit = self.max_tilt if planned_acc is None else self.max_tilt_planned
        want = self.lean_into * tray_acc - self.stiffness * events.load_pos - self.damping * events.load_vel
        goal = np.clip(np.array([-want[1], want[0]]) / G, -limit, limit)
        # Ease toward the goal: smoothed, and never faster than the arm's wrist can follow.
        step = (goal - self._tilt) * min(dt / self.smoothing, 1.0)
        self._tilt = self._tilt + np.clip(step, -self.max_rate * dt, self.max_rate * dt)
        return self._tilt
