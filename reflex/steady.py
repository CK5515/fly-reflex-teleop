"""Keep a loose load on a carried tray: the steadying reflex.

Three parts, all continuous:
- hold the tray level whatever the operator's wrist does;
- lean into the tray's own acceleration, the way a waiter does, so the load is not left behind;
- lean a little more to roll the load back to the middle and stop it.
"""

import numpy as np

from reflex.interface import SceneEvents

G = 9.81


class Steady:
    def __init__(self, stiffness: float = 16.0, damping: float = 6.5, lean_into: float = 1.0, max_tilt: float = 0.6):
        self.stiffness = stiffness   # pull toward the tray's centre, 1/s^2 (a 4 rad/s spring)
        self.damping = damping       # braking of the load's roll, 1/s
        self.lean_into = lean_into   # share of the tray's acceleration cancelled by leaning
        self.max_tilt = max_tilt     # radians

    def tilt(self, events: SceneEvents, planned_acc: np.ndarray | None = None) -> np.ndarray:
        """Lean about the world x and y axes. Leaning by +a about x pushes the load toward -y at g*a;
        leaning by +a about y pushes it toward +x."""
        if not events.load_on:
            return np.zeros(2)
        # The acceleration we want the load to have: follow the tray, and return to its centre. When the
        # reflex itself is moving the arm, its planned acceleration is known ahead of the measured one.
        tray_acc = events.gripper_acc[:2] if planned_acc is None else planned_acc[:2]
        want = (self.lean_into * tray_acc
                - self.stiffness * events.load_pos - self.damping * events.load_vel)
        return np.clip(np.array([-want[1], want[0]]) / G, -self.max_tilt, self.max_tilt)
