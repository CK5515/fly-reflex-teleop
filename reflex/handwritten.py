"""Hand-written reflexes: the fallback that keeps the demo whole without the brain model.

Escape: something is about to hit the gripper, so jump out of its path.
Groom:  something touched the arm, so shake it off.
"""

import numpy as np

from reflex.interface import ReflexCommand, SceneEvents

UP = np.array([0.0, 0.0, 1.0])


GRAVITY = np.array([0.0, 0.0, -9.81])
_LOOKAHEAD = np.linspace(0.0, 1.0, 101)[1:, None]  # seconds


def time_to_contact(events: SceneEvents) -> tuple[float, np.ndarray]:
    """Seconds until the object is closest, and where it will pass (relative to the gripper).

    Assumes the object is in free flight and the gripper stays put.
    """
    if float(events.object_vel @ events.object_vel) < 0.25:  # slower than 0.5 m/s: not a throw
        return np.inf, events.object_pos
    path = events.object_pos + _LOOKAHEAD * events.object_vel + 0.5 * _LOOKAHEAD**2 * GRAVITY
    nearest = int(np.argmin(np.linalg.norm(path, axis=1)))
    if nearest == len(path) - 1:  # still closing at the end of the lookahead: too far off to matter
        return np.inf, events.object_pos
    return float(_LOOKAHEAD[nearest, 0]), path[nearest]


class HandwrittenReflex:
    def __init__(
        self,
        escape_within: float = 0.45,   # trigger when impact is this many seconds away...
        escape_if_closer: float = 0.25,  # ...and the object would pass this close, metres
        escape_jump: float = 0.30,     # metres
        escape_hold: float = 0.5,      # seconds at full override
        escape_release: float = 0.5,   # seconds handing control back
        groom_force: float = 0.5,      # newtons of touch that trigger a shake
        groom_swing: float = 0.06,     # metres
        groom_rate: float = 4.0,       # shakes per second
        groom_time: float = 0.75,      # seconds
    ):
        self.p = dict(locals())
        del self.p["self"]
        self.active: str | None = None
        self.touched = False  # touched while escaping: shake it off once the escape is over
        self.started = 0.0
        self.offset = np.zeros(3)

    def step(self, events: SceneEvents) -> ReflexCommand:
        p = self.p
        ttc, passing = time_to_contact(events)
        miss = float(np.linalg.norm(passing))
        on_course = miss < p["escape_if_closer"]
        escape_drive = float(np.clip(p["escape_within"] / ttc, 0, 2)) if on_course else 0.0
        groom_drive = float(np.clip(events.contact_force / p["groom_force"], 0, 2))
        activations = {"escape": escape_drive, "groom": groom_drive}

        if self.active == "escape" and groom_drive >= 1.0:
            self.touched = True

        if self.active is None:
            if self.touched:
                self.active, self.started, self.touched = "groom", events.t, False
            elif escape_drive >= 1.0:
                self.active, self.started = "escape", events.t
                self.offset = p["escape_jump"] * self._way_out(passing, events.object_vel, events.gripper_pos)
            elif groom_drive >= 1.0:
                self.active, self.started = "groom", events.t

        if self.active == "escape":
            age = events.t - self.started
            if age < p["escape_hold"]:
                gain = 1.0
            else:
                gain = 1.0 - (age - p["escape_hold"]) / p["escape_release"]
            if gain <= 0:
                self.active = None
                if self.touched:
                    self.active, self.started, self.touched = "groom", events.t, False
                return ReflexCommand(activations=activations)
            return ReflexCommand("escape", gain, self.offset, activations)

        if self.active == "groom":
            age = events.t - self.started
            if age >= p["groom_time"]:
                self.active = None
                return ReflexCommand(activations=activations)
            swing = p["groom_swing"] * np.sin(2 * np.pi * p["groom_rate"] * age)
            return ReflexCommand("groom", 1.0, np.array([0.0, swing, 0.0]), activations)

        return ReflexCommand(activations=activations)

    @staticmethod
    def _way_out(passing: np.ndarray, velocity: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
        """Unit sidestep, level with the table and square to the object's flight.

        Level because a lobbed object comes down from above, and because the
        arm has little reach left upward. Steps away from the side the object
        will pass on; on a dead-centre hit, steps toward the middle of the
        workspace, where the arm has the most room.
        """
        heading = np.array([velocity[0], velocity[1], 0.0])
        heading /= np.linalg.norm(heading) + 1e-9
        side = np.cross(UP, heading)
        lean = -float(passing @ side)
        if abs(lean) < 0.03:
            lean = -float(gripper_pos @ side)
        return side if lean >= 0 else -side
