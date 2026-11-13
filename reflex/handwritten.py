"""Hand-written reflexes: the fallback that keeps the demo whole without the brain model.

Escape: something is about to hit the gripper, so jump out of its path.
Groom:  something touched the arm, so shake it off.
"""

import numpy as np

from reflex.interface import ReflexCommand, SceneEvents
from reflex.steady import Steady

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


def _smooth_jump(age: float, onset: float, hold: float, release: float) -> tuple[float, float]:
    """Share of the jump made at `age` seconds, and that share's second derivative (1/s^2).

    Out along a minimum-jerk curve over `onset`, held until `hold`, back the same way over `release`.
    """
    def curve(x: float, duration: float) -> tuple[float, float]:
        x = min(max(x, 0.0), 1.0)
        return 10 * x**3 - 15 * x**4 + 6 * x**5, (60 * x - 180 * x**2 + 120 * x**3) / duration**2

    if age < max(onset, hold):
        share, acc = curve(age / onset, onset)
        return share, acc if age < onset else 0.0
    share, acc = curve((age - max(onset, hold)) / release, release)
    return 1.0 - share, -acc


class HandwrittenReflex:
    def __init__(
        self,
        escape_within: float = 0.45,   # trigger when impact is this many seconds away...
        escape_if_closer: float = 0.25,  # ...and the object would pass this close, metres
        escape_jump: float = 0.30,     # metres
        escape_hold: float = 0.5,      # seconds at full override
        escape_release: float = 0.5,   # seconds handing control back
        carry_within: float = 0.6,     # when carrying a load: trigger this early instead...
        carry_onset: float = 0.4,      # ...and take this long to make the jump, smoothly
        carry_release: float = 0.9,    # ...and this long to come back
        groom_force: float = 0.5,      # newtons of touch that trigger a shake
        groom_swing: float = 0.06,     # metres
        groom_rate: float = 4.0,       # shakes per second
        groom_time: float = 0.75,      # seconds
    ):
        self.p = dict(locals())
        del self.p["self"]
        self.active: str | None = None
        self.touched = False  # touched while escaping: shake it off once the escape is over
        self.steady = Steady()
        self.gentle = False   # the current escape is being made smoothly, for a carried load
        self.started = 0.0
        self.offset = np.zeros(3)

    def judge(self, events: SceneEvents) -> tuple[float, float, np.ndarray, dict]:
        """Decide how strongly each reflex is called for: (escape drive, groom drive, which way to
        jump, extra detail). A drive of 1 or more fires the reflex. Subclasses replace this decision
        and inherit the movements below.
        """
        p = self.p
        ttc, passing = time_to_contact(events)
        on_course = float(np.linalg.norm(passing)) < p["escape_if_closer"]
        within = p["carry_within"] if events.load_on else p["escape_within"]
        escape_drive = float(np.clip(within / ttc, 0, 2)) if on_course else 0.0
        groom_drive = float(np.clip(events.contact_force / p["groom_force"], 0, 2))
        return escape_drive, groom_drive, self._way_out(passing, events.object_vel, events.gripper_pos), {}

    def step(self, events: SceneEvents) -> ReflexCommand:
        """The triggered reflexes, plus continuous steadying whenever a loose load is carried."""
        command = self._triggered(events)
        if events.load_on:
            command.level = True
            moving = command.expert == "escape"
            command.tilt = self.steady.tilt(events, command.planned_acc if moving else None)
            command.activations["steady"] = float(np.abs(command.tilt).max() / self.steady.max_tilt * 2)
        return command

    def _triggered(self, events: SceneEvents) -> ReflexCommand:
        p = self.p
        escape_drive, groom_drive, way_out, detail = self.judge(events)
        if events.load_on:
            groom_drive = 0.0  # shaking would throw the load off
        activations = {"escape": escape_drive, "groom": groom_drive}
        idle = ReflexCommand(activations=activations, detail=detail)

        if self.active == "escape" and groom_drive >= 1.0:
            self.touched = True

        if self.active is None:
            if self.touched:
                self.active, self.started, self.touched = "groom", events.t, False
            elif escape_drive >= 1.0:
                self.active, self.started = "escape", events.t
                self.gentle = events.load_on
                self.offset = p["escape_jump"] * way_out
            elif groom_drive >= 1.0:
                self.active, self.started = "groom", events.t

        if self.active == "escape":
            age = events.t - self.started
            if self.gentle:
                gain, gain_acc = _smooth_jump(age, p["carry_onset"], p["escape_hold"], p["carry_release"])
            else:
                gain = 1.0 if age < p["escape_hold"] else 1.0 - (age - p["escape_hold"]) / p["escape_release"]
                gain_acc = 0.0
            if gain <= 0 and age > p["escape_hold"]:
                self.active = None
                if self.touched:
                    self.active, self.started, self.touched = "groom", events.t, False
                return idle
            command = ReflexCommand("escape", gain, self.offset, activations, detail=detail)
            command.planned_acc = gain_acc * self.offset
            return command

        if self.active == "groom":
            age = events.t - self.started
            if age >= p["groom_time"]:
                self.active = None
                return idle
            swing = p["groom_swing"] * np.sin(2 * np.pi * p["groom_rate"] * age)
            return ReflexCommand("groom", 1.0, np.array([0.0, swing, 0.0]), activations, detail)

        return idle

    @staticmethod
    def _side(velocity: np.ndarray) -> np.ndarray:
        """Unit vector level with the table and square to the object's flight (its left-hand side)."""
        heading = np.array([velocity[0], velocity[1], 0.0])
        heading /= np.linalg.norm(heading) + 1e-9
        return np.cross(UP, heading)

    @staticmethod
    def _way_out(passing: np.ndarray, velocity: np.ndarray, gripper_pos: np.ndarray) -> np.ndarray:
        """Unit sidestep, level with the table and square to the object's flight.

        Level because a lobbed object comes down from above, and because the
        arm has little reach left upward. Steps away from the side the object
        will pass on; on a dead-centre hit, steps toward the middle of the
        workspace, where the arm has the most room.
        """
        side = HandwrittenReflex._side(velocity)
        lean = -float(passing @ side)
        if abs(lean) < 0.03:
            lean = -float(gripper_pos @ side)
        return side if lean >= 0 else -side
