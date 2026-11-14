"""Continuous avoidance: keep sliding to wherever the incoming objects will not be.

Unlike the single escape jump, this weighs every object in the air at every
moment. For a ring of directions it asks "if the gripper were a step that way,
how close would each object pass?", then eases toward the safest one. Because
the motion is smooth and known in advance, the steadying reflex can lean into
it and keep a carried load on its tray.

`danger()` is the part a brain model can replace: it returns how dangerous it
is to stay put and to step in each direction.
"""

import numpy as np

from reflex.interface import ReflexCommand, SceneEvents
from reflex.steady import Steady

GRAVITY = np.array([0.0, 0.0, -9.81])
N_DIRECTIONS = 16
ANGLES = np.arange(N_DIRECTIONS) * 2 * np.pi / N_DIRECTIONS
DIRECTIONS = np.stack([np.cos(ANGLES), np.sin(ANGLES), np.zeros(N_DIRECTIONS)], axis=1)  # level with the table
LOOKAHEAD = np.linspace(0.0, 1.0, 51)[1:, None]   # seconds
WORKSPACE_CENTRE = np.array([0.5, 0.0])           # where the arm has the most room, metres from its base


def looming(pos: np.ndarray, vel: np.ndarray, starts: float, full: float, reach: float) -> tuple[float, np.ndarray]:
    """How urgently one object looms (0..1), and its predicted path relative to the gripper [time, 3].

    Urgency rises from 0 when the object is `starts` seconds from its closest approach to 1 at `full`
    seconds, and only if it will pass within `reach` metres.
    """
    path = pos + LOOKAHEAD * vel + 0.5 * LOOKAHEAD**2 * GRAVITY
    if float(vel @ vel) < 0.25:
        return 0.0, path
    distance = np.linalg.norm(path, axis=1)
    nearest = int(np.argmin(distance))
    if nearest == len(path) - 1 or distance[nearest] > reach:
        return 0.0, path
    ttc = float(LOOKAHEAD[nearest, 0])
    return float(np.clip((starts - ttc) / (starts - full), 0.0, 1.0)), path


class ContinuousReflex:
    def __init__(
        self,
        step: float = 0.30,          # how far the gripper will move to get clear, metres
        clear: float = 0.22,         # an object passing farther than this is no danger, metres
        starts: float = 0.8,         # seconds before closest approach at which an object starts to matter
        full: float = 0.6,           # ... and at which it matters fully
        reach: float = 0.35,         # objects passing farther than this are ignored, metres
        quickness: float = 14.0,     # how fast the gripper eases to its goal, rad/s
        carry_quickness: float = 12.0,  # ... when carrying a load
        carry_step: float = 0.24,    # a shorter step when carrying, metres
        carry_acc: float = 6.0,      # and no harder than this, m/s^2: about what leaning the tray can cancel
        acc_smoothing: float = 0.05,  # seconds over which the acceleration builds up (the lean needs time too)
        handling_quickness: float = 11.0,  # how fast a carried tray follows the hand, rad/s
        handling_acc: float = 4.0,   # the hardest a carried tray is allowed to accelerate, m/s^2
        stay_bias: float = 0.15,     # reluctance to move away from the middle of the workspace
        stickiness: float = 0.15,    # reluctance to change direction once moving
    ):
        self.p = {k: v for k, v in locals().items() if k != "self"}
        self.steady = Steady()
        self.offset = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.choice: int | None = None
        self.lag = np.zeros(3)            # how far the smoothed tray target trails the hand's target
        self.lag_velocity = np.zeros(3)
        self._target: np.ndarray | None = None
        self._time: float | None = None

    def danger(self, events: SceneEvents) -> tuple[float, np.ndarray, dict]:
        """(danger of staying put, danger of stepping in each of DIRECTIONS, extra detail). 0 = safe."""
        p = self.p
        stay, steps = 0.0, np.zeros(N_DIRECTIONS)
        for pos, vel in events.objects:
            # Judge from where the operator wants the gripper, not from where the reflex has already
            # moved it: otherwise a successful dodge looks safe and the arm drifts back into the path.
            urgency, path = looming(pos + self._shift(), vel, p["starts"], p["full"], p["reach"])
            if urgency == 0.0:
                continue
            closeness = lambda gap: np.clip(1.0 - gap / p["clear"], 0.0, 1.0)
            stay += urgency * closeness(np.linalg.norm(path, axis=1).min())
            gaps = np.linalg.norm(path[None] - self._step(events) * DIRECTIONS[:, None], axis=2).min(1)
            steps += urgency * closeness(gaps)
        return stay, steps, {}

    relative_to_target = False   # True when objects are already measured from the operator's target point

    def _shift(self) -> np.ndarray:
        """From gripper-relative to target-relative positions."""
        return np.zeros(3) if self.relative_to_target else self.offset

    def _step(self, events: SceneEvents) -> float:
        return self.p["carry_step"] if events.load_on else self.p["step"]

    def step(self, events: SceneEvents) -> ReflexCommand:
        p = self.p
        dt = 0.0 if self._time is None else max(events.t - self._time, 0.0)
        self._time = events.t
        stay, steps, detail = self.danger(events)

        if stay < 0.05:
            self.choice, goal = None, np.zeros(3)
        else:
            # Prefer directions that keep the arm near the middle of its workspace, and the current one.
            outward = (events.gripper_pos[:2] - self.offset[:2] - self.lag[:2] - WORKSPACE_CENTRE) / self._step(events)
            cost = steps + p["stay_bias"] * (DIRECTIONS[:, :2] @ outward)
            if self.choice is not None:
                cost[self.choice] -= p["stickiness"]
            self.choice = int(np.argmin(cost))
            goal = self._step(events) * DIRECTIONS[self.choice]   # commit to a full step: half a dodge is no dodge

        # Ease toward the goal like a critically damped spring: smooth, and its acceleration is known.
        w = p["carry_quickness"] if events.load_on else p["quickness"]
        wanted = w * w * (goal - self.offset) - 2 * w * self.velocity
        if events.load_on:   # no harder than the tray's lean can cancel, and built up smoothly
            size = np.linalg.norm(wanted)
            if size > p["carry_acc"]:
                wanted *= p["carry_acc"] / size
            self.acceleration += (wanted - self.acceleration) * min(dt / p["acc_smoothing"], 1.0)
        else:
            self.acceleration = wanted
        acceleration = self.acceleration
        self.velocity = self.velocity + acceleration * dt
        self.offset = self.offset + self.velocity * dt

        # While a load is carried, the tray does not copy the hand's every jerk: it follows the hand
        # through a smooth, acceleration-limited path, so the lean always knows what is coming.
        handling = np.zeros(3)
        if events.load_on and events.target_pos is not None:
            if self._target is None:
                self._target = events.target_pos.copy()
            hand_step = events.target_pos - self._target      # how far the hand's target just moved
            self._target = events.target_pos.copy()
            self.lag -= hand_step                              # the smoothed target stays put; the hand moved on
            hw = p["handling_quickness"]
            handling = -hw * hw * self.lag - 2 * hw * self.lag_velocity
            size = np.linalg.norm(handling)
            if size > p["handling_acc"]:
                handling *= p["handling_acc"] / size
            self.lag_velocity = self.lag_velocity + handling * dt
            self.lag = self.lag + self.lag_velocity * dt
            if np.linalg.norm(self.lag) > 0.25:                # never trail the hand by more than this
                self.lag *= 0.25 / np.linalg.norm(self.lag)
        else:
            self.lag[:] = 0.0
            self.lag_velocity[:] = 0.0
            self._target = None

        dodging = float(np.linalg.norm(self.offset)) > 0.02
        holding_back = float(np.linalg.norm(self.lag)) > 0.03
        expert = "escape" if dodging else "steady" if holding_back else None
        command = ReflexCommand(expert, 1.0 if expert else 0.0, self.offset + self.lag,
                                {"escape": 2.0 * min(stay, 1.0)}, detail=detail)
        command.planned_acc = acceleration + handling
        if events.load_on:
            command.level = True
            # The tray's whole motion is planned here (smoothed hand + dodge), so lean into exactly that.
            command.tilt = self.steady.tilt(events, command.planned_acc, planned_only=True)
            command.activations["steady"] = float(np.abs(command.tilt).max() / self.steady.max_tilt * 2)
        return command
