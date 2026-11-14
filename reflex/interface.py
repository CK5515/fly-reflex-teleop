"""The contract between the scene and whatever supplies the reflexes.

The hand-written reflexes and the fly-brain model both implement `Reflex`,
so either can be plugged into the teleop loop without changing it.
"""

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class SceneEvents:
    """What the arm can sense this instant, relative to its gripper."""

    t: float
    gripper_pos: np.ndarray  # where the gripper is, metres from the arm's base
    object_pos: np.ndarray   # thrown object's position minus the gripper's, metres
    object_vel: np.ndarray   # its velocity minus the gripper's, m/s
    object_radius: float
    contact_force: float     # newtons pressing on the arm from the thrown object (0 = none)
    # Every thrown object, as (position, velocity) relative to the gripper. `object_pos` and
    # `object_vel` above are the most urgent of these.
    objects: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    sights: list | None = None   # per thrown object, the eye's recent frames (sim.eye.Frame), if an eye is fitted
    touching: frozenset[int] = frozenset()   # which thrown objects are touching the arm right now
    target_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))  # acceleration the operator is asking for, m/s^2
    load_on: bool = False    # something loose is being carried and is still on the tray
    load_pos: np.ndarray = field(default_factory=lambda: np.zeros(2))  # its offset from the tray's centre (x, y), metres
    load_vel: np.ndarray = field(default_factory=lambda: np.zeros(2))  # its velocity over the tray, m/s


@dataclass
class ReflexCommand:
    """How a reflex wants to override the operator this instant.

    The arm goes to: operator's target + gain * offset. gain 0 = operator in
    full control. `activations` is every expert's drive, for the live view.

    Steadying a carried load is continuous, not all-or-nothing: `level` asks
    for the gripper to be held level whatever the operator's wrist does, and
    `tilt` then leans it (radians about the world x and y axes) to keep the
    load in place.
    """

    expert: str | None = None
    gain: float = 0.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    activations: dict[str, float] = field(default_factory=dict)
    level: bool = False
    planned_acc: np.ndarray = field(default_factory=lambda: np.zeros(3))  # acceleration the reflex is about to cause
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    detail: dict = field(default_factory=dict)  # anything extra the live view may show (e.g. which experts)


class Reflex(Protocol):
    def step(self, events: SceneEvents) -> ReflexCommand: ...


class NoReflex:
    def step(self, events: SceneEvents) -> ReflexCommand:
        return ReflexCommand()
