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


@dataclass
class ReflexCommand:
    """How a reflex wants to override the operator this instant.

    The arm goes to: operator's target + gain * offset. gain 0 = operator in
    full control. `activations` is every expert's drive, for the live view.
    """

    expert: str | None = None
    gain: float = 0.0
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    activations: dict[str, float] = field(default_factory=dict)


class Reflex(Protocol):
    def step(self, events: SceneEvents) -> ReflexCommand: ...


class NoReflex:
    def step(self, events: SceneEvents) -> ReflexCommand:
        return ReflexCommand()
