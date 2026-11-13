"""The operator's hand as plain numbers, and how it steers the gripper.

Every input (glove, script, later a replayed tape) produces a HandPose, so
the simulation never needs to know where the motion came from.
"""

from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


@dataclass
class HandPose:
    pos: np.ndarray   # metres, in the source's own frame
    quat: np.ndarray  # w, x, y, z
    grip: float       # 0 = open hand, 1 = closed pinch


class HandSource(Protocol):
    def latest(self, t: float) -> HandPose | None: ...


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, a, b)
    return out


def _inv(q: np.ndarray) -> np.ndarray:
    out = np.zeros(4)
    mujoco.mju_negQuat(out, q)
    return out


class Retarget:
    """Map hand motion onto the gripper, relative to where both started.

    The first pose seen is the reference: from then on the gripper target
    moves and turns by however much the hand has moved and turned since.
    `frame` rotates the source's axes into the simulation's (identity until
    calibrated against a real glove).
    """

    def __init__(self, home_pos, home_quat, scale: float = 1.0, frame: np.ndarray | None = None):
        self.home_pos = np.array(home_pos, dtype=float)
        self.home_quat = np.array(home_quat, dtype=float)
        self.scale = scale
        self.frame = np.eye(3) if frame is None else frame
        self.frame_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.frame_quat, self.frame.flatten())
        self.ref: HandPose | None = None

    def recalibrate(self) -> None:
        self.ref = None

    def __call__(self, hand: HandPose) -> tuple[np.ndarray, np.ndarray, float]:
        if self.ref is None:
            self.ref = HandPose(hand.pos.copy(), hand.quat.copy(), hand.grip)
        pos = self.home_pos + self.scale * self.frame @ (hand.pos - self.ref.pos)
        turned = _mul(hand.quat, _inv(self.ref.quat))  # rotation since the reference, source frame
        turned = _mul(self.frame_quat, _mul(turned, _inv(self.frame_quat)))
        quat = _mul(turned, self.home_quat)
        return pos, quat, float(np.clip(hand.grip, 0.0, 1.0))


class ScriptedHand:
    """A stand-in operator: moves through timed waypoints, looping.

    Waypoints are (seconds, displacement xyz, grip); the hand does not turn.
    """

    def __init__(self, waypoints: list[tuple[float, tuple[float, float, float], float]], loop: bool = True):
        self.times = np.array([w[0] for w in waypoints])
        self.points = np.array([w[1] for w in waypoints], dtype=float)
        self.grips = np.array([w[2] for w in waypoints], dtype=float)
        self.loop = loop

    def latest(self, t: float) -> HandPose:
        if self.loop:
            t = t % self.times[-1]
        pos = np.array([np.interp(t, self.times, self.points[:, i]) for i in range(3)])
        grip = float(np.interp(t, self.times, self.grips))
        return HandPose(pos, np.array([1.0, 0.0, 0.0, 0.0]), grip)
