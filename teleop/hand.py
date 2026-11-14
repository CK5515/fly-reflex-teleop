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


# How far the hand may carry the target from its starting point, metres: (low, high) on x, y, z.
# Keeps the arm inside the space it can reach comfortably; `floor` keeps it off the table.
WORKSPACE = np.array([[-0.25, 0.25], [-0.35, 0.35], [-0.45, 0.25]])


class Retarget:
    """Map hand motion onto the gripper, relative to where both started.

    The first pose seen is the reference: from then on the gripper target
    moves and turns by however much the hand has moved and turned since.
    `frame` rotates the source's position axes into the simulation's, and
    `turn_frame` does the same for its orientation axes (a glove reports the
    two in different frames). Both are the identity until calibrated; see
    mocap/calibrate.py.
    """

    def __init__(self, home_pos, home_quat, scale: float = 1.0, frame: np.ndarray | None = None,
                 turn_frame: np.ndarray | None = None, floor: float = 0.0,
                 tilt_scale: float = 1.0, max_tilt: float | None = None, follow_yaw: bool = True):
        self.floor = floor   # the lowest the target may go, metres above the ground
        # How the wrist's turning is passed on. A real wrist turns far more than a carried tray
        # should: `tilt_scale` shrinks tipping, `max_tilt` (radians) caps it, and `follow_yaw=False`
        # ignores twisting about the vertical, which would swing a tray held out to one side.
        self.tilt_scale, self.max_tilt, self.follow_yaw = tilt_scale, max_tilt, follow_yaw
        self.home_pos = np.array(home_pos, dtype=float)
        self.home_quat = np.array(home_quat, dtype=float)
        self.scale = scale
        self.frame = np.eye(3) if frame is None else np.asarray(frame, dtype=float)
        turn = self.frame if turn_frame is None else np.asarray(turn_frame, dtype=float)
        self.turn_quat = np.zeros(4)
        mujoco.mju_mat2Quat(self.turn_quat, turn.flatten())
        self.ref: HandPose | None = None

    def recalibrate(self) -> None:
        self.ref = None

    def __call__(self, hand: HandPose) -> tuple[np.ndarray, np.ndarray, float]:
        if self.ref is None:
            self.ref = HandPose(hand.pos.copy(), hand.quat.copy(), hand.grip)
        moved = self.scale * self.frame @ (hand.pos - self.ref.pos)
        pos = self.home_pos + np.clip(moved, WORKSPACE[:, 0], WORKSPACE[:, 1])
        pos[2] = max(pos[2], self.floor)
        turned = _mul(hand.quat, _inv(self.ref.quat))  # rotation since the reference, source frame
        turned = _mul(self.turn_quat, _mul(turned, _inv(self.turn_quat)))
        if self.tilt_scale != 1.0 or self.max_tilt is not None or not self.follow_yaw:
            turn = np.zeros(3)
            mujoco.mju_quat2Vel(turn, turned, 1.0)          # as a rotation vector: about x, y, z
            turn[:2] *= self.tilt_scale
            if not self.follow_yaw:
                turn[2] = 0.0
            tilt = np.linalg.norm(turn[:2])
            if self.max_tilt is not None and tilt > self.max_tilt:
                turn[:2] *= self.max_tilt / tilt
            angle = np.linalg.norm(turn)
            turned = np.array([1.0, 0.0, 0.0, 0.0])
            if angle > 1e-9:
                mujoco.mju_axisAngle2Quat(turned, turn / angle, angle)
        quat = _mul(turned, self.home_quat)
        return pos, quat, float(np.clip(hand.grip, 0.0, 1.0))


def level(quat: np.ndarray, home_quat: np.ndarray) -> np.ndarray:
    """`quat` with any tipping removed: only its turn about the vertical, relative to `home_quat`, is kept."""
    turned = _mul(quat, _inv(home_quat))
    twist = np.array([turned[0], 0.0, 0.0, turned[3]])
    norm = np.linalg.norm(twist)
    twist = twist / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0, 0.0])
    return _mul(twist, home_quat)


def lean(quat: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """`quat` leaned by `tilt` radians about the world x and y axes."""
    angle = float(np.linalg.norm(tilt))
    if angle < 1e-9:
        return quat
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, np.array([tilt[0], tilt[1], 0.0]) / angle, angle)
    return _mul(q, quat)


class ScriptedHand:
    """A stand-in operator: moves through timed waypoints, looping.

    Waypoints are (seconds, displacement xyz, grip). `wobble` adds an unsteady
    wrist: it rocks by that many radians, slowly, about both level axes.
    """

    def __init__(self, waypoints: list[tuple[float, tuple[float, float, float], float]], loop: bool = True,
                 wobble: float = 0.0, wobble_from: float = 0.0):
        self.wobble, self.wobble_from = wobble, wobble_from   # the wrist is steady before `wobble_from` seconds
        self.times = np.array([w[0] for w in waypoints])
        self.points = np.array([w[1] for w in waypoints], dtype=float)
        self.grips = np.array([w[2] for w in waypoints], dtype=float)
        self.loop = loop

    def latest(self, t: float) -> HandPose:
        if self.loop:
            t = t % self.times[-1]
        pos = np.array([np.interp(t, self.times, self.points[:, i]) for i in range(3)])
        grip = float(np.interp(t, self.times, self.grips))
        quat = np.array([1.0, 0.0, 0.0, 0.0])
        if self.wobble and t >= self.wobble_from:
            quat = lean(quat, self.wobble * np.array([np.sin(2 * np.pi * 0.4 * t), np.sin(2 * np.pi * 0.3 * t + 1.0)]))
        return HandPose(pos, quat, grip)
