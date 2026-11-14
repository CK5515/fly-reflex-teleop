"""Read the scene the way the arm's senses would, and throw things at it."""

import mujoco
import numpy as np

from reflex.interface import SceneEvents
from sim.scene import BALL_PARKED, BALL_RADIUS, TRAY_HALF, ball_names, control_site

GRAVITY = np.array([0.0, 0.0, -9.81])


class Senses:
    def __init__(self, model: mujoco.MjModel, eye=None):
        self.eye = eye   # sim.eye.Eye, or None when the reflex is given exact positions
        self.model = model
        self.site = model.site("grasp").id
        self.hand = model.body("hand").id
        names = ball_names(model)
        self.ball_bodies = [model.body(n).id for n in names]
        self.ball_geoms = {model.geom(n).id for n in names}
        self.ball_index = {model.geom(n).id: i for i, n in enumerate(names)}
        self._touching: set[int] = set()
        joints = [model.body(n).jntadr[0] for n in names]
        self.ball_qpos = [model.jnt_qposadr[j] for j in joints]
        self.ball_qvel = [model.jnt_dofadr[j] for j in joints]
        self.ball_body = self.ball_bodies[0]
        self._next_ball = 0
        self.arm_root = model.body_rootid[self.hand]
        self._force = np.zeros(6)
        self._vel = np.zeros(6)
        self.carrying = control_site(model) == "tray"
        if self.carrying:
            self.site = model.site("tray").id       # everything is sensed relative to the tray's centre
            self.load_body = model.body("load").id
            self.load_qvel = model.jnt_dofadr[model.body("load").jntadr[0]]
            self.load_geom = model.geom("load").id
        self._target = None            # the operator's target and its smoothed velocity and acceleration
        self._target_vel = np.zeros(3)
        self._target_acc = np.zeros(3)

    def read(self, data: mujoco.MjData) -> SceneEvents:
        m = self.model
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_SITE, self.site, self._vel, 0)
        hand_vel = self._vel[3:].copy()
        # With several balls about, attend to the one closing fastest for its distance.
        site_pos = data.site_xpos[self.site]
        urgency, object_pos, object_vel = -np.inf, None, None
        objects = []
        for body, qvel in zip(self.ball_bodies, self.ball_qvel):
            pos = data.xpos[body] - site_pos
            vel = data.qvel[qvel:qvel + 3] - hand_vel
            objects.append((pos, vel))
            closing = -float(pos @ vel) / max(float(pos @ pos), 1e-9)   # 1 / seconds to reach us, roughly
            if closing > urgency:
                urgency, object_pos, object_vel = closing, pos, vel
        # What the operator is asking for, differenced twice and smoothed over about 50 ms each time.
        # Taken from the target, not the arm: the arm's own motion includes the steadying reflex's
        # leaning, and feeding that back made the reflex shake the arm.
        target = data.mocap_pos[0].copy()
        if self._target is not None:
            vel = (target - self._target) / m.opt.timestep
            new_vel = self._target_vel + 0.04 * (vel - self._target_vel)
            self._target_acc += 0.04 * ((new_vel - self._target_vel) / m.opt.timestep - self._target_acc)
            self._target_vel = new_vel
        self._target = target
        load = {}
        if self.carrying:
            offset = data.xpos[self.load_body] - data.site_xpos[self.site]
            on = bool(np.abs(offset[:2]).max() < TRAY_HALF + 0.01 and abs(offset[2]) < 0.06)
            load = dict(load_on=on, load_pos=offset[:2].copy(),
                        load_vel=(data.qvel[self.load_qvel:self.load_qvel + 3] - hand_vel)[:2].copy())
        return SceneEvents(
            target_acc=self._target_acc.copy(),
            **load,
            t=data.time,
            gripper_pos=data.site_xpos[self.site].copy(),
            object_pos=object_pos,
            object_vel=object_vel,
            object_radius=BALL_RADIUS,
            contact_force=self.touch(data),
            objects=objects,
            sights=None if self.eye is None else self.eye.look(
                data.time, [data.xpos[b] - data.mocap_pos[0] for b in self.ball_bodies], BALL_RADIUS),
            touching=frozenset(self._touching),
        )

    def touch(self, data: mujoco.MjData) -> float:
        """Total normal force between thrown balls and any part of the arm."""
        total = 0.0
        self._touching = set()
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self.ball_geoms:
                other = c.geom2
            elif c.geom2 in self.ball_geoms:
                other = c.geom1
            else:
                continue
            if self.model.body_rootid[self.model.geom_bodyid[other]] != self.arm_root:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._force)
            total += abs(self._force[0])
            ball = c.geom1 if c.geom1 in self.ball_geoms else c.geom2
            self._touching.add(self.ball_index[ball])
        return total

    def load_struck(self, data: mujoco.MjData) -> bool:
        """Whether a thrown ball is touching the carried load right now."""
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 == self.load_geom and c.geom2 in self.ball_geoms) or (c.geom2 == self.load_geom and c.geom1 in self.ball_geoms):
                return True
        return False

    def park(self, data: mujoco.MjData, ball: int) -> None:
        """Take a spent ball out of the scene, so it cannot lie on the tray or roll back into the arm."""
        qpos, qvel = self.ball_qpos[ball], self.ball_qvel[ball]
        data.qpos[qpos:qpos + 3] = BALL_PARKED + [0.0, 0.15 * ball, 0.0]
        data.qvel[qvel:qvel + 6] = 0
        mujoco.mj_forward(self.model, data)

    def throw(self, data: mujoco.MjData, rng: np.random.Generator, flight: float = 0.6,
              aim_at: str = "gripper", unavailable: set[int] | None = None) -> int | None:
        """Launch an available ball. Return its index, or None if every ball is still in flight."""
        unavailable = unavailable or set()
        thrown = None
        for i in range(len(self.ball_bodies)):
            candidate = (self._next_ball + i) % len(self.ball_bodies)
            if candidate not in unavailable:
                thrown = candidate
                break
        if thrown is None:
            return None
        qpos, qvel = self.ball_qpos[thrown], self.ball_qvel[thrown]
        self._next_ball = (thrown + 1) % len(self.ball_bodies)
        if aim_at == "gripper":
            target = data.site_xpos[self.site].copy()
        else:  # the forearm, to provoke a touch instead of a near miss
            target = data.xpos[self.model.body("link5").id].copy()
        angle = rng.uniform(-1.2, 1.2)
        start = target + np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), rng.uniform(0.0, 0.4)])
        velocity = (target - start - 0.5 * GRAVITY * flight**2) / flight
        data.qpos[qpos:qpos + 3] = start
        data.qpos[qpos + 3:qpos + 7] = [1, 0, 0, 0]
        data.qvel[qvel:qvel + 3] = velocity
        data.qvel[qvel + 3:qvel + 6] = 0
        mujoco.mj_forward(self.model, data)
        return thrown
