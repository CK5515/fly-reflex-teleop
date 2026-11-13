"""Read the scene the way the arm's senses would, and throw things at it."""

import mujoco
import numpy as np

from reflex.interface import SceneEvents
from sim.scene import BALL_RADIUS

GRAVITY = np.array([0.0, 0.0, -9.81])


class Senses:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.site = model.site("grasp").id
        self.hand = model.body("hand").id
        self.ball_body = model.body("ball").id
        self.ball_geom = model.geom("ball").id
        joint = model.body("ball").jntadr[0]
        self.ball_qpos = model.jnt_qposadr[joint]
        self.ball_qvel = model.jnt_dofadr[joint]
        self.arm_root = model.body_rootid[self.hand]
        self._force = np.zeros(6)
        self._vel = np.zeros(6)

    def read(self, data: mujoco.MjData) -> SceneEvents:
        m = self.model
        mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_SITE, self.site, self._vel, 0)
        hand_vel = self._vel[3:].copy()
        ball_vel = data.qvel[self.ball_qvel:self.ball_qvel + 3]
        return SceneEvents(
            t=data.time,
            gripper_pos=data.site_xpos[self.site].copy(),
            object_pos=data.xpos[self.ball_body] - data.site_xpos[self.site],
            object_vel=ball_vel - hand_vel,
            object_radius=BALL_RADIUS,
            contact_force=self.touch(data),
        )

    def touch(self, data: mujoco.MjData) -> float:
        """Total normal force between the ball and any part of the arm."""
        total = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            if self.ball_geom not in (c.geom1, c.geom2):
                continue
            other = c.geom2 if c.geom1 == self.ball_geom else c.geom1
            if self.model.body_rootid[self.model.geom_bodyid[other]] != self.arm_root:
                continue
            mujoco.mj_contactForce(self.model, data, i, self._force)
            total += abs(self._force[0])
        return total

    def throw(self, data: mujoco.MjData, rng: np.random.Generator, flight: float = 0.6, aim_at: str = "gripper") -> None:
        """Launch the ball from in front of the arm so it arrives after `flight` seconds."""
        if aim_at == "gripper":
            target = data.site_xpos[self.site].copy()
        else:  # the forearm, to provoke a touch instead of a near miss
            target = data.xpos[self.model.body("link5").id].copy()
        angle = rng.uniform(-1.2, 1.2)
        start = target + np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), rng.uniform(0.0, 0.4)])
        velocity = (target - start - 0.5 * GRAVITY * flight**2) / flight
        data.qpos[self.ball_qpos:self.ball_qpos + 3] = start
        data.qpos[self.ball_qpos + 3:self.ball_qpos + 7] = [1, 0, 0, 0]
        data.qvel[self.ball_qvel:self.ball_qvel + 3] = velocity
        data.qvel[self.ball_qvel + 3:self.ball_qvel + 6] = 0
