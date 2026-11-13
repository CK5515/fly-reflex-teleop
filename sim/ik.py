"""Turn a wanted gripper pose into joint angles for the Panda's seven arm joints."""

import mujoco
import numpy as np

from sim.scene import HOME_ARM, control_site

ARM_DOFS = 7


class ArmIK:
    """Damped least-squares inverse kinematics, one small step per call.

    The solution is tracked on a private copy of the model state, so it does
    not wobble with the physics; the arm's position actuators chase it.
    """

    def __init__(self, model: mujoco.MjModel, damping: float = 0.05, max_step: float = 0.02):
        self.model = model
        self.scratch = mujoco.MjData(model)
        self.site = model.site(control_site(model)).id
        self.damping = damping
        self.max_step = max_step  # radians per call, per joint
        self.q = HOME_ARM.copy()
        self.lower = model.jnt_range[:ARM_DOFS, 0]
        self.upper = model.jnt_range[:ARM_DOFS, 1]
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def reset(self, q: np.ndarray) -> None:
        self.q = np.array(q[:ARM_DOFS], dtype=float)

    def step(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        d = self.scratch
        d.qpos[:ARM_DOFS] = self.q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)

        error = np.zeros(6)
        error[:3] = target_pos - d.site_xpos[self.site]
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, d.site_xmat[self.site])
        inv = np.zeros(4)
        mujoco.mju_negQuat(inv, site_quat)
        diff = np.zeros(4)
        mujoco.mju_mulQuat(diff, target_quat, inv)
        mujoco.mju_quat2Vel(error[3:], diff, 1.0)

        mujoco.mj_jacSite(self.model, d, self._jacp, self._jacr, self.site)
        jac = np.vstack([self._jacp[:, :ARM_DOFS], self._jacr[:, :ARM_DOFS]])
        gram = jac @ jac.T + self.damping**2 * np.eye(6)
        dq = jac.T @ np.linalg.solve(gram, error)
        # Use the spare degree of freedom to stay near the home posture.
        null = np.eye(ARM_DOFS) - np.linalg.pinv(jac) @ jac
        dq += null @ (0.05 * (HOME_ARM - self.q))

        biggest = np.abs(dq).max()
        if biggest > self.max_step:
            dq *= self.max_step / biggest
        self.q = np.clip(self.q + dq, self.lower, self.upper)
        return self.q
