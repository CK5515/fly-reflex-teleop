"""Turn a wanted gripper pose into joint angles for the Panda's seven arm joints."""

import mujoco
import numpy as np

from sim.scene import HOME_ARM, control_site

ARM_DOFS = 7
SHOULDER = np.array([0.0, 0.0, 0.333])
REACH = 0.80   # metres from the shoulder that the steered point may be asked to go


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
        self._stuck = 0
        self.restarts = 0   # times the solver had to start again from the home posture
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def steer(self, site: str) -> None:
        """Change which point on the arm is steered (between the fingertips, or the tray's centre)."""
        self.site = self.model.site(site).id

    def reset(self, q: np.ndarray) -> None:
        self.q = np.array(q[:ARM_DOFS], dtype=float)

    def step(self, target_pos: np.ndarray, target_quat: np.ndarray) -> np.ndarray:
        # Keep the target within reach: asking for more locks the elbow straight.
        from_shoulder = np.asarray(target_pos, dtype=float) - SHOULDER
        distance = np.linalg.norm(from_shoulder)
        if distance > REACH:
            target_pos = SHOULDER + from_shoulder * (REACH / distance)
        miss = self._update(target_pos, target_quat, self.max_step)
        # Small steps can get stuck against a joint limit. If the solution has been well off the
        # target for a while, solve again from the home posture and continue from there.
        self._stuck = self._stuck + 1 if miss > 0.10 else 0
        if self._stuck > 100:
            self.q = HOME_ARM.copy()
            for _ in range(80):
                self._update(target_pos, target_quat, 0.1)
            self._stuck = 0
            self.restarts += 1
        return self.q

    def _update(self, target_pos: np.ndarray, target_quat: np.ndarray, max_step: float) -> float:
        """One damped least-squares step. Returns how far the solution was from the target, metres."""
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
        if biggest > max_step:
            dq *= max_step / biggest
        self.q = np.clip(self.q + dq, self.lower, self.upper)
        return float(np.linalg.norm(error[:3]))
