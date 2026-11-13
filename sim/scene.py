"""Build the simulated scene: Panda arm, table, a cube to pick up, a ball to throw.

The arm comes from MuJoCo Menagerie (third_party/, fetched separately); the
rest is added here so the Menagerie files stay untouched.
"""

from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
PANDA_XML = REPO / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

HOME_ARM = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
GRIPPER_OPEN = 255.0
TABLE_TOP_Z = 0.20
CUBE_START = np.array([0.55, 0.0, TABLE_TOP_Z + 0.02])
BALL_PARKED = np.array([3.0, 0.0, 0.05])
BALL_RADIUS = 0.04

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
PLANE = mujoco.mjtGeom.mjGEOM_PLANE


def build_model() -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):  # the stock keyframe no longer fits once bodies are added
        spec.delete(key)

    # Where a viewer's camera starts: on the arm and table, not stretched out to the parked ball.
    spec.stat.center = [0.4, 0.0, 0.35]
    spec.stat.extent = 0.9

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2.5])
    world.add_light(pos=[1.0, -1.0, 2.0], dir=[-0.4, 0.4, -1.0])
    world.add_geom(name="floor", type=PLANE, size=[0, 0, 0.05], rgba=[0.18, 0.2, 0.24, 1])
    world.add_geom(
        name="table",
        type=BOX,
        pos=[0.55, 0, TABLE_TOP_Z / 2],
        size=[0.22, 0.35, TABLE_TOP_Z / 2],
        rgba=[0.55, 0.45, 0.35, 1],
    )

    # Where the arm grips: between the fingertips.
    spec.body("hand").add_site(name="grasp", pos=[0, 0, 0.1034], size=[0.005, 0.005, 0.005])

    cube = world.add_body(name="cube", pos=CUBE_START)
    cube.add_freejoint()
    cube.add_geom(
        name="cube", type=BOX, size=[0.02, 0.02, 0.02], mass=0.05,
        rgba=[0.9, 0.3, 0.2, 1], friction=[1.5, 0.02, 0.001],
    )

    ball = world.add_body(name="ball", pos=BALL_PARKED)
    ball.add_freejoint()
    ball.add_geom(name="ball", type=SPHERE, size=[BALL_RADIUS, 0, 0], mass=0.05, rgba=[0.2, 0.8, 0.3, 1])

    # The pose the operator is asking for. Drag it in the viewer to steer by mouse.
    target = world.add_body(name="target", mocap=True, pos=[0.5, 0, 0.5])
    target.add_geom(type=BOX, size=[0.02, 0.02, 0.02], rgba=[0.2, 0.6, 1.0, 0.35], contype=0, conaffinity=0)

    return spec.compile()


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Arm at home with the gripper open, target on the gripper, objects at rest."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_ARM
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = HOME_ARM
    data.ctrl[7] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    site = model.site("grasp").id
    data.mocap_pos[0] = data.site_xpos[site]
    mujoco.mju_mat2Quat(data.mocap_quat[0], data.site_xmat[site])
