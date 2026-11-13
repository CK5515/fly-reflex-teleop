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
TRAY_HALF = 0.09          # the tray is a square plate, this far from centre to edge
TRAY_OFFSET = [0.17, 0.0, 0.09]   # tray centre in the gripper's frame: held out to one side, level at home
LOAD_RADIUS = 0.02

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
PLANE = mujoco.mjtGeom.mjGEOM_PLANE


def build_model(load: str | None = None) -> mujoco.MjModel:
    """`load="tray"` fixes a rimless tray to the gripper with a loose ball on it (the thing to keep steady)."""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):  # the stock keyframe no longer fits once bodies are added
        spec.delete(key)

    # Where a viewer's camera starts: on the arm and table, not stretched out to the parked ball.
    spec.stat.center = [0.4, 0.0, 0.35]
    spec.stat.extent = 0.9

    # Gravity compensation, as real arms have: without it the arm sags about a degree under a
    # carried tray, which is enough to roll a ball off it.
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0

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

    if load == "tray":
        hand = spec.body("hand")
        hand.add_geom(name="tray", type=BOX, pos=TRAY_OFFSET, size=[TRAY_HALF, TRAY_HALF, 0.004],
                      mass=0.1, rgba=[0.85, 0.85, 0.9, 1], friction=[0.6, 0.005, 0.001], condim=6)
        # The top face is the gripper's -z side (the gripper points down).
        hand.add_site(name="tray", pos=[TRAY_OFFSET[0], TRAY_OFFSET[1], TRAY_OFFSET[2] - 0.004], size=[0.004, 0.004, 0.004])
        ball = world.add_body(name="load", pos=[0.4, 0.3, 1.0])
        ball.add_freejoint()
        ball.add_geom(name="load", type=SPHERE, size=[LOAD_RADIUS, 0, 0], mass=0.03, rgba=[0.95, 0.75, 0.1, 1],
                      friction=[0.6, 0.005, 0.001], condim=6)   # rolls once the tray tilts about 3 degrees

    return spec.compile()


def control_site(model: mujoco.MjModel) -> str:
    """The point the operator steers: the tray's centre when there is a tray, else between the fingertips."""
    return "tray" if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tray") >= 0 else "grasp"


def place_load(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the loose ball back on the middle of the tray, moving with it."""
    tray = model.site("tray").id
    joint = model.body("load").jntadr[0]
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[q:q + 3] = data.site_xpos[tray] + [0, 0, LOAD_RADIUS + 0.001]
    data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tray, velocity, 0)
    data.qvel[v:v + 3] = velocity[3:]
    data.qvel[v + 3:v + 6] = 0


def reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Arm at home with the gripper open, target on the gripper, objects at rest."""
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_ARM
    data.qpos[7:9] = 0.04
    data.ctrl[:7] = HOME_ARM
    data.ctrl[7] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    site = model.site(control_site(model)).id
    data.mocap_pos[0] = data.site_xpos[site]
    mujoco.mju_mat2Quat(data.mocap_quat[0], data.site_xmat[site])
    if control_site(model) == "tray":
        place_load(model, data)
        mujoco.mj_forward(model, data)
