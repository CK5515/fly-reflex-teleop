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
LOAD_PARKED = np.array([3.0, 1.0, 0.03])   # where the loose ball waits when no paddle is held
# Collision groups. The robot's own parts are in group 1. A paddle that can be picked up is in group 2
# only, so it never collides with the hand that holds it; everything else is in both.
THINGS = dict(contype=3, conaffinity=3)
LOOKS = dict(contype=0, conaffinity=0)   # drawn only: touches nothing
PADDLE = dict(contype=2, conaffinity=2)
PEDESTAL_TOP = np.array([0.50, 0.44, 0.29])   # where the paddle's plate rests until it is picked up
BLOCK_HALF = 0.0225       # wooden blocks are 4.5 cm cubes
WOOD = [[0.76, 0.58, 0.36, 1], [0.66, 0.47, 0.28, 1], [0.84, 0.68, 0.45, 1]]
HANDLE_HALF_WIDTH = 0.012  # the tray's handle, which the fingers close on
CARRY_GRIP = 255.0 * HANDLE_HALF_WIDTH / 0.04   # gripper command that brings the fingers onto the handle

BOX = mujoco.mjtGeom.mjGEOM_BOX
SPHERE = mujoco.mjtGeom.mjGEOM_SPHERE
PLANE = mujoco.mjtGeom.mjGEOM_PLANE
HIDDEN = [0, 0, 0, 0]   # reference points are for the controller, not the viewer: by default they draw as grey balls


def ball_names(model: mujoco.MjModel) -> list[str]:
    """Names of the balls that can be thrown: "ball", "ball1", "ball2", ..."""
    names, i = ["ball"], 1
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ball{i}") >= 0:
        names.append(f"ball{i}")
        i += 1
    return names


def _dress(spec: mujoco.MjSpec) -> None:
    """Sky, floor and bench materials. Looks only: none of it takes part in the physics."""
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.13, 0.16, 0.22], rgb2=[0.02, 0.025, 0.035], width=256, height=1536)
    spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.15, 0.17, 0.21], rgb2=[0.13, 0.15, 0.19], mark=mujoco.mjtMark.mjMARK_EDGE,
                     markrgb=[0.26, 0.29, 0.36], width=300, height=300)
    floor = spec.add_material(name="floor", texrepeat=[4, 4], texuniform=True, reflectance=0.08)
    floor.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    # A cool mid grey, so the wooden blocks, the pale tray and the yellow ball all stand out against it.
    spec.add_material(name="benchtop", rgba=[0.40, 0.45, 0.52, 1], specular=0.25, shininess=0.4)
    spec.add_material(name="steel", rgba=[0.22, 0.23, 0.26, 1], specular=0.6, shininess=0.7)


def build_model(load: str | None = None, balls: int = 1, target_marker: bool = False, blocks: int = 0) -> mujoco.MjModel:
    """`load="tray"` fixes a rimless tray to the gripper with a loose ball on it (the thing to keep steady).
    `load="grab"` leaves the same tray on a pedestal as a paddle to be picked up: see `grab` and `release`.
    `balls` is how many balls can be in the air at once. `blocks` puts that many wooden blocks on
    the table to stack, in place of the single cube. `target_marker` draws a small grey ball where the
    operator is asking the gripper to be; without it that point is invisible."""
    spec = mujoco.MjSpec.from_file(str(PANDA_XML))
    for key in list(spec.keys):  # the stock keyframe no longer fits once bodies are added
        spec.delete(key)

    # Where a viewer's camera starts: on the arm and table, not stretched out to the parked ball.
    spec.stat.center = [0.4, 0.0, 0.35]
    spec.stat.extent = 0.9
    # Seen as if the arm were your own right arm: from where your head would be, above and behind
    # the shoulder and a little to the left, looking down the arm at what it holds.
    spec.stat.center = [0.50, 0.08, 0.36]
    spec.stat.extent = 0.62
    spec.visual.global_.azimuth = -26
    spec.visual.global_.elevation = -38

    # Gravity compensation, as real arms have: without it the arm sags about a degree under a
    # carried tray, which is enough to roll a ball off it.
    for body in spec.bodies:
        if body.name.startswith("link") or body.name in ("hand", "left_finger", "right_finger"):
            body.gravcomp = 1.0

    world = spec.worldbody
    world.add_light(pos=[0, 0, 2.5])
    world.add_light(pos=[1.0, -1.0, 2.0], dir=[-0.4, 0.4, -1.0])
    _dress(spec)
    world.add_geom(name="floor", type=PLANE, size=[0, 0, 0.05], material="floor", **THINGS)
    # The table the physics sees is a solid block, hidden (group 3, like the arm's own collision
    # shapes). What is drawn in its place is a bench of the same footprint and height.
    centre, half = np.array([0.55, 0.0]), np.array([0.22, 0.35])
    world.add_geom(name="table", type=BOX, pos=[*centre, TABLE_TOP_Z / 2], size=[*half, TABLE_TOP_Z / 2], group=3, **THINGS)
    top = 0.025   # thickness of the bench top
    world.add_geom(type=BOX, pos=[*centre, TABLE_TOP_Z - top / 2], size=[*half, top / 2], material="benchtop", **LOOKS)
    world.add_geom(type=BOX, pos=[*centre, 0.07], size=[*(half - 0.03), 0.006], material="steel", **LOOKS)   # a shelf
    for x in (-1, 1):
        for y in (-1, 1):
            world.add_geom(type=BOX, pos=[*(centre + [x, y] * (half - 0.025)), (TABLE_TOP_Z - top) / 2],
                           size=[0.018, 0.018, (TABLE_TOP_Z - top) / 2], material="steel", **LOOKS)

    # Where the arm grips: between the fingertips.
    spec.body("hand").add_site(name="grasp", pos=[0, 0, 0.1034], size=[0.005, 0.005, 0.005], rgba=HIDDEN)

    if blocks:   # wooden blocks to stack, in a row across the table
        for i in range(blocks):
            block = world.add_body(name=f"block{i}", pos=[0.50, (i - (blocks - 1) / 2) * 0.13, TABLE_TOP_Z + BLOCK_HALF])
            block.add_freejoint()
            block.add_geom(name=f"block{i}", type=BOX, size=[BLOCK_HALF] * 3, mass=0.06,
                           rgba=WOOD[i % len(WOOD)], friction=[1.5, 0.02, 0.001], **THINGS)
    else:
        cube = world.add_body(name="cube", pos=CUBE_START)
        cube.add_freejoint()
        cube.add_geom(
            name="cube", type=BOX, size=[0.02, 0.02, 0.02], mass=0.05,
            rgba=[0.9, 0.3, 0.2, 1], friction=[1.5, 0.02, 0.001], **THINGS,
        )

    for i in range(balls):
        name = "ball" if i == 0 else f"ball{i}"
        ball = world.add_body(name=name, pos=BALL_PARKED + [0.0, 0.15 * i, 0.0])
        ball.add_freejoint()
        ball.add_geom(name=name, type=SPHERE, size=[BALL_RADIUS, 0, 0], mass=0.05, rgba=[0.2, 0.8, 0.3, 1], **THINGS)

    # The pose the operator is asking for. Drag it in the viewer to steer by mouse.
    target = world.add_body(name="target", mocap=True, pos=[0.5, 0, 0.5])
    if target_marker:   # only where it must be grabbed: dragging it is how the mouse steers
        target.add_geom(type=SPHERE, size=[0.012, 0, 0], rgba=[0.8, 0.8, 0.8, 0.5], contype=0, conaffinity=0)

    if load in ("tray", "grab"):
        hand = spec.body("hand")
        plate = dict(type=BOX, pos=TRAY_OFFSET, size=[TRAY_HALF, TRAY_HALF, 0.004], mass=0.1,
                     rgba=[0.85, 0.85, 0.9, 1], friction=[0.6, 0.005, 0.001], condim=6)
        # A handle from between the fingers to the tray's near edge, so the gripper visibly holds it.
        reach = TRAY_OFFSET[0] - TRAY_HALF
        handle = dict(type=BOX, pos=[reach / 2, 0.0, TRAY_OFFSET[2]], size=[reach / 2 + 0.005, HANDLE_HALF_WIDTH, 0.006],
                      mass=0.02, rgba=[0.3, 0.3, 0.35, 1])
        if load == "tray":      # fixed to the hand from the start
            hand.add_geom(name="tray", **plate)
            hand.add_geom(name="tray_handle", **handle)
        else:                   # a separate paddle, held by a weld that is switched on when it is grabbed
            # Its frame is the hand's frame while held, so "held" means the two frames coincide. The
            # weld takes its resting offset from the starting poses, so start the paddle on the hand.
            bare = mujoco.MjSpec.from_file(str(PANDA_XML)).compile()
            pose = mujoco.MjData(bare)
            mujoco.mj_forward(bare, pose)
            paddle = world.add_body(name="paddle", pos=pose.xpos[bare.body("hand").id], quat=pose.xquat[bare.body("hand").id])
            paddle.add_freejoint()
            paddle.add_geom(name="tray", **plate, **PADDLE)
            paddle.add_geom(name="tray_handle", **handle, **PADDLE)
            hold = spec.add_equality()
            hold.name, hold.type, hold.objtype = "hold", mujoco.mjtEq.mjEQ_WELD, mujoco.mjtObj.mjOBJ_BODY
            hold.name1, hold.name2, hold.active = "hand", "paddle", False
            hold.solref = [0.004, 1.0]   # stiff: the plate must follow the hand's lean closely
            world.add_geom(name="pedestal", type=BOX, pos=[PEDESTAL_TOP[0], PEDESTAL_TOP[1], PEDESTAL_TOP[2] / 2],
                           size=[0.06, 0.06, PEDESTAL_TOP[2] / 2], rgba=[0.3, 0.3, 0.33, 1], **THINGS)
        # The top face is the gripper's -z side (the gripper points down).
        hand.add_site(name="tray", pos=[TRAY_OFFSET[0], TRAY_OFFSET[1], TRAY_OFFSET[2] - 0.004], size=[0.004, 0.004, 0.004], rgba=HIDDEN)
        ball = world.add_body(name="load", pos=LOAD_PARKED)
        ball.add_freejoint()
        ball.add_geom(name="load", type=SPHERE, size=[LOAD_RADIUS, 0, 0], mass=0.03, rgba=[0.95, 0.75, 0.1, 1],
                      friction=[0.6, 0.005, 0.001], condim=6, **THINGS)   # rolls once the tray tilts about 3 degrees

    return spec.compile()


def can_grab(model: mujoco.MjModel) -> bool:
    """Whether the scene has a paddle to pick up (as opposed to none, or one fixed to the hand)."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle") >= 0


def control_site(model: mujoco.MjModel) -> str:
    """The point the operator steers at the start: the tray's centre when a tray is fixed to the hand,
    else between the fingertips."""
    fixed_tray = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tray") >= 0 and not can_grab(model)
    return "tray" if fixed_tray else "grasp"


def _set_free(model, data, body: str, pos, quat, linear=(0, 0, 0), angular=(0, 0, 0)) -> None:
    joint = model.body(body).jntadr[0]
    q, v = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    data.qpos[q:q + 3], data.qpos[q + 3:q + 7] = pos, quat
    data.qvel[v:v + 3], data.qvel[v + 3:v + 6] = linear, angular


def paddle_in_reach(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    """Whether the hand is on the paddle's handle the right way round: close enough to take hold."""
    hand, paddle = model.body("hand").id, model.body("paddle").id
    close = np.linalg.norm(data.xpos[hand] - data.xpos[paddle]) < 0.07
    aligned = abs(float(data.xquat[hand] @ data.xquat[paddle])) > 0.9
    return bool(close and aligned)


def grab(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Take hold of the paddle: it snaps into the hand and is welded there, with the loose ball on it."""
    hand = model.body("hand").id
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, hand, velocity, 0)
    _set_free(model, data, "paddle", data.xpos[hand], data.xquat[hand], velocity[3:], velocity[:3])
    data.eq_active[model.equality("hold").id] = 1
    mujoco.mj_forward(model, data)
    place_load(model, data)
    mujoco.mj_forward(model, data)


def release(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Let go of the paddle; it and the loose ball fall where they are."""
    data.eq_active[model.equality("hold").id] = 0


def rest_paddle(model: mujoco.MjModel, data: mujoco.MjData, hand_quat: np.ndarray) -> None:
    """Put the paddle back on its pedestal, turned the way the hand holds it at home."""
    turn = np.zeros(9)
    mujoco.mju_quat2Mat(turn, hand_quat)
    origin = PEDESTAL_TOP + [0, 0, 0.004] - turn.reshape(3, 3) @ np.array(TRAY_OFFSET)
    _set_free(model, data, "paddle", origin, hand_quat)
    _set_free(model, data, "load", LOAD_PARKED, [1, 0, 0, 0])


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
    if can_grab(model):
        rest_paddle(model, data, data.xquat[model.body("hand").id].copy())
        mujoco.mj_forward(model, data)
    if control_site(model) == "tray":
        data.qpos[7:9] = HANDLE_HALF_WIDTH
        data.ctrl[7] = CARRY_GRIP
        mujoco.mj_forward(model, data)
        place_load(model, data)
        mujoco.mj_forward(model, data)
