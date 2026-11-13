"""Steer the simulated arm.

    python run_teleop.py                    # scripted stand-in operator picks up the cube
    python run_teleop.py --source mouse     # drag the blue box in the viewer
    python run_teleop.py --headless --seconds 12   # no window; prints whether the cube was lifted
    python run_teleop.py --throw-every 4    # a ball is thrown at the gripper every 4 s
    python run_teleop.py --throw-every 4 --reflex none   # same, with the reflexes off

Add --view to any of these to run everything from the live web page (http://127.0.0.1:8000)
instead of the viewer window: it shows the arm in 3D, who is in control, and has buttons to throw.
With --source mouse, double-click the blue box there and Ctrl+right-drag it to steer.
In the viewer window, press T to throw a ball at the gripper and F to throw one at the forearm.
"""

import argparse
import socket
import time

import mujoco
import mujoco.viewer
import numpy as np

from reflex.handwritten import HandwrittenReflex, time_to_contact
from reflex.interface import NoReflex
from sim.events import Senses
from sim.ik import ArmIK
from sim.scene import CUBE_START, GRIPPER_OPEN, TABLE_TOP_Z, build_model, place_load, reset
from teleop.hand import Retarget, ScriptedHand, lean, level


def pick_script(home_pos: np.ndarray) -> ScriptedHand:
    """Reach over the cube, come down, pinch, lift, carry, put down, let go."""
    over = CUBE_START - home_pos + np.array([0.0, 0.0, 0.15])
    at = CUBE_START - home_pos + np.array([0.0, 0.0, 0.005])
    aside = over + np.array([0.0, 0.2, 0.0])
    down = at + np.array([0.0, 0.2, 0.01])
    zero = (0.0, 0.0, 0.0)
    return ScriptedHand([
        (0.0, zero, 0.0),
        (2.0, tuple(over), 0.0),
        (3.5, tuple(at), 0.0),
        (4.5, tuple(at), 1.0),
        (6.0, tuple(over), 1.0),
        (8.0, tuple(aside), 1.0),
        (9.5, tuple(down), 1.0),
        (10.5, tuple(down), 0.0),
        (12.0, tuple(aside), 0.0),
        (14.0, zero, 0.0),
    ])


def waiter_script(wobble: float) -> ScriptedHand:
    """Carry a tray briskly around above the table with an unsteady wrist."""
    stops = [(0.0, 0.0, 0.0), (0.05, 0.25, 0.05), (-0.1, 0.25, -0.05), (-0.1, -0.25, 0.05), (0.05, -0.25, -0.05)]
    waypoints, t = [], 0.0
    for stop in [*stops, stops[0]]:
        waypoints.append((t, stop, 0.0))
        waypoints.append((t + 0.6, stop, 0.0))   # pause, then a quick move to the next stop
        t += 1.5
    return ScriptedHand(waypoints, wobble=wobble)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["scripted", "mouse", "glove"], default="scripted")
    parser.add_argument("--hand", choices=["left", "right"], default=None, help="which glove steers (default: the first seen)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until closed)")
    parser.add_argument("--reflex", choices=["handwritten", "brain", "none"], default="handwritten")
    parser.add_argument("--brain", default="runs/live/model.pt", help="trained model used by --reflex brain")
    parser.add_argument("--throw-every", type=float, default=0.0, help="seconds between throws (0 = only on keypress)")
    parser.add_argument("--aim-at", choices=["gripper", "forearm"], default="gripper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load", choices=["none", "tray"], default="none", help="tray: carry a loose ball on a tray")
    parser.add_argument("--wobble", type=float, default=0.15, help="scripted wrist unsteadiness when carrying, radians")
    parser.add_argument("--view", action="store_true", help="serve the live web page")
    parser.add_argument("--view-port", type=int, default=8000)
    args = parser.parse_args()

    model = build_model(None if args.load == "none" else args.load)
    carrying = args.load == "tray"
    data = mujoco.MjData(model)
    reset(model, data)
    ik = ArmIK(model)
    home_pos, home_quat = data.mocap_pos[0].copy(), data.mocap_quat[0].copy()
    retarget = Retarget(home_pos, home_quat)

    if args.source == "scripted":
        source = waiter_script(args.wobble) if carrying else pick_script(home_pos)
    elif args.source == "glove":
        from mocap.glove import GloveHand
        source = GloveHand(hand=args.hand)
    else:
        source = None  # the viewer moves the target directly

    senses = Senses(model)
    view = scene_view = None
    if args.view:
        from mujoco.studio import launch_passive, viewer_protocol
        from view.server import LiveView
        scene_port = args.view_port + 1  # MuJoCo's own web viewer, shown inside the page
        try:
            with socket.socket() as probe:  # MuJoCo reports a taken port from another thread, too late to stop here
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("", scene_port))
        except OSError:
            raise SystemExit(f"port {scene_port} is in use; pick another with --view-port") from None
        scene_view = launch_passive.launch(viewer_protocol.ViewerConfig(gfx="web", http_port=scene_port))
        view = LiveView(args.view_port, viewer_port=scene_port)
    scene_every = round(1 / 60 / model.opt.timestep)  # the 3D view is refreshed 60 times a second
    view_every = round(0.05 / model.opt.timestep)  # 20 snapshots a second
    steps = 0
    peak: dict[str, float] = {}  # highest drive since the last snapshot, so brief spikes are not missed
    if args.reflex == "brain":
        from brain.reflex import BrainReflex
        reflex = BrainReflex(args.brain, threaded=not args.headless)
    else:
        reflex = HandwrittenReflex() if args.reflex == "handwritten" else NoReflex()
    rng = np.random.default_rng(args.seed)
    cube_qpos = model.jnt_qposadr[model.body("cube").jntadr[0]]
    highest = 0.0
    next_throw = args.throw_every
    throws = hits = fired = 0
    hit_counted = True
    in_control = None
    drops = 0
    off_since = None

    def throw(aim_at: str) -> None:
        nonlocal throws, hit_counted
        hit_counted = False
        senses.throw(data, rng, aim_at=aim_at)
        throws += 1

    def step() -> None:
        nonlocal highest, next_throw, hits, fired, hit_counted, in_control, steps, drops, off_since
        if view is not None:
            for aim in view.take_throws():  # asked for from the page
                throw(aim)
        if args.throw_every > 0 and data.time >= next_throw:
            next_throw += args.throw_every
            throw(args.aim_at)
        if source is not None:
            hand = source.latest(data.time)
            if hand is not None:
                pos, quat, grip = retarget(hand)
                data.mocap_pos[0], data.mocap_quat[0] = pos, quat
                data.ctrl[7] = GRIPPER_OPEN * (1.0 - grip)
        events = senses.read(data)
        command = reflex.step(events)
        if command.expert != in_control:
            in_control = command.expert
            fired += command.expert is not None
            print(f"[{data.time:6.2f} s] {in_control or 'operator'} in control")
        if events.contact_force > 0 and not hit_counted:  # one hit per throw, however often it bounces
            hits += 1
            hit_counted = True
        steps += 1
        for name, drive in command.activations.items():
            peak[name] = max(peak.get(name, 0.0), drive)
        if view is not None and steps % view_every == 0:
            ttc, passing = time_to_contact(events)
            if np.linalg.norm(passing) > 0.5:  # passing well clear: not worth reporting
                ttc = np.inf
            view.publish({
                "t": round(data.time, 3),
                "control": command.expert or "operator",
                "gain": round(command.gain, 3),
                "drive": {k: round(v, 3) for k, v in peak.items()},
                "ttc": None if np.isinf(ttc) else round(ttc, 3),
                "miss": round(float(np.linalg.norm(passing)), 3),
                "force": round(events.contact_force, 2),
                "throws": throws,
                "hits": hits,
            })
            peak.clear()
        # The blue box stays where the operator wants; the arm goes where the reflex allows.
        quat = data.mocap_quat[0]
        if command.level:
            quat = lean(level(quat, home_quat), command.tilt)
        data.ctrl[:7] = ik.step(data.mocap_pos[0] + command.gain * command.offset, quat)
        mujoco.mj_step(model, data)
        if carrying:  # a load that has been off the tray for a moment counts as dropped and is put back
            if events.load_on:
                off_since = None
            elif off_since is None:
                off_since = data.time
            elif data.time - off_since > 0.5:
                drops += 1
                off_since = None
                place_load(model, data)
                print(f"[{data.time:6.2f} s] load dropped")
        highest = max(highest, data.qpos[cube_qpos + 2] - TABLE_TOP_Z - 0.02)

    def done() -> bool:
        return args.seconds > 0 and data.time >= args.seconds

    if scene_view is not None:  # the web page is the viewer: keep to real time
        began = time.monotonic()
        try:
            while scene_view.is_running() and not done():
                step()
                if steps % scene_every == 0:
                    scene_view.sync(model, data)  # also brings back a target dragged in the page
                time.sleep(max(0.0, data.time - (time.monotonic() - began)))
        except KeyboardInterrupt:
            pass
        finally:
            scene_view.close()
    elif args.headless:
        while not done():
            step()
    else:
        def on_key(key: int) -> None:
            if key == ord("T"):
                throw("gripper")
            elif key == ord("F"):
                throw("forearm")

        with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
            while viewer.is_running() and not done():
                tick = time.monotonic()
                step()
                viewer.sync()
                time.sleep(max(0.0, model.opt.timestep - (time.monotonic() - tick)))

    print(f"cube lifted {highest * 100:.1f} cm above the table at its highest; "
          f"ended at {np.round(data.qpos[cube_qpos:cube_qpos + 3], 3)}")
    if carrying:
        print(f"load dropped {drops} times in {data.time:.0f} s")
    if throws:
        print(f"{throws} throws, {hits} touched the arm, reflexes fired {fired} times")


if __name__ == "__main__":
    main()
