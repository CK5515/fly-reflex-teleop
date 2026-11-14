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
from sim.scene import CARRY_GRIP, CUBE_START, GRIPPER_OPEN, TABLE_TOP_Z, build_model, place_load, reset
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
    parser.add_argument("--reflex", choices=["continuous", "spectral", "handwritten", "brain", "none"], default="continuous",
                        help="continuous: smooth avoidance by geometry; spectral: the same, with the danger map "
                             "computed by the connectome model; handwritten / brain: the earlier single-jump reflexes")
    parser.add_argument("--threat-model", default="runs/threat/model.npz", help="fitted model used by --reflex spectral")
    parser.add_argument("--brain", default="runs/live/model.pt", help="trained model used by --reflex brain")
    parser.add_argument("--throw-every", type=float, default=0.0, help="seconds between throws (0 = only on keypress)")
    parser.add_argument("--aim-at", choices=["gripper", "forearm"], default="gripper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--volley", type=int, default=1, help="balls thrown each time, a fifth of a second apart")
    parser.add_argument("--load", choices=["none", "tray"], default="none", help="tray: carry a loose ball on a tray")
    parser.add_argument("--wobble", type=float, default=0.15, help="scripted wrist unsteadiness when carrying, radians")
    parser.add_argument("--view", action="store_true", help="serve the live web page")
    parser.add_argument("--view-port", type=int, default=8000)
    args = parser.parse_args()
    if args.volley < 1:
        parser.error("--volley must be at least 1")
    if args.throw_every < 0:
        parser.error("--throw-every cannot be negative")

    model = build_model(None if args.load == "none" else args.load, balls=args.volley)
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
    if args.reflex == "continuous":
        from reflex.continuous import ContinuousReflex
        reflex = ContinuousReflex()
    elif args.reflex == "spectral":
        from brain.threat import SpectralThreatReflex
        reflex = SpectralThreatReflex(args.threat_model)
    elif args.reflex == "brain":
        from brain.reflex import BrainReflex
        reflex = BrainReflex(args.brain, threaded=not args.headless)
    else:
        reflex = HandwrittenReflex() if args.reflex == "handwritten" else NoReflex()
    rng = np.random.default_rng(args.seed)
    cube_qpos = model.jnt_qposadr[model.body("cube").jntadr[0]]
    highest = 0.0
    next_throw = args.throw_every
    throws = hits = fired = 0
    unhit: set[int] = set()   # balls thrown and not yet counted as having touched the arm
    in_flight: dict[int, float] = {}   # ball -> when it was thrown
    SPENT_AFTER = 1.5         # seconds after which a thrown ball is removed from the scene
    in_control = None
    drops = 0
    off_since = 0.0
    load_state, cause = "on", "neither"
    last_hit = last_dodge = -10.0          # when a ball last touched the arm, tray or load; when the reflex last moved the arm
    causes = {"after a ball hit": 0, "during a dodge": 0, "neither": 0}

    queued: list[tuple[float, str]] = []   # (time, aim) of the rest of a volley

    def throw(aim_at: str) -> None:
        """Queue a volley: one ball now, the others at fifth-of-a-second intervals."""
        queued.extend((data.time + 0.2 * i, aim_at) for i in range(args.volley))
        queued.sort(key=lambda item: item[0])

    def launch(aim_at: str) -> bool:
        nonlocal throws
        ball = senses.throw(data, rng, aim_at=aim_at, unavailable=set(in_flight))
        if ball is None:
            return False
        unhit.add(ball)
        in_flight[ball] = data.time
        throws += 1
        return True

    def step() -> None:
        nonlocal highest, next_throw, hits, fired, in_control, steps, drops, off_since, last_hit, last_dodge, load_state, cause
        if view is not None:
            for aim in view.take_throws():  # asked for from the page
                throw(aim)
        for ball, thrown_at in list(in_flight.items()):
            if data.time - thrown_at > SPENT_AFTER:
                senses.park(data, ball)
                del in_flight[ball]
                unhit.discard(ball)
        if args.throw_every > 0 and data.time >= next_throw:
            next_throw += args.throw_every
            throw(args.aim_at)
        while queued and queued[0][0] <= data.time:
            if not launch(queued[0][1]):
                break  # keep the request queued until a ball has completed its flight
            queued.pop(0)
        if source is not None:
            hand = source.latest(data.time)
            if hand is not None:
                pos, quat, grip = retarget(hand)
                data.mocap_pos[0], data.mocap_quat[0] = pos, quat
                data.ctrl[7] = CARRY_GRIP if carrying else GRIPPER_OPEN * (1.0 - grip)
        events = senses.read(data)
        command = reflex.step(events)
        if command.expert != in_control:
            in_control = command.expert
            fired += command.expert is not None
            print(f"[{data.time:6.2f} s] {in_control or 'operator'} in control")
        for ball in events.touching & unhit:  # one hit per thrown ball, however often it bounces
            hits += 1
            unhit.discard(ball)
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
        if carrying:
            # A load off the tray for half a second counts as dropped, once. It is put back only when
            # things are calm, so one drop is not counted again while the tray is still mid-dodge.
            if events.touching or senses.load_struck(data):
                last_hit = data.time
            if command.expert is not None:
                last_dodge = data.time
            if load_state == "on" and not events.load_on:
                load_state, off_since = "off", data.time
                cause = ("after a ball hit" if data.time - last_hit < 0.6 else
                         "during a dodge" if data.time - last_dodge < 0.6 else "neither")
            elif load_state == "off" and events.load_on:
                load_state = "on"
            elif load_state == "off" and data.time - off_since > 0.5:
                load_state = "dropped"
                drops += 1
                causes[cause] += 1
                print(f"[{data.time:6.2f} s] load dropped ({cause})")
            elif load_state == "dropped" and data.time - max(off_since + 0.5, last_hit, last_dodge) > 0.7:
                load_state = "on"
                place_load(model, data)
                mujoco.mj_forward(model, data)
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
    if ik.restarts:
        print(f"inverse kinematics restarted {ik.restarts} times")
    if carrying:
        print(f"load dropped {drops} times in {data.time:.0f} s {causes}")
    if throws:
        print(f"{throws} throws, {hits} touched the arm, reflexes fired {fired} times")


if __name__ == "__main__":
    main()
