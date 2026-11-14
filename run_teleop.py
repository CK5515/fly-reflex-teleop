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
from pathlib import Path
import socket
import time

import mujoco
import mujoco.viewer
import numpy as np

from reflex.handwritten import HandwrittenReflex, time_to_contact
from reflex.interface import NoReflex
from sim.events import Senses
from sim.ik import ArmIK
from sim.scene import (CARRY_GRIP, CUBE_START, GRIPPER_OPEN, PEDESTAL_TOP, TABLE_TOP_Z, TRAY_OFFSET, build_model, grab,
                       paddle_in_reach, place_load, release, reset, rest_paddle)
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


def grab_script(home_pos: np.ndarray, data, model, wobble: float) -> ScriptedHand:
    """Reach for the paddle's handle, pinch to pick it up, carry it about unsteadily, then let go."""
    hand = model.body("paddle").id
    turn = data.xmat[hand].reshape(3, 3)
    handle = data.xpos[hand] + turn @ [0, 0, 0.1034] - home_pos      # where the fingertips take hold
    above = handle + [0, 0, 0.12]
    carry = [(handle + [0.0, -0.2, 0.2]), (handle + [0.1, -0.35, 0.25]), (handle + [-0.1, -0.1, 0.2])]
    zero = (0.0, 0.0, 0.0)
    return ScriptedHand([
        (0.0, zero, 0.0), (2.0, tuple(above), 0.0), (3.2, tuple(handle), 0.0), (3.8, tuple(handle), 1.0),
        (5.0, tuple(carry[0]), 1.0), (6.5, tuple(carry[1]), 1.0), (8.0, tuple(carry[2]), 1.0), (9.5, tuple(carry[0]), 1.0),
        (11.0, tuple(carry[1]), 1.0), (12.5, tuple(carry[2]), 1.0), (14.0, tuple(carry[0]), 1.0),
        (14.5, tuple(carry[0]), 0.0), (16.5, zero, 0.0), (18.0, zero, 0.0),
    ], wobble=wobble, wobble_from=5.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=["scripted", "mouse", "glove"], default="scripted")
    parser.add_argument("--hand-scale", type=float, default=2.0, help="how far the arm moves per unit of glove movement")
    parser.add_argument("--hand", choices=["left", "right"], default=None, help="which glove steers (default: the first seen)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=0.0, help="stop after this long (0 = until closed)")
    parser.add_argument("--reflex", choices=["continuous", "spectral", "eye-geometric", "eye-fly", "handwritten", "brain", "none"], default=None,
                        help="continuous: smooth avoidance by geometry; spectral: the same, with the danger map "
                             "computed by the connectome model; handwritten / brain: the earlier single-jump reflexes")
    parser.add_argument("--eye-noise", choices=["none", "moderate", "heavy"], default="moderate",
                        help="for the eye-* reflexes: how noisy the eye's bearing and apparent-size measurements are")
    parser.add_argument("--eye-model", default="models/eye_model.npz", help="fitted model used by --reflex eye-fly")
    parser.add_argument("--threat-model", default="models/model.npz", help="fitted model used by --reflex spectral")
    parser.add_argument("--brain", default="runs/live/model.pt", help="trained model used by --reflex brain")
    parser.add_argument("--throw-every", type=float, default=0.0, help="seconds between throws (0 = only on keypress)")
    parser.add_argument("--aim-at", choices=["gripper", "forearm"], default="gripper")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--volley", type=int, default=1, help="balls thrown each time, a fifth of a second apart")
    parser.add_argument("--flight", type=float, default=0.6, help="seconds a thrown ball takes to arrive")
    parser.add_argument("--blocks", type=int, default=0, help="wooden blocks on the table to stack (without a tray)")
    parser.add_argument("--load", choices=["none", "tray", "grab"], default=None,
                        help="tray: a tray with a loose ball is fixed to the gripper. grab: the tray waits on a pedestal as a "
                             "paddle; pinch on its handle to pick it up (the ball appears on it), let go to drop it. "
                             "Default: grab with a glove, none otherwise")
    parser.add_argument("--wobble", type=float, default=0.15, help="scripted wrist unsteadiness when carrying, radians")
    parser.add_argument("--view", action="store_true", help="serve the live web page")
    parser.add_argument("--view-port", type=int, default=8000)
    args = parser.parse_args()
    if args.volley < 1:
        parser.error("--volley must be at least 1")
    if args.throw_every < 0:
        parser.error("--throw-every cannot be negative")

    if args.load is None:
        args.load = "grab" if args.source == "glove" else "none"
    if args.load == "grab" and args.blocks == 0:
        args.blocks = 3
    model = build_model(None if args.load == "none" else args.load, balls=args.volley,
                        target_marker=args.source == "mouse",
                        blocks=0 if args.load == "tray" else args.blocks)
    grabbable = args.load == "grab"
    carrying = args.load == "tray"     # whether a tray is in the hand; changes as the paddle is grabbed and dropped
    data = mujoco.MjData(model)
    reset(model, data)
    ik = ArmIK(model)
    home_pos, home_quat = data.mocap_pos[0].copy(), data.mocap_quat[0].copy()
    FLOOR = {True: TABLE_TOP_Z + 0.10, False: TABLE_TOP_Z + 0.015}   # a tray is kept well clear of the table
    floor = FLOOR[carrying]
    retarget = Retarget(home_pos, home_quat, floor=floor)
    if args.source == "glove":
        from mocap.calibrate import load_calibration
        calibration = load_calibration() or {"frame": None, "turn_frame": None}
        if calibration["frame"] is None:
            print("no glove calibration found: run `python -m mocap.calibrate` first (or Recalibrate glove on the page), "
                  "or the arm will move the wrong way")
        # A small hand movement moves the arm twice as far, so it can be steered from a rested elbow.
        # Wrist tipping is halved and capped at 25 degrees, and wrist twist is ignored.
        retarget = Retarget(home_pos, home_quat, frame=calibration["frame"], turn_frame=calibration["turn_frame"],
                            floor=floor, scale=args.hand_scale, tilt_scale=0.5, max_tilt=np.radians(25), follow_yaw=False)

    if args.source == "scripted":
        source = (waiter_script(args.wobble) if carrying else grab_script(home_pos, data, model, args.wobble) if grabbable
                  else pick_script(home_pos))
    elif args.source == "glove":
        from mocap.glove import GloveHand
        source = GloveHand(hand=args.hand)
    else:
        source = None  # the viewer moves the target directly

    from sim.eye import Eye
    senses = Senses(model, eye=Eye(max(args.volley, 1), args.eye_noise, seed=args.seed))
    view = scene_view = watcher = None
    descending_peak = 1e-6
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
        from brain.layout import load_layout
        show_neurons = Path(args.eye_model).exists()
        view = LiveView(args.view_port, viewer_port=scene_port, layout=load_layout() if show_neurons else None)
        if show_neurons:   # the fly circuit is always watching, whichever mode is acting
            from brain.threat import EyeFlyReflex
            watcher = EyeFlyReflex(args.eye_model)
    scene_every = round(1 / 60 / model.opt.timestep)  # the 3D view is refreshed 60 times a second
    view_every = round(0.05 / model.opt.timestep)  # 20 snapshots a second
    steps = 0
    peak: dict[str, float] = {}  # highest drive since the last snapshot, so brief spikes are not missed
    def make_reflex(mode: str):
        if mode == "continuous":
            from reflex.continuous import ContinuousReflex
            return ContinuousReflex()
        if mode == "spectral":
            from brain.threat import SpectralThreatReflex
            return SpectralThreatReflex(args.threat_model)
        if mode == "eye-geometric":
            from reflex.eye import EyeGeometricReflex
            return EyeGeometricReflex()
        if mode == "eye-fly":
            from brain.threat import EyeFlyReflex
            return EyeFlyReflex(args.eye_model)
        if mode == "brain":
            from brain.reflex import BrainReflex
            return BrainReflex(args.brain, threaded=not args.headless)
        return HandwrittenReflex() if mode == "handwritten" else NoReflex()

    mode = args.reflex
    if mode is None:   # the fly reflex if its fitted model is here, else plain geometric avoidance
        mode = "eye-fly" if Path(args.eye_model).exists() else "continuous"
    reflex = make_reflex(mode)
    # Outcomes per reflex mode, so modes switched between on the page can be compared.
    scores: dict[str, dict[str, int]] = {}
    # "carried" counts the steps a load was carried in that mode, so the page shows drops only where there could be some.
    score = lambda m: scores.setdefault(m, {"throws": 0, "hits": 0, "drops": 0, "carried": 0})
    thrown_in: dict[int, str] = {}   # ball -> the mode that was active when it was thrown
    score(mode)
    rng = np.random.default_rng(args.seed)
    stacking = args.blocks > 0 and args.load != "tray"
    watched = "block0" if stacking else "cube"   # the object whose height is reported at the end
    cube_qpos = model.jnt_qposadr[model.body(watched).jntadr[0]]
    block_qpos = [model.jnt_qposadr[model.body(f"block{i}").jntadr[0]] for i in range(args.blocks)] if stacking else []
    grabs = 0
    glove_state = "waiting" if args.source == "glove" else "none"   # none: no glove in use
    home_hand_quat = data.xquat[model.body("hand").id].copy()
    calibrating = None     # the glove calibration started from the page, while it runs and until its result is applied
    finished_calibration, finished_at = None, 0.0   # its outcome, shown on the page for a few seconds afterwards
    pinched = False        # whether the hand was pinched on the last step (carrying: a pinch throws a ball)
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
        ball = senses.throw(data, rng, flight=args.flight, aim_at=aim_at, unavailable=set(in_flight))
        if ball is None:
            return False
        unhit.add(ball)
        in_flight[ball] = data.time
        thrown_in[ball] = mode
        score(mode)["throws"] += 1
        throws += 1
        return True

    def neuron_activity(events) -> dict:
        """What the looming detectors and descending neurons are doing, as 0-99 per neuron, for the page."""
        nonlocal descending_peak
        if watcher is None:
            return {}
        from brain.threat import eye_activity
        detectors = eye_activity(watcher.retina, events.sights or [], events.t, watcher.p)
        descending = np.abs((detectors.reshape(2, -1) @ watcher.through.T)[0])
        descending_peak = max(descending_peak * 0.999, float(descending.max()))
        scale = lambda v: np.clip(np.round(v * 99), 0, 99).astype(int).tolist()
        return {"neurons": {"detectors": scale(detectors[:len(detectors) // 2]),
                            "descending": scale((descending / descending_peak) ** 2)}}   # squared: the strongest stand out

    def hold(take: bool) -> None:
        """Pick the paddle up or let it go, and move the steered point between the fingertips and the tray."""
        nonlocal carrying, grabs, load_state
        tray, fingers = model.site("tray").id, model.site("grasp").id
        between = data.site_xpos[tray] - data.site_xpos[fingers]
        if take:
            grab(model, data)
            grabs += 1
        else:
            release(model, data)
        carrying = take
        load_state = "on" if take else "away"
        senses.carry(take)
        ik.steer("tray" if take else "grasp")
        retarget.home_pos += between if take else -between   # so the arm does not jump as the point changes
        retarget.floor = FLOOR[take]
        print(f"[{data.time:6.2f} s] paddle {'picked up' if take else 'let go'}")

    def tidy() -> None:
        """Put back anything that has fallen off the table: the paddle on its pedestal, blocks where they began."""
        if grabbable and not carrying and data.xpos[model.body("paddle").id][2] < 0.15:
            rest_paddle(model, data, home_hand_quat)
        for i, q in enumerate(block_qpos):
            if data.qpos[q + 2] < 0.1:
                data.qpos[q:q + 3] = [0.50, (i - (len(block_qpos) - 1) / 2) * 0.13, TABLE_TOP_Z + 0.03]
                data.qpos[q + 3:q + 7] = [1, 0, 0, 0]
                v = model.jnt_dofadr[model.body(f"block{i}").jntadr[0]]
                data.qvel[v:v + 6] = 0

    def step() -> None:
        nonlocal highest, next_throw, hits, fired, in_control, steps, drops, off_since, last_hit, last_dodge, load_state, cause, mode, reflex, pinched, carrying, grabs, glove_state, calibrating, finished_calibration, finished_at
        if view is not None:
            for aim in view.take_throws():  # asked for from the page
                throw(aim)
            if view.take_recentre():   # the hand's current position becomes "arm at home"
                retarget.recalibrate()
                print(f"[{data.time:6.2f} s] recentred on the hand")
            for asked in view.take_calibration():
                if asked == "start" and args.source != "glove":
                    print(f"[{data.time:6.2f} s] no glove to calibrate: start with --source glove")
                elif asked == "start" and (calibrating is None or not calibrating.running):
                    from mocap.calibrate import PageCalibration
                    calibrating = PageCalibration(source)
                    print(f"[{data.time:6.2f} s] glove calibration started from the page; the arm holds still")
                elif asked == "ready" and calibrating is not None:
                    calibrating.ready()
                elif asked == "cancel" and calibrating is not None:
                    calibrating.cancel()
            wanted = view.take_mode()
            if wanted is not None and wanted != mode:
                try:
                    reflex, mode = make_reflex(wanted), wanted
                    score(mode)
                    print(f"[{data.time:6.2f} s] reflex mode: {mode}")
                except FileNotFoundError as missing:
                    print(f"cannot switch to {wanted}: {missing}")
        for ball, thrown_at in list(in_flight.items()):
            if data.time - thrown_at > SPENT_AFTER:
                senses.park(data, ball)
                del in_flight[ball]
                unhit.discard(ball)
        if args.throw_every > 0 and data.time >= next_throw:
            next_throw += args.throw_every
            if carrying or not grabbable:   # with a paddle to grab, the balls only come while it is held
                throw(args.aim_at)
        while queued and queued[0][0] <= data.time:
            if not launch(queued[0][1]):
                break  # keep the request queued until a ball has completed its flight
            queued.pop(0)
        if calibrating is not None and not calibrating.running:   # finished: use the new axes if it saved them
            result = calibrating.snapshot()["result"]
            if result == "saved":
                from mocap.calibrate import load_calibration
                fresh = load_calibration()
                retarget.set_frames(fresh["frame"], fresh["turn_frame"])
            retarget.home_pos = data.mocap_pos[0].copy()   # the hand moved about meanwhile: carry on from where the arm is
            retarget.recalibrate()
            print(f"[{data.time:6.2f} s] glove calibration {result}")
            finished_calibration, finished_at, calibrating = calibrating.snapshot(), data.time, None
        if source is not None:
            hand = source.latest(data.time)
            if calibrating is not None:   # the hand is busy with the calibration's movements: hold the arm still
                hand = None
            elif not getattr(source, "live", True):   # glove unplugged or out of range: hold the arm where it is
                hand = None
                if glove_state != "waiting":
                    glove_state = "waiting"
                    print(f"[{data.time:6.2f} s] waiting for the glove")
            elif glove_state == "waiting" and hand is not None:
                glove_state = "live"
                retarget.home_pos = data.mocap_pos[0].copy()   # carry on from where the arm stopped
                retarget.recalibrate()
                print(f"[{data.time:6.2f} s] glove found: arm follows the hand from here")
            if hand is not None:
                pos, quat, grip = retarget(hand)
                data.mocap_pos[0], data.mocap_quat[0] = pos, quat
                data.ctrl[7] = CARRY_GRIP if carrying else GRIPPER_OPEN * (1.0 - grip)
                if grabbable:
                    # Pinch on the handle to take the paddle; relax the pinch to let it go.
                    if not carrying and grip > 0.6 and paddle_in_reach(model, data):
                        hold(True)
                    elif carrying and grip < 0.3:
                        hold(False)
                # With a tray fixed to the hand, the gripper is busy, so a pinch is free to mean
                # "throw a ball at me": pinch past 0.75 to throw, relax below 0.4 to arm it again.
                elif carrying and args.source == "glove":
                    if grip > 0.75 and not pinched:
                        pinched = True
                        throw("gripper")
                    elif grip < 0.4:
                        pinched = False
        if steps % 250 == 0:
            tidy()
        events = senses.read(data)
        command = reflex.step(events)
        if command.expert != in_control:
            in_control = command.expert
            fired += command.expert is not None
            print(f"[{data.time:6.2f} s] {in_control or 'operator'} in control")
        for ball in events.touching & unhit:  # one hit per thrown ball, however often it bounces
            hits += 1
            score(thrown_in.get(ball, mode))["hits"] += 1
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
                "mode": mode,
                "glove": "calibrating" if calibrating is not None else glove_state,
                "calibration": (calibrating.snapshot() if calibrating is not None else
                                finished_calibration if data.time - finished_at < 8 else None),
                **neuron_activity(events),
                "carrying": carrying,
                "scores": scores,
            })
            peak.clear()
        # The blue box stays where the operator wants; the arm goes where the reflex allows.
        quat = data.mocap_quat[0]
        if command.level:
            quat = lean(level(quat, home_quat), command.tilt)
        data.ctrl[:7] = ik.step(data.mocap_pos[0] + command.gain * command.offset, quat)
        mujoco.mj_step(model, data)
        if carrying:
            score(mode)["carried"] += 1
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
                score(mode)["drops"] += 1
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

    if stacking:
        tops = sorted(data.qpos[q + 2] for q in block_qpos)
        layers = 1 + sum(tops[i + 1] - tops[i] > 0.03 for i in range(len(tops) - 1))
        print(f"blocks stacked {layers} high")
    print(f"{watched} lifted {highest * 100:.1f} cm above the table at its highest; "
          f"ended at {np.round(data.qpos[cube_qpos:cube_qpos + 3], 3)}")
    if ik.restarts:
        print(f"inverse kinematics restarted {ik.restarts} times")
    if grabbable:
        print(f"paddle picked up {grabs} times")
    if carrying or grabbable or args.load == "tray":
        print(f"load dropped {drops} times in {data.time:.0f} s {causes}")
    if throws:
        print(f"{throws} throws, {hits} touched the arm, reflexes fired {fired} times")


if __name__ == "__main__":
    main()
