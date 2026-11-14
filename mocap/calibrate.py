"""Line the glove's axes up with the simulated arm's.

The glove reports position relative to the Coil Pro and orientation relative to north and gravity.
Neither frame is the arm's, so the same hand movement would move the arm in some other direction.
This asks for four short movements and works out the two rotations from them:

    python -m mocap.calibrate [left|right]

Sit or stand the way you will when operating, facing the screen. There is no countdown: each step
waits until you have made the movement and held still, and shows how far you have moved. The result is saved to data/glove_calibration.json and used by
`run_teleop.py --source glove`.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
SAVED = REPO / "data" / "glove_calibration.json"
MOVE = 0.15    # metres the hand must move for a step to count
TILT = 30      # degrees the fingers must tip
STILL = 0.7    # seconds the hand must then stay still


def rotation_between(q_from: np.ndarray, q_to: np.ndarray) -> tuple[np.ndarray, float]:
    """Axis (unit) and angle (radians) of the rotation taking orientation `q_from` to `q_to`, world frame."""
    w0, v0 = q_from[0], q_from[1:]
    w1, v1 = q_to[0], q_to[1:]
    w = w1 * w0 + v1 @ v0                       # q_to * conj(q_from)
    v = -w1 * v0 + w0 * v1 - np.cross(v1, v0)
    if w < 0:
        w, v = -w, -v
    angle = 2 * np.arctan2(np.linalg.norm(v), w)
    return v / (np.linalg.norm(v) + 1e-12), float(angle)


def solve(rest, forward, left, up, rest_quat, tilted_quat) -> dict:
    """The two rotations, from where the hand was at each step."""
    moves = np.stack([forward - rest, left - rest, up - rest], axis=1)        # columns, in the coil's frame
    lengths = np.linalg.norm(moves, axis=0)
    # The rotation that best sends the three movements to the arm's x (forward), y (left), z (up).
    u, _, vt = np.linalg.svd((moves / lengths).T)
    frame = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    squareness = float(np.abs((frame @ (moves / lengths)) - np.eye(3)).max())

    # Tipping the fingers down is a turn about the operator's left-pointing axis. Orientation comes
    # in a north-east-down frame, so up is (0, 0, -1) there.
    axis, angle = rotation_between(rest_quat, tilted_quat)
    up_ned = np.array([0.0, 0.0, -1.0])
    left_ned = axis - (axis @ up_ned) * up_ned
    level = float(np.linalg.norm(left_ned))
    left_ned /= level + 1e-12
    turn_frame = np.stack([np.cross(left_ned, up_ned), left_ned, up_ned])     # rows: arm x, y, z in that frame
    return {"frame": frame.tolist(), "turn_frame": turn_frame.tolist(),
            "check": {"movement_cm": (lengths * 100).round(1).tolist(), "movements_off_square": round(squareness, 3),
                      "tilt_deg": round(float(np.degrees(angle)), 1), "tilt_axis_level": round(level, 3)}}


def load_calibration() -> dict | None:
    if not SAVED.exists():
        return None
    saved = json.loads(SAVED.read_text())
    return {"frame": np.array(saved["frame"]), "turn_frame": np.array(saved["turn_frame"])}


def wait_for(glove, instruction: str, done, progress, give_up: float = 40.0) -> tuple[np.ndarray, np.ndarray] | None:
    """Say what to do, then wait (no countdown) until `done(pos, quat)` holds and the hand has been
    still for STILL seconds. Returns the hand's average position and orientation while still, or None
    if that never happened."""
    print(f"\n{instruction}", flush=True)
    recent: list[np.ndarray] = []
    started = said = time.monotonic()
    while time.monotonic() - started < give_up:
        time.sleep(0.01)
        pose = glove.latest()
        now = time.monotonic()
        if pose is None or not glove.live:
            continue
        recent.append(np.r_[now, pose.pos, pose.quat])
        recent = [r for r in recent if now - r[0] <= STILL]
        window = np.array(recent)
        still = (now - window[0, 0] > STILL * 0.9 and np.linalg.norm(window[:, 1:4].std(0)) < 0.008
                 and np.abs(window[:, 4:] - window[0, 4:]).max() < 0.05)
        if done(pose.pos, pose.quat) and still:
            quat = window[:, 4:].mean(0)
            print("  got it", flush=True)
            return window[:, 1:4].mean(0), quat / np.linalg.norm(quat)
        if now - said > 0.5:
            said = now
            print(f"  {progress(pose.pos, pose.quat)}", flush=True)
    return None


def still_hand(glove, instruction: str):
    return wait_for(glove, instruction, lambda pos, quat: True, lambda pos, quat: "waiting for your hand to be still")


def check(glove, calibration: dict) -> bool:
    """A ten-second test of a saved calibration: push the hand forward and see that it reads as forward."""
    rest = still_hand(glove, "Hold your hand out comfortably, palm down, fingers pointing at the screen. Keep still.")
    if rest is None:
        return False
    far = lambda pos, quat: np.linalg.norm(pos - rest[0]) > 0.12
    moved = lambda pos, quat: f"moved {np.linalg.norm(pos - rest[0]) * 100:4.0f} cm (need 12, then hold still)"
    pushed = wait_for(glove, "Push your hand FORWARD, toward the screen, and hold it there.", far, moved, give_up=20)
    if pushed is None:
        return False
    step = calibration["frame"] @ (pushed[0] - rest[0])
    forward = float(step[0] / np.linalg.norm(step))
    print(f"  that push reads as {forward:.0%} forward" + (": calibration still good" if forward > 0.85 else ": calibration is off"))
    return forward > 0.85


def calibrate(glove) -> bool:
    """The full routine. Saves and returns True if the movements were square enough."""
    rest = still_hand(glove, "Hold your hand out comfortably, palm down, fingers pointing at the screen. Keep still.")
    if rest is None:
        return False
    rest, rest_quat = rest
    far = lambda pos, quat: np.linalg.norm(pos - rest) > MOVE
    near = lambda pos, quat: np.linalg.norm(pos - rest) < 0.05
    moved = lambda pos, quat: f"moved {np.linalg.norm(pos - rest) * 100:4.0f} cm (need {MOVE * 100:.0f}, then hold still)"
    away = lambda pos, quat: f"{np.linalg.norm(pos - rest) * 100:4.0f} cm from the start (come back, then hold still)"
    back = "Back to where you started, and hold still."
    steps = []
    for name in ("FORWARD, toward the screen,", "to your LEFT,", "UP,"):
        there = wait_for(glove, f"Move your hand {name} and hold it there.", far, moved)
        home = wait_for(glove, back, near, away)
        if there is None or home is None:
            return False
        steps.append(there[0])
    rest2, rest_quat = home
    tipped = lambda pos, quat: rotation_between(rest_quat, quat)[1] > np.radians(TILT)
    tilt = lambda pos, quat: f"tipped {np.degrees(rotation_between(rest_quat, quat)[1]):3.0f} degrees (need {TILT}, then hold still)"
    tilted = wait_for(glove, "Keeping your arm where it is, TIP YOUR FINGERS DOWN at the wrist, and hold.", tipped, tilt)
    if tilted is None:
        return False

    result = solve((rest + rest2) / 2, *steps, rest_quat, tilted[1])
    report = result["check"]
    print(f"\nmovements measured: forward {report['movement_cm'][0]} cm, left {report['movement_cm'][1]} cm, up {report['movement_cm'][2]} cm")
    print(f"how far the three movements were from square: {report['movements_off_square']} (0 is perfect, 0.35 is the limit)")
    print(f"finger tilt measured: {report['tilt_deg']} degrees, axis {report['tilt_axis_level']:.2f} level (1 is perfect, 0.8 is the limit)")
    if report["movements_off_square"] > 0.35 or report["tilt_axis_level"] < 0.8:
        print("NOT SAVED: the movements were not square enough. Move straight along each direction.")
        return False
    SAVED.parent.mkdir(parents=True, exist_ok=True)
    SAVED.write_text(json.dumps(result, indent=1))
    print(f"saved {SAVED}")
    return True


def main() -> None:
    from mocap.glove import GloveHand
    try:
        glove = GloveHand(hand=sys.argv[1] if len(sys.argv) > 1 else None)
    except ConnectionRefusedError:
        raise SystemExit("The glove driver is not running. Run `python demo.py`, which starts it, or start it yourself:\n"
                         "  ~/.local/share/rokoko-device-sdk/bin/rokoko-sdk --auto-stream-usb") from None
    ok = calibrate(glove)
    glove.close()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
