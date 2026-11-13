"""Read one Rokoko Smartglove from the Device SDK and report it as a HandPose.

Needs `rokoko-sdk` running (or `python -m mocap.tape replay ...` standing in
for it). The wire decoding is Rokoko's own client library, which ships with
the SDK as an example and is imported from the install directory.

    python -m mocap.glove [left|right]   # print the hand pose a few times a second
"""

import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

from teleop.hand import HandPose

SDK_HOME = Path(os.environ.get("ROKOKO_SDK_HOME", Path.home() / ".local/share/rokoko-device-sdk"))
sys.path.append(str(SDK_HOME / "examples/python"))

from rgmp_core.client import DEFAULT_PORT, RgmpClient  # noqa: E402
from rgmp_core.layout import read_slot  # noqa: E402

GROUP = "default"
# Thumb-to-index fingertip distance, in metres, that counts as fully open / fully pinched.
PINCH_OPEN = 0.09
PINCH_CLOSED = 0.025

# (target_frame, reference_frame, measure_type) of the streams we use.
WRIST = ("arm_sensor", "coil_pro", "TRANSFORM")   # position needs a Coil Pro
HUB = ("hub", "LTP_NED", "ORIENTATION")           # hand orientation, glove alone
THUMB = ("thumb_finger_sensor", "hub", "TRANSFORM")
INDEX = ("index_finger_sensor", "hub", "TRANSFORM")


def _wxyz(xyzw) -> np.ndarray:
    x, y, z, w = xyzw
    return np.array([w, x, y, z])


class GloveHand:
    """Keeps the newest hand pose from the stream; `latest()` never blocks."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, hand: str | None = None):
        self.hand = hand            # "left", "right", or None for the first glove that announces itself
        self.device_id: int | None = None
        self._pose: HandPose | None = None
        self._slots: dict[tuple, object] = {}
        self._group_id: int | None = None
        self._lock = threading.Lock()
        self.frames = 0
        self._client = RgmpClient(host, port)
        self._client.on_definition = self._on_definition
        self._client.on_data = self._on_data
        self._client.on_error = lambda e: print(f"glove stream error: {e}", file=sys.stderr)
        self._client.__enter__()
        threading.Thread(target=self._client.run, daemon=True).start()

    def close(self) -> None:
        self._client.close()

    def latest(self, t: float = 0.0) -> HandPose | None:
        with self._lock:
            return self._pose

    def _on_definition(self, definition, layout) -> None:
        if self.device_id is None:
            announced = (definition.get("device_info") or {}).get("hand")
            if self.hand is None or announced == self.hand:
                self.device_id = definition["device_id"]
        if definition["device_id"] != self.device_id:
            return
        for gid, slots in layout.groups.items():
            if layout.group_names[gid] != GROUP:
                continue
            self._group_id = gid
            for slot in slots:
                e = slot.entry
                key = (e.get("target_frame"), e.get("reference_frame"), e.get("measure_type"))
                if key in (WRIST, HUB, THUMB, INDEX):
                    self._slots[key] = slot
        missing = [k for k in (WRIST, HUB, THUMB, INDEX) if k not in self._slots]
        if missing:
            print(f"glove stream is missing {missing}", file=sys.stderr)

    def _on_data(self, device_id, group_id, ts_us, payload, layout) -> None:
        if device_id != self.device_id or group_id != self._group_id or len(self._slots) < 4:
            return
        wrist = read_slot(self._slots[WRIST], payload)
        if not any(wrist[:3]):  # exactly zero: the Coil Pro has not located the hand yet
            return
        hub = read_slot(self._slots[HUB], payload)
        thumb = np.array(read_slot(self._slots[THUMB], payload)[:3])
        index = np.array(read_slot(self._slots[INDEX], payload)[:3])
        gap = float(np.linalg.norm(thumb - index))
        grip = (PINCH_OPEN - gap) / (PINCH_OPEN - PINCH_CLOSED)
        pose = HandPose(np.array(wrist[:3]), _wxyz(hub), float(np.clip(grip, 0.0, 1.0)))
        with self._lock:
            self._pose = pose
            self.frames += 1


if __name__ == "__main__":
    hand = GloveHand(hand=sys.argv[1] if len(sys.argv) > 1 else None)
    np.set_printoptions(precision=3, suppress=True)
    start = time.monotonic()
    try:
        while True:
            time.sleep(0.25)
            p = hand.latest()
            rate = hand.frames / (time.monotonic() - start)
            print("waiting for a glove..." if p is None else f"pos {p.pos}  quat {p.quat}  grip {p.grip:.2f}  ({rate:.0f} Hz)")
    except KeyboardInterrupt:
        hand.close()
