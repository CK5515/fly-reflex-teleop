"""The trained fly-brain model as the reflex layer of the teleop loop.

The scene is turned into sensor input for real neuron populations: a looming
object drives the LPLC2 looming detectors on the side it will pass, and touch
drives the bristle neurons. Activity spreads through the connectome, the
spectral model reads it, and its decision (nothing, escape left, escape
right, groom) triggers the same movements the hand-written reflexes use.
"""

import threading
from pathlib import Path

import numpy as np
import torch

from brain.circuit import load
from brain.data import CLASSES, SNAPSHOTS, Simulator
from brain.model import SpectralMoE
from reflex.handwritten import HandwrittenReflex, time_to_contact
from reflex.interface import SceneEvents

LOOM_STARTS = 0.75   # seconds to contact at which the looming detectors begin to respond
LOOM_FULL = 0.2      # ... and at which they respond fully
LOOM_RANGE = 0.35    # objects passing farther than this from the gripper do not loom, metres
TOUCH_FULL = 2.0     # newtons of touch that drive the bristle neurons fully
EVERY = 0.05         # seconds between brain updates
LATENCY = 0.10       # seconds a decision takes to compute on the demo laptop (measured: 0.08-0.10)


class BrainReflex(HandwrittenReflex):
    """`threaded=False` (tests, headless runs): decisions are computed in step with the simulation and
    take effect LATENCY seconds of simulated time later, so results are repeatable and still honest
    about the delay. `threaded=True` (live viewer): a worker thread computes them while the simulation
    keeps running, and each takes effect when it is ready.
    """

    def __init__(self, checkpoint: Path, threaded: bool = False, **movement):
        super().__init__(**movement)
        self.threaded = threaded
        self._pending: list[tuple[float, tuple]] = []     # (time it takes effect, decision)
        self._inbox: SceneEvents | None = None
        self._wake = threading.Event()
        if threaded:
            threading.Thread(target=self._work, daemon=True).start()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        a = saved["args"]
        self.circuit = load(a["min_synapses"], a["hops_out"], a["hops_back"], a.get("shuffle", 0))
        self.relations = saved["relations"]
        modes = [torch.from_numpy(self.circuit.basis(r)[1]) for r in self.relations]
        self.sim = Simulator(self.circuit)
        self.readout = torch.as_tensor(self.circuit.groups["descending"])
        self.model = SpectralMoE(modes, SNAPSHOTS, len(CLASSES), len(self.readout), a["bands"], a["width"], a["depth"])
        self.model.load_state_dict(saved["model"])
        self.model.eval()
        self.bands = a["bands"]
        self._last_time = -1.0
        self._last = (0.0, 0.0, np.zeros(3), {})

    def judge(self, events: SceneEvents) -> tuple[float, float, np.ndarray, dict]:
        if self.threaded:
            self._inbox = events
            self._wake.set()
            return self._last
        while self._pending and self._pending[0][0] <= events.t:
            self._last = self._pending.pop(0)[1]
        if not 0 <= events.t - self._last_time < EVERY:
            self._last_time = events.t
            self._pending.append((events.t + LATENCY, self._decide(events)))
        return self._last

    def _work(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            events = self._inbox
            if events is not None:
                self._last = self._decide(events)

    def _decide(self, events: SceneEvents) -> tuple[float, float, np.ndarray, dict]:
        ttc, passing = time_to_contact(events)
        side = self._side(events.object_vel)
        loom = 0.0
        if np.linalg.norm(passing) < LOOM_RANGE:
            loom = float(np.clip((LOOM_STARTS - ttc) / (LOOM_STARTS - LOOM_FULL), 0.0, 1.0))
        touch = float(np.clip(events.contact_force / TOUCH_FULL, 0.0, 1.0))
        # The object passes on the gripper's left or right; ties go to the side with less room to dodge.
        lean = float(passing @ side)
        if abs(lean) < 0.03:
            lean = float(events.gripper_pos @ side)
        from_left = lean >= 0

        if loom == 0.0 and touch == 0.0:
            return 0.0, 0.0, np.zeros(3), {"sensors": {"loom": 0.0, "touch": 0.0}}

        drive = torch.zeros(self.sim.n)
        if loom > 0:
            drive += self.sim.drive_for("escape_left" if from_left else "escape_right", strength=loom)
        if touch > 0:
            drive += self.sim.drive_for("groom", strength=touch)
        with torch.no_grad():
            scores, gates = self.model.decide(self.sim.spread(drive[None]), self.readout)
        prob = torch.softmax(scores[0], 0).numpy()
        p = dict(zip(CLASSES, prob))
        # Dodge away from the side the model says the threat is on.
        way_out = -side if p["escape_left"] >= p["escape_right"] else side
        top = torch.topk(gates[0], 3)
        experts = [{"expert": f"{self.relations[int(e) // self.bands]} band {int(e) % self.bands}", "gate": round(float(g), 3)}
                   for g, e in zip(top.values, top.indices)]
        detail = {"sensors": {"loom": round(loom, 2), "touch": round(touch, 2), "side": "left" if from_left else "right"},
                  "decision": {k: round(float(v), 3) for k, v in p.items()}, "experts": experts}
        # A drive of 1 fires the reflex, so 1 corresponds to the model being 50% sure.
        return 2.0 * float(p["escape_left"] + p["escape_right"]), 2.0 * float(p["groom"]), way_out, detail
