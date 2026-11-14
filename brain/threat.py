"""The connectome as the threat sensor for continuous avoidance.

    objects in the air
      -> looming detectors (LPLC2), laid out as a retina from their anatomical positions
      -> band filters on the precomputed Laplacians of each neurotransmitter relation
      -> descending neurons
      -> danger of staying put and of stepping in each of 16 directions

Everything is linear after the retina, so several objects simply add up, and
once fitted the whole chain folds into two small matrices: no spectral bases
are needed at run time. What is learned is one gain per (relation, band) and
the readout of the descending neurons; the wiring fixes everything else.

    python -m brain.threat                 # fit on the real wiring, save runs/threat/model.npz
    python -m brain.threat --compare       # also fit rewired wiring and a no-brain baseline
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from brain.circuit import FLYWIRE, RELATIONS, Circuit, load
from reflex.continuous import DIRECTIONS, N_DIRECTIONS, ContinuousReflex, looming
from reflex.interface import SceneEvents

REPO = Path(__file__).resolve().parent.parent
BANDS = 8
TUNING = np.radians(20.0)        # width of each looming detector's receptive field
FIELD_AZIMUTH = np.radians([-10.0, 110.0])    # each eye sees from just across the midline to behind its side
FIELD_ELEVATION = np.radians([-30.0, 80.0])


class Retina:
    """Where each looming detector looks. Azimuth is positive to the left (+y), 0 is straight ahead (+x)."""

    def __init__(self, circuit: Circuit):
        self.cells = np.r_[circuit.groups["loom_left"], circuit.groups["loom_right"]]
        n_left = len(circuit.groups["loom_left"])
        ann = pd.read_csv(FLYWIRE / "neuron_annotations.tsv", sep="\t", low_memory=False,
                          usecols=["root_id", "pos_x", "pos_y", "pos_z"]).drop_duplicates("root_id").set_index("root_id")
        position = ann.loc[circuit.root_ids[self.cells], ["pos_x", "pos_y", "pos_z"]].values.astype(float)
        self.azimuth = np.zeros(len(self.cells))
        self.elevation = np.zeros(len(self.cells))
        for side, rows in ((1.0, slice(0, n_left)), (-1.0, slice(n_left, None))):
            # The detectors of one eye lie in a sheet. Its long axis is taken as azimuth and its short
            # axis as elevation, spread evenly over the eye's field. This layout is an assumption: the
            # sheet is real anatomy, but which way it maps onto the visual field was not looked up.
            p = position[rows] - position[rows].mean(0)
            axes = np.linalg.svd(p, full_matrices=False)[2]
            rank = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)
            self.azimuth[rows] = side * (FIELD_AZIMUTH[0] + rank(p @ axes[0]) * np.ptp(FIELD_AZIMUTH))
            self.elevation[rows] = FIELD_ELEVATION[0] + rank(p @ axes[1]) * np.ptp(FIELD_ELEVATION)

    def encode(self, objects: list[tuple[np.ndarray, float]]) -> np.ndarray:
        """Activity of every detector for objects given as (position relative to the gripper, urgency)."""
        activity = np.zeros(len(self.cells))
        for pos, urgency in objects:
            if urgency <= 0:
                continue
            azimuth = np.arctan2(pos[1], pos[0])
            elevation = np.arctan2(pos[2], np.hypot(pos[0], pos[1]))
            turn = np.angle(np.exp(1j * (self.azimuth - azimuth)))
            activity += urgency * np.exp(-(turn**2 + (self.elevation - elevation) ** 2) / (2 * TUNING**2))
        return activity


def band_operators(circuit: Circuit) -> np.ndarray:
    """[relation * band, descending neurons, detectors]: how each band of each relation's Laplacian
    carries detector activity to the descending neurons."""
    cells = np.r_[circuit.groups["loom_left"], circuit.groups["loom_right"]]
    out = circuit.groups["descending"]
    operators = []
    for relation in RELATIONS:
        modes = circuit.basis(relation)[1]
        k = modes.shape[1]
        edges = [-(-band * k // BANDS) for band in range(BANDS + 1)]
        for start, stop in zip(edges[:-1], edges[1:]):
            operators.append(modes[out, start:stop] @ modes[cells, start:stop].T)
    return np.stack(operators).astype(np.float32)


def scenes(n: int, rng: np.random.Generator, teacher: ContinuousReflex, retina: Retina):
    """Random moments from random throws: detector activity, and the geometric danger map to match."""
    gravity = np.array([0.0, 0.0, -9.81])
    activity = np.zeros((n, len(retina.cells)), np.float32)
    target = np.zeros((n, 1 + N_DIRECTIONS), np.float32)
    p = teacher.p
    for i in range(n):
        objects = []
        for _ in range(rng.integers(1, 4)):
            aim = rng.normal(0.0, 0.10, 3)                       # thrown at the gripper, not perfectly
            angle = rng.uniform(-1.2, 1.2)
            start = np.array([1.5 * np.cos(angle), 1.5 * np.sin(angle), rng.uniform(0.0, 0.4)])
            flight = rng.uniform(0.5, 0.8)
            velocity = (aim - start - 0.5 * gravity * flight**2) / flight
            t = rng.uniform(0.0, flight)
            objects.append((start + velocity * t + 0.5 * gravity * t * t, velocity + gravity * t))
        events = SceneEvents(t=0.0, gripper_pos=np.zeros(3), object_pos=objects[0][0], object_vel=objects[0][1],
                             object_radius=0.04, contact_force=0.0, objects=objects)
        stay, steps, _ = teacher.danger(events)
        target[i, 0], target[i, 1:] = stay, steps
        activity[i] = retina.encode([(pos, looming(pos, vel, p["starts"], p["full"], p["reach"])[0]) for pos, vel in objects])
    return activity, target


def fit(operators: np.ndarray | None, activity, target, steps: int = 150, seed: int = 0, ridge: float = 1e-3):
    """Fit band gains and the descending-neuron readout. `operators=None` fits a direct map with no brain.

    The readout is solved exactly (ridge regression) for whatever the gains are, and the gains are
    improved by gradient descent through that solution. Gains start as a low-pass, like diffusion
    along the wiring: equal gains would add the bands back up to the identity, which passes nothing
    from one neuron to another.
    """
    import torch
    torch.manual_seed(seed)
    x, y = torch.from_numpy(activity).double(), torch.from_numpy(target).double()

    def solve(hidden):
        design = torch.cat([hidden, torch.ones(len(hidden), 1, dtype=hidden.dtype)], dim=1)
        gram = design.T @ design + ridge * len(hidden) * torch.eye(design.shape[1], dtype=hidden.dtype)
        weights = torch.linalg.solve(gram, design.T @ y)
        return weights, ((design @ weights - y) ** 2).mean()

    if operators is None:
        through, gains = torch.eye(x.shape[1], dtype=x.dtype), None
    else:
        ops = torch.from_numpy(operators).double()
        start = torch.exp(-3.0 * torch.arange(BANDS) / BANDS).repeat(len(ops) // BANDS)
        ops = ops / torch.einsum("e,eod,nd->no", start.double(), ops, x[:512]).std()   # descending activity of order one
        gains = torch.nn.Parameter(start.double())
        optim = torch.optim.Adam([gains], lr=0.03)
        for _ in range(steps):
            _, loss = solve(x @ torch.einsum("e,eod->od", gains, ops).T)
            optim.zero_grad(); loss.backward(); optim.step()
        through = torch.einsum("e,eod->od", gains.detach(), ops)
    with torch.no_grad():
        weights, _ = solve(x @ through.T)
    return dict(through=through.numpy().astype(np.float32), readout=weights[:-1].T.numpy().astype(np.float32),
                bias=weights[-1].numpy().astype(np.float32), gains=None if gains is None else gains.detach().numpy())


def score(model: dict, activity, target, teacher: ContinuousReflex) -> dict:
    """How well a fitted model matches the geometric danger map on scenes it was not fitted on."""
    predicted = activity @ model["through"].T @ model["readout"].T + model["bias"]
    explained = 1 - ((predicted - target) ** 2).sum() / ((target - target.mean(0)) ** 2).sum()
    threatened = target[:, 0] > 0.3
    chosen = predicted[threatened, 1:].argmin(1)
    best = target[threatened, 1:].min(1)
    picked = target[threatened, 1:][np.arange(threatened.sum()), chosen]
    return {"danger map explained": float(explained),
            "chosen direction's true danger, above the best available": float((picked - best).mean()),
            "share of choices within 0.1 of the best": float((picked - best < 0.1).mean())}


class SpectralThreatReflex(ContinuousReflex):
    """Continuous avoidance whose danger map comes from the fitted connectome model."""

    def __init__(self, model_path: str | Path, **kwargs):
        super().__init__(**kwargs)
        z = np.load(model_path)
        self.through, self.readout, self.bias = z["through"], z["readout"], z["bias"]
        self.azimuth, self.elevation = z["azimuth"], z["elevation"]
        self.retina = Retina.__new__(Retina)
        self.retina.cells, self.retina.azimuth, self.retina.elevation = np.arange(len(self.azimuth)), self.azimuth, self.elevation

    def danger(self, events: SceneEvents) -> tuple[float, np.ndarray, dict]:
        p = self.p
        seen = [(pos + self.offset, looming(pos + self.offset, vel, p["starts"], p["full"], p["reach"])[0])
                for pos, vel in events.objects]
        detectors = self.retina.encode(seen)
        descending = self.through @ detectors
        danger = np.maximum(self.readout @ descending + self.bias, 0.0)
        if not detectors.any():
            danger[:] = 0.0
        detail = {"detectors": float(detectors.sum()), "descending": float(np.abs(descending).sum())}
        return float(danger[0]), danger[1:], detail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--compare", action="store_true", help="also fit rewired wiring and a no-brain baseline")
    parser.add_argument("--scenes", type=int, default=20000)
    args = parser.parse_args()
    circuit = load(20, 2, 1)
    retina, teacher, rng = Retina(circuit), ContinuousReflex(), np.random.default_rng(0)
    activity, target = scenes(args.scenes, rng, teacher, retina)
    test_activity, test_target = scenes(4000, rng, teacher, retina)
    print(f"{circuit.name}: {len(retina.cells)} looming detectors -> {len(circuit.groups['descending'])} descending neurons; "
          f"{(target[:, 0] > 0.3).mean():.0%} of scenes threatening")

    model = fit(band_operators(circuit), activity, target)
    print("real wiring:", {k: round(v, 3) for k, v in score(model, test_activity, test_target, teacher).items()})
    out = REPO / "runs" / "threat"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "model.npz", through=model["through"], readout=model["readout"], bias=model["bias"],
             gains=model["gains"], azimuth=retina.azimuth, elevation=retina.elevation)
    numbers = model["through"].size + model["readout"].size + model["bias"].size
    print(f"saved {out / 'model.npz'}: {numbers} numbers at run time, {len(model['gains']) + model['readout'].size + model['bias'].size} fitted")
    top = np.argsort(-np.abs(model["gains"]))[:5]
    print("largest band gains:", [f"{list(RELATIONS)[e // BANDS]} band {e % BANDS}: {model['gains'][e]:+.2f}" for e in top])

    if args.compare:
        for seed in (1, 2):
            rewired = fit(band_operators(load(20, 2, 1, shuffle=seed)), activity, target)
            print(f"rewired wiring (seed {seed}):", {k: round(v, 3) for k, v in score(rewired, test_activity, test_target, teacher).items()})
        direct = fit(None, activity, target)
        print("no brain (detectors read out directly):", {k: round(v, 3) for k, v in score(direct, test_activity, test_target, teacher).items()})


if __name__ == "__main__":
    main()
