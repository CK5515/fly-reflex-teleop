"""Do looming and touch excite different spectral bands of the reflex circuit?

If they do, a mixture of band experts can be routed by measuring where the
evoked activity lands, with no learned router. This script stimulates each
sensor population many times with random subsets and strengths, lets the
activity spread through the real wiring, and compares how its energy divides
across bands. Random populations of the same sizes are the yardstick: two
arbitrary groups of neurons also differ somewhat, so the real pair has to
differ by more than that to mean anything.

    python -m brain.band_experiment
"""

import numpy as np

from brain.circuit import Circuit, load

BANDS = 8
TRIALS = 200
STEPS = 3  # synaptic hops the activity is allowed to spread


def evoke(circuit: Circuit, population: np.ndarray, rng: np.random.Generator) -> list[np.ndarray]:
    """Firing rates after 0..STEPS hops, driving a random part of `population`."""
    n = len(circuit.root_ids)
    drive = np.zeros(n, np.float32)
    active = population[rng.random(len(population)) < rng.uniform(0.2, 1.0)]
    drive[active] = rng.lognormal(0.0, 0.5, len(active))
    rates, out = drive.copy(), [drive.copy()]
    for _ in range(STEPS):
        rates = np.maximum(circuit.weights @ rates + drive, 0.0)
        spread = rates.copy()
        spread[population] = 0.0  # only what the stimulus caused elsewhere
        out.append(spread)
    return out


def profiles(circuit, analysis, population, rng, edges) -> np.ndarray:
    """Share of the evoked energy in each band, as [trial, hop, band].

    `analysis` is the modes transposed with the constant mode left out. All
    trials of one hop are projected in a single matrix product.
    """
    evoked = [evoke(circuit, population, rng) for _ in range(TRIALS)]
    out = np.zeros((TRIALS, STEPS + 1, BANDS))
    for hop in range(STEPS + 1):
        rates = np.stack([trial[hop] for trial in evoked], axis=1)        # [neuron, trial]
        energy = (analysis @ rates) ** 2                                   # [mode, trial]
        per_band = np.add.reduceat(energy, edges[:-1], axis=0).T           # [trial, band]
        out[:, hop] = per_band / np.maximum(per_band.sum(1, keepdims=True), 1e-20)
    return out


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total variation between two mean band profiles: 0 = identical, 1 = no overlap."""
    return 0.5 * float(np.abs(a.mean(0) - b.mean(0)).sum())


def accuracy(a: np.ndarray, b: np.ndarray) -> float:
    """Nearest-centroid classification of single trials, fitted on one half, scored on the other."""
    half = len(a) // 2
    ca, cb = a[:half].mean(0), b[:half].mean(0)
    test = np.vstack([a[half:], b[half:]])
    truth = np.r_[np.zeros(len(a) - half), np.ones(len(b) - half)]
    guess = np.linalg.norm(test - cb, axis=1) < np.linalg.norm(test - ca, axis=1)
    return float((guess == truth).mean())


def main() -> None:
    circuit = load()
    rng = np.random.default_rng(0)
    n = len(circuit.root_ids)
    edges = np.linspace(0, n - 1, BANDS + 1).astype(int)  # equal number of modes per band
    g = circuit.groups
    analysis = np.ascontiguousarray(circuit.basis("all")[1][:, 1:].T)
    loom = np.r_[g["loom_left"], g["loom_right"]]
    print(f"circuit: {n} neurons; looming sensors {len(loom)}, touch sensors {len(g['touch'])}")

    p = {name: profiles(circuit, analysis, pop, rng, edges) for name, pop in
         [("loom", loom), ("touch", g["touch"]), ("loom_left", g["loom_left"]), ("loom_right", g["loom_right"])]}

    senders = np.flatnonzero(np.asarray((circuit.weights != 0).sum(0)).ravel() > 0)
    null_dist, null_acc = [], []
    for _ in range(20):
        a = profiles(circuit, analysis, rng.choice(senders, len(loom), replace=False), rng, edges)
        b = profiles(circuit, analysis, rng.choice(senders, len(g["touch"]), replace=False), rng, edges)
        null_dist.append([distance(a[:, h], b[:, h]) for h in range(STEPS + 1)])
        null_acc.append([accuracy(a[:, h], b[:, h]) for h in range(STEPS + 1)])
    null_dist, null_acc = np.array(null_dist), np.array(null_acc)

    np.set_printoptions(precision=3, suppress=True, linewidth=140)
    for hop in range(STEPS + 1):
        what = "the stimulus itself" if hop == 0 else f"activity spread {hop} hop(s), sensors excluded"
        print(f"\n== {what}")
        print("  band share, looming:", p["loom"][:, hop].mean(0))
        print("  band share, touch:  ", p["touch"][:, hop].mean(0))
        d = distance(p["loom"][:, hop], p["touch"][:, hop])
        print(f"  looming vs touch distance {d:.3f} | random pairs: median {np.median(null_dist[:, hop]):.3f}, "
              f"max {null_dist[:, hop].max():.3f} | random pairs beaten: {(d > null_dist[:, hop]).mean():.0%}")
        print(f"  single-trial accuracy looming vs touch {accuracy(p['loom'][:, hop], p['touch'][:, hop]):.0%} "
              f"| random pairs median {np.median(null_acc[:, hop]):.0%}")
        print(f"  single-trial accuracy left vs right looming {accuracy(p['loom_left'][:, hop], p['loom_right'][:, hop]):.0%}")


if __name__ == "__main__":
    main()
