"""Does knowing the real wiring help? Activity always comes from the real circuit; only the model's
spectral bases change.

The task is the pre-training one: hide half the neurons' activity and reconstruct it from the rest.

    --basis true      bases of the real wiring, one per neurotransmitter relation
    --basis single    one basis over all real connections
    --basis rewired   bases of a rewired circuit (same neurons, same connection counts per neuron)
    --basis none      no wiring: every neuron is its own mode, so nothing passes between neurons

Test A (default): train and test on the same kinds of stimulus.
Test B: `--train looming` trains on looming only; touch and random-neuron stimuli are then never seen
in training, so reconstructing them needs knowledge of where activity goes.

    python -m brain.structure_test --basis true --train all --seed 0
    python -m brain.structure_test --summarise
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from brain.circuit import RELATIONS, load
from brain.data import SNAPSHOTS, Simulator
from brain.model import SpectralMoE

OUT = Path(__file__).resolve().parent.parent / "runs" / "structure"
KINDS = ["escape_left", "escape_right", "groom", "random"]   # "random" = 200 arbitrary neurons driven
TRAIN_SETS = {"all": ["escape_left", "escape_right", "groom"], "looming": ["escape_left", "escape_right"]}
RANDOM_SIZE = 200


def stimuli(sim: Simulator, kinds: list[str], size: int, senders: torch.Tensor, generator: torch.Generator):
    """Activity [size, neurons, snapshots] for stimuli drawn evenly from `kinds`."""
    rand = lambda: torch.rand(1, generator=generator, device=sim.device).item()
    drive = torch.zeros(size, sim.n, device=sim.device)
    for i in range(size):
        kind = kinds[i % len(kinds)]
        if kind == "random":
            chosen = senders[torch.randperm(len(senders), generator=generator, device=sim.device)[:RANDOM_SIZE]]
            drive[i, chosen] = 0.3 + 0.7 * rand()
        else:
            drive[i] = sim.drive_for(kind, strength=0.3 + 0.7 * rand(), share=0.3 + 0.7 * rand(), generator=generator)
    return sim.spread(drive)


def run(args) -> None:
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = dict(min_synapses=args.min_synapses, hops_out=2, hops_back=2)
    real = load(**size)
    n = len(real.root_ids)
    if args.basis == "true":
        modes = [torch.from_numpy(real.basis(r)[1]) for r in RELATIONS]
    elif args.basis == "single":
        modes = [torch.from_numpy(real.basis("all")[1])]
    elif args.basis == "rewired":
        rewired = load(**size, shuffle=args.rewire_seed)
        assert (rewired.root_ids == real.root_ids).all()
        modes = [torch.from_numpy(rewired.basis(r)[1]) for r in RELATIONS]
    else:
        modes = [torch.eye(n)]
    modes = [m.to(device) for m in modes]
    sim = Simulator(real, device)                       # the data always come from the real wiring
    senders = torch.as_tensor(np.flatnonzero(np.asarray((real.weights != 0).sum(0)).ravel() > 0), device=device)
    model = SpectralMoE(modes, SNAPSHOTS, 2, 1, args.bands, args.width, args.depth).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_kinds = TRAIN_SETS[args.train]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    for step in range(1, args.steps + 1):
        activity = stimuli(sim, train_kinds, args.batch, senders, generator)
        hidden = torch.rand(activity.shape[:2], device=device, generator=generator) < 0.5
        loss = F.mse_loss(model.reconstruction(activity, hidden)[hidden], activity[hidden])
        optim.zero_grad(); loss.backward(); optim.step()
        if step % 250 == 0:
            print(f"{args.name} step {step} loss {loss.item():.5f}", flush=True)

    # Held-out stimuli, the same ones for every model (fixed seed). Explained = share of the hidden
    # activity's variance recovered: 1 is perfect, 0 is no better than predicting its overall mean.
    model.eval()
    test = torch.Generator(device=device).manual_seed(99_999)
    result = {"args": vars(args), "explained": {}, "per_neuron_average": {}}
    with torch.no_grad():
        for kind in KINDS:
            reference = stimuli(sim, [kind], 64, senders, test).mean(0, keepdim=True)   # per-neuron average
            error = baseline = spread = 0.0
            for _ in range(20):
                activity = stimuli(sim, [kind], args.batch, senders, test)
                hidden = torch.rand(activity.shape[:2], device=device, generator=test) < 0.5
                target = activity[hidden]
                error += (model.reconstruction(activity, hidden)[hidden] - target).pow(2).sum().item()
                baseline += (reference.expand_as(activity)[hidden] - target).pow(2).sum().item()
                spread += (target - activity.mean()).pow(2).sum().item()
            result["explained"][kind] = 1 - error / spread
            result["per_neuron_average"][kind] = 1 - baseline / spread
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{args.name}.json").write_text(json.dumps(result, indent=1))
    print(args.name, {k: round(v, 3) for k, v in result["explained"].items()}, flush=True)


def summarise() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    for train in TRAIN_SETS:
        print(f"\ntrained on: {train} ({', '.join(TRAIN_SETS[train])}); unseen in training: "
              f"{', '.join(k for k in KINDS if k not in TRAIN_SETS[train])}")
        print(f"{'model':<26}{'runs':>5}" + "".join(f"{k:>16}" for k in KINDS))
        chosen = [r for r in runs if r["args"]["train"] == train]
        for basis in ["true", "single", "rewired", "none"]:
            group = [r for r in chosen if r["args"]["basis"] == basis]
            if not group:
                continue
            cells = ""
            for kind in KINDS:
                values = [r["explained"][kind] for r in group]
                cells += f"{np.mean(values):>9.3f} ±{np.std(values):.3f}"
            print(f"{basis:<26}{len(group):>5}{cells}")
        if chosen:
            print(f"{'(per-neuron average)':<26}{'':>5}" + "".join(
                f"{np.mean([r['per_neuron_average'][k] for r in chosen]):>16.3f}" for k in KINDS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--summarise", action="store_true")
    p.add_argument("--basis", choices=["true", "single", "rewired", "none"], default="true")
    p.add_argument("--rewire-seed", type=int, default=1)
    p.add_argument("--train", choices=list(TRAIN_SETS), default="all")
    p.add_argument("--min-synapses", type=int, default=10)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.summarise:
        summarise()
    else:
        tag = f"rewired{args.rewire_seed}" if args.basis == "rewired" else args.basis
        args.name = f"{args.train}_{tag}_seed{args.seed}"
        run(args)
