"""Pre-train the spectral model as a masked autoencoder, then teach it the reflex decision.

    python -m brain.train --min-synapses 20 --hops-back 1 --pretrain-steps 200 --finetune-steps 200

Stage 1 (self-supervised): hide half the neurons' simulated activity and
reconstruct it from the rest through the wiring's spectral bases.
Stage 2: from the full activity, read out which stimulus occurred (nothing,
looming from the left, looming from the right, touch) at the descending neurons.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from brain.circuit import RELATIONS, load
from brain.data import CLASSES, SNAPSHOTS, Simulator
from brain.model import SpectralMoE, routing_loss

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--min-synapses", type=int, default=10)
    p.add_argument("--hops-out", type=int, default=2)
    p.add_argument("--hops-back", type=int, default=2)
    p.add_argument("--bands", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--pretrain-steps", type=int, default=1000)
    p.add_argument("--finetune-steps", type=int, default=500)
    p.add_argument("--mask", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--routing", type=float, default=0.3, help="weight of the sharp-but-balanced routing objective")
    p.add_argument("--shuffle", type=int, default=0, help="seed for the rewired control (0 = real wiring)")
    p.add_argument("--single-relation", action="store_true", help="ablation: one Laplacian over all connections")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="run")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    circuit = load(args.min_synapses, args.hops_out, args.hops_back, args.shuffle)
    relations = ["all"] if args.single_relation else list(RELATIONS)
    modes = [torch.from_numpy(circuit.basis(r)[1]).to(device) for r in relations]
    sim = Simulator(circuit, device)
    readout = torch.as_tensor(circuit.groups["descending"], device=device)
    model = SpectralMoE(modes, SNAPSHOTS, len(CLASSES), len(readout), args.bands, args.width, args.depth).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"{circuit.name}: {len(circuit.root_ids)} neurons, relations {relations}, {params} parameters, on {device}")

    out = REPO / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    log = {"args": vars(args), "pretrain": [], "finetune": []}
    began = time.monotonic()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    for step in range(1, args.pretrain_steps + 1):
        batch = sim.batch(args.batch)
        hidden = torch.rand(batch.activity.shape[:2], device=device) < args.mask
        predicted = model.reconstruction(batch.activity, hidden)
        loss = F.mse_loss(predicted[hidden], batch.activity[hidden])
        optim.zero_grad(); loss.backward(); optim.step()
        if step % 25 == 0 or step == args.pretrain_steps:
            # How much of the hidden activity's variance is recovered (1 = all, 0 = no better than the mean).
            target = batch.activity[hidden]
            explained = 1 - loss.item() / target.var().item()
            log["pretrain"].append({"step": step, "loss": loss.item(), "explained": explained})
            print(f"pretrain {step:5d}  loss {loss.item():.5f}  hidden activity explained {explained:6.1%}  "
                  f"{time.monotonic() - began:5.0f} s")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr / 3, weight_decay=1e-4)
    for step in range(1, args.finetune_steps + 1):
        batch = sim.batch(args.batch)
        scores, gates = model.decide(batch.activity, readout)
        loss = F.cross_entropy(scores, batch.label)
        optim.zero_grad(); (loss + args.routing * routing_loss(gates)).backward(); optim.step()
        if step % 25 == 0 or step == args.finetune_steps:
            accuracy = (scores.argmax(1) == batch.label).float().mean().item()
            log["finetune"].append({"step": step, "loss": loss.item(), "accuracy": accuracy})
            print(f"finetune {step:5d}  loss {loss.item():.4f}  batch accuracy {accuracy:6.1%}  "
                  f"{time.monotonic() - began:5.0f} s")

    # Held-out check on fresh stimuli, and the average gate pattern per stimulus class.
    model.eval()
    generator = torch.Generator(device=device).manual_seed(10_000 + args.seed)
    correct, total = 0, 0
    gate_sum = torch.zeros(len(CLASSES), model.basis.n_relations * args.bands, device=device)
    count = torch.zeros(len(CLASSES), device=device)
    confusion = np.zeros((len(CLASSES), len(CLASSES)), int)
    with torch.no_grad():
        for _ in range(25):
            batch = sim.batch(args.batch, generator)
            scores, gates = model.decide(batch.activity, readout)
            guess = scores.argmax(1)
            correct += (guess == batch.label).sum().item(); total += len(guess)
            gate_sum.index_add_(0, batch.label, gates); count.index_add_(0, batch.label, torch.ones_like(count)[batch.label])
            for t, g in zip(batch.label.tolist(), guess.tolist()):
                confusion[t, g] += 1
    gates = (gate_sum / count.clamp_min(1).unsqueeze(1)).cpu().numpy()
    log["test_accuracy"] = correct / total
    log["confusion"] = confusion.tolist()
    log["gates_by_class"] = gates.tolist()
    print(f"\nheld-out accuracy {correct / total:.1%} on {total} fresh stimuli (chance {1 / len(CLASSES):.0%})")
    print("rows = true class, columns = guessed:", CLASSES)
    print(confusion)
    top = np.argsort(-gates, axis=1)[:, :3]
    for c, name in enumerate(CLASSES):
        labels = [f"{relations[e // args.bands]}/band{e % args.bands} {gates[c, e]:.2f}" for e in top[c]]
        print(f"  strongest experts for {name}: {labels}")

    # How differently the router treats the classes: 0 = identical gates, 1 = no expert in common.
    separation = {f"{CLASSES[a]} vs {CLASSES[b]}": round(0.5 * float(np.abs(gates[a] - gates[b]).sum()), 3)
                  for a in range(len(CLASSES)) for b in range(a + 1, len(CLASSES))}
    log["gate_separation"] = separation
    print("  gate difference between classes (0 = same, 1 = disjoint):", separation)
    torch.save({"model": model.state_dict(), "args": vars(args), "relations": relations}, out / "model.pt")
    (out / "log.json").write_text(json.dumps(log, indent=1))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
