"""Does the wiring alone send looming to the escape neurons? No training involved.

Stimulate each sensor population in the simulated circuit and read the firing
rate of the two giant-fibre neurons (the fly's escape command neurons) and of
the average descending neuron. Repeat on the rewired control circuit.

    python -m brain.giant_fibre_check --min-synapses 10
"""

import argparse

import torch

from brain.circuit import load
from brain.data import CLASSES, Simulator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--min-synapses", type=int, default=10)
    p.add_argument("--hops-out", type=int, default=2)
    p.add_argument("--hops-back", type=int, default=2)
    p.add_argument("--shuffles", type=int, nargs="*", default=[1])
    args = p.parse_args()
    for shuffle in [0, *args.shuffles]:
        circuit = load(args.min_synapses, args.hops_out, args.hops_back, shuffle)
        sim = Simulator(circuit)
        giant = torch.as_tensor(circuit.groups["giant_fibre"])
        descending = torch.as_tensor(circuit.groups["descending"])
        generator = torch.Generator().manual_seed(0)
        print(f"\n{'real wiring' if shuffle == 0 else f'rewired control (seed {shuffle})'}: {circuit.name}")
        for name in CLASSES[1:]:
            drive = torch.stack([sim.drive_for(name, strength=0.3 + 0.7 * torch.rand(1, generator=generator).item(),
                                               share=0.3 + 0.7 * torch.rand(1, generator=generator).item(),
                                               generator=generator) for _ in range(100)])
            final = sim.spread(drive)[:, :, -1]
            print(f"  {name:13s} giant fibre rate {final[:, giant].mean():.4f} | "
                  f"average descending neuron {final[:, descending].mean():.4f} | "
                  f"giant fibre is in the top {100 * (final[:, descending].mean(0) >= final[:, giant].mean()).float().mean():.0f}% of descending neurons")


if __name__ == "__main__":
    main()
