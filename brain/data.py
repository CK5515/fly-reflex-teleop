"""Simulated activity in the reflex circuit, for training.

A stimulus drives some sensor neurons; activity then spreads through the
signed wiring as firing rates. Each example is the circuit's activity at a
few moments after stimulus onset, plus what the stimulus was.
"""

from dataclasses import dataclass

import numpy as np
import torch

from brain.circuit import Circuit

CLASSES = ["none", "escape_left", "escape_right", "groom"]
SNAPSHOTS = 4       # moments of activity kept per example (the node features)
STEPS_BETWEEN = 2   # simulation steps between snapshots
LEAK = 0.5          # how much of a neuron's rate is replaced each step
GAIN = 4.0          # scales synaptic input before the rate saturates


@dataclass
class Batch:
    drive: torch.Tensor      # [batch, neurons] sensor input
    activity: torch.Tensor   # [batch, neurons, SNAPSHOTS]
    label: torch.Tensor      # [batch] index into CLASSES


class Simulator:
    def __init__(self, circuit: Circuit, device: str = "cpu"):
        self.n = len(circuit.root_ids)
        self.device = device
        w = circuit.weights.tocoo()
        self.weights = torch.sparse_coo_tensor(
            np.vstack([w.row, w.col]), w.data.astype(np.float32), (self.n, self.n)).coalesce().to(device)
        g = circuit.groups
        self.populations = {
            "escape_left": torch.as_tensor(g["loom_left"], device=device),
            "escape_right": torch.as_tensor(g["loom_right"], device=device),
            "groom": torch.as_tensor(g["touch"], device=device),
        }
        sensors = np.r_[g["loom_left"], g["loom_right"], g["touch"]]
        self.sensors = torch.as_tensor(sensors, device=device)

    def spread(self, drive: torch.Tensor) -> torch.Tensor:
        """Activity snapshots [batch, neurons, SNAPSHOTS] for sensor input [batch, neurons]."""
        rates = torch.zeros_like(drive)
        frames = []
        for step in range(SNAPSHOTS * STEPS_BETWEEN):
            synaptic = torch.sparse.mm(self.weights, rates.T).T
            rates = (1 - LEAK) * rates + LEAK * torch.tanh(torch.relu(GAIN * synaptic + drive))
            if (step + 1) % STEPS_BETWEEN == 0:
                frames.append(rates)
        return torch.stack(frames, dim=-1)

    def drive_for(self, name: str, strength: float = 1.0, share: float = 1.0,
                  generator: torch.Generator | None = None) -> torch.Tensor:
        """Sensor input [neurons] for one stimulus: `share` of the population active at `strength`."""
        drive = torch.zeros(self.n, device=self.device)
        population = self.populations[name]
        active = torch.rand(len(population), generator=generator, device=self.device) < share
        drive[population[active]] = strength
        return drive

    def batch(self, size: int, generator: torch.Generator | None = None) -> Batch:
        """Random stimuli: each class equally often, varied in strength, extent and background noise."""
        label = torch.randint(len(CLASSES), (size,), generator=generator, device=self.device)
        drive = torch.zeros(size, self.n, device=self.device)
        rand = lambda *shape: torch.rand(*shape, generator=generator, device=self.device)
        for i in range(size):
            name = CLASSES[label[i]]
            if name != "none":
                drive[i] = self.drive_for(name, strength=0.3 + 0.7 * rand(1).item(),
                                          share=0.3 + 0.7 * rand(1).item(), generator=generator)
        # Background: every sensor fires a little at random, so "none" is not simply silence.
        noise = 0.15 * rand(size, len(self.sensors)) * (rand(size, len(self.sensors)) < 0.2)
        drive[:, self.sensors] += noise
        return Batch(drive, self.spread(drive), label)
