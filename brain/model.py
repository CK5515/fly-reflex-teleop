"""A spectral mixture of band experts over a multi-relation connectome.

Each neurotransmitter relation has its own graph Laplacian, decomposed in
advance. A layer moves the neurons' features into each relation's spectral
basis, where the modes are grouped into bands; every (relation, band) pair is
one expert with its own mixing matrix. A router reads how much energy each
band holds and decides how much each expert contributes.

Following MORGaN (Occhetta et al., 2025), the experts' matrices are built
from a few shared basis matrices, and the model is pre-trained as a masked
autoencoder: hide half the neurons' activity and reconstruct it from the rest.
"""

import torch
from torch import nn


class SpectralBasis(nn.Module):
    """The precomputed modes of every relation, with each mode assigned to a band."""

    def __init__(self, modes: list[torch.Tensor], n_bands: int):
        super().__init__()
        self.n_relations = len(modes)
        self.n_bands = n_bands
        for r, m in enumerate(modes):
            self.register_buffer(f"modes_{r}", m, persistent=False)   # [neurons, modes], eigenvalue ascending
            band = torch.arange(m.shape[1]) * n_bands // m.shape[1]   # equal number of modes per band
            self.register_buffer(f"band_{r}", band, persistent=False)

    def modes(self, r: int) -> torch.Tensor:
        return getattr(self, f"modes_{r}")

    def band(self, r: int) -> torch.Tensor:
        return getattr(self, f"band_{r}")


class BandExpertLayer(nn.Module):
    def __init__(self, basis: SpectralBasis, d_in: int, d_out: int, n_shared: int = 2, router_modes: int = 16):
        super().__init__()
        self.basis = basis
        experts = basis.n_relations * basis.n_bands
        self.shared = nn.Parameter(torch.empty(n_shared, d_in, d_out))       # shared basis matrices
        self.mix = nn.Parameter(torch.empty(experts, n_shared))              # each expert's blend of them
        self.own = nn.Linear(d_in, d_out)                                    # a neuron's own features
        # The router reads two things per relation: how the energy divides across bands, and the signed
        # coefficients of the lowest modes. Low modes are the circuit's coarse spatial patterns, so their
        # signs say which part of the circuit is active; band energy alone does not (see band_experiment).
        self.router_modes = router_modes
        self.squeeze = nn.Linear(d_in, 4)
        router_in = experts + basis.n_relations * router_modes * 4
        self.router = nn.Sequential(nn.LayerNorm(router_in), nn.Linear(router_in, 64), nn.PReLU(), nn.Linear(64, experts))
        self.norm = nn.LayerNorm(d_out)
        self.act = nn.PReLU()
        self.residual = d_in == d_out
        nn.init.xavier_uniform_(self.shared)
        nn.init.xavier_uniform_(self.mix)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: [batch, neurons, d_in] -> ([batch, neurons, d_out], gates [batch, relations * bands])."""
        b = self.basis
        coeffs, energy, coarse = [], [], []
        for r in range(b.n_relations):
            z = torch.einsum("nk,bnd->bkd", b.modes(r), h)                  # features in spectral coordinates
            per_mode = z.pow(2).sum(-1)                                      # [batch, modes]
            per_band = torch.zeros(h.shape[0], b.n_bands, device=h.device).index_add_(1, b.band(r), per_mode)
            coeffs.append(z)
            energy.append(per_band)
            coarse.append(self.squeeze(z[:, :self.router_modes]).flatten(1))
        energy = torch.cat(energy, dim=1)                                    # [batch, experts]
        share = energy / energy.sum(1, keepdim=True).clamp_min(1e-12)
        router_in = torch.cat([torch.log(share + 1e-6), *coarse], dim=1)
        gates = torch.softmax(self.router(router_in), dim=1)                 # how much each expert contributes

        matrices = torch.einsum("es,sio->eio", self.mix, self.shared)        # [experts, d_in, d_out]
        out = self.own(h)
        for r in range(b.n_relations):
            expert = r * b.n_bands + b.band(r)                               # which expert owns each mode
            z = torch.einsum("bki,kio->bko", coeffs[r], matrices[expert])
            z = z * gates[:, expert].unsqueeze(-1) * gates.shape[1]          # gate 1/experts = neutral
            out = out + torch.einsum("nk,bko->bno", b.modes(r), z)
        out = self.act(self.norm(out))
        return (out + h if self.residual else out), gates


class SpectralMoE(nn.Module):
    def __init__(self, modes: list[torch.Tensor], n_features: int, n_classes: int, n_readout: int,
                 n_bands: int = 8, width: int = 32, depth: int = 2):
        super().__init__()
        self.basis = SpectralBasis(modes, n_bands)
        self.embed = nn.Linear(n_features, width)
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.encoder = nn.ModuleList([BandExpertLayer(self.basis, width, width) for _ in range(depth)])
        self.decoder = BandExpertLayer(self.basis, width, width)
        self.reconstruct = nn.Linear(width, n_features)
        experts = self.basis.n_relations * n_bands
        # Each output neuron votes for each class with its own weight, so which neuron is active matters
        # (left and right outputs are different neurons; an average over them would hide the side).
        self.vote = nn.Linear(width, n_classes)
        self.neuron_weight = nn.Parameter(torch.ones(n_readout, n_classes))
        self.from_gates = nn.Linear(experts, n_classes)

    def encode(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [batch, neurons, features]; hidden: [batch, neurons] bool, neurons whose features are masked."""
        h = self.embed(x)
        if hidden is not None:
            h = torch.where(hidden.unsqueeze(-1), self.mask_token.expand_as(h), h)
        gates = None
        for layer in self.encoder:
            h, gates = layer(h)
        return h, gates                                                      # gates of the last encoder layer

    def reconstruction(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        h, _ = self.encode(x, hidden)
        h = torch.where(hidden.unsqueeze(-1), self.mask_token.expand_as(h), h)   # re-mask before decoding
        h, _ = self.decoder(h)
        return self.reconstruct(h)

    def decide(self, x: torch.Tensor, readout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Class scores from the embeddings of the `readout` neurons (the brain's outputs) and the gates."""
        h, gates = self.encode(x)
        votes = self.vote(h[:, readout]) * self.neuron_weight               # [batch, outputs, classes]
        return votes.mean(1) + self.from_gates(gates), gates


def routing_loss(gates: torch.Tensor) -> torch.Tensor:
    """Negative mutual information between example and expert, scaled to [-1, 0].

    Lowest when each example routes to few experts (sharp) while the batch as a
    whole uses all of them (balanced). Sharpness alone collapses the router
    onto one expert; balance alone flattens it to uniform; this asks for both.
    """
    entropy = lambda p: -(p * (p + 1e-9).log()).sum(-1)
    information = entropy(gates.mean(0)) - entropy(gates).mean()
    return -information / torch.log(torch.tensor(float(gates.shape[1])))
