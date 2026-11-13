"""Cut the reflex circuit out of the FlyWire connectome and decompose it.

The circuit is every neuron on a short path from the sensors we stimulate
(looming detectors, touch bristles) to the descending neurons that leave the
brain. Connections are split by neurotransmitter into relations, and each
relation's graph Laplacian is decomposed in full, so every spectral basis is
exact rather than truncated.

    python -m brain.circuit --min-synapses 20 --hops-back 1     # small, runs on a laptop
    python -m brain.circuit --min-synapses 5                    # 17,700 neurons, needs a big machine

Results are cached under data/brain/<name>/.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

REPO = Path(__file__).resolve().parent.parent
FLYWIRE = REPO / "data/flywire"
BRAIN = REPO / "data/brain"

TOUCH_CLASSES = ["grooming", "head bristle"]   # mechanosensory sub-classes used as touch sensors
# Relation -> the neurotransmitter columns of the connection table that make it up.
# Dopamine, serotonin and octopamine are rare, so they share one "modulatory" relation.
RELATIONS = {
    "ach": ["ach_avg"],
    "gaba": ["gaba_avg"],
    "glut": ["glut_avg"],
    "mod": ["da_avg", "ser_avg", "oct_avg"],
}
NT_COLUMNS = [c for cols in RELATIONS.values() for c in cols]


@dataclass
class Circuit:
    name: str
    root_ids: np.ndarray                    # FlyWire id of each neuron in the circuit
    weights: sp.csr_matrix                  # weights[post, pre]: signed share of post's input coming from pre
    relations: dict[str, sp.csr_matrix]     # synapse counts per relation, [post, pre]
    groups: dict[str, np.ndarray]           # named neuron populations, as indices into the circuit

    @property
    def folder(self) -> Path:
        return BRAIN / self.name

    def basis(self, relation: str) -> tuple[np.ndarray, np.ndarray]:
        """(eigenvalues ascending, modes as columns) of one relation's Laplacian; "all" = every connection."""
        z = np.load(self.folder / f"basis_{relation}.npz")
        return z["eigenvalues"], z["modes"]


def circuit_name(min_synapses: int, hops_out: int, hops_back: int, shuffle: int = 0) -> str:
    return f"syn{min_synapses}_out{hops_out}_back{hops_back}" + (f"_shuffle{shuffle}" if shuffle else "")


def _shuffled(sub: sp.csr_matrix, relations: dict[str, sp.csr_matrix], seed: int):
    """Rewire at random: every connection keeps its sender and strength but gets another one's target.

    Each neuron keeps its number of outgoing and incoming connections, so only
    the specific wiring is destroyed. This is the control for "does the real
    wiring matter".
    """
    coo = sub.tocoo()
    order = np.random.default_rng(seed).permutation(coo.nnz)
    post, pre = coo.row[order], coo.col
    rebuild = lambda values: sp.csr_matrix((values, (post, pre)), shape=sub.shape)
    new_relations = {rel: rebuild(np.asarray(m[coo.row, coo.col]).ravel()) for rel, m in relations.items()}
    return rebuild(coo.data), new_relations


def _edges() -> pd.DataFrame:
    """Neuron-to-neuron synapse counts (5 or more), summed over brain regions and cached."""
    cache = FLYWIRE / "edges_min5.feather"
    if cache.exists():
        return pd.read_feather(cache)
    edges = pd.read_feather(FLYWIRE / "proofread_connections_783.feather",
                            columns=["pre_pt_root_id", "post_pt_root_id", "syn_count"])
    edges = edges[edges.syn_count >= 2]  # drops single-synapse contacts before summing, to fit in memory
    edges = edges.groupby(["pre_pt_root_id", "post_pt_root_id"], as_index=False, sort=False).syn_count.sum()
    edges = edges[edges.syn_count >= 5].reset_index(drop=True)
    edges.to_feather(cache)
    return edges


def _reach(seed: np.ndarray, step: sp.csr_matrix, hops: int) -> np.ndarray:
    seen = seed.copy()
    for _ in range(hops):
        seen |= np.asarray(step @ seen.astype(np.float32)).ravel() > 0
    return seen


def _relation_counts(root_ids: np.ndarray) -> dict[str, sp.csr_matrix]:
    """Synapse counts between circuit neurons, split by transmitter. Reads the big table piece by piece."""
    order = np.argsort(root_ids)
    sorted_ids = root_ids[order]
    n = len(root_ids)
    totals = {name: sp.csr_matrix((n, n), dtype=np.float32) for name in RELATIONS}
    reader = pa.ipc.open_file(FLYWIRE / "proofread_connections_783.feather")
    columns = ["pre_pt_root_id", "post_pt_root_id", "syn_count", *NT_COLUMNS]
    for i in range(reader.num_record_batches):
        batch = reader.get_batch(i).select(columns).to_pandas()
        inside = batch.pre_pt_root_id.isin(sorted_ids) & batch.post_pt_root_id.isin(sorted_ids)
        batch = batch[inside.values]
        if batch.empty:
            continue
        pre = order[np.searchsorted(sorted_ids, batch.pre_pt_root_id.values)]
        post = order[np.searchsorted(sorted_ids, batch.post_pt_root_id.values)]
        for name, cols in RELATIONS.items():
            value = (batch.syn_count.values * batch[cols].sum(axis=1).values).astype(np.float32)
            totals[name] = totals[name] + sp.csr_matrix((value, (post, pre)), shape=(n, n))
    return totals


def decompose(counts: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Eigenvalues and modes of the symmetric normalised Laplacian of a connection matrix.

    Uses the GPU when PyTorch finds one. A neuron with no connections in this
    relation gets its own mode with eigenvalue 1.
    """
    sym = (counts + counts.T).toarray().astype(np.float32)
    scale = (1.0 / np.sqrt(np.maximum(sym.sum(1), 1e-12))).astype(np.float32)
    sym *= -scale[:, None]
    sym *= scale[None, :]
    sym[np.diag_indices_from(sym)] += 1.0  # now the Laplacian, built in place
    try:
        import torch
        if torch.cuda.is_available():
            values, modes = torch.linalg.eigh(torch.from_numpy(sym).cuda())
            return values.cpu().numpy(), modes.cpu().numpy()
    except ImportError:
        pass
    return scipy.linalg.eigh(sym, overwrite_a=True, check_finite=False)


def build(min_synapses: int, hops_out: int, hops_back: int, shuffle: int = 0) -> Circuit:
    name = circuit_name(min_synapses, hops_out, hops_back, shuffle)
    ann = pd.read_csv(FLYWIRE / "neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "super_class", "cell_class", "cell_sub_class", "cell_type", "side"])
    ids = np.load(FLYWIRE / "proofread_root_ids_783.npy")
    index = pd.Series(np.arange(len(ids)), index=ids)
    ann = ann[ann.root_id.isin(index.index)].drop_duplicates("root_id")
    ann_index = index[ann.root_id].values
    n = len(ids)

    edges = _edges()
    edges = edges[edges.syn_count >= min_synapses]
    pre, post = index[edges.pre_pt_root_id].values, index[edges.post_pt_root_id].values
    counts = sp.csr_matrix((edges.syn_count.values.astype(np.float32), (post, pre)), shape=(n, n))

    def mask(rows: pd.Series) -> np.ndarray:
        out = np.zeros(n, bool)
        out[ann_index[rows.values]] = True
        return out

    populations = {
        "loom_left": mask((ann.cell_type == "LPLC2") & (ann.side == "left")),
        "loom_right": mask((ann.cell_type == "LPLC2") & (ann.side == "right")),
        "touch": mask((ann.cell_class == "mechanosensory") & ann.cell_sub_class.isin(TOUCH_CLASSES)),
        "giant_fibre": mask(ann.cell_type == "DNp01"),
        "descending": mask(ann.super_class == "descending"),
    }
    sensors = populations["loom_left"] | populations["loom_right"] | populations["touch"]
    downstream = _reach(sensors, counts, hops_out)                    # counts sends pre -> post
    upstream = _reach(populations["descending"], counts.T.tocsr(), hops_back)
    keep = (downstream & upstream) | sensors

    sub = counts[keep][:, keep]
    _, label = connected_components(sub + sub.T, directed=False)
    biggest = label == np.bincount(label).argmax()
    keep_idx = np.flatnonzero(keep)[biggest]
    sub = sub[biggest][:, biggest].tocsr()
    root_ids = ids[keep_idx]

    kept = (sub > 0).astype(np.float32)   # only connections that passed the synapse threshold
    relations = {rel: m.multiply(kept).tocsr() for rel, m in _relation_counts(root_ids).items()}
    if shuffle:
        sub, relations = _shuffled(sub, relations, shuffle)

    # Fast dynamics: acetylcholine excites, GABA and glutamate inhibit, each as a share of the target's input.
    signed = relations["ach"] - relations["gaba"] - relations["glut"]
    total_in = np.asarray(sub.sum(1)).ravel()
    weights = (sp.diags(1.0 / np.maximum(total_in, 1.0)) @ signed).tocsr().astype(np.float32)

    position = np.full(n, -1)
    position[keep_idx] = np.arange(len(keep_idx))
    groups = {g: position[np.flatnonzero(m)] for g, m in populations.items()}
    groups = {g: v[v >= 0] for g, v in groups.items()}

    circuit = Circuit(name, root_ids, weights, relations, groups)
    circuit.folder.mkdir(parents=True, exist_ok=True)
    for rel, matrix in {"all": sub, **relations}.items():
        start = time.monotonic()
        values, modes = decompose(matrix)
        np.savez(circuit.folder / f"basis_{rel}.npz", eigenvalues=values, modes=modes.astype(np.float32))
        print(f"  {rel}: {matrix.nnz} connections, decomposed in {time.monotonic() - start:.0f} s")
    sp.save_npz(circuit.folder / "weights.npz", weights)
    for rel, matrix in relations.items():
        sp.save_npz(circuit.folder / f"relation_{rel}.npz", matrix)
    np.savez(circuit.folder / "meta.npz", root_ids=root_ids, **{f"group_{g}": v for g, v in groups.items()})
    return circuit


def load(min_synapses: int = 10, hops_out: int = 2, hops_back: int = 2, shuffle: int = 0) -> Circuit:
    name = circuit_name(min_synapses, hops_out, hops_back, shuffle)
    folder = BRAIN / name
    if not (folder / "meta.npz").exists():
        return build(min_synapses, hops_out, hops_back, shuffle)
    meta = np.load(folder / "meta.npz")
    groups = {k[len("group_"):]: meta[k] for k in meta.files if k.startswith("group_")}
    relations = {rel: sp.load_npz(folder / f"relation_{rel}.npz") for rel in RELATIONS}
    return Circuit(name, meta["root_ids"], sp.load_npz(folder / "weights.npz"), relations, groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--min-synapses", type=int, default=10)
    parser.add_argument("--hops-out", type=int, default=2)
    parser.add_argument("--hops-back", type=int, default=2)
    parser.add_argument("--shuffle", type=int, default=0, help="seed for the rewired control (0 = real wiring)")
    args = parser.parse_args()
    began = time.monotonic()
    c = load(args.min_synapses, args.hops_out, args.hops_back, args.shuffle)
    print(f"{c.name}: {len(c.root_ids)} neurons, {c.weights.nnz} signed connections, {time.monotonic() - began:.0f} s")
    print("populations:", {k: len(v) for k, v in c.groups.items()})
    print("connections per relation:", {k: v.nnz for k, v in c.relations.items()})
    print("synapses per relation:", {k: int(v.sum()) for k, v in c.relations.items()})
