"""Where the circuit's neurons sit in the brain, for drawing them on the live page.

Positions are the FlyWire annotation coordinates seen from the front (x across, y down), scaled into
a unit box. Cached as JSON next to the fitted models.

    python -m brain.layout
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from brain.circuit import FLYWIRE, load

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "runs" / "threat" / "layout.json"


def build() -> dict:
    circuit = load(20, 2, 1)
    ann = pd.read_csv(FLYWIRE / "neuron_annotations.tsv", sep="\t", low_memory=False,
                      usecols=["root_id", "pos_x", "pos_y"]).drop_duplicates("root_id").set_index("root_id")
    xy = ann.loc[circuit.root_ids, ["pos_x", "pos_y"]].values.astype(float)
    low, high = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    span = (high - low).max()
    xy = np.clip((xy - (low + high) / 2) / span + 0.5, 0.0, 1.0)   # same scale on both axes, centred
    detectors = np.r_[circuit.groups["loom_left"], circuit.groups["loom_right"]]
    descending = circuit.groups["descending"]
    rest = np.setdiff1d(np.arange(len(xy)), np.r_[detectors, descending])
    pack = lambda rows: np.round(xy[rows], 4).tolist()
    return {"detectors": pack(detectors), "descending": pack(descending), "rest": pack(rest),
            "aspect": float((high - low)[1] / (high - low)[0])}


def load_layout() -> dict | None:
    """The cached layout, building it if the connectome data is here; None if neither is available."""
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    try:
        layout = build()
    except (FileNotFoundError, KeyError):
        return None
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(layout))
    return layout


if __name__ == "__main__":
    layout = load_layout()
    print({k: (len(v) if isinstance(v, list) else v) for k, v in layout.items()})
    d = np.array(layout["detectors"]); n = len(d) // 2
    print("left-eye detectors centre", d[:108].mean(0).round(2), "| right-eye", d[108:].mean(0).round(2),
          "| descending centre", np.array(layout["descending"]).mean(0).round(2))
