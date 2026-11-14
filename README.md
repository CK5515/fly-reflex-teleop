# Fly Reflex Teleop

Junction 2026, Rokoko teleoperation track.

A motion-capture glove mirrors to a robot what the operator does, but nothing tells the operator gets no feedback from the robot.
You cannot feel a load sliding and cannot react to an object that suddenly appears in half a second.

This project puts a "reflex layer" that runs in parallel with the regular operator. A person steers a simulated robot arm with a Rokoko glove.
Meanwhile, the arm keeps what it carries steady and steps out of the way of incoming objects for a brief moment before handing back control. The incoming objects are sensed through the fruit fly's looming circuit, taken from the FlyWire connectome of *Drosophila*.

## What you see

`python demo.py` opens one page:

- **The arm**, seen from above its shoulder as if it were your own. It starts empty. Pinch on the paddle's handle to
  pick it up, which puts a loose ball on it, or pinch to pick up and stack three wooden blocks.
- **Who is in control**: the glove operator or the fly reflex, with one sentence saying why.
- **The fly's looming circuit**: 210 looming detectors in the two optic lobes and 245 descending neurons, drawn at
  their positions in the brain, lighting up as objects approach.
- **A switch** to turn the reflex off and on... and a score for each: balls thrown, balls that hit the arm, loads dropped.

Press `T` to throw a ball at the arm, `1` and `3` to switch the reflex off and on, `C` to recentre on the hand.

## How it works

```
glove ──► hand pose ──► target for the arm ──┐
                                             ├──► arm (inverse kinematics, MuJoCo)
objects in the air ──► eye ──► fly circuit ──► reflex ──┘
```

- **Eye.** Each ball is reduced to what an eye can measure: its bearing and its apparent size, 60 times a second,
  one frame late, with noise. (No distances, velocities or 3D tracking)
- **Fly circuit.** A ball excites the looming detectors (LPLC2) looking its way, in proportion to how soon it will
  arrive... apparent size divided by its growth rate. Band filters on the graph Laplacians of the circuit's wiring,
  one per neurotransmitter, carry that activity to the descending neurons. A linear readout of those neurons gives
  the danger of staying put and of stepping in each of 16 directions.
- **Reflex.** The arm eases toward the safest direction along a smooth path. While a load is carried, the tray is
  held level, follows the hand along an acceleration-limited path and leans into every movement so the load stays on.

After fitting the fly circuit is two small matrices... ~ 55,000 numbers. This makes one decision take about a
millisecond on a laptop CPU. Perfect for edge AI.

## Results

In simulation, with a scripted operator. Each cell is the mean over 5 runs of 62 seconds, with the lowest and highest
run in brackets. "Reflex on" is the fly circuit working from eye-only input. Reproduce with `python bench.py`.

| Situation | Measure | Reflex off | Reflex on |
|---|---|---|---|
| Single balls thrown at a still arm | hit the arm, of 20 | 20 (20–20) | 4.6 (3–6) |
| Volleys of three balls at a still arm | hit the arm, of 45 | 45 (45–45) | 13.8 (13–15) |
| Unsteady operator carrying a load, no balls | load drops | 35.8 (30–38) | 0 (0–0) |
| Unsteady operator carrying a load, single balls | load drops | 32.4 (31–33) | 6.4 (6–8) |
| | hit the arm, of 20 | 18.2 (18–19) | 13.6 (11–15) |

The ball runs differ in where the balls come from and in the eye's sensor noise. The no-balls carrying runs have
nothing random in them, so they differ in how unsteady the wrist is (0.10 to 0.30 radians of rocking).

The reflex is strongest at steadying a load and at clearing single objects from rest. It is weakest when both
happen at once: with the operator moving and a load on the tray, about two balls in three still touch the arm,
because the dodge is kept gentle enough not to throw the load.

## Limits

- **The circuit does the sensing; the movements are hand-written control.** Choosing the direction from the danger
  map, moving smoothly and steadying the load involve no connectome.
- **It does not beat geometry.** A standard method given the same eye-only input (recover distance from apparent
  size, fit a ballistic flight) does as well in clean conditions and better with sensor noise.
- **The wiring does not improve the avoidance.** Fitted on the real wiring, on rewired wiring and with no brain at
  all, the model chooses directions equally well. In a separate test the real wiring does carry information: a
  model given it recovers a third of the activity evoked by a stimulus it never saw, and none with rewired wiring.
- **Where each looming detector looks is assumed**, from the detectors' positions in the optic lobe. The neural
  activity comes from a rate simulation on the connectome, not from recordings of a fly.
- **Everything is simulated**, and the eye is told which ball is which.

## Install

Python 3.12 or later, on Linux.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# The arm (Franka Panda, from MuJoCo Menagerie)
mkdir -p third_party && cd third_party
git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
cd mujoco_menagerie && git sparse-checkout set franka_emika_panda && cd ../..
```

The fitted fly models are in `models/`, so the connectome data is not needed to run the demo.

For the glove: Rokoko's Device SDK, installed and activated from your Rokoko ID page, a Smartglove II on USB and a
Coil Pro.

## Run

```
.venv/bin/python demo.py --no-glove    # scripted operator, no hardware
.venv/bin/python demo.py               # with the glove
```

With the glove, `demo.py` waits for it to be plugged in, starts Rokoko's driver, checks the saved calibration with
one forward push (running the 30-second calibration if there is none), then opens the page. The glove can be
unplugged and replugged while it runs.

Other entry points:

```
.venv/bin/python run_teleop.py --help     # every scene, input and reflex mode, with or without the page
.venv/bin/python bench.py                 # the results table over several seeds
.venv/bin/python -m unittest              # the page server's tests
```

## Refitting the fly model

This needs the FlyWire data (about 900 MB) in `data/flywire/`, plus `torch`:

```
cd data/flywire
curl -L -o proofread_root_ids_783.npy https://zenodo.org/api/records/10676866/files/proofread_root_ids_783.npy/content
curl -L -o proofread_connections_783.feather https://zenodo.org/api/records/10676866/files/proofread_connections_783.feather/content
curl -L -o neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
cd ../..
.venv/bin/python -m brain.threat --eye --compare
```

## Layout

| Path | What is in it |
|---|---|
| `demo.py` | One-command start |
| `run_teleop.py` | The teleoperation loop |
| `mocap/` | Glove reader, calibration, stream recorder |
| `teleop/` | Hand pose to arm target |
| `sim/` | Scene, inverse kinematics, sensing, the eye |
| `reflex/` | Avoidance and steadying |
| `brain/` | Circuit extraction, the fly threat model, wiring experiments |
| `view/` | The page and its server |
| `models/` | Fitted fly models and the neuron layout |
| `PLAN.md` | Working notes: what was planned, what changed, what is still untested |

## Sources

Connectome and biology

- Dorkenwald, S. et al. (2024). Neuronal wiring diagram of an adult brain. *Nature* 634, 124–138.
  [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y). The FlyWire connectome.
- Schlegel, P. et al. (2024). Whole-brain annotation and multi-connectome cell typing of *Drosophila*. *Nature* 634,
  139–152. [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5). The neuron annotations
  (cell types, sides, positions), from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- Eckstein, N. et al. (2024). Neurotransmitter classification from electron microscopy images at synaptic sites in
  *Drosophila melanogaster*. *Cell* 187, 2574–2594. [doi:10.1016/j.cell.2024.03.016](https://doi.org/10.1016/j.cell.2024.03.016).
  The per-connection neurotransmitter probabilities that split the wiring into relations.
- Klapoetke, N. C. et al. (2017). Ultra-selective looming detection from radial motion opponency. *Nature* 551,
  237–241. [doi:10.1038/nature24626](https://doi.org/10.1038/nature24626). LPLC2 as looming detectors that feed
  the giant fibre escape pathway.
- Data: FlyWire whole-brain connectome connectivity data, release 783,
  [Zenodo record 10676866](https://zenodo.org/records/10676866), CC BY 4.0.

Simulation and hardware

- Todorov, E., Erez, T. and Tassa, Y. (2012). MuJoCo: A physics engine for model-based control. *IEEE/RSJ IROS*,
  5026–5033. [doi:10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109).
- Zakka, K., Tassa, Y. and MuJoCo Menagerie Contributors (2022). MuJoCo Menagerie: A collection of high-quality
  simulation models for MuJoCo. [github.com/google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie).
  The Franka Emika Panda arm model.
- Rokoko Smartgloves II, Coil Pro and Device SDK, [sdk.rokoko.com](https://sdk.rokoko.com/).

Method

- Occhetta, M., Mudaliar, M., Myatt, A. and Bessant, C. (2025). MORGaN: self-supervised multi-relational graph
  learning for druggable gene discovery. bioRxiv. [doi:10.1101/2025.09.10.675402](https://doi.org/10.1101/2025.09.10.675402).
  The masked-reconstruction test of the wiring, over several relation types, follows this idea.

MIT licence.
