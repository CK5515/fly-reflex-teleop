# Fly Reflex Teleop

Junction 2026, Rokoko teleoperation track.

A motion-capture glove tells a robot what the operator does, but nothing tells the operator what the robot feels.
They cannot feel a load sliding, and cannot react to an object that arrives in half a second.

This project puts a reflex layer under the operator. A person steers a simulated robot arm with a Rokoko glove.
Underneath, the arm keeps what it carries steady and steps out of the way of incoming objects, then hands control
straight back. The incoming objects are sensed through the fruit fly's looming circuit, taken from the FlyWire
connectome of *Drosophila*.

## What you see

`python demo.py` opens one page:

- **The arm**, seen from above its shoulder as if it were your own. It starts empty. Pinch on the paddle's handle to
  pick it up, which puts a loose ball on it, or pinch to pick up and stack three wooden blocks.
- **Who is in control**: the glove operator or the fly reflex, with one sentence saying why.
- **The fly's looming circuit**: 210 looming detectors in the two optic lobes and 245 descending neurons, drawn at
  their positions in the brain, lighting up as objects approach.
- **A switch** to turn the reflex off and on, and a score for each: balls thrown, balls that hit the arm, loads dropped.

Press `T` to throw a ball at the arm, `1` and `3` to switch the reflex off and on, `C` to recentre on the hand.

## How it works

```
glove ──► hand pose ──► target for the arm ──┐
                                             ├──► arm (inverse kinematics, MuJoCo)
objects in the air ──► eye ──► fly circuit ──► reflex ──┘
```

- **Eye.** Each ball is reduced to what an eye can measure: its bearing and its apparent size, 60 times a second,
  one frame late, with noise. No distances, no velocities, no 3D tracking.
- **Fly circuit.** A ball excites the looming detectors (LPLC2) looking its way, in proportion to how soon it will
  arrive: apparent size divided by its growth rate. Band filters on the graph Laplacians of the circuit's wiring,
  one per neurotransmitter, carry that activity to the descending neurons. A linear readout of those neurons gives
  the danger of staying put and of stepping in each of 16 directions.
- **Reflex.** The arm eases toward the safest direction along a smooth path. While a load is carried, the tray is
  held level, follows the hand along an acceleration-limited path, and leans into every movement so the load stays on.

After fitting, the fly circuit is two small matrices, about 55,000 numbers, and one decision takes about a
millisecond on a laptop CPU.

## Results

In simulation, with a scripted operator. One 62-second run per cell so far; `python bench.py` repeats them over seeds.

| Situation | Reflex off | Reflex on |
|---|---|---|
| Balls thrown at the arm: how many hit, of 20 | 20 | 3 |
| Unsteady operator carrying a load: drops in a minute | 36 | 0 |
| Volleys of three balls: how many hit, of 45 | 45 | 18 |

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

- Connectome: FlyWire, release 783 (Zenodo record 10676866), and the FlyWire neuron annotations.
- Arm model: Franka Emika Panda from MuJoCo Menagerie. Simulator: MuJoCo.
- Motion capture: Rokoko Smartgloves II, Coil Pro and Device SDK.
- The masked-reconstruction wiring test follows the idea of MORGaN (Occhetta et al., 2025).

MIT licence.
