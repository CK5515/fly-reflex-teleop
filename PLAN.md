# Plan

Junction 2026, Rokoko teleoperation track, solo. All code is written at the event; the brain model starts from
scratch from public connectome data.

Times assume a Sunday-morning submission deadline. Confirm on the Junction platform and shift the gates if needed.

## The demo

You wear a Rokoko glove and steer a simulated arm. Something flies at the arm or touches it, and the arm reacts by
reflex before you can. A live view shows which fly-brain expert took over.

Pitch: teleoperation has latency and operators make mistakes. Biology handles this with reflexes underneath
voluntary control, so this is shared autonomy with a real connectome as the reflex layer.

## Scope

In:

- Mocap input: one Smartglove (plus Coil Pro for wrist position if available) through the Rokoko Device SDK.
- Simulation: MuJoCo with an open-source arm and gripper model.
- Reflex interface: scene events in, expert name plus override command out.
- Two reflexes: a looming object triggers escape (withdrawal); contact on the arm triggers a grooming-style brush-off.
- Live view: a web page showing who is in control and the expert activations.

Out: the physical G1, the full-body suit, a dexterous five-finger hand, training a policy from demonstrations.

## Order of work

The hand-written reflex goes in before the brain model, so the demo is complete before the model is touched. The
model replaces it behind the same interface only if it passes the Saturday 22:00 gate.

| When | Work | Gate |
|---|---|---|
| Fri, to ~01:00 | Install the SDK and MuJoCo. Stream example printing glove data. Record 2-3 clips to file. Load the arm model. Start the connectome download. | A recorded clip replays from file |
| Sat 08-11 | Wrist pose to arm through inverse kinematics; finger curl to gripper. Replay first, then live. | You pick up an object in sim, live |
| Sat 11-13 | Scene events (thrown object, contact), the reflex interface, the hand-written reflexes, the blend between operator and reflex. | Full demo works with the hand-written reflex |
| Sat 13-17 | Brain pipeline, small: top few hundred spectral modes, two experts, a synthetic training task generated from the sim's own events. | Parallel sweep of 6-8 configurations launched on Verda by 17:00 |
| Sat 17-19 | Live view. | Page shows control state and expert activations |
| Sat 19-20 | Record a backup demo video. Book a glove slot for Sunday. | Video on disk |
| Sat 20-24 | Plug in the brain model. | Go/no-go on the model at 22:00 |
| Sun, to deadline minus 2 h | Bug fixes only. Then freeze, record the 2-minute pitch video, finish the README, submit. | Submitted an hour early |

## Cut order if behind

1. The brain model (ship the hand-written reflex and say so in the pitch).
2. The grooming reflex.
3. Coil Pro wrist tracking (use glove orientation only).
4. Live-view polish.

## Risks

- Glove time is shared. Record clips the first time a glove is on, and develop against the replay.
- The brain model is the part most likely to slip. Verda credits (EUR 200) buy parallel runs, not writing time.
- The earlier research repo must stay closed while the model is written.

## Status, Friday night

Done ahead of schedule (all in simulation; nothing has been tried with a physical glove):

- Arm follows a hand pose from a script, the mouse, the SDK's emulated glove, or Rokoko's recorded two-glove sessions
  (`data/rokoko_factory_sim/*.rkkrdd`, played with `rokoko-sdk --playback`).
- Thrown ball, touch sensing, hand-written escape and grooming reflexes, live view (`--view`).
- Fly-brain reflex (`--reflex brain`): a 1,340-neuron circuit cut from FlyWire, four neurotransmitter relations, band
  experts with a learned router, trained on the Verda instance. The spectral bases must be copied from the machine
  that trained the model (`data/brain/<circuit>/`): recomputing them elsewhere flips eigenvector signs.

Open, in priority order:

1. Calibrate glove axes to the simulation and limit the target to the arm's workspace (recorded hand drives the
   target into the table at present).
2. Glove time: record our own clips, check Coil Pro position, tune pinch thresholds.
3. Show the brain's sensors, decision and experts in the live view.
4. Brain claims: the classification task is too easy to tell real wiring from rewired wiring (all reach ~100%).
   Needs a harder test before any claim that the wiring matters. The simulated rates are also very low.
5. Backup demo video, then pitch video and README.
