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

## Status, Saturday morning

The demo has changed shape since Friday. The Friday sections above describe the first design (escape and grooming
experts, a spectral mixture-of-experts classifier) and are kept as history.

What exists now, started with `python demo.py`:

- A Rokoko glove steers the simulated arm, seen from above its shoulder. The arm starts empty. Pinch on the paddle's
  handle to pick it up (a loose ball appears on it), or pinch to pick up and stack three wooden blocks.
- The fly reflex has two parts. Threat sensing runs through the fly's looming circuit (210 LPLC2 detectors, 245
  descending neurons, FlyWire wiring) from eye-only input: each ball's bearing and apparent size. Steadying a
  carried load and the avoiding movement itself are hand-written control.
- The page shows the arm, who is in control (glove operator or fly reflex), the neurons lighting up at their real
  positions, an off/on switch and a score per mode.

Measured so far, one seeded 62-second scripted run each: thrown balls hit the arm 20 of 20 times with the reflex off
and 3 of 20 with it on; an unsteady operator dropped a carried load 36 times with it off and 0 with it on.

Not shown: the fly circuit does not beat a geometric baseline (it ties in clean conditions and loses with sensor
noise), and real wiring, rewired wiring and no brain choose avoidance directions equally well.

## Next steps, before hosting a demo

In this order. Hosting comes last, and only if time is left.

1. Run `python demo.py` with the glove on. Nothing since the calibration has been tried by hand: the grab scene, the
   doubled hand-to-arm scale, the tamed wrist rotation, the stronger steadying, the launcher's glove path.
2. Record a backup video with `python demo.py --no-glove` before changing anything else.
3. README: what it is, how to install (including fetching the Panda arm model into `third_party/`), how to run, the
   results with their limits.
4. Merge `teleop-sim-and-live-view` into `main`. The submission's source link points at `main`, which is empty.
5. Pitch video from the two-minute script, filmed with the glove, numbers matched to what is filmed.
6. Submission form text (drafted in the session: name, punchline, description).

Hosting, if attempted. The live demo needs the glove and a local simulator, so only the scripted operator can be
hosted. Before putting it on a public address:

- The page's server and MuJoCo's viewer must listen on the public interface (the page's server binds to localhost now).
- `/throw`, `/mode` and `/recentre` take requests from anyone, and MuJoCo's viewer warns that it has no
  authentication and exposes logs and file paths. Decide whether that is acceptable or put it behind a password.
- There is one simulation: every visitor shares it and sees the others' throws.
- It must restart itself if it stops, and be watched for memory growth over hours.
- A CPU-only machine is enough (rendering happens in the visitor's browser); check the cost against the Verda credits.

The alternative that costs nothing is to leave the form's demo field empty and link the video.

## Rigorous tests still to perform

Every number so far is a single seeded run with a scripted operator. Before quoting any of them in the pitch:

Outcomes
- Repeat each headline case over at least 5 seeds and report the mean and the range: thrown balls with and without
  the reflex; an unsteady operator carrying a load with and without it; volleys of three.
- Measure one combined case for the pitch (carrying a load while balls are thrown, reflex off and on), so the spoken
  numbers come from a single scenario and not two.
- Repeat the eye-only comparison (geometry against fly circuit, three noise levels) over seeds. One run per cell
  cannot support "ties" or "loses".
- Count how often the arm's solver restarts per minute in each case; each restart is a visible lurch.

With the glove
- Direction check after calibration: forward, left, up and finger tilt each move the arm the expected way.
- Delay from hand to arm, measured from the logs, and whether it is noticeable.
- Behaviour near the edge of the Coil Pro's range, and after unplugging and replugging the glove mid-run.
- Grab scene by hand: how often a pinch on the handle picks the paddle up, and whether three blocks can be stacked.
- Reflex off against on with a real unsteady hand: drops per minute, the same measure as the scripted runs.

The brain claims
- The reconstruction test (real wiring recovers activity from a stimulus never seen; rewired wiring does not) used
  2 seeds, 4 for rewired. Repeat with more seeds and more rewirings.
- The assumed layout of where each looming detector looks: either look up the real mapping or state the assumption.

The software
- Fresh clone on a clean machine: install, `python demo.py --no-glove`, page and brain panel appear.
- The page in a real browser (Chrome and Firefox), including the 3D frame, which has only been seen in screenshots
  that could not render it.
- A fifteen-minute unattended run watching memory; the laptop has 7 GB and has run out once.
- Automated checks cover only the page's server. Add one that starts each reflex mode and each scene for a few
  seconds and fails on any error.
