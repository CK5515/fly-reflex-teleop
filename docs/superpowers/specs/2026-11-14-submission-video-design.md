# Submission video — design

Date: 2026-11-14. Junction 2026, Rokoko track.

## Intent

A 2:00 video that the Rokoko-track judges watch to decide whether Fly Reflex is impressive. The main message is
**"the careful robot"**: when you steer a robot with a glove you can see what it does but can't feel it, so the arm
looks after itself. It keeps the load steady and steps out of the way of things thrown at it, then hands control
straight back. The fruit fly comes in halfway through, as the answer to how the arm notices things coming. It is not
the headline.

Why lead with this: Rokoko's brief names precise manipulation of fragile objects first, and the project's strongest
result is load drops falling from 36 to 0.

## Goals

- Within the first 10 s, a judge understands the problem: you can't feel what the robot feels.
- Show the real glove driving the real sim. The judges need to see a hand and the arm moving together.
- Show both reflexes with the switch flipped off and on, each with one number.
- Reveal the fly, with a low-poly 3D scene and the live looming-circuit panel.
- Look sleek and on brand, matching the demo page (see Brand).

## Non-goals

- Explaining the maths (graph Laplacians, band filters, readouts).
- Covering every result in the README table.

## Anti-goals: what would make this a failure even if it's polished

- **Overclaiming.** The README states that the wiring does not improve avoidance compared with rewired wiring or no
  brain at all, that geometry does as well or better, and that the movements are hand-written control. The voiceover
  must not say or imply that the connectome makes the robot dodge better, or that this runs on a real robot. Judges
  may read the README after watching.
- Footage of the desktop, terminal or OBS on screen.
- The venue audio. All five recordings have audio that is near full volume the whole time and unusable.
- Stock footage of unrelated robots that could be mistaken for this project's own.

## Constraints

- 2:00 or less, 1920×1080, 30 fps, H.264 MP4 (exact submission format still to be confirmed against the Junction rules).
- No Blender on this machine and the NVIDIA driver isn't loaded, so local 3D is CPU-rendered three.js (hyperframes).
  Verda GPUs are available for a Blender/Cycles version of a hero shot if wanted.
- Downloaded clips and music must be licensed for reuse (Pexels/Pixabay licence, CC0 or CC-BY with a credit on the
  end card).
- Existing footage: `~/2026-11-14 12-37-11.mp4`, about 200–330 s in, is live glove-driven sim: paddle with ball, the
  orange "Fly reflex" handover, block stacking. `~/2026-11-14 12-28-52.mp4` is a webcam shot of the user at the venue.
  The rest is desktop and terminal.

## Brand (taken from `view/index.html`)

| Token | Value | Use |
|---|---|---|
| page | `#0c0c0c` | backgrounds, placeholders |
| surface / line | `#151515` / `#2a2a2a` | panels, rules |
| text / muted | `#f2f2f2` / `#b8b8b8` | type |
| glove | `#ffffff` | everything about the operator |
| fly-body | `#e9a23b` | everything about the reflex (matches the orange handover frame) |
| fly-eye | `#ff3b30` | threats and incoming objects, sparingly |

Type: Inter for headings and captions, monospace for numbers and labels. Low-poly 3D uses flat shading in the same
palette: a white arm, amber for the reflex, red for incoming objects, on near-black with a soft floor grid like the
demo's.

## Approaches considered

1. **hyperframes for the whole edit, the real sim for the robot shots, Verda optional (chosen).** Titles, overlays,
   placeholders and the low-poly scenes are HTML/three.js compositions rendered by hyperframes. The dodge and
   load-drop shots are rendered from `run_teleop.py`, so they are the real system, not an illustration. Quick to
   change, and everything uses one design language.
2. **Blender on Verda for all the 3D.** Higher fidelity and nicer lighting, but slow to iterate and it needs a separate
   asset pipeline. Kept as an upgrade for the cold-open shot only.
3. **ffmpeg cut of existing footage plus stock clips.** Fastest, but it can't produce the low-poly scenes the user asked
   for, and it looks generic.

The user picked "careful robot" as the story and said Verda is available. The tool choice above is Claude's call.

## Storyboard (2:00)

`FULL` = full-screen shot. `PIP` = the user's hand in a corner inset (about 30% width, bottom-right, rounded, thin
white border) over sim footage, so the arm is the focus. `[REC-n]` = the user films it (see the shot list). Until it's
filmed, each `[REC-n]` is a black `#0c0c0c` card in the brand type that says what the shot is and whether it's FULL or PIP.

| # | Time | Picture | Voiceover |
|---|---|---|---|
| 1 | 0:00–0:09 | **3D, low-poly.** A dim warehouse. A low-poly arm carries a tray with a rack of sample vials and a beaker of water. The tray tilts, the water climbs the side of the beaker, a loose vial slides to the open edge and tips, and the image freezes on it. (Only the real sim footage has a ball; the 3D scenes carry everyday fragile loads to make judges think of applications.) | "When you steer a robot with your hands, you can see everything it does. You can't feel any of it." |
| 2 | 0:09–0:20 | `[REC-1]` FULL: close-up of the gloved hand reaching and pinching, then cut to **sim, reflex off**: the unsteady carry, the ball sliding off the paddle. Title chip "Reflex off" (white). | "You can't feel a load start to slide. You can't react to something that arrives in half a second." |
| 3 | 0:20–0:26 | **Title card.** "Fly Reflex", then the subtitle "a reflex layer under the operator". Amber underline draws in. | "So I gave the robot a reflex." |
| 4 | 0:26–0:46 | New screen capture of the paddle and ball, recorded at the same time as `[REC-2]` and shown with it as a PIP. Then a block-stacking capture with `[REC-3]` as a PIP. (The rough cut uses footage from 12-37-11 in its place; that footage has no matching hand video.) Captions: "Rokoko Smartglove II + Coil Pro", "Franka arm · MuJoCo". | "I steer a simulated robot arm with a Rokoko Smartglove and Coil Pro. Pinch the handle and it picks up the paddle. Underneath my hand, the arm holds the tray level and leans into every move, so the ball stays on." |
| 5 | 0:46–1:00 | **Split screen, sim renders.** The same unsteady operator, reflex off on the left and on on the right. The ball falls on the left and stays on the right. Counter: **load drops 36 → 0** (amber). | "Same shaky hand, reflex off and on. Over a minute of carrying, the load fell about thirty-six times. With the reflex, zero." |
| 6 | 1:00–1:20 | **3D, low-poly.** A fruit fly on a surface. In slow motion, a low-poly human hand swings down to swat it; the fly's red eyes heat up as the hand looms, and it leaps clear just before the hand lands. Cut to the demo's looming-circuit panel (from screen capture), zoomed in, lighting up as a ball approaches. Captions: "FlyWire connectome", "210 looming detectors · 245 descending neurons". | "To notice things coming, it borrows from a fruit fly. A fly escapes because a small circuit in its eyes reacts to things that grow fast. I took that circuit from the FlyWire map of the fly's brain, and gave it only what an eye can see: where something is, and how fast it's growing." |
| 7 | 1:20–1:40 | `[REC-4]` FULL split: the hand held still on the left, the sim on the right. A ball is thrown, the arm steps aside, the panel flips to "Fly reflex" (amber frame), then back to "Glove operator". Then a quick off/on comparison render and the counter **hits 20 of 20 → under 5** (mean 4.6). | "My hand doesn't move. The arm steps aside, and hands control straight back. Out of twenty balls, all twenty hit without the reflex. With it, fewer than five. Each decision takes about a millisecond on a laptop." |
| 8 | 1:40–1:52 | `[REC-5]` FULL wide shot of the user at the venue with the glove (or the webcam clip as a fallback). Then a text card of honest limits in muted type. | "It's all in simulation, and the fly does the sensing, not the steering. But the idea carries over: a reflex layer under the operator, so the person can stay in charge without having to feel everything." |
| 8b | 1:47–1:53.5 | **3D, low-poly applications.** Three trays held level side by side, each with an amber level line: a rack of lab vials, a brim-full cup of coffee, a carton of eggs. Heading "The idea carries over — a reflex layer under any operator"; labels Lab samples · Liquids · Eggs. (REC-5 shortens to 1:40–1:43.5 and the limits card to 1:43.5–1:47.) | (continues: "…a reflex layer under the operator, so the person can stay in charge…") |
| 9 | 1:53.5–2:00 | **End card.** Fly Reflex · Junction 2026 · Rokoko track. The author's name, the repo link, and credits (FlyWire, MuJoCo Menagerie, music, stock). | (music only) |

The voiceover is about 230 words, around 120 words a minute, which leaves room for the music.

**Music:** one licensed, minimal electronic track (Pixabay Music or the YouTube Audio Library). It sits under the
voice at about −18 dB, swells on the title card and the end card, and the master is normalized to −16 LUFS.

**Stock:** optional. One licensed macro clip of a real fruit fly (Pexels, Pixabay or Wikimedia CC) can be cut for 1 s
before the low-poly fly in beat 6. Nothing else.

## What the user records

Recording setup for every shot:
- Phone, landscape, 1080p at 60 fps, with the exposure and focus locked (long-press on the screen).
- A plain dark background with the glove lit from the side.
- No laptop screen in the frame, unless the shot says so.

To sync the phone with the sim, run OBS at 1080p60 on the browser in full screen (F11) at the same time. Start each
take by opening the hand wide and then snapping into a pinch, which gives the same sync point on both recordings.
Record two or three takes of each shot.

| ID | Shot | Framing | Use | Length | What to do |
|---|---|---|---|---|---|
| REC-1 | Hero hand | Side view at table height, tight on the glove and Coil Pro | FULL | 6–8 s | Hand rests open. Slowly reach forward, pinch firmly as if taking a handle, lift a little and hold. Move slowly; it should look deliberate. |
| REC-2 | Paddle carry | Top-down or over the shoulder on the hand only | PIP (sync) | 25 s | Reflex on. Pinch to pick up the paddle (the ball appears), carry it left and right with a slight wobble, put it down. |
| REC-3 | Block stack | Same as REC-2 | PIP (sync) | 20 s | Run `--blocks 3`. Pinch, lift and stack three blocks. Keep it calm and precise. |
| REC-4 | Still hand, ball dodge | Side view, hand in the left half of the frame | FULL split (sync) | 20 s | Hold the hand completely still in a light pinch. Someone else presses `T` 3–4 times, or use `--throw-every 4`. Do not flinch. The point is that the hand stays still while the arm moves. |
| REC-5 | Establishing | Wide shot of you at the venue desk with the glove on, the screen visible behind you | FULL | 4–6 s | Look at the screen and make a pinch or two. Have a friend hold the phone. |
| VO | Voiceover | — | — | — | In a quiet room, phone about 20 cm from the mouth, with a soft surface behind it (a wardrobe of clothes works). One take per row of the script, read twice. Not at the venue. |

## What Claude makes

- Sim renders at 1080p with a fixed seed, from the page in full screen: reflex off and on with an unsteady carry
  (`--load tray --wobble 0.2`), a single throw with reflex off and on, and the circuit panel lighting up. Captured with
  a headless browser or OBS, whichever gives clean frames.
- Low-poly scenes 1 (the warehouse carry) and 6 (the fly escape) as hyperframes three.js compositions.
- The title card, captions, number counters, split-screen and PIP layouts, the limits card, the end card, and the
  placeholder cards.
- The assembled rough cut, with placeholders and a scratch voiceover from text-to-speech for timing, then the final
  cut once the user's recordings arrive.

Location: a separate folder, `~/WEIRD PROJECTS/fly-reflex-video/`, so the repo stays clean. The source footage is not
copied into the repo.

## Calls Claude made (check these)

- 1080p30 output, and the 0:00–2:00 timing of each beat.
- The PIP position and size, and that REC-1, REC-4 and REC-5 are full screen.
- Numbers on screen: load drops 36 → 0 (rounded from the mean 35.8; the README's "35.8 (30–38)" goes on the limits
  card), and hits 20 → 4.6.
- The low-poly fly scene, and the use of a stock fly clip at all (optional).
- Music style and levels.
- Keeping the video project outside the repo.

## Still open

- Junction's submission rules: maximum length, format and where it's uploaded.
- Whether a Verda/Blender version of scene 1 is worth doing after the rough cut.
- The name to put on the end card.
